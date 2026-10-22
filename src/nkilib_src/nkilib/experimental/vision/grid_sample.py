# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grid sample (torch.nn.functional.grid_sample, 4-D input) on trn2, by batched indirect DMA gathers."""

import re

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import get_verified_program_sharding_info


def nki_at_least(*version: int) -> bool:
    """Whether the installed nki is at least `version` (major, minor, patch); an unparsable version passes."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", getattr(nki, "__version__", ""))
    return m is None or tuple(int(g) for g in m.groups()) >= version


# nki 0.6.0 rejects indirect DMA gathers with more than one index per partition in MLIR
# verification, and every gather here has more; nki 0.7.0 (Neuron SDK 2.33) accepts them.
GRID_SAMPLE_NKI_SUPPORTED = nki_at_least(0, 7, 0)

_PMAX = 128
_MODES = ("bilinear", "nearest")
_PADDING_MODES = ("zeros", "border", "reflection")
_MAX_COLS = 64  # output pixels per partition of a quad chunk spanning batches: at most 8192 rows per gather
# ... and of every other chunk: smaller gathers keep more chunks in flight, and a one-batch quad chunk's gather index
# comes from the sample layout (see grid_sample_sample_layout)
_MAX_COLS_DEEP = 32
_GATHER_BYTES = 24 * 1024  # SBUF bytes per partition of one chunk's gathered rows
_SCRATCH_ROW_ELEMS = 8192  # elements per partition of one scratch-build tile
_MAX_EXACT = 1 << 24  # indices and pixel counters held in fp32 stay exact below this
_MAX_ROWS = 1 << 32  # scratch rows per batch of an exact plan (uint32 gather indices), at most
_MAX_PIXELS = 1 << 31  # output pixels per batch (int32 chunk counters), below this
_MAX_SCRATCH_BYTES = 1 << 31  # quad scratch cap (the plain layout is 4x smaller)
_TRN2_SBUF_BYTES = 212984  # SBUF bytes per partition a trn2 kernel may allocate
_SBUF_SHARE = 0.9  # share of SBUF the estimate (grid_sample_sbuf_bytes) may fill
_BUILD_UNROLL = 128  # scratch-build tiles that are unrolled (more: a device loop over them)
# a batch's image plus its scratch, bytes, at most, for the batch-per-partition build (see _build_bp)
_BP_BYTES = 24 * 1024
_MAX_SEG_PIX = 1 << 21  # output pixels of a multi-batch segment: floor((pixel + 0.5) / (Ho Wo)) stays exact in fp32
_MAX_SEGMENTS = 64  # most single-batch segments a call may unroll (a poor divisor of Ho * Wo, see _plan_layout)
# quad scratch only while a batch's padded input (bytes) is at most this per output pixel: building the quad scratch
# writes 4x the input, which pays off through the 3 gathered rows per sample it saves
_QUAD_BYTES_PER_SAMPLE = 256
_PSUM_BANK_BYTES = 2048  # PSUM bytes per partition of one bank: a group of channel-major store transposes
_TSTORE_TILE_BYTES = 8192  # SBUF bytes per partition of one buffer set's channel-major store tile, at most
_MAX_STAGE_BYTES = 1 << 30  # channel-major staging buffer of a multi-batch segment, at most
_MAX_RUN_BYTES = 32768  # contiguous bytes of one staging-copy DMA descriptor, at most
_BUILD_SETS = 2  # tile sets of the pipelined scratch build (_build_pipelined)
_MAX_DEPTH = 8  # chunks in flight in the sampling pipeline (one tile set each), at most; fewer when SBUF runs short
_UNROLL_CHUNKS = 512  # chunks of a call that run unrolled; beyond, each segment loops (see _sample_segment)
_LOOP_ROUNDS = 8  # rounds of `depth` chunks per device-loop iteration
_MIN_LOOP_ITERS = 3  # iterations a segment's device loop runs at least (else fewer rounds per iteration)
_BLEND4_ELEMS = 512  # quad chunks of at most this many output elements per partition blend in 4 instructions


class GridSamplePlan(nl.NKIObject):
    """Scratch layout and chunking of one grid_sample call (see plan_grid_sample)."""

    def __init__(
        self,
        quad,
        hs,
        ws,
        row,
        taps,
        cols,
        seg,
        bk,
        segments,
        tstore,
        direct,
        pipelined,
        depth=2,
        gp=1,
        bp=False,
        exact=False,
    ):
        self.quad = quad  # True: one scratch row per bilinear sample (its 2 x 2 neighbourhood, 4 C elements)
        # True: no scratch, the gathers read the input itself (one channel: its rows are already channels-last)
        self.direct = direct
        self.hs = hs  # scratch rows per batch (padded image rows, or quad rows)
        self.ws = ws  # scratch columns
        self.row = row  # elements per scratch entry (4 C for quad, C otherwise)
        self.taps = taps  # gathers per chunk: 1 (quad or nearest) or 4 (plain bilinear)
        self.cols = cols  # output pixels per partition of a chunk, at most (a segment may use fewer)
        self.seg = seg  # scratch-build tile width in scratch columns
        self.bk = bk  # batches per scratch-build tile (more than 1 only when a batch's scratch fits one tile)
        # groups of bk batches stacked along a build tile's partitions (group g on partitions g * hs ..): more than 1
        # when a batch's scratch rows fill at most half the partitions
        self.gp = gp
        # True: small images, the scratch built one batch per partition (_build_bp)
        self.bp = bp
        self.bp_chw = 0  # C * H * W of a bp plan (its image tile), set by plan_grid_sample
        # [(first batch, end batch)]: one batch, or several whose output goes through a channel-major staging buffer
        self.segments = segments
        # True: chunks are transposed to channel-major on the Tensor Engine before the store (see _store_channel_major)
        self.tstore = tstore
        # True: the scratch build runs software-pipelined over _BUILD_SETS tile sets (see _build_pipelined)
        self.pipelined = pipelined
        # chunks in flight in the sampling pipeline (tile sets), 2 .. _MAX_DEPTH (see _sample_segment)
        self.depth = depth
        # True: 2^24 scratch rows per batch or more (6K images and up), past fp32's exact integers: stage A adds each
        # row base and column in integer arithmetic, ws rounded up so that the row bases stay exact in fp32 (see
        # grid_sample_scratch_width and _exact_rows)
        self.exact = exact
        # True: quad chunks multiply all four taps by their weights in one instruction (see _stage_b)
        self.blend4 = quad and cols * row <= 4 * _BLEND4_ELEMS


def grid_sample_chunks(S: int, cols: int):
    """
    Chunking of a segment of S output pixels at up to cols pixels per partition: (partitions, pixels per partition,
    full chunks, tail start or -1). A batched gather (several indices per partition) needs all 128 partitions; the
    last chunk ends at the segment's end and may overlap the one before it.
    """
    seg_cols = max(1, min(cols, S // _PMAX))
    parts = min(_PMAX, S // seg_cols)
    chunk = parts * seg_cols
    n_full = S // chunk
    tail = -1
    if S % chunk > 0:
        tail = S - chunk
    return parts, seg_cols, n_full, tail


def grid_sample_work(plan: GridSamplePlan, T: int, n_prgs: int, prg_id: int):
    """
    The share of program prg_id (of n_prgs) in each segment of plan, for T output pixels per batch: [(first batch,
    end batch, partitions per chunk, pixels per partition, full chunks, tail start or -1, first chunk, chunk count)].
    The programs must share one control flow, so they take the same number of batches of a multi-batch segment (the
    last range ends at the segment's end and may overlap the one before it, those batches then written twice with
    the same values), or the same number of a single batch's chunks, consecutively.
    """
    work = []
    for si in range(len(plan.segments)):
        n0, n1 = plan.segments[si]
        split_batches = n_prgs > 1 and n1 - n0 > 1
        if split_batches:
            nbp = (n1 - n0 + n_prgs - 1) // n_prgs
            n0 = min(n0 + nbp * prg_id, n1 - nbp)
            n1 = n0 + nbp
        parts, cols, n_full, tail = grid_sample_chunks((n1 - n0) * T, plan.cols)
        n_chunks = n_full
        if tail >= 0:
            n_chunks = n_full + 1
        first = 0
        count = n_chunks
        if n_prgs > 1 and not split_batches:
            count = (n_chunks + n_prgs - 1) // n_prgs
            first = min(count * prg_id, n_chunks - count)
        work.append((n0, n1, parts, cols, n_full, tail, first, count))
    return work


def grid_sample_loop(count: int, depth: int, unroll: bool):
    """
    Schedule of a segment's count chunks through a sampling pipeline depth chunks deep (see _sample_segment):
    (per_it, n_loop), a device loop of n_loop iterations of per_it chunks, or (0, 0) for every chunk unrolled. Every
    engine meets at the loop's back edge, draining the pipeline, so per_it is the most rounds of depth chunks (at most
    _LOOP_ROUNDS) that still loop _MIN_LOOP_ITERS times; the loop leaves out the pipeline's depth - 1 leading stage
    A's and the last chunk, which starts at a static pixel. Unrolled: at most (_LOOP_ROUNDS + 1) * depth chunks.
    """
    per_it = 0
    if not unroll:
        for r in range(1, _LOOP_ROUNDS + 1):
            if (count - depth) // (r * depth) >= _MIN_LOOP_ITERS:
                per_it = r * depth
    n_loop = 0
    if per_it > 0:
        n_loop = (count - depth) // per_it
    return per_it, n_loop


def grid_sample_sample_layout(plan: GridSamplePlan, nb: int, parts: int, cols: int) -> bool:
    """
    True: stage A of a work item's chunks (nb batches, parts x cols pixels) runs in the sample layout
    (_stage_a_sample_layout): quad chunks of one batch whose partitions hold whole rows of cols pixels, cols a multiple
    of 32 (each block of cols index partitions must start at a multiple of 32: engine accesses do).
    """
    return plan.quad and nb == 1 and parts % cols == 0 and cols % 32 == 0


def grid_sample_scratch_width(hs: int, ws: int) -> int:
    """
    The width of a scratch of hs rows of ws entries per batch: ws below 2^24 rows, where fp32 holds every index
    exactly. From 2^24 rows on (6K images and up) stage A adds a row base y * ws (y < hs) and a column in integer
    arithmetic (_exact_rows), and only the row base comes from fp32: ws rounds up to a multiple of 2^k, for the least
    k with hs * ws / 2^k < 2^24, so that every row base is fewer than 24 significant bits times a power of two,
    exact in fp32. ws itself when no k does (2^24 rows or more), which the kernel rejects.
    """
    if hs * ws < _MAX_EXACT:
        return ws
    width = 0
    p = 1
    for k in range(32):
        m = (ws + p - 1) // p
        if width == 0 and hs * m < _MAX_EXACT:
            width = m * p
        p = p * 2
    if width == 0:
        return ws
    return width


def grid_sample_exact_rows(plan: GridSamplePlan) -> bool:
    """
    True when stage A computes every scratch row of plan exactly: in fp32 below 2^24 rows per batch; past it
    (plan.exact), each row base y * ws (y < hs) in fp32, exact while y times the odd part of ws stays below 2^24 (the
    rest of ws is a power of two), and its sum with the column in uint32, below 2^32.
    """
    if not plan.exact:
        return plan.hs * plan.ws < _MAX_EXACT
    odd = plan.ws
    for k in range(32):
        if odd % 2 == 0:
            odd = odd // 2
    return (plan.hs - 1) * odd < _MAX_EXACT and plan.hs * plan.ws <= _MAX_ROWS


def grid_sample_build_ranges(plan: GridSamplePlan, n_prgs: int, prg_id: int):
    """
    The batch ranges [(first batch, end batch)] whose scratch program prg_id (of n_prgs) builds: every batch its
    chunks read. A segment the programs split by batch (see grid_sample_work) gives each program its share, so each
    builds half of it with LNC=2; runs of the other segments merge into one range. Every program gets ranges of the
    same sizes in the same order (one control flow).
    """
    ranges = []
    run0 = -1
    run1 = -1
    for si in range(len(plan.segments)):
        n0, n1 = plan.segments[si]
        if n_prgs > 1 and n1 - n0 > 1:
            if run0 >= 0:
                ranges.append((run0, run1))
                run0 = -1
            nbp = (n1 - n0 + n_prgs - 1) // n_prgs
            a = min(n0 + nbp * prg_id, n1 - nbp)
            ranges.append((a, a + nbp))
        else:
            if run0 < 0:
                run0 = n0
            run1 = n1
    if run0 >= 0:
        ranges.append((run0, run1))
    return ranges


def plan_grid_sample(
    N: int, C: int, H: int, W: int, Ho: int, Wo: int, mode: str, elem_bytes: int, sbuf_bytes: int = _TRN2_SBUF_BYTES
) -> GridSamplePlan:
    """
    Scratch layout and chunking of grid_sample. Bilinear samples gather from a "quad" scratch, whose entry (y, x)
    holds the 2 x 2 neighbourhood of padded pixel (y, x) (4 C contiguous elements, so one gathered row per sample),
    when it fits _MAX_SCRATCH_BYTES, its tiles fit SBUF, and the input is not much larger than the output (building
    it writes 4x the input; see _QUAD_BYTES_PER_SAMPLE); otherwise from the zero-padded channels-last image (4
    gathered rows of C per sample). Nearest samples gather one row of C.
    The output pixels of a segment of batches form one flat range, cut into chunks of parts x cols pixels (cols
    consecutive pixels per partition, see grid_sample_chunks; the last chunk may overlap the one before it, whose
    pixels are then written twice with the same values). Every batch is a segment while each fills a chunk alone
    and there are at most _MAX_SEGMENTS; otherwise a segment holds as many batches as keep its scratch rows and
    pixels exact in fp32, and its chunks (which may span batches) store into a channel-major staging buffer that
    one copy moves to the output (with one channel, straight into the output, where the segment's pixels are
    consecutive). Chunks with more channels than gathered rows per partition are stored channel-major
    (plan.tstore): their many small stores would cost more than the transposes. A one-channel input that would get
    the plain layout needs no scratch (plan.direct): the gathers read the input itself. Many tiny images build
    their scratch one batch per partition (plan.bp), small ones in tiles of plan.gp groups of batches stacked
    along the partitions, and more build tiles than are unrolled run in a software pipeline (plan.pipelined).
    A scratch of 2^24 rows per batch or more (plan.exact: 6K images and up) is past fp32's exact integers: its rows
    are formed in integer arithmetic from row bases that stay exact in fp32 (its width rounds up for that, see
    grid_sample_scratch_width), every batch is a segment, and a one-channel input takes the plain layout instead of
    the direct read, whose input rows cannot be widened.
    """
    layouts = [False]
    # a batch's padded input, bytes, at the quad scratch's width
    in_bytes = (H + 2) * grid_sample_scratch_width(H + 2, W + 2) * C * elem_bytes
    if mode == "bilinear" and N * 4 * in_bytes <= _MAX_SCRATCH_BYTES and in_bytes <= _QUAD_BYTES_PER_SAMPLE * Ho * Wo:
        layouts = [True, False]
    plan = None
    for quad in layouts:
        if plan is None:
            # the channel-major store's tile, then the quad layout, give way when SBUF runs short
            for allow_tstore in (True, False):
                cand = _plan_layout(N, C, H, W, Ho, Wo, mode, elem_bytes, quad, allow_tstore)
                fits = grid_sample_sbuf_bytes(cand, C, elem_bytes) <= _SBUF_SHARE * sbuf_bytes
                if plan is None and (fits or (not quad and not allow_tstore)):
                    plan = cand
    # then as many chunks in flight as SBUF holds
    depth = 2
    for d in range(3, _MAX_DEPTH + 1):
        plan.depth = d
        if depth == d - 1 and grid_sample_sbuf_bytes(plan, C, elem_bytes) <= _SBUF_SHARE * sbuf_bytes:
            depth = d
    plan.depth = depth
    return plan


def _plan_layout(N, C, H, W, Ho, Wo, mode, elem_bytes, quad, allow_tstore):
    """plan_grid_sample for one scratch layout (quad: bilinear only), with or without the channel-major store."""
    T = Ho * Wo
    hs = H + 2
    ws = W + 2
    row = C
    taps = 1
    if quad:
        row = 4 * C
    elif mode == "bilinear":
        hs = H + 3
        ws = W + 3
        taps = 4
    # bilinear, one channel and no quads: the padded scratch would only add a zero border, so the gathers read the
    # input itself, at taps clamped into it, and stage B zeroes the clamped taps (below 2^24 pixels, where the fp32
    # indices of _direct_indices are exact)
    direct = C == 1 and not quad and mode == "bilinear" and H * W < _MAX_EXACT
    if direct:
        hs = H
        ws = W
    # 2^24 scratch rows per batch or more: integer row arithmetic in stage A, at a width that keeps the row bases
    # exact in fp32 (zero-filled past the padded image)
    exact = hs * ws >= _MAX_EXACT
    ws = grid_sample_scratch_width(hs, ws)
    cols = max(1, min(_MAX_COLS, _GATHER_BYTES // (taps * row * elem_bytes)))
    # channel-major store when a chunk's store would move more pieces (C per partition) than its gather moves rows
    tstore = allow_tstore and C > taps * cols
    # several batches per segment while its scratch rows (< 2^24) and pixels (_MAX_SEG_PIX: the batch of a pixel,
    # floor((pixel + 0.5) / T), stays exact in fp32) allow, and (several channels) its staging buffer stays within
    # _MAX_STAGE_BYTES; one batch per segment while each fills a chunk alone
    nb = max(1, min(N, (_MAX_EXACT - 1) // (hs * ws), _MAX_SEG_PIX // T))
    if C > 1:
        nb = max(1, min(nb, _MAX_STAGE_BYTES // (C * T * elem_bytes)))
    # one batch per segment (no staging copy) while a batch's own chunks waste at most a tenth of their pixels
    chunk = _PMAX * cols
    if N <= _MAX_SEGMENTS and T >= chunk and 10 * chunk * ((T + chunk - 1) // chunk) <= 11 * T:
        nb = 1
    # narrower chunks, except for quads spanning batches (RAFT's level-3 lookup: 3.5 ms at 32 columns, 3.1 at 64)
    if nb == 1 or not quad:
        cols = min(cols, _MAX_COLS_DEEP)
        tstore = allow_tstore and C > taps * cols
    segments = []
    for n0 in range(0, N, nb):
        segments.append((n0, min(N, n0 + nb)))
    seg = max(1, min(ws, _SCRATCH_ROW_ELEMS // row))
    # batches per scratch-build tile: side by side along the free dimension while a batch's scratch is one tile wide
    bk = 1
    if seg == ws:
        bk = max(1, min(N, _SCRATCH_ROW_ELEMS // (ws * row)))
    # tiny images, many batches: one batch per partition, contiguous loads and stores (_build_bp)
    bp = (
        not direct
        and N >= 32
        and (C * H * W + hs * ws * row) * elem_bytes <= _BP_BYTES
        and (N + _PMAX - 1) // _PMAX <= _BUILD_UNROLL
    )
    # small images: as many groups of bk batches as fit stacked along the partitions, while they fill them
    gp = 1
    if not direct and seg == ws and 2 * hs <= _PMAX:
        gp = max(1, min(_PMAX // hs, (N + bk - 1) // bk))
    # many build tiles: pipelined (more than are unrolled), one group per tile
    pipelined = not direct and N // (gp * bk) > _BUILD_UNROLL and hs <= _PMAX and seg == ws
    if pipelined or bp:
        gp = 1
    if bp:
        pipelined = False
    p = GridSamplePlan(
        quad, hs, ws, row, taps, cols, seg, bk, segments, tstore, direct, pipelined, gp=gp, bp=bp, exact=exact
    )
    p.bp_chw = C * H * W
    return p


def _tstore_regions(plan, C, elem_bytes):
    """Channel-major store tile regions (one per block of 128 channels, reused cyclically past _TSTORE_TILE_BYTES)."""
    per = _PMAX * plan.cols * elem_bytes
    return max(1, min((C + _PMAX - 1) // _PMAX, _TSTORE_TILE_BYTES // per))


def grid_sample_sbuf_bytes(plan: GridSamplePlan, C: int, elem_bytes: int) -> int:
    """
    Estimated SBUF bytes per partition of grid_sample: plan.depth sets of chunk tiles (coordinates in both layouts,
    scratch rows and their transposed indices, gathered rows, working tiles, the blend's fp32 accumulator, products
    and weights for bilinear, the output tile when the result is not the accumulator, the channel-major store tile,
    and an exact plan's tiles for its integer row arithmetic), the multi-batch constant and the chunk counters, and
    the scratch build's tiles.
    """
    cols, taps, row = plan.cols, plan.taps, plan.row
    bilinear = taps == 4 or plan.quad
    one = 2 * _PMAX * 4 + 2 * cols * 4 + taps * _PMAX * 4 + taps * cols * 4 + taps * cols * row * elem_bytes
    one = one + 8 * _PMAX * 4 + 8 * cols * 4
    if elem_bytes < 4:
        one = one + (2 * _PMAX + 2 * cols) * elem_bytes
    if bilinear:
        one = one + 2 * C * cols * 4 + 4 * cols * 4
    if plan.blend4:
        one = one + 3 * C * cols * 4
    if not plan.tstore and (not bilinear or elem_bytes < 4):
        one = one + C * cols * elem_bytes
    if plan.tstore:
        one = one + _tstore_regions(plan, C, elem_bytes) * _PMAX * cols * elem_bytes
    if plan.direct:
        one = one + 4 * cols + 4 * cols * elem_bytes
    if plan.exact:
        # the columns' transposed block beside the row bases', and their uint32 evictions (_exact_rows)
        one = one + (max(taps, 2) - taps) * _PMAX * 4 + 4 * cols * 4
    consts = 2 * _PMAX * 4 + 16
    build = 2 * plan.bk * plan.seg * C * elem_bytes
    if plan.quad:
        build = (plan.bk * plan.seg * row + 2 * plan.bk * C * (plan.seg + 1)) * elem_bytes
    if plan.direct:
        build = 0
    if plan.pipelined:
        build = _BUILD_SETS * build
    if plan.bp:
        build = 2 * (plan.bp_chw + plan.hs * plan.ws * plan.row) * elem_bytes
    return plan.depth * one + consts + build


@nki.jit
def grid_sample(
    input: nl.NkiTensor,
    grid: nl.NkiTensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> nl.NkiTensor:
    """
    Sample input at the grid's normalized coordinates, as torch.nn.functional.grid_sample for 4-D input.

    Any coordinates are exact (no displacement bound): the input is first rewritten into an HBM scratch in
    channels-last order with a zero border, and every chunk of output pixels fetches its samples with batched
    indirect DMA gathers (software DGE, 128 x cols rows per instruction). For bilinear samples the scratch normally
    holds each pixel's 2 x 2 neighbourhood, so a sample is one contiguous row of 4 C elements; the bilinear weights
    then blend the four taps on the vector engine. An input much larger than the output, or whose neighbourhood
    scratch would pass 2 GiB or not fit SBUF's tiles, gathers the four taps from the padded image instead, and a
    one-channel input there is read directly, without a scratch (see plan_grid_sample). Padding modes act on the
    source coordinates (border clamps, reflection reflects then clamps); zeros relies on the zero border, onto which
    coordinates outside [-1, size] are clamped. Coordinates follow ATen's fp32 operation order; reflection is exact
    for source coordinates within 2^23 pixels of the input (beyond, where fp32 holds no fraction, they saturate at
    that distance). Scratch rows (the gathered entries) are indexed in fp32 below 2^24 per batch; past it (6K images
    and up) each index is a row base exact in fp32 plus a column, added in integer arithmetic: an input is rejected
    only when its padded height reaches 2^24 or its scratch 2^32 entries per batch, an output from 2^31 pixels per
    batch.
    A chunk stores each channel's consecutive output pixels as runs; when it has more channels than gathered rows
    per partition, the Tensor Engine first transposes it channel-major (bit-exact), so that each channel leaves in
    one run. Small batches share chunks, which then store into a channel-major staging buffer that one copy moves
    to the output. Sampling is software-pipelined over up to 8 tile sets (as many as fit SBUF): a chunk's
    coordinates, scratch rows and gather run up to 7 chunks ahead of its blend and store. A program of at most 512
    chunks runs them unrolled; beyond, each segment's steady state runs in a device loop. With LNC=2 each program
    builds its own scratch of the batches it samples and takes half of a single batch's chunks, or half of the
    batches that share chunks.

    Dimensions:
        N: Batch size
        C: Channels
        H, W: Input height and width
        Ho, Wo: Output height and width (the grid's)

    Args:
        input (nl.NkiTensor): [N, C, H, W], float32, bfloat16 or float16.
        grid (nl.NkiTensor): [N, Ho, Wo, 2], normalized (x, y) sample coordinates (-1 and 1 are the input's edges),
            the dtype of input.
        mode (str): "bilinear" or "nearest" (round half to even, as torch).
        padding_mode (str): "zeros", "border" or "reflection".
        align_corners (bool): As torch: True maps -1 and 1 to the centers of the corner pixels, False to their
            outer edges.

    Returns:
        output (nl.NkiTensor): [N, C, Ho, Wo] in input's dtype.

    Pseudocode:
        scratch[n, y, x] = 2 x 2 neighbourhood of zero-padded input[n] at (y, x), channels last
        for chunk of output pixels (128 partitions x cols):
            ix, iy = unnormalize(grid[chunk]); apply padding_mode; clamp to [-1, W] x [-1, H]
            x0, y0 = floor(ix), floor(iy)
            rows = gather(scratch, n * Hs * Ws + (y0 + 1) * Ws + (x0 + 1))    # indirect DMA, one row per sample
            out[chunk] = sum over taps of w_tap(ix - x0, iy - y0) * rows[tap]
    """
    _validate_grid_sample_inputs(input, grid, mode, padding_mode)
    N, C, H, W = input.shape
    Ho = grid.shape[1]
    Wo = grid.shape[2]
    dt = input.dtype
    elem_bytes = 2
    if dt == nl.float32:
        elem_bytes = 4
    budget = nl.tile_size.total_available_sbuf_size
    plan = plan_grid_sample(N, C, H, W, Ho, Wo, mode, elem_bytes, budget)
    kernel_assert(
        grid_sample_exact_rows(plan),
        f"grid_sample: input {H} x {W}: a scratch of {plan.hs} x {plan.ws} entries per batch is past exact index "
        f"arithmetic (2^24 rows, 2^32 entries)",
    )
    kernel_assert(Ho * Wo < _MAX_PIXELS, f"grid_sample: output {Ho} x {Wo} has 2^31 pixels or more")
    kernel_assert(
        grid_sample_sbuf_bytes(plan, C, elem_bytes) <= _SBUF_SHARE * budget,
        f"grid_sample: C = {C} channels do not fit in SBUF",
    )
    output = nl.ndarray((N, C, Ho, Wo), dtype=dt, buffer=nl.shared_hbm)
    # 2-D, one scratch entry per row: an indirect DMA's vector offset counts rows of the indexed tensor's dim 0
    _, n_prgs, prg_id = get_verified_program_sharding_info("grid_sample", (0, 1))
    if plan.direct:
        scratch = input.reshape((N * H * W, 1))
    else:
        scratch = nl.ndarray((N * plan.hs * plan.ws, plan.row), dtype=dt, buffer=nl.private_hbm)
        ranges = grid_sample_build_ranges(plan, n_prgs, prg_id)
        for ri in range(len(ranges)):
            r0, r1 = ranges[ri]
            _build_scratch(input, scratch, plan, r0, r1)
    ctx = _ChunkContext(input, grid, output, scratch, plan, mode, padding_mode, align_corners)
    T = Ho * Wo
    work = grid_sample_work(plan, T, n_prgs, prg_id)
    total = 0
    for si in range(len(work)):
        total = total + work[si][7]
    for si in range(len(work)):
        n0, n1, parts, cols, n_full, tail, first, count = work[si]
        S = (n1 - n0) * T
        # chunks store channel-major into the output (one batch: the [C, T] rows of batch n0; one channel: the
        # segment's consecutive pixels) or a staging buffer [C, S]
        stage = None
        dst = _StoreTarget(output.reshape((N * C * T, 1)), n0 * C * T, T)
        if n1 - n0 > 1 and C > 1:
            stage = nl.ndarray((C * S, 1), dtype=dt, buffer=nl.private_hbm)
            dst = _StoreTarget(stage, 0, S)
        _sample_segment(ctx, n0, n1, parts, cols, first, count, n_full, tail, dst, total <= _UNROLL_CHUNKS)
        if stage is not None:
            _unstage(output, stage, n0, n1, S, elem_bytes)
    return output


def _unstage(output, stage, n0, n1, S, elem_bytes):
    """
    output[n, c, t] = stage[c, (n - n0) T + t] for the batches [n0, n1) of a staged segment of S pixels, in
    contiguous runs of at most _MAX_RUN_BYTES: the runtime rejects longer descriptors on the output tensor. The
    split of T into runs is the outermost dimension, so that the lowering cannot merge consecutive runs again.
    """
    N, C, Ho, Wo = output.shape
    T = Ho * Wo
    out = output.reshape((N * C * T, 1))
    k = _MAX_RUN_BYTES // elem_bytes
    full = T // k
    if full > 0:
        nisa.dma_copy(
            dst=out.ap(pattern=[[k, full], [C * T, n1 - n0], [T, C], [1, k]], offset=n0 * C * T),
            src=stage.ap(pattern=[[k, full], [T, n1 - n0], [S, C], [1, k]], offset=0),
        )
    rem = T - full * k
    if rem > 0:
        nisa.dma_copy(
            dst=out.ap(pattern=[[C * T, n1 - n0], [T, C], [1, rem]], offset=n0 * C * T + full * k),
            src=stage.ap(pattern=[[T, n1 - n0], [S, C], [1, rem]], offset=full * k),
        )


class _StoreTarget(nl.NKIObject):
    """Where a segment's chunks store: tensor t viewed as [-1, 1], channel c's pixel p of the segment at element
    base + c * stride + p."""

    def __init__(self, t, base, stride):
        self.t = t
        self.base = base
        self.stride = stride


class _ChunkContext(nl.NKIObject):
    """The tensors and options every chunk of a call shares."""

    def __init__(self, input, grid, output, scratch, plan, mode, padding_mode, align_corners):
        self.input = input
        self.grid = grid
        self.output = output
        self.scratch = scratch
        self.plan = plan
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners


def _validate_grid_sample_inputs(input, grid, mode, padding_mode):
    """Validate shapes, dtypes and options of grid_sample."""
    kernel_assert(nisa.get_nc_version() == nisa.nc_version.gen3, "grid_sample supports trn2 (NeuronCore-v3) only")
    kernel_assert(
        GRID_SAMPLE_NKI_SUPPORTED,
        f"grid_sample needs nki 0.7.0 or later (Neuron SDK 2.33), got {nki.__version__}: earlier releases reject its "
        "batched indirect DMA gathers",
    )
    kernel_assert(len(input.shape) == 4, f"input must be [N, C, H, W], got {input.shape}")
    kernel_assert(len(grid.shape) == 4 and grid.shape[3] == 2, f"grid must be [N, Ho, Wo, 2], got {grid.shape}")
    kernel_assert(grid.shape[0] == input.shape[0], f"batch mismatch: input {input.shape} vs grid {grid.shape}")
    kernel_assert(mode in _MODES, f"mode must be one of {_MODES}, got {mode}")
    kernel_assert(padding_mode in _PADDING_MODES, f"padding_mode must be one of {_PADDING_MODES}, got {padding_mode}")
    kernel_assert(
        input.dtype == nl.float32 or input.dtype == nl.bfloat16 or input.dtype == nl.float16,
        f"input dtype must be float32, bfloat16 or float16, got {input.dtype}",
    )
    kernel_assert(grid.dtype == input.dtype, f"grid dtype {grid.dtype} must match input dtype {input.dtype}")


def _build_scratch(input, scratch, plan, r0, r1):
    """
    The scratch of batches r0 .. r1 - 1: one batch per partition for tiny images (_build_bp); for small ones,
    tiles of plan.gp groups of plan.bk batches stacked along the partitions (_build_stacked), then one tile of the
    remaining whole groups and one group of the last batches; else in groups of plan.bk batches per tile, unrolled
    for up to _BUILD_UNROLL groups (static descriptors), else in a device loop whose batch counters offset the
    input and scratch accesses (hardware DGE), pipelined when a group is one tile (_build_pipelined); the last
    (r1 - r0) mod bk batches form one more, static group.
    """
    bk = plan.bk
    if plan.bp:
        _build_bp(input, scratch, plan, r0, r1)
        return
    if plan.gp > 1:
        per = plan.gp * bk
        n_t = (r1 - r0) // per
        for ti in range(n_t):
            _build_stacked(input, scratch, r0 + ti * per, plan.gp, bk, plan)
        rem = r1 - r0 - n_t * per
        ng = rem // bk
        if ng > 0:
            _build_stacked(input, scratch, r0 + n_t * per, ng, bk, plan)
        if rem > ng * bk:
            _build_group(input, scratch, r0 + n_t * per + ng * bk, rem - ng * bk, plan, None)
        return
    groups = (r1 - r0) // bk
    if plan.pipelined and groups > 0:
        _build_pipelined(input, scratch, plan, r0, groups)
    elif groups <= _BUILD_UNROLL:
        for g in range(groups):
            _build_group(input, scratch, r0 + g * bk, bk, plan, None)
    else:
        b = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=b, value=r0)

        def body(_):
            _build_group(input, scratch, 0, bk, plan, b)
            nisa.tensor_scalar(dst=b, data=b, op0=nl.add, operand0=bk)

        nl.fori_loop(0, groups, body)
    if r1 - r0 > groups * bk:
        _build_group(input, scratch, r0 + groups * bk, r1 - r0 - groups * bk, plan, None)


def _build_group(input, scratch, n, nbk, plan, b):
    """The scratch of batches n .. n + nbk - 1 (n at the batch counter b), tile by tile on fresh tiles."""
    for y0 in range(0, plan.hs, _PMAX):
        for x0 in range(0, plan.ws, plan.seg):
            tiles = _build_tiles(input, plan)
            _build_load(input, tiles, n, nbk, plan, b, y0, x0, True)
            _build_finish(input, scratch, tiles, n, nbk, plan, b, y0, x0)


def _interleave_engine(dx, dt):
    """The engine of a quad interleave copy: dx = 0 on Vector, dx = 1 on Scalar (GpSimd shares Vector's SBUF port
    and would slow it), except at 16 bits, where Scalar's copies run 2.5x slower and Vector takes all four."""
    if dx == 1 and dt == nl.float32:
        return nisa.engine.scalar
    return nisa.engine.vector


def _build_bp(input, scratch, plan, r0, r1):
    """
    The scratch of batches r0 .. r1 - 1 for tiny images, one batch per partition: per tile of up to 128 batches one
    contiguous load of their images, the interleave on all of them (the zero border stays from one memset per tile
    set: the interleave writes the same entries every tile), and one contiguous store of their scratch rows.
    """
    N, C, H, W = input.shape
    hs, ws, row = plan.hs, plan.ws, plan.row
    per = hs * ws * row
    chw = C * H * W
    nt = (r1 - r0 + _PMAX - 1) // _PMAX
    sets = []
    for si in range(min(2, nt)):
        img = nl.ndarray((_PMAX, chw), dtype=input.dtype, buffer=nl.sbuf)
        q = nl.ndarray((_PMAX, per), dtype=input.dtype, buffer=nl.sbuf)
        nisa.memset(dst=q, value=0.0)
        sets.append((img, q))
    flat_in = input.reshape((N, chw))
    flat_sc = scratch.reshape((N, per))
    for ti in range(nt):
        n = r0 + ti * _PMAX
        nb = min(_PMAX, r1 - n)
        img, q = sets[ti % 2]
        nisa.dma_copy(dst=img[0:nb, 0:chw], src=flat_in[n : n + nb, 0:chw])
        src = img[0:nb, 0:chw].reshape_dim(1, (C, H, W)).permute((0, 2, 3, 1))
        if plan.quad:
            qv = q[0:nb, 0:per].reshape_dim(1, (hs, ws, 4, C))
            for dy in range(2):
                for dx in range(2):
                    # entry (y, x) tap (dy, dx) = input (y + dy - 1, x + dx - 1), inside it
                    dst = qv.slice(dim=1, start=1 - dy, end=H + 1 - dy, step=1)
                    dst = dst.slice(dim=2, start=1 - dx, end=W + 1 - dx, step=1).select(3, 2 * dy + dx)
                    nisa.tensor_copy(dst=dst, src=src, engine=_interleave_engine(dx, input.dtype))
        else:
            qv = q[0:nb, 0:per].reshape_dim(1, (hs, ws, C))
            dst = qv.slice(dim=1, start=1, end=H + 1, step=1).slice(dim=2, start=1, end=W + 1, step=1)
            nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.vector)
        nisa.dma_copy(dst=flat_sc[n : n + nb, 0:per], src=q[0:nb, 0:per])


def _build_stacked(input, scratch, n, ng, nbk, plan):
    """
    The scratch of batches n .. n + ng * nbk - 1 in one build tile: ng groups of nbk batches stacked along the
    partitions (group g on partitions g * hs .. g * hs + hs - 1, its batches side by side as in _build_group), so the
    interleave runs on ng times the partitions; zeroed whole first, then one load and one store per group.
    """
    hs = plan.hs
    C = input.shape[1]
    tiles = _build_tiles(input, plan)
    _, _, wsrc = _tile_extent(plan, 0, 0)
    for i in range(len(tiles.src)):
        nisa.memset(dst=tiles.src[i][0 : ng * hs, 0 : nbk * C * wsrc], value=0.0)
    for g in range(ng):
        _build_load(input, tiles, n + g * nbk, nbk, plan, None, 0, 0, False, g * hs)
    _build_interleave(input, tiles, nbk, plan, ng * hs, 0, 0)
    for g in range(ng):
        _build_store(input, scratch, tiles, n + g * nbk, nbk, plan, None, 0, 0, g * hs)


def _build_pipelined(input, scratch, plan, r0, groups):
    """
    The scratch of batches r0 .. r0 + groups * bk - 1, each group a single tile, software-pipelined over two tile sets:
    group g loads into set g % 2 while group g - 1 interleaves and stores from the other set. The loads always fill
    the same region of a set, so the zero border is set once. Batch counters per set ([1, 1] int32) advance by two
    groups after each store.
    """
    bk = plan.bk
    sets = []
    cnt = []
    for s in range(2):
        tiles = _build_tiles(input, plan)
        _build_zero(tiles, input.shape[1], bk, plan, 0, 0)
        sets.append(tiles)
        c = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=c, value=r0 + s * bk)
        cnt.append(c)
    _build_load(input, sets[0], 0, bk, plan, cnt[0], 0, 0, False)
    # step g (1 <= g < groups): load group g, then finish group g - 1; two steps per loop iteration
    n_pairs = (groups - 1) // 2
    if n_pairs > 0:

        def body(_):
            for s in range(2):
                _build_step(input, scratch, sets, cnt, plan, 1 - s)

        nl.fori_loop(0, n_pairs, body)
    if (groups - 1) % 2 == 1:
        _build_step(input, scratch, sets, cnt, plan, 1)
    last = (groups - 1) % 2
    _build_finish(input, scratch, sets[last], 0, bk, plan, cnt[last], 0, 0)


def _build_step(input, scratch, sets, cnt, plan, s):
    """One pipeline step of _build_pipelined: load the next group into set s, finish the previous one (set 1 - s),
    and move that set's counter two groups on."""
    bk = plan.bk
    _build_load(input, sets[s], 0, bk, plan, cnt[s], 0, 0, False)
    _build_finish(input, scratch, sets[1 - s], 0, bk, plan, cnt[1 - s], 0, 0)
    nisa.tensor_scalar(dst=cnt[1 - s], data=cnt[1 - s], op0=nl.add, operand0=2 * bk)


def _batch_ap(t, n, b, pattern, offset):
    """Access pattern on HBM tensor t, viewed as [N, -1], of batch n (static) or of the batch counter b."""
    N = t.shape[0]
    per = 1
    for d in t.shape[1:]:
        per = per * d
    flat = t.reshape((N, per))
    if b is None:
        return flat.ap(pattern=pattern, offset=n * per + offset)
    return flat.ap(pattern=pattern, offset=offset, scalar_offset=b, indirect_dim=0)


def _build_dma(dst, src, b):
    """A scratch-build copy: descriptors as the compiler picks for a static batch, hardware DGE at the batch counter
    b. (Hardware DGE issued by the Sync engine for the static ones measured no faster, and slower for the plain
    layout.)"""
    if b is None:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.unknown)
    else:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge)


class _BuildTiles(nl.NKIObject):
    """One set of scratch-build tiles: the channel-planar input rows (src: one tile, or the top and bottom rows of
    the quads) and the channels-last tile q."""

    def __init__(self, src, q):
        self.src = src
        self.q = q


def _build_tiles(input, plan):
    """A fresh set of scratch-build tiles for plan.bk batches of one tile (up to 128 rows by plan.seg columns)."""
    C = input.shape[1]
    bk, seg, row = plan.bk, plan.seg, plan.row
    src = []
    if plan.quad:
        for dy in range(2):
            src.append(nl.ndarray((_PMAX, bk * C * (seg + 1)), dtype=input.dtype, buffer=nl.sbuf))
    else:
        src.append(nl.ndarray((_PMAX, bk * C * seg), dtype=input.dtype, buffer=nl.sbuf))
    q = nl.ndarray((_PMAX, bk * seg * row), dtype=input.dtype, buffer=nl.sbuf)
    return _BuildTiles(src, q)


def _tile_extent(plan, y0, x0):
    """Rows and input-tile columns of the build tile at scratch row y0 and column x0 (quads read one column more)."""
    r = min(_PMAX, plan.hs - y0)
    wseg = min(plan.seg, plan.ws - x0)
    wsrc = wseg
    if plan.quad:
        wsrc = wseg + 1
    return r, wseg, wsrc


def _build_zero(tiles, C, nbk, plan, y0, x0):
    """Zero the input tiles of the build tile at (y0, x0): the loads then fill all but the zero border."""
    r, wseg, wsrc = _tile_extent(plan, y0, x0)
    for i in range(len(tiles.src)):
        nisa.memset(dst=tiles.src[i][0:r, 0 : nbk * C * wsrc], value=0.0)


def _build_load(input, tiles, n, nbk, plan, b, y0, x0, zero, po=0):
    """
    Load the input under the build tile at (y0, x0) of batches n .. n + nbk - 1 (n at the batch counter b) into
    tiles.src, channel-planar: row p of input tile dy holds input row y0 + p + dy - 1 (quad, dy = 0, 1) or
    y0 + p - 1 (plain), column j input column x0 - 1 + j, zeros outside the input (zero: set the zero border
    first, on the tiles the load does not cover whole).
    """
    N, C, H, W = input.shape
    r, wseg, wsrc = _tile_extent(plan, y0, x0)
    for dy in range(len(tiles.src)):
        p_lo = max(0, 1 - dy - y0)
        p_hi = min(r, H + 1 - dy - y0)
        j_lo = max(0, 1 - x0)
        j_hi = min(wsrc, W + 1 - x0)
        covered = p_lo == 0 and p_hi == r and j_lo == 0 and j_hi == wsrc
        if zero and not covered:
            nisa.memset(dst=tiles.src[dy][0:r, 0 : nbk * C * wsrc], value=0.0)
        if p_lo < p_hi and j_lo < j_hi:
            _build_dma(
                tiles.src[dy][po + p_lo : po + p_hi, 0 : nbk * C * wsrc]
                .reshape_dim(1, (nbk, C, wsrc))
                .slice(dim=3, start=j_lo, end=j_hi, step=1),
                _batch_ap(
                    input,
                    n,
                    b,
                    [[W, p_hi - p_lo], [C * H * W, nbk], [H * W, C], [1, j_hi - j_lo]],
                    (y0 + p_lo + dy - 1) * W + x0 - 1 + j_lo,
                ),
                b,
            )


def _build_finish(input, scratch, tiles, n, nbk, plan, b, y0, x0):
    """
    Interleave the loaded input tiles channels-last into tiles.q and store it as the scratch rows y0 .. y0 + r - 1,
    columns x0 .. x0 + wseg - 1 of batches n .. n + nbk - 1 (n at the batch counter b).
    """
    r, _, _ = _tile_extent(plan, y0, x0)
    _build_interleave(input, tiles, nbk, plan, r, y0, x0)
    _build_store(input, scratch, tiles, n, nbk, plan, b, y0, x0, 0)


def _build_interleave(input, tiles, nbk, plan, r, y0, x0):
    """
    tiles.q = the loaded input tiles' first r partitions, channels-last. Quad: scratch[n, y, x, (dy, dx), c] =
    input[n, c, y + dy - 1, x + dx - 1] (four strided copies, on the engines of _interleave_engine); plain:
    scratch[n, y, x, c] = input[n, c, y - 1, x - 1].
    """
    N, C, H, W = input.shape
    row = plan.row
    _, wseg, wsrc = _tile_extent(plan, y0, x0)
    q = tiles.q
    if plan.quad:
        quads = q[0:r, 0 : nbk * wseg * row].reshape_dim(1, (nbk, wseg, 4 * C))
        for dy in range(2):
            planar = tiles.src[dy][0:r, 0 : nbk * C * wsrc].reshape_dim(1, (nbk, C, wsrc))
            for dx in range(2):
                k = dy * 2 + dx
                dst = quads.slice(dim=3, start=k * C, end=(k + 1) * C, step=1)
                nisa.tensor_copy(
                    dst=dst.permute((0, 1, 3, 2)),
                    src=planar.slice(dim=3, start=dx, end=dx + wseg, step=1),
                    engine=_interleave_engine(dx, input.dtype),
                )
    else:
        nisa.tensor_copy(
            dst=q[0:r, 0 : nbk * wseg * C].reshape_dim(1, (nbk, wseg, C)).permute((0, 1, 3, 2)),
            src=tiles.src[0][0:r, 0 : nbk * C * wseg].reshape_dim(1, (nbk, C, wseg)),
            engine=nisa.engine.vector,
        )


def _build_store(input, scratch, tiles, n, nbk, plan, b, y0, x0, po):
    """Store tiles.q's partitions po .. po + r - 1 as the scratch rows y0 .. y0 + r - 1, columns x0 .. x0 + wseg - 1
    of batches n .. n + nbk - 1 (n at the batch counter b)."""
    N = input.shape[0]
    hs, ws, row = plan.hs, plan.ws, plan.row
    r, wseg, _ = _tile_extent(plan, y0, x0)
    q = tiles.q
    sc = scratch.reshape((N, hs * ws * row))
    _build_dma(
        _batch_ap(sc, n, b, [[ws * row, r], [hs * ws * row, nbk], [1, wseg * row]], (y0 * ws + x0) * row),
        q[po : po + r, 0 : nbk * wseg * row].reshape_dim(1, (nbk, wseg * row)),
        b,
    )


def _sample_segment(ctx, n0, n1, parts, cols, first, count, n_full, tail, dst, unroll):
    """
    The chunks first .. first + count - 1 of batches [n0, n1) (chunk j starts at pixel j * parts * cols of the
    segment for j < n_full, chunk n_full at pixel tail), stored channel-major at dst (a _StoreTarget). Every chunk
    runs as stage A (coordinates, scratch rows, gather, weights) and stage B (blend, store), software-pipelined
    over D = min(plan.depth, count) tile sets: stage A runs D - 1 chunks ahead of stage B (chunk j on set j mod D),
    so D - 1 gathers are in flight while a chunk blends; the in-order engine queues reach A(i + D - 1) before B(i).
    The chunks before the last follow each other through the chunk counters (one per stage, each advanced by its
    stage: [1, 1] int32, the chunk's first pixel within the segment, a DMA offset, and the same in fp32 on every
    partition); the last starts at a static pixel. unroll: every chunk unrolled; else the steady state runs in a
    device loop scheduled by grid_sample_loop. The control flow depends on count only.
    """
    plan = ctx.plan
    chunk = parts * cols
    D = min(plan.depth, count)
    bufs = []
    for s in range(D):
        bufs.append(_chunk_buffers(plan, ctx.input.shape[1], ctx.input.dtype, parts, ctx.mode))
    consts = _segment_consts(n0, n1, parts, cols)
    cnt = []
    for s in range(2):
        f_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        f_f = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=f_i, value=first * chunk)
        nisa.memset(dst=f_f, value=float(first * chunk))
        cnt.append((f_i, f_f))
    seg = (n0, n1, parts, cols, consts, dst)
    last_start = tail
    if first + count - 1 < n_full:
        last_start = (first + count - 1) * chunk
    for j in range(D - 1):
        _stage_at(True, ctx, seg, bufs[j % D], cnt[0], chunk, j == count - 1, last_start)
    per_it, n_loop = grid_sample_loop(count, D, unroll)
    if n_loop > 0:

        def body(_):
            for r in range(per_it):
                _stage_a(ctx, seg, bufs[(r + D - 1) % D], cnt[0], chunk)
                _stage_b(ctx, seg, bufs[r % D], cnt[1], chunk)

        nl.fori_loop(0, n_loop, body)
    for i in range(n_loop * per_it, count):
        j = i + D - 1
        if j < count:
            _stage_at(True, ctx, seg, bufs[j % D], cnt[0], chunk, j == count - 1, last_start)
        _stage_at(False, ctx, seg, bufs[i % D], cnt[1], chunk, i == count - 1, last_start)


def _stage_at(is_a, ctx, seg, t, f, chunk, static, start):
    """Stage A (is_a) or B of a chunk at the counters f, first moved to the static pixel start when static."""
    if static:
        nisa.memset(dst=f[0], value=start)
        nisa.memset(dst=f[1], value=float(start))
    if is_a:
        _stage_a(ctx, seg, t, f, chunk)
    else:
        _stage_b(ctx, seg, t, f, chunk)


def _advance(f, chunk):
    """Move a stage's chunk counters by one chunk."""
    nisa.tensor_scalar(dst=f[0], data=f[0], op0=nl.add, operand0=chunk)
    nisa.tensor_scalar(dst=f[1], data=f[1], op0=nl.add, operand0=float(chunk))


def _segment_consts(n0, n1, parts, cols):
    """The per-segment constant pix_i[m, p] = m * parts + p (fp32, the index layout's pixel within the chunk), only
    for multi-batch segments."""
    if n1 - n0 == 1:
        return None
    pix_i_i = nl.ndarray((_PMAX, _PMAX), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pix_i_i[0:cols, 0:parts], pattern=[[1, parts]], offset=0, channel_multiplier=parts)
    pix_i = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pix_i[0:cols, 0:parts], src=pix_i_i[0:cols, 0:parts], engine=nisa.engine.vector)
    return pix_i


def _chunk_buffers(plan, C, dt, parts, mode):
    """One set of per-chunk tiles."""
    cols, taps = plan.cols, plan.taps
    t = {}
    t["cx"] = nl.ndarray((_PMAX, 2 * _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    t["cn"] = nl.ndarray((_PMAX, 2 * cols), dtype=nl.float32, buffer=nl.sbuf)
    t["cx16"] = None
    t["cn16"] = None
    if dt != nl.float32:
        t["cx16"] = nl.ndarray((_PMAX, 2 * _PMAX), dtype=dt, buffer=nl.sbuf)
        t["cn16"] = nl.ndarray((_PMAX, 2 * cols), dtype=dt, buffer=nl.sbuf)
    idx_taps = taps
    if plan.exact:
        idx_taps = max(taps, 2)
    t["idx"] = nl.ndarray((_PMAX, idx_taps * _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    t["vidx"] = nl.ndarray((_PMAX, taps * cols), dtype=nl.uint32, buffer=nl.sbuf)
    # an exact plan's uint32 row bases (two, one per dy) and columns (two, one per dx), see _exact_rows
    t["vxy"] = None
    if plan.exact:
        t["vxy"] = nl.ndarray((_PMAX, 4 * cols), dtype=nl.uint32, buffer=nl.sbuf)
    t["rows"] = nl.ndarray((_PMAX, taps * cols * plan.row), dtype=dt, buffer=nl.sbuf)
    bilinear = mode == "bilinear"
    if bilinear:
        t["acc"] = nl.ndarray((_PMAX, C * cols), dtype=nl.float32, buffer=nl.sbuf)
        t["prod"] = nl.ndarray((_PMAX, C * cols), dtype=nl.float32, buffer=nl.sbuf)
        if plan.blend4:
            t["prod"] = nl.ndarray((_PMAX, 4 * C * cols), dtype=nl.float32, buffer=nl.sbuf)
        t["wts"] = nl.ndarray((_PMAX, 4 * cols), dtype=nl.float32, buffer=nl.sbuf)
    # the stored tile, when it is neither the fp32 accumulator nor transposed straight from the gathered rows
    t["res"] = None
    if not plan.tstore and (not bilinear or dt != nl.float32):
        t["res"] = nl.ndarray((_PMAX, C * cols), dtype=dt, buffer=nl.sbuf)
    t["valid"] = None
    t["zeros"] = None
    if plan.direct:
        t["valid"] = nl.ndarray((_PMAX, 4 * cols), dtype=nl.uint8, buffer=nl.sbuf)
        t["zeros"] = nl.ndarray((_PMAX, 4 * cols), dtype=dt, buffer=nl.sbuf)
        nisa.memset(dst=t["zeros"], value=0.0)
    t["tst"] = None
    if plan.tstore:
        eb = 4
        if dt != nl.float32:
            eb = 2
        t["tst"] = nl.ndarray((_PMAX, _tstore_regions(plan, C, eb) * _PMAX * cols), dtype=dt, buffer=nl.sbuf)
    # coordinate working tiles: 8 in the index layout [cols, parts], 8 in the sample layout [parts, cols]
    t["ti"] = nl.ndarray((_PMAX, 8 * _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    t["ts"] = nl.ndarray((_PMAX, 8 * cols), dtype=nl.float32, buffer=nl.sbuf)
    return t


def _work_tiles(t, rows, cols, stride):
    """Eight fp32 [rows, cols] working tiles out of t (each stride columns apart)."""
    out = []
    for i in range(8):
        out.append(t[0:rows, i * stride : i * stride + cols])
    return out


def _stage_a(ctx, seg, t, f, chunk):
    """
    Stage A of the chunk at counters f: its coordinates in both layouts, its scratch rows, the gather, and (bilinear)
    the tap weights; then advances f. The index layout puts the chunk's pixel m * parts + p on partition m, so its
    transpose has it at [p, m], the order a batched gather reads its indices in; the gathered rows then land in the
    sample layout, partition r holding the chunk's pixels r * cols + j. An exact plan's scratch rows come from
    _exact_rows.
    """
    n0, n1, parts, cols, pix_i, _ = seg
    plan = ctx.plan
    if grid_sample_sample_layout(plan, n1 - n0, parts, cols):
        _stage_a_sample_layout(ctx, seg, t, f, chunk)
        return
    N, C, H, W = ctx.input.shape
    T = ctx.grid.shape[1] * ctx.grid.shape[2]
    taps, row = plan.taps, plan.row
    grid2 = ctx.grid.reshape((N * T, 2))
    cx = t["cx"][0:cols, 0 : 2 * parts]
    cx = _load_coords(
        cx,
        t["cx16"],
        grid2.ap(pattern=[[2 * parts, cols], [1, 2 * parts]], offset=n0 * T * 2, scalar_offset=f[0], indirect_dim=0),
    )
    cn = t["cn"][0:parts, 0 : 2 * cols]
    if ctx.mode == "bilinear":
        cn = _load_coords(
            cn,
            t["cn16"],
            grid2.ap(pattern=[[2 * cols, parts], [1, 2 * cols]], offset=n0 * T * 2, scalar_offset=f[0], indirect_dim=0),
        )
    ix, iy, x0, y0, tmp, t2, t3, tb = _work_tiles(t["ti"], cols, parts, _PMAX)
    _coords(cx, ix, iy, x0, y0, tmp, t2, t3, W, H, ctx)
    idx = t["idx"]
    vidx = t["vidx"]
    if plan.exact:
        _exact_rows(t, x0, y0, parts, cols, plan)
    else:
        if plan.direct:
            _direct_indices(idx, x0, y0, (ix, iy, tmp, t2, t3, tb), pix_i, f, n1 - n0 > 1, parts, cols, W, H, T)
        else:
            _scratch_indices(idx, x0, y0, (tmp, tb), pix_i, f, n1 - n0 > 1, parts, cols, plan, T)
        for k in range(taps):
            ps = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(
                dst=ps[0:parts, 0:cols], data=idx[0:cols, k * parts : (k + 1) * parts], engine=nisa.engine.tensor
            )
            nisa.tensor_copy(
                dst=vidx[0:parts, k * cols : (k + 1) * cols], src=ps[0:parts, 0:cols], engine=nisa.engine.scalar
            )
    rows = t["rows"]
    for k in range(taps):
        nisa.dma_copy(
            dst=rows[0:parts, k * cols * row : (k + 1) * cols * row],
            src=ctx.scratch.ap(
                pattern=[[row, parts * cols], [1, row]],
                offset=n0 * plan.hs * plan.ws * row,
                vector_offset=vidx[0:parts, k * cols : (k + 1) * cols],
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.swdge,
        )
    if ctx.mode == "bilinear":
        sx, sy, sx0, sy0, stmp, s2, s3, s4 = _work_tiles(t["ts"], parts, cols, cols)
        _coords(cn, sx, sy, sx0, sy0, stmp, s2, s3, W, H, ctx)
        if plan.direct:
            _direct_valid(t["valid"], sx0, sy0, (stmp, s2, s3, s4), parts, cols, W, H)
        # fractions, then the four weights w[k] = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy)
        nisa.tensor_tensor(dst=sx, data1=sx, data2=sx0, op=nl.subtract)
        nisa.tensor_tensor(dst=sy, data1=sy, data2=sy0, op=nl.subtract)
        nisa.tensor_scalar(dst=sx0, data=sx, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
        nisa.tensor_scalar(dst=sy0, data=sy, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
        wts = t["wts"]
        for k in range(4):
            wx = sx0
            if k % 2 == 1:
                wx = sx
            wy = sy0
            if k // 2 == 1:
                wy = sy
            nisa.tensor_tensor(dst=wts[0:parts, k * cols : (k + 1) * cols], data1=wx, data2=wy, op=nl.multiply)
    _advance(f, chunk)


def _stage_a_sample_layout(ctx, seg, t, f, chunk):
    """
    _stage_a for quad chunks of one batch whose partitions hold q = parts / cols rows of cols pixels (cols a
    multiple of 32): everything in
    the sample layout. The scratch row of pixel r * cols + j is X[r, j], placed q times side by side, so the gather
    order's index of pixel m * parts + h * cols + j (sample position [m q + h, j]) is X^T[h cols + j, m q + h]: one PE
    transpose and q strided copies (Scalar) replace the index layout's coordinate pass, which computes the same
    integers on only cols partitions. The floors run over both axes at once (conversions on Scalar), and so do the
    fractions and the four weights (one instruction, broadcast access patterns); the same values as _stage_a. An
    exact plan's scratch rows come from _exact_rows_sample_layout.
    """
    n0, n1, parts, cols, _, _ = seg
    plan = ctx.plan
    N, C, H, W = ctx.input.shape
    T = ctx.grid.shape[1] * ctx.grid.shape[2]
    row = plan.row
    q = parts // cols
    grid2 = ctx.grid.reshape((N * T, 2))
    cn = t["cn"][0:parts, 0 : 2 * cols]
    cn = _load_coords(
        cn,
        t["cn16"],
        grid2.ap(pattern=[[2 * cols, parts], [1, 2 * cols]], offset=n0 * T * 2, scalar_offset=f[0], indirect_dim=0),
    )
    # ts: [ix | iy | x0 | y0 | scratch ...], cols columns each
    ts = t["ts"]
    tmp = ts[0:parts, 4 * cols : 5 * cols]
    t2 = ts[0:parts, 5 * cols : 6 * cols]
    t3 = ts[0:parts, 6 * cols : 7 * cols]
    sc = (tmp, t2, t3)
    _source_coords(
        ts[0:parts, 0:cols], cn.slice(dim=1, start=0, end=2 * cols, step=2), W, ctx.align_corners, ctx.padding_mode, sc
    )
    _source_coords(
        ts[0:parts, cols : 2 * cols],
        cn.slice(dim=1, start=1, end=2 * cols, step=2),
        H,
        ctx.align_corners,
        ctx.padding_mode,
        sc,
    )
    coord2 = ts[0:parts, 0 : 2 * cols]
    floor2 = ts[0:parts, 2 * cols : 4 * cols]
    _floor(floor2, coord2, ts[0:parts, 4 * cols : 6 * cols], nisa.engine.scalar)
    X = t["idx"]
    vidx = t["vidx"]
    if plan.exact:
        _exact_rows_sample_layout(t, floor2, parts, cols, q, plan)
    else:
        xa = X[0:parts, 0:cols]
        nisa.tensor_scalar(
            dst=xa,
            data=ts[0:parts, 3 * cols : 4 * cols],
            op0=nl.add,
            operand0=1.0,
            op1=nl.multiply,
            operand1=float(plan.ws),
        )
        nisa.tensor_tensor(dst=xa, data1=xa, data2=ts[0:parts, 2 * cols : 3 * cols], op=nl.add)
        nisa.tensor_scalar(dst=xa, data=xa, op0=nl.add, operand0=1.0)
        for h in range(1, q):
            nisa.tensor_copy(dst=X[0:parts, h * cols : (h + 1) * cols], src=xa, engine=nisa.engine.scalar)
        ps = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=ps[0:parts, 0:parts], data=X[0:parts, 0:parts], engine=nisa.engine.tensor)
        for h in range(q):
            nisa.tensor_copy(
                dst=vidx[h * cols : (h + 1) * cols, 0:cols],
                src=ps[h * cols : (h + 1) * cols, 0:parts].slice(dim=1, start=h, end=parts, step=q),
                engine=nisa.engine.scalar,
            )
    rows = t["rows"]
    nisa.dma_copy(
        dst=rows[0:parts, 0 : cols * row],
        src=ctx.scratch.ap(
            pattern=[[row, parts * cols], [1, row]],
            offset=n0 * plan.hs * plan.ws * row,
            vector_offset=vidx[0:parts, 0:cols],
            indirect_dim=0,
        ),
        dge_mode=nisa.dge_mode.swdge,
    )
    # F = [fx | fy] in place of the floors, G = [1 - fx | 1 - fy] in place of the coordinates, then w[dy, dx] =
    # (dx ? F : G)_x * (dy ? F : G)_y
    nisa.tensor_tensor(dst=floor2, data1=coord2, data2=floor2, op=nl.subtract)
    nisa.tensor_scalar(dst=coord2, data=floor2, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
    t4 = ts[0:parts, 0 : 4 * cols].reshape_dim(1, (2, 2, cols))
    wx = t4.slice(dim=2, start=0, end=1, step=1).permute((0, 2, 1, 3)).broadcast(1, 2)
    wy = t4.slice(dim=2, start=1, end=2, step=1).broadcast(2, 2)
    nisa.tensor_tensor(
        dst=t["wts"][0:parts, 0 : 4 * cols].reshape_dim(1, (2, 2, cols)), data1=wx, data2=wy, op=nl.multiply
    )
    _advance(f, chunk)


def _scratch_indices(idx, x0, y0, work, pix_i, f, multi, parts, cols, plan, T):
    """Scratch rows of the chunk's taps in the index layout: tap (dx, dy) of pixel m * parts + p at idx[m, k * parts
    + p], k = 2 dy + dx, the zero-padded scratch's entry (y0 + 1 + dy, x0 + 1 + dx) of its batch."""
    tmp, tb = work
    taps = plan.taps
    i0 = idx[0:cols, 0:parts]
    nisa.tensor_scalar(dst=i0, data=y0, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=float(plan.ws))
    nisa.tensor_tensor(dst=i0, data1=i0, data2=x0, op=nl.add)
    if multi:
        # + batch * scratch rows per batch: the batch of pixel f + pix is floor((f + pix + 0.5) / T)
        nisa.tensor_scalar(dst=tb, data=pix_i[0:cols, 0:parts], op0=nl.add, operand0=f[1][0:cols, 0:1])
        nisa.tensor_scalar(dst=tb, data=tb, op0=nl.add, operand0=0.5, op1=nl.multiply, operand1=1.0 / T)
        _floor(tb, tb, tmp)
        nisa.scalar_tensor_tensor(
            dst=i0, data=tb, op0=nl.multiply, operand0=float(plan.hs * plan.ws), op1=nl.add, operand1=i0
        )
    nisa.tensor_scalar(dst=i0, data=i0, op0=nl.add, operand0=1.0)
    if taps == 4:
        for k in range(1, 4):
            off = float((k // 2) * plan.ws + k % 2)
            nisa.tensor_scalar(dst=idx[0:cols, k * parts : (k + 1) * parts], data=i0, op0=nl.add, operand0=off)


def _exact_rows(t, x0, y0, parts, cols, plan):
    """
    The gather indices t["vidx"] (uint32, gather order) of an exact plan's chunk in the index layout (one batch:
    every exact plan's segments are): tap k = 2 dy + dx of the pixel with floors (or nearest integers) x0, y0 reads
    scratch row (y0 + 1 + dy) ws + x0 + 1 + dx, past fp32's exact integers. The small integers y0 + 1 and x0 + 1 are
    transposed separately; evicting them, the Scalar Engine forms the row bases (y0 + 1 + dy) ws, exact in fp32 (see
    grid_sample_scratch_width), and the columns x0 + 1 + dx as uint32, and GpSimd adds each tap's pair in integer
    arithmetic.
    """
    idx = t["idx"]
    vxy = t["vxy"]
    vidx = t["vidx"]
    nisa.tensor_scalar(dst=idx[0:cols, 0:parts], data=y0, op0=nl.add, operand0=1.0)
    nisa.tensor_scalar(dst=idx[0:cols, parts : 2 * parts], data=x0, op0=nl.add, operand0=1.0)
    psy = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
    psx = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=psy[0:parts, 0:cols], data=idx[0:cols, 0:parts], engine=nisa.engine.tensor)
    nisa.nc_transpose(dst=psx[0:parts, 0:cols], data=idx[0:cols, parts : 2 * parts], engine=nisa.engine.tensor)
    nd = 1
    if plan.taps == 4:
        nd = 2
    for d in range(nd):
        # the row base (y0 + 1 + d) ws at columns d cols .., the column x0 + 1 + d at (2 + d) cols ..
        nisa.activation(
            dst=vxy[0:parts, d * cols : (d + 1) * cols],
            op=nl.copy,
            data=psy[0:parts, 0:cols],
            scale=float(plan.ws),
            bias=float(d * plan.ws),
        )
        nisa.activation(
            dst=vxy[0:parts, (2 + d) * cols : (3 + d) * cols], op=nl.copy, data=psx[0:parts, 0:cols], bias=float(d)
        )
    for k in range(plan.taps):
        dy = k // 2
        dx = k % 2
        nisa.tensor_tensor(
            dst=vidx[0:parts, k * cols : (k + 1) * cols],
            data1=vxy[0:parts, dy * cols : (dy + 1) * cols],
            data2=vxy[0:parts, (2 + dx) * cols : (3 + dx) * cols],
            op=nl.add,
            engine=nisa.engine.gpsimd,
        )


def _exact_rows_sample_layout(t, floor2, parts, cols, q, plan):
    """
    _exact_rows for _stage_a_sample_layout: from the floors floor2 = [x0 | y0] (sample layout), x0 + 1 and y0 + 1 go
    side by side q times into two blocks of t["idx"] and are transposed separately (as one block of the fp32 path);
    the Scalar Engine evicts the row bases (y0 + 1) ws, exact in fp32, into t["vidx"] and the columns x0 + 1 into
    t["vxy"] as uint32, and GpSimd adds them in integer arithmetic.
    """
    X = t["idx"]
    vidx = t["vidx"]
    vx = t["vxy"]
    xy = X[0:parts, 0 : 2 * parts].reshape_dim(1, (2, parts))
    nisa.tensor_scalar(
        dst=xy.slice(dim=2, start=0, end=cols, step=1),
        data=floor2.reshape_dim(1, (2, cols)),
        op0=nl.add,
        operand0=1.0,
    )
    for h in range(1, q):
        nisa.tensor_copy(
            dst=xy.slice(dim=2, start=h * cols, end=(h + 1) * cols, step=1),
            src=xy.slice(dim=2, start=0, end=cols, step=1),
            engine=nisa.engine.scalar,
        )
    psx = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
    psy = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=psx[0:parts, 0:parts], data=X[0:parts, 0:parts], engine=nisa.engine.tensor)
    nisa.nc_transpose(dst=psy[0:parts, 0:parts], data=X[0:parts, parts : 2 * parts], engine=nisa.engine.tensor)
    for h in range(q):
        nisa.activation(
            dst=vidx[h * cols : (h + 1) * cols, 0:cols],
            op=nl.copy,
            data=psy[h * cols : (h + 1) * cols, 0:parts].slice(dim=1, start=h, end=parts, step=q),
            scale=float(plan.ws),
        )
        nisa.tensor_copy(
            dst=vx[h * cols : (h + 1) * cols, 0:cols],
            src=psx[h * cols : (h + 1) * cols, 0:parts].slice(dim=1, start=h, end=parts, step=q),
            engine=nisa.engine.scalar,
        )
    nisa.tensor_tensor(
        dst=vidx[0:parts, 0:cols],
        data1=vidx[0:parts, 0:cols],
        data2=vx[0:parts, 0:cols],
        op=nl.add,
        engine=nisa.engine.gpsimd,
    )


def _direct_indices(idx, x0, y0, work, pix_i, f, multi, parts, cols, W, H, T):
    """
    Input rows of the chunk's taps for a direct plan (one channel, no scratch), in the index layout: tap (dx, dy)
    of pixel m * parts + p at idx[m, k * parts + p], k = 2 dy + dx, the input pixel (x0 + dx, y0 + dy) of its batch,
    clamped into the input; _direct_valid marks the clamped taps, whose values stage B zeroes.
    """
    xa, xb, ya, yb, ra, tb = work
    nisa.tensor_scalar(dst=xa, data=x0, op0=nl.maximum, operand0=0.0, op1=nl.minimum, operand1=float(W - 1))
    nisa.tensor_scalar(dst=xb, data=x0, op0=nl.add, operand0=1.0, op1=nl.minimum, operand1=float(W - 1))
    nisa.tensor_scalar(dst=ya, data=y0, op0=nl.maximum, operand0=0.0, op1=nl.minimum, operand1=float(H - 1))
    nisa.tensor_scalar(dst=yb, data=y0, op0=nl.add, operand0=1.0, op1=nl.minimum, operand1=float(H - 1))
    rb = y0
    if multi:
        # the batch of pixel f + pix is floor((f + pix + 0.5) / T); its first input row is batch * H * W
        nisa.tensor_scalar(dst=tb, data=pix_i[0:cols, 0:parts], op0=nl.add, operand0=f[1][0:cols, 0:1])
        nisa.tensor_scalar(dst=tb, data=tb, op0=nl.add, operand0=0.5, op1=nl.multiply, operand1=1.0 / T)
        _floor(tb, tb, x0)
        nisa.tensor_scalar(dst=tb, data=tb, op0=nl.multiply, operand0=float(H * W))
        nisa.scalar_tensor_tensor(dst=ra, data=ya, op0=nl.multiply, operand0=float(W), op1=nl.add, operand1=tb)
        nisa.scalar_tensor_tensor(dst=rb, data=yb, op0=nl.multiply, operand0=float(W), op1=nl.add, operand1=tb)
    else:
        nisa.tensor_scalar(dst=ra, data=ya, op0=nl.multiply, operand0=float(W))
        nisa.tensor_scalar(dst=rb, data=yb, op0=nl.multiply, operand0=float(W))
    for k in range(4):
        r = ra
        if k // 2 == 1:
            r = rb
        x = xa
        if k % 2 == 1:
            x = xb
        nisa.tensor_tensor(dst=idx[0:cols, k * parts : (k + 1) * parts], data1=r, data2=x, op=nl.add)


def _direct_valid(valid, sx0, sy0, work, parts, cols, W, H):
    """valid[r, k * cols + j] (uint8) = 1 when tap k = 2 dy + dx of the sample-layout pixel r * cols + j, input
    pixel (sx0 + dx, sy0 + dy), lies outside the input (sx0, sy0 >= -1: the coordinates are clamped), else 0."""
    vx0, vx1, vy0, vy1 = work
    nisa.tensor_scalar(dst=vx0, data=sx0, op0=nl.greater_equal, operand0=0.0)
    nisa.scalar_tensor_tensor(
        dst=vx0, data=sx0, op0=nl.less_equal, operand0=float(W - 1), op1=nl.multiply, operand1=vx0
    )
    nisa.tensor_scalar(dst=vx1, data=sx0, op0=nl.less_equal, operand0=float(W - 2))
    nisa.tensor_scalar(dst=vy0, data=sy0, op0=nl.greater_equal, operand0=0.0)
    nisa.scalar_tensor_tensor(
        dst=vy0, data=sy0, op0=nl.less_equal, operand0=float(H - 1), op1=nl.multiply, operand1=vy0
    )
    nisa.tensor_scalar(dst=vy1, data=sy0, op0=nl.less_equal, operand0=float(H - 2))
    for k in range(4):
        vx = vx0
        if k % 2 == 1:
            vx = vx1
        vy = vy0
        if k // 2 == 1:
            vy = vy1
        vk = valid[0:parts, k * cols : (k + 1) * cols]
        nisa.tensor_tensor(dst=vk, data1=vx, data2=vy, op=nl.multiply)
        # 1 for the taps outside the input (the predicate of stage B's zeroing)
        nisa.tensor_scalar(dst=vk, data=vk, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)


def _load_coords(dst, stage, src):
    """The grid coordinates src (an access pattern at a chunk counter) in dst (fp32), or, for 16-bit coordinates,
    in stage (their dtype, dst's shape: the hardware DGE copies without a cast); returns the tile that holds them.
    16-bit coordinates are not widened first: every engine reads them as fp32, exactly."""
    if stage is None:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge)
        return dst
    st = stage[0 : dst.shape[0], 0 : dst.shape[1]]
    nisa.dma_copy(dst=st, src=src, dge_mode=nisa.dge_mode.hwdge)
    return st


def _stage_b(ctx, seg, t, f, chunk):
    """Stage B of the chunk at counters f: blend the gathered taps (bilinear) channel-planar, store (channel-major
    through the Tensor Engine with plan.tstore, else straight from the sample layout), and advance f."""
    n0, n1, parts, cols, _, dst = seg
    plan = ctx.plan
    C = ctx.input.shape[1]
    taps, row = plan.taps, plan.row
    rows = t["rows"]
    # the stored values: res [parts, C cols] channel-planar (pixel j of channel c at c * cols + j), or (nearest with
    # plan.tstore) the gathered rows themselves, channels last (j * C + c)
    res = None
    if ctx.mode == "nearest":
        if not plan.tstore:
            res = t["res"][0:parts, 0 : C * cols]
            nisa.tensor_copy(
                dst=res.reshape_dim(1, (C, cols)),
                src=rows[0:parts, 0 : cols * C].reshape_dim(1, (cols, C)).permute((0, 2, 1)),
                engine=nisa.engine.vector,
            )
    else:
        if plan.direct:
            # taps clamped into the input (outside it) count as zeros, as torch skips them
            nisa.tensor_copy_predicated(
                dst=rows[0:parts, 0 : 4 * cols],
                src=t["zeros"][0:parts, 0 : 4 * cols],
                predicate=t["valid"][0:parts, 0 : 4 * cols],
            )
        acc = t["acc"][0:parts, 0 : C * cols]
        acc3 = acc.reshape_dim(1, (C, cols))
        wts = t["wts"]
        prod3 = t["prod"][0:parts, 0 : C * cols].reshape_dim(1, (C, cols))
        n = C * cols
        # the tile the blend's last add writes: the stored 16-bit tile (the add's own rounding is the cast), else acc
        last = acc
        if ctx.input.dtype != nl.float32 and not plan.tstore:
            last = t["res"][0:parts, 0 : C * cols]
        if plan.blend4:
            # the four tap products in one instruction (taps [k, c, j] times weights broadcast over c), then the
            # same three adds in the same order: ((p0 + p1) + p2) + p3
            tap4 = rows[0:parts, 0 : cols * row].reshape_dim(1, (cols, 4, C)).permute((0, 2, 3, 1))
            w4 = wts[0:parts, 0 : 4 * cols].reshape_dim(1, (4, cols)).expand_dim(2).broadcast(2, C)
            p4 = t["prod"]
            nisa.tensor_tensor(
                dst=p4[0:parts, 0 : 4 * n].reshape_dim(1, (4, C, cols)), data1=tap4, data2=w4, op=nl.multiply
            )
            nisa.tensor_tensor(dst=acc, data1=p4[0:parts, 0:n], data2=p4[0:parts, n : 2 * n], op=nl.add)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=p4[0:parts, 2 * n : 3 * n], op=nl.add)
            nisa.tensor_tensor(dst=last, data1=acc, data2=p4[0:parts, 3 * n : 4 * n], op=nl.add)
        else:
            for k in range(4):
                if taps == 1:
                    tap = rows[0:parts, 0 : cols * row].reshape_dim(1, (cols, 4 * C))
                    tap = tap.slice(dim=2, start=k * C, end=(k + 1) * C, step=1).permute((0, 2, 1))
                else:
                    tap = (
                        rows[0:parts, k * cols * row : (k + 1) * cols * row]
                        .reshape_dim(1, (cols, C))
                        .permute((0, 2, 1))
                    )
                w = wts[0:parts, k * cols : (k + 1) * cols].expand_dim(1).broadcast(1, C)
                if k == 0:
                    nisa.tensor_tensor(dst=acc3, data1=tap, data2=w, op=nl.multiply)
                else:
                    nisa.tensor_tensor(dst=prod3, data1=tap, data2=w, op=nl.multiply)
                    out3 = acc3
                    if k == 3:
                        out3 = last.reshape_dim(1, (C, cols))
                    nisa.tensor_tensor(dst=out3, data1=acc3, data2=prod3, op=nl.add)
        res = last
    if plan.tstore:
        if res is None:
            _store_channel_major(ctx, t, rows[0:parts, 0 : cols * C], True, parts, cols, f, dst)
        else:
            _store_channel_major(ctx, t, res, False, parts, cols, f, dst)
    else:
        # partition r's run of cols pixels per channel: contiguous at dst (the batch's [C, T] rows, or the staging)
        nisa.dma_copy(
            dst=dst.t.ap(
                pattern=[[cols, parts], [dst.stride, C], [1, cols]], offset=dst.base, scalar_offset=f[0], indirect_dim=0
            ),
            src=res,
            dge_mode=nisa.dge_mode.hwdge,
        )
    _advance(f, chunk)


def _store_channel_major(ctx, t, src2, jc, parts, cols, f, dst):
    """
    Store a chunk channel-major: src2 [parts, C cols] holds pixel r * cols + j of channel c at column c * cols + j
    (jc False) or j * C + c (jc True). Per block of up to 128 channels, the Tensor Engine transposes each pixel
    column j (bit-accurate; its input a single strided free dimension) into a PSUM bank, as many columns as fill
    it, the Scalar Engine evicts them into a region of t["tst"] (channel c on partition c, pixel r * cols + j at
    column r * cols + j, cast to the output dtype), and one DMA stores the block's parts * cols contiguous pixels
    per channel at dst.
    """
    C = ctx.input.shape[1]
    dt = ctx.input.dtype
    pdt = src2.dtype
    peb = 4
    if pdt != nl.float32:
        peb = 2
    chunk = parts * cols
    per_bank = max(1, _PSUM_BANK_BYTES // (peb * parts))
    tst = t["tst"]
    eb = 4
    if dt != nl.float32:
        eb = 2
    n_reg = _tstore_regions(ctx.plan, C, eb)
    region = _PMAX * ctx.plan.cols
    for bi in range((C + _PMAX - 1) // _PMAX):
        c0 = bi * _PMAX
        cb = min(_PMAX, C - c0)
        r0 = (bi % n_reg) * region
        stb = tst[0:cb, r0 : r0 + chunk].reshape_dim(1, (parts, cols))
        for j0 in range(0, cols, per_bank):
            nj = min(per_bank, cols - j0)
            ps = nl.ndarray((_PMAX, per_bank * parts), dtype=pdt, buffer=nl.psum)
            for jl in range(nj):
                j = j0 + jl
                if jc:
                    data = src2[0:parts, j * C + c0 : j * C + c0 + cb]
                else:
                    data = src2.slice(dim=1, start=c0 * cols + j, end=(c0 + cb) * cols, step=cols)
                nisa.nc_transpose(dst=ps[0:cb, jl * parts : (jl + 1) * parts], data=data, engine=nisa.engine.tensor)
            if nj == 1:
                evict_dst = tst[0:cb, r0 + j0 : r0 + chunk].slice(dim=1, start=0, end=chunk - j0, step=cols)
                evict_src = ps[0:cb, 0:parts]
            else:
                evict_dst = stb.slice(dim=2, start=j0, end=j0 + nj, step=1).permute((0, 2, 1))
                evict_src = ps[0:cb, 0 : nj * parts].reshape_dim(1, (nj, parts))
            nisa.tensor_copy(dst=evict_dst, src=evict_src, engine=nisa.engine.scalar)
        nisa.dma_copy(
            dst=dst.t.ap(
                pattern=[[dst.stride, cb], [1, chunk]],
                offset=dst.base + c0 * dst.stride,
                scalar_offset=f[0],
                indirect_dim=0,
            ),
            src=tst[0:cb, r0 : r0 + chunk],
            dge_mode=nisa.dge_mode.hwdge,
        )


def _source_coords(dst, g, size, align_corners, padding_mode, tmp):
    """
    dst = the input-pixel coordinate of the normalized coordinates g (ATen's operation order), after the padding
    transform (border: clamp to [0, size - 1]; reflection: reflect, then clamp), clamped to [-1, size] (a coordinate
    outside it samples only zeros, as it does at -1 or size). tmp: three scratch tiles of g's shape.
    """
    if align_corners:
        nisa.tensor_scalar(dst=dst, data=g, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=0.5)
        nisa.tensor_scalar(dst=dst, data=dst, op0=nl.multiply, operand0=float(size - 1))
    else:
        nisa.tensor_scalar(dst=dst, data=g, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=float(size))
        nisa.tensor_scalar(dst=dst, data=dst, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=0.5)
    if padding_mode == "reflection":
        _reflect(dst, size, align_corners, tmp)
    if padding_mode == "zeros":
        nisa.tensor_scalar(dst=dst, data=dst, op0=nl.maximum, operand0=-1.0, op1=nl.minimum, operand1=float(size))
    else:
        nisa.tensor_scalar(dst=dst, data=dst, op0=nl.maximum, operand0=0.0, op1=nl.minimum, operand1=float(size - 1))


def _reflect(x, size, align_corners, tmp):
    """ATen's reflect_coordinates in place: reflect x about the edges (pixel centers with align_corners, else pixel
    outer edges) until it lands in range."""
    twice_low = -1
    twice_high = 2 * size - 1
    if align_corners:
        twice_low = 0
        twice_high = 2 * (size - 1)
    if twice_low == twice_high:
        nisa.memset(dst=x, value=0.0)
        return
    lo = twice_low / 2.0
    span = (twice_high - twice_low) / 2.0
    t, extra, flips = tmp
    # t = |x - lo|, saturated at 2^23 (below it every step is exact: span is an integer, so flips * span and
    # t - flips * span are exact in fp32); flips = floor(t / span) through the reciprocal, then corrected so that
    # extra = t - flips * span lands in [0, span) (fmod for t >= 0)
    nisa.tensor_scalar(dst=t, data=x, op0=nl.subtract, operand0=lo)
    nisa.tensor_scalar(dst=extra, data=t, op0=nl.multiply, operand0=-1.0)
    nisa.tensor_tensor(dst=t, data1=t, data2=extra, op=nl.maximum)
    nisa.tensor_scalar(dst=t, data=t, op0=nl.minimum, operand0=float(1 << 23))
    nisa.tensor_scalar(dst=extra, data=t, op0=nl.multiply, operand0=1.0 / span)
    _floor(flips, extra, x)
    nisa.tensor_scalar(dst=extra, data=flips, op0=nl.multiply, operand0=-span)
    nisa.tensor_tensor(dst=extra, data1=extra, data2=t, op=nl.add)
    for i in range(2):
        # i = 0: extra < 0 -> flips - 1, extra + span; i = 1: extra >= span -> flips + 1, extra - span
        if i == 0:
            nisa.tensor_scalar(dst=x, data=extra, op0=nl.less, operand0=0.0)
            sign = -1.0
        else:
            nisa.tensor_scalar(dst=x, data=extra, op0=nl.greater_equal, operand0=span)
            sign = 1.0
        nisa.scalar_tensor_tensor(dst=flips, data=x, op0=nl.multiply, operand0=sign, op1=nl.add, operand1=flips)
        nisa.scalar_tensor_tensor(dst=extra, data=x, op0=nl.multiply, operand0=-sign * span, op1=nl.add, operand1=extra)
    # parity of flips: flips - 2 floor(flips / 2), 0 or 1
    nisa.tensor_scalar(dst=t, data=flips, op0=nl.multiply, operand0=0.5)
    _floor(t, t, x)
    nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=-2.0)
    nisa.tensor_tensor(dst=flips, data1=flips, data2=t, op=nl.add)
    # odd: span - extra + lo, even: extra + lo, i.e. extra + lo + parity * (span - 2 extra)
    nisa.tensor_scalar(dst=t, data=extra, op0=nl.multiply, operand0=-2.0, op1=nl.add, operand1=span)
    nisa.tensor_tensor(dst=t, data1=t, data2=flips, op=nl.multiply)
    nisa.tensor_scalar(dst=x, data=extra, op0=nl.add, operand0=lo)
    nisa.tensor_tensor(dst=x, data1=x, data2=t, op=nl.add)


def _floor(dst, src, tmp, eng=nisa.engine.vector):
    """dst = floor(src) for |src| < 2^23: the fp32 -> int32 conversion (on eng) lands within 1 of src (rounding to
    nearest or toward zero), and the result is corrected downward wherever it exceeds src."""
    i32 = tmp.view(nl.int32)
    nisa.tensor_copy(dst=i32, src=src, engine=eng)
    nisa.tensor_copy(dst=tmp, src=i32, engine=eng)
    nisa.tensor_tensor(dst=dst, data1=tmp, data2=src, op=nl.greater)
    nisa.tensor_tensor(dst=dst, data1=tmp, data2=dst, op=nl.subtract)


def _coords(c, ix, iy, x0, y0, tmp, t2, t3, W, H, ctx):
    """From interleaved (x, y) coordinates c [rows, 2 cols]: ix, iy (source coordinates, see _source_coords) and
    their floors x0, y0 (bilinear) or nearest integers (nearest, ix, iy unchanged)."""
    cols = ix.shape[1]
    gx = c.slice(dim=1, start=0, end=2 * cols, step=2)
    gy = c.slice(dim=1, start=1, end=2 * cols, step=2)
    sc = (tmp, t2, t3)
    _source_coords(ix, gx, W, ctx.align_corners, ctx.padding_mode, sc)
    _source_coords(iy, gy, H, ctx.align_corners, ctx.padding_mode, sc)
    if ctx.mode == "nearest":
        # the fp32 -> int32 conversion rounds half to even, as torch's nearbyint
        pairs = ((ix, x0), (iy, y0))
        for i in range(2):
            nisa.tensor_copy(dst=tmp.view(nl.int32), src=pairs[i][0], engine=nisa.engine.vector)
            nisa.tensor_copy(dst=pairs[i][1], src=tmp.view(nl.int32), engine=nisa.engine.vector)
    else:
        _floor(x0, ix, tmp)
        _floor(y0, iy, tmp)
