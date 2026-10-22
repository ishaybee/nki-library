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

"""Multi-scale deformable attention forward on trn2 through grid_sample's quad scratch: one gathered row per sample."""

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
MSDA_QUAD_NKI_SUPPORTED = nki_at_least(0, 7, 0)

_PMAX = 128
_MAX_SEG = 8  # quad-scratch build tile width, padded columns, at most
_BUILD_TILE_ELEMS = 8192  # elements per partition of a build tile's quad entries, at most
_MAX_DEPTH = 3  # (query tile, level) work items in flight: stage A runs depth - 1 items ahead of stage B
_MAX_EXACT = 1 << 24  # scratch rows held in fp32 stay exact below this
_MAX_GATHER_SLOTS = 64  # indices per partition of one batched indirect DMA, at most (8192 rows)
_TRN2_SBUF_BYTES = 212984  # SBUF bytes per partition a trn2 kernel may allocate
_SBUF_SHARE = 0.85  # share of SBUF the sampling tiles may fill


class MsdaQuadPlan(nl.NKIObject):
    """Tiling of one ms_deformable_attention_quad call (see plan_ms_deformable_attention_quad)."""

    def __init__(self, jp, groups, depth, seg, ok, reason):
        self.jp = jp  # gather slots per query and level: N_h * N_p padded to 32, 64 or 128
        self.groups = groups  # heads per blend instruction
        self.depth = depth  # work items in flight
        self.seg = seg  # build tile width in padded columns
        self.ok = ok  # False: the call is not supported (reason says why)
        self.reason = reason


def msda_quad_sbuf_bytes(NH, CH, NL, NP, jp, groups, depth, value_bytes, loc_bytes, aw_bytes):
    """SBUF bytes per partition of the sampling phase: depth sets of work-item tiles (coordinates, weights, indices,
    gathered rows), depth + 1 query-tile sets (locations, attention weights, sum, result), and the tiles stage B
    shares (one group's products, a level's sum) and the slot constants."""
    per_item = (6 * jp + 8 * jp + _PMAX + jp) * 4 + jp * 4 * CH * value_bytes
    if jp > _MAX_GATHER_SLOTS:
        per_item = per_item + _PMAX * 4
    K = NH * NL * NP
    per_query = 2 * K * loc_bytes + K * aw_bytes + NH * CH * value_bytes
    shared = groups * CH * NP * 4 * 4 + 2 * jp * 4
    if NL > 1:
        per_query = per_query + NH * CH * 4
        shared = shared + NH * CH * 4
    return depth * per_item + (depth + 1) * per_query + shared


def plan_ms_deformable_attention_quad(
    NH, CH, NL, NP, value_bytes, loc_bytes, aw_bytes, sbuf_bytes=_TRN2_SBUF_BYTES
) -> MsdaQuadPlan:
    """
    Gather slots per query and level, heads per blend instruction and work items in flight: the deepest pipeline
    (3, else 2) whose tiles fit _SBUF_SHARE of SBUF with the most heads per blend; build tiles of at most
    _BUILD_TILE_ELEMS quad elements per partition.
    """
    J = NH * NP
    jp = 128
    if J <= 32:
        jp = 32
    elif J <= 64:
        jp = 64
    seg = max(1, min(_MAX_SEG, _BUILD_TILE_ELEMS // (4 * NH * CH)))
    if J > _PMAX:
        return MsdaQuadPlan(jp, 1, 2, seg, False, f"N_h * N_p = {J} exceeds 128")
    budget = _SBUF_SHARE * sbuf_bytes
    for depth in range(_MAX_DEPTH, 1, -1):
        for groups in range(NH, 0, -1):
            nbytes = msda_quad_sbuf_bytes(NH, CH, NL, NP, jp, groups, depth, value_bytes, loc_bytes, aw_bytes)
            if nbytes <= budget:
                return MsdaQuadPlan(jp, groups, depth, seg, True, "")
    return MsdaQuadPlan(jp, 1, 2, seg, False, f"N_h = {NH}, C_h = {CH}, N_p = {NP} do not fit in SBUF")


def msda_quad_work(B, NQ, n_prgs, prg_id):
    """
    The query tiles [(batch, first query, queries)] of program prg_id (of n_prgs): tiles of 128 queries per batch
    (the last overlapping the one before it), or one tile of N_q queries when N_q < 128. Every program takes the
    same number of consecutive tiles (one control flow); the last range may overlap the one before it, those tiles
    then stored twice with the same values.
    """
    tiles = []
    for b in range(B):
        if NQ < _PMAX:
            tiles.append((b, 0, NQ))
        else:
            for q0 in range(0, NQ - _PMAX + 1, _PMAX):
                tiles.append((b, q0, _PMAX))
            if NQ % _PMAX > 0:
                tiles.append((b, NQ - _PMAX, _PMAX))
    count = (len(tiles) + n_prgs - 1) // n_prgs
    first = min(count * prg_id, len(tiles) - count)
    return tiles[first : first + count]


@nki.jit
def ms_deformable_attention_quad(
    value: nl.NkiTensor,
    spatial_shapes: tuple,
    level_start_index: tuple,
    sampling_locations: nl.NkiTensor,
    attention_weights: nl.NkiTensor,
    value_layout: str = "BLNC",
    sampling_locations_layout: str = "BQHLP2",
    align_corners: bool = False,
    padding_mode: str = "zeros",
) -> nl.NkiTensor:
    """
    Multi-scale deformable attention forward, the same call as ms_deformable_attention, through a quad scratch.

    As in grid_sample (experimental/vision), every level of the value is first rewritten into an HBM scratch whose
    entry (y, x) for head h holds the 2 x 2 neighbourhood of padded pixel (y, x) (taps (dy, dx) = pixel
    (y + dy - 1, x + dx - 1), zero outside the level), 4 C_h contiguous elements, so a bilinear sample is one
    gathered row and the zero border stands in for the zeros padding's out-of-range corners. Per tile of up to 128
    queries and per level, the coordinates, the bilinear weights times the attention weights, and the scratch rows
    of all N_h x N_p samples are computed with the queries on the partitions; one batched indirect DMA gathers the
    N_h N_p rows of every query (the indices transposed into the gather's order on the Tensor Engine), and the blend
    is one multiply and one reduction over (point, tap) per group of heads. Border padding clamps the coordinates to
    the level, which equals clamping each corner (the corner weights sum to one). Coordinates, weights and sums are
    fp32; the output is rounded once. With LNC=2 each program builds its own scratch and takes half of the query
    tiles.

    Dimensions:
        B, N_q, N_h, C_h, N_l, N_p: batch, queries, heads, channels per head, levels, points per level
        L: sum of H_l * W_l

    Args:
        value (nl.NkiTensor): (B, L, N_h, C_h) ("BLNC") or (B, N_h, L, C_h) ("BNLC"), float32, bfloat16 or float16.
        spatial_shapes (tuple): ((H_l, W_l), ...) per level.
        level_start_index (tuple): First row of each level in L.
        sampling_locations (nl.NkiTensor): (B, N_q, N_h, N_l, N_p, 2) ("BQHLP2") or (B, 2, N_q, N_h, N_l, N_p)
            ("B2QHLP"), normalized (x, y) in [0, 1].
        attention_weights (nl.NkiTensor): (B, N_q, N_h, N_l, N_p).
        value_layout (str): "BLNC" or "BNLC".
        sampling_locations_layout (str): "BQHLP2" or "B2QHLP".
        align_corners (bool): True: [0, 1] maps to [0, size - 1]; False: to [-0.5, size - 0.5].
        padding_mode (str): "zeros" or "border".

    Returns:
        output (nl.NkiTensor): (B, N_q, N_h * C_h) in value's dtype.

    Notes:
        N_h * N_p <= 128 (padded to 32, 64 or 128 gather slots per query); the work-item tiles must fit SBUF.

    Pseudocode:
        scratch[l][b, y, x, h] = 2 x 2 neighbourhood of zero-padded value level l at (y, x), head h
        for tile of 128 queries, for level l:
            x, y = loc * size - 0.5 (or loc * (size - 1)); clamp; x0, y0 = floor(x), floor(y)
            rows[q, (h, p)] = gather(scratch[l], (y0 + 1) * (W + 2) + x0 + 1, h)    # one row of 4 C_h per sample
            out[q, h] += sum over (p, tap) of bilinear_w[q, h, p, tap] * attn_w[q, h, l, p] * rows[q, (h, p), tap]
    """
    if value_layout == "BLNC":
        B, L, NH, CH = value.shape
    else:
        B, NH, L, CH = value.shape
    if sampling_locations_layout == "BQHLP2":
        NQ = sampling_locations.shape[1]
        NL = sampling_locations.shape[3]
        NP = sampling_locations.shape[4]
    else:
        NQ = sampling_locations.shape[2]
        NL = sampling_locations.shape[4]
        NP = sampling_locations.shape[5]
    kernel_assert(nisa.get_nc_version() == nisa.nc_version.gen3, "ms_deformable_attention_quad supports trn2 only")
    kernel_assert(
        MSDA_QUAD_NKI_SUPPORTED,
        f"ms_deformable_attention_quad needs nki 0.7.0 or later (Neuron SDK 2.33), got {nki.__version__}: earlier "
        "releases reject its batched indirect DMA gathers",
    )
    kernel_assert(
        value_layout == "BLNC" or value_layout == "BNLC", f"value_layout must be BLNC or BNLC, got {value_layout}"
    )
    kernel_assert(
        sampling_locations_layout == "BQHLP2" or sampling_locations_layout == "B2QHLP",
        f"sampling_locations_layout must be BQHLP2 or B2QHLP, got {sampling_locations_layout}",
    )
    kernel_assert(
        padding_mode == "zeros" or padding_mode == "border", f"padding_mode must be zeros or border, got {padding_mode}"
    )
    kernel_assert(len(spatial_shapes) == NL and len(level_start_index) == NL, "one spatial shape and start per level")
    eb = _dtype_bytes(value.dtype)
    plan = plan_ms_deformable_attention_quad(
        NH,
        CH,
        NL,
        NP,
        eb,
        _dtype_bytes(sampling_locations.dtype),
        _dtype_bytes(attention_weights.dtype),
        nl.tile_size.total_available_sbuf_size,
    )
    kernel_assert(plan.ok, f"ms_deformable_attention_quad: {plan.reason}")
    offs = []
    S = 0
    for li in range(NL):
        H, W = spatial_shapes[li]
        offs.append(S)
        S = S + (H + 2) * (W + 2)
    kernel_assert(
        B * S * NH < _MAX_EXACT, f"ms_deformable_attention_quad: quad scratch of {B * S * NH} rows exceeds 2^24"
    )
    _, n_prgs, prg_id = get_verified_program_sharding_info("ms_deformable_attention_quad", (0, 1))
    dt = value.dtype
    scratch = nl.ndarray((B * S * NH, 4 * CH), dtype=dt, buffer=nl.private_hbm)
    for b in range(B):
        for li in range(NL):
            _build_level(value, scratch, value_layout, b, spatial_shapes[li], level_start_index[li], offs[li], S, plan)
    output = nl.ndarray((B, NQ, NH * CH), dtype=dt, buffer=nl.shared_hbm)
    geo = _Geometry(B, NQ, NH, CH, NL, NP, S, offs, spatial_shapes, plan)
    tiles = msda_quad_work(B, NQ, n_prgs, prg_id)
    _sample(
        scratch,
        sampling_locations,
        attention_weights,
        output,
        geo,
        tiles,
        sampling_locations_layout,
        align_corners,
        padding_mode,
    )
    return output


def _dtype_bytes(dt):
    if dt == nl.float32:
        return 4
    return 2


class _Geometry(nl.NKIObject):
    """The call's dimensions, the quad scratch's level offsets and the plan."""

    def __init__(self, B, NQ, NH, CH, NL, NP, S, offs, shapes, plan):
        self.B = B
        self.NQ = NQ
        self.NH = NH
        self.CH = CH
        self.NL = NL
        self.NP = NP
        self.J = NH * NP  # samples (head, point) per query and level
        self.JP = plan.jp
        self.S = S  # quad entries per head and batch over all levels
        self.offs = offs  # first quad entry of each level
        self.shapes = shapes
        self.plan = plan


def _interleave_engine(dx, dt):
    """dx = 0 copies on Vector, dx = 1 on Scalar, except at 16 bits (Scalar's 16-bit copies are slow)."""
    if dx == 1 and dt == nl.float32:
        return nisa.engine.scalar
    return nisa.engine.vector


def _build_level(value, scratch, value_layout, b, shape, start, off, S, plan):
    """
    The quad scratch of one level of batch b, in tiles of up to 128 padded rows by plan.seg padded columns: per tile
    two loads (the input rows under the quads' top and bottom taps, zero outside the level), four interleave copies
    and one store of all heads' entries.
    """
    if value_layout == "BLNC":
        _, L, NH, CH = value.shape
    else:
        _, NH, L, CH = value.shape
    H, W = shape
    Hs = H + 2
    Ws = W + 2
    E = NH * CH
    dt = value.dtype
    seg = plan.seg
    flat_v = value.reshape((value.shape[0] * L * E, 1))
    flat_s = scratch.reshape((scratch.shape[0] * 4 * CH, 1))
    for y0 in range(0, Hs, _PMAX):
        r = min(_PMAX, Hs - y0)
        for x0 in range(0, Ws, seg):
            wseg = min(seg, Ws - x0)
            wsrc = wseg + 1
            srcs = []
            for dy in range(2):
                t = nl.ndarray((_PMAX, (seg + 1) * E), dtype=dt, buffer=nl.sbuf)
                # row p of tile dy holds input row y0 + p + dy - 1, column j input column x0 + j - 1
                p_lo = max(0, 1 - dy - y0)
                p_hi = min(r, H + 1 - dy - y0)
                j_lo = max(0, 1 - x0)
                j_hi = min(wsrc, W + 1 - x0)
                covered = p_lo == 0 and p_hi == r and j_lo == 0 and j_hi == wsrc
                if not covered:
                    nisa.memset(dst=t[0:r, 0 : wsrc * E], value=0.0)
                if p_lo < p_hi and j_lo < j_hi:
                    row0 = y0 + p_lo + dy - 1
                    col0 = x0 + j_lo - 1
                    nr = p_hi - p_lo
                    nc = j_hi - j_lo
                    if value_layout == "BLNC":
                        nisa.dma_copy(
                            dst=t[p_lo:p_hi, j_lo * E : j_hi * E],
                            src=flat_v.ap(
                                pattern=[[W * E, nr], [1, nc * E]], offset=(b * L + start + row0 * W + col0) * E
                            ),
                        )
                    else:
                        nisa.dma_copy(
                            dst=t[p_lo:p_hi, j_lo * E : j_hi * E].reshape_dim(1, (nc, NH, CH)).permute((0, 2, 1, 3)),
                            src=flat_v.ap(
                                pattern=[[W * CH, nr], [L * CH, NH], [CH, nc], [1, CH]],
                                offset=(b * NH * L + start + row0 * W + col0) * CH,
                            ),
                        )
                srcs.append(t)
            q = nl.ndarray((_PMAX, seg * 4 * E), dtype=dt, buffer=nl.sbuf)
            qv = q[0:r, 0 : wseg * 4 * E].reshape_dim(1, (wseg, NH, 4, CH))
            for dy in range(2):
                sv = srcs[dy][0:r, 0 : wsrc * E].reshape_dim(1, (wsrc, NH, CH))
                for dx in range(2):
                    nisa.tensor_copy(
                        dst=qv.select(3, 2 * dy + dx),
                        src=sv.slice(dim=1, start=dx, end=dx + wseg, step=1),
                        engine=_interleave_engine(dx, dt),
                    )
            nisa.dma_copy(
                dst=flat_s.ap(
                    pattern=[[Ws * NH * 4 * CH, r], [1, wseg * NH * 4 * CH]],
                    offset=((b * S + off + y0 * Ws + x0) * NH) * 4 * CH,
                ),
                src=q[0:r, 0 : wseg * 4 * E],
            )


def _floor(dst, src, tmp, eng):
    """dst = floor(src) for |src| < 2^23 (the int32 conversion lands within 1, then a downward correction)."""
    i32 = tmp.view(nl.int32)
    nisa.tensor_copy(dst=i32, src=src, engine=eng)
    nisa.tensor_copy(dst=tmp, src=i32, engine=eng)
    nisa.tensor_tensor(dst=dst, data1=tmp, data2=src, op=nl.greater)
    nisa.tensor_tensor(dst=dst, data1=tmp, data2=dst, op=nl.subtract)


def _item_tiles(geo, dt):
    """One set of per-(query tile, level) tiles; the padded slots stay zero (index 0, weight 0)."""
    JP = geo.JP
    t = {}
    t["xy"] = nl.ndarray((_PMAX, 2 * JP), dtype=nl.float32, buffer=nl.sbuf)
    t["fl"] = nl.ndarray((_PMAX, 2 * JP), dtype=nl.float32, buffer=nl.sbuf)
    t["tmp"] = nl.ndarray((_PMAX, 2 * JP), dtype=nl.float32, buffer=nl.sbuf)
    t["fg"] = nl.ndarray((_PMAX, 4 * JP), dtype=nl.float32, buffer=nl.sbuf)
    t["w4"] = nl.ndarray((_PMAX, 4 * JP), dtype=nl.float32, buffer=nl.sbuf)
    t["X"] = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    t["Xr"] = None
    if JP > _MAX_GATHER_SLOTS:
        t["Xr"] = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    t["vidx"] = nl.ndarray((_PMAX, JP), dtype=nl.uint32, buffer=nl.sbuf)
    t["rows"] = nl.ndarray((_PMAX, JP * 4 * geo.CH), dtype=dt, buffer=nl.sbuf)
    if geo.J < JP:
        for k in ("xy", "fl", "fg", "X"):
            nisa.memset(dst=t[k], value=0.0)
    return t


def _query_tiles(geo, loc_dt, aw_dt, out_dt):
    """One set of per-query-tile tiles: sampling locations (x plane, y plane), attention weights, the fp32 sum and
    the stored result. Short tiles (N_q < 128) keep zero locations and weights on the partitions past N_q."""
    K = geo.NH * geo.NL * geo.NP
    t = {}
    t["loc"] = nl.ndarray((_PMAX, 2 * K), dtype=loc_dt, buffer=nl.sbuf)
    t["aw"] = nl.ndarray((_PMAX, K), dtype=aw_dt, buffer=nl.sbuf)
    if geo.NQ < _PMAX:
        nisa.memset(dst=t["loc"], value=0.0)
        nisa.memset(dst=t["aw"], value=0.0)
    t["acc"] = None
    if geo.NL > 1:
        t["acc"] = nl.ndarray((_PMAX, geo.NH * geo.CH), dtype=nl.float32, buffer=nl.sbuf)
    t["res"] = nl.ndarray((_PMAX, geo.NH * geo.CH), dtype=out_dt, buffer=nl.sbuf)
    return t


def _sample(scratch, locs, aw, output, geo, tiles, loc_layout, align_corners, padding_mode):
    """
    Every (query tile, level) work item through stage A (coordinates, weights, scratch rows, gather) and stage B
    (blend, accumulate, store after the last level), software-pipelined over plan.depth tile sets: stage A runs
    depth - 1 items ahead of stage B, so the gathers of the next items are in flight while one blends.
    """
    NH, NL, NP, J, JP = geo.NH, geo.NL, geo.NP, geo.J, geo.JP
    items = []
    for ti in range(len(tiles)):
        b, q0, nq = tiles[ti]
        for li in range(NL):
            items.append((b, q0, nq, li, ti))
    D = min(geo.plan.depth, len(items))
    sets = []
    for s in range(D):
        sets.append(_item_tiles(geo, scratch.dtype))
    QS = D + 1
    qsets = []
    for s in range(QS):
        qsets.append(_query_tiles(geo, locs.dtype, aw.dtype, output.dtype))
    # the head of each slot (h, p), as fp32
    hc_i = nl.ndarray((_PMAX, JP), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=hc_i[0:_PMAX, 0:J].reshape_dim(1, (NH, NP)), pattern=[[1, NH], [0, NP]], offset=0)
    hc = nl.ndarray((_PMAX, JP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=hc[0:_PMAX, 0:J], src=hc_i[0:_PMAX, 0:J], engine=nisa.engine.vector)
    # stage B's tiles, shared by the work items (their blends follow each other on Vector)
    prod = nl.ndarray((_PMAX, geo.plan.groups * geo.CH * NP * 4), dtype=nl.float32, buffer=nl.sbuf)
    red = None
    if NL > 1:
        red = nl.ndarray((_PMAX, NH * geo.CH), dtype=nl.float32, buffer=nl.sbuf)
    ctx = (scratch, locs, aw, output, geo, loc_layout, align_corners, padding_mode, hc, prod, red)
    n = len(items)
    for i in range(D - 1):
        _stage_a(ctx, items[i], sets[i % D], qsets[items[i][4] % QS])
    for i in range(n):
        j = i + D - 1
        if j < n:
            _stage_a(ctx, items[j], sets[j % D], qsets[items[j][4] % QS])
        _stage_b(ctx, items[i], sets[i % D], qsets[items[i][4] % QS])


def _stage_a(ctx, item, t, qt):
    """Coordinates, weights, scratch rows and the gather of one (query tile, level) work item."""
    scratch, locs, aw, output, geo, loc_layout, align_corners, padding_mode, hc, _, _ = ctx
    b, q0, nq, li, _ = item
    NQ, NH, NL, NP, J, JP, CH = geo.NQ, geo.NH, geo.NL, geo.NP, geo.J, geo.JP, geo.CH
    K = NH * NL * NP
    H, W = geo.shapes[li]
    Ws = W + 2
    if li == 0:
        # the query tile's sampling locations (x plane, then y plane, or interleaved) and attention weights
        if loc_layout == "BQHLP2":
            lf = locs.reshape((geo.B * NQ, 2 * K))
            nisa.dma_copy(dst=qt["loc"][0:nq, 0 : 2 * K], src=lf[b * NQ + q0 : b * NQ + q0 + nq, 0 : 2 * K])
        else:
            lf = locs.reshape((geo.B * 2 * NQ, K))
            for a in range(2):
                r0 = (b * 2 + a) * NQ + q0
                nisa.dma_copy(dst=qt["loc"][0:nq, a * K : (a + 1) * K], src=lf[r0 : r0 + nq, 0:K])
        af = aw.reshape((geo.B * NQ, K))
        nisa.dma_copy(dst=qt["aw"][0:nq, 0:K], src=af[b * NQ + q0 : b * NQ + q0 + nq, 0:K])
    # (x, y) of this level's samples, [128, N_h, N_p] each
    if loc_layout == "BQHLP2":
        lv = qt["loc"][0:_PMAX, 0 : 2 * K].reshape_dim(1, (NH, NL, NP, 2)).select(2, li)
        u = lv.select(3, 0)
        v = lv.select(3, 1)
    else:
        u = qt["loc"][0:_PMAX, 0:K].reshape_dim(1, (NH, NL, NP)).select(2, li)
        v = qt["loc"][0:_PMAX, K : 2 * K].reshape_dim(1, (NH, NL, NP)).select(2, li)
    awl = qt["aw"][0:_PMAX, 0:K].reshape_dim(1, (NH, NL, NP)).select(2, li)
    xy = t["xy"]
    for a in range(2):
        size = W
        src = u
        if a == 1:
            size = H
            src = v
        dst = xy[0:_PMAX, a * JP : a * JP + J].reshape_dim(1, (NH, NP))
        if align_corners:
            nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=float(size - 1))
        else:
            nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=float(size), op1=nl.add, operand1=-0.5)
        if padding_mode == "zeros":
            # beyond [-1, size] a sample reads only zeros, as at -1 or size
            nisa.tensor_scalar(dst=dst, data=dst, op0=nl.maximum, operand0=-1.0, op1=nl.minimum, operand1=float(size))
        else:
            nisa.tensor_scalar(
                dst=dst, data=dst, op0=nl.maximum, operand0=0.0, op1=nl.minimum, operand1=float(size - 1)
            )
    fl = t["fl"]
    _floor(fl[0:_PMAX, 0 : 2 * JP], xy[0:_PMAX, 0 : 2 * JP], t["tmp"][0:_PMAX, 0 : 2 * JP], nisa.engine.scalar)
    # fg [s][a][j]: s = 0: 1 - frac, s = 1: frac; a = 0: x, a = 1: y (times the attention weight)
    fg4 = t["fg"][0:_PMAX, 0 : 4 * JP].reshape_dim(1, (2, 2, JP))
    xy2 = xy[0:_PMAX, 0 : 2 * JP].reshape_dim(1, (2, JP)).slice(dim=2, start=0, end=J, step=1)
    fl2 = fl[0:_PMAX, 0 : 2 * JP].reshape_dim(1, (2, JP)).slice(dim=2, start=0, end=J, step=1)
    frac = fg4.select(1, 1).slice(dim=2, start=0, end=J, step=1)
    omf = fg4.select(1, 0).slice(dim=2, start=0, end=J, step=1)
    nisa.tensor_tensor(dst=frac, data1=xy2, data2=fl2, op=nl.subtract)
    nisa.tensor_scalar(dst=omf, data=frac, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
    fy = fg4.select(2, 1).slice(dim=2, start=0, end=J, step=1).reshape_dim(2, (NH, NP))
    nisa.tensor_tensor(dst=fy, data1=fy, data2=awl.expand_dim(1).broadcast(1, 2), op=nl.multiply)
    # w4[j][dy][dx] = x factor (dx) * y factor (dy)
    wx = fg4.select(2, 0).permute((0, 2, 1)).expand_dim(2).broadcast(2, 2)
    wy = fg4.select(2, 1).permute((0, 2, 1)).expand_dim(3).broadcast(3, 2)
    nisa.tensor_tensor(dst=t["w4"][0:_PMAX, 0 : 4 * JP].reshape_dim(1, (JP, 2, 2)), data1=wx, data2=wy, op=nl.multiply)
    # scratch row of each sample: ((b S + off + (y0 + 1) Ws + x0 + 1) NH + h, in the sample layout X[q, j]
    X = t["X"]
    xa = X[0:_PMAX, 0:J]
    base = float((b * geo.S + geo.offs[li] + Ws + 1) * NH)
    nisa.tensor_scalar(
        dst=xa, data=fl[0:_PMAX, JP : JP + J], op0=nl.multiply, operand0=float(Ws * NH), op1=nl.add, operand1=base
    )
    nisa.scalar_tensor_tensor(
        dst=xa, data=fl[0:_PMAX, 0:J], op0=nl.multiply, operand0=float(NH), op1=nl.add, operand1=xa
    )
    nisa.tensor_tensor(dst=xa, data1=xa, data2=hc[0:_PMAX, 0:J], op=nl.add)
    # a gather of 128 x JG rows (JG = min(JP, 64) slots: at most 8192 rows) reads index (q, j) of destination row q
    # at vidx[p, m], m * 128 + p = q * JG + j: the group's slots placed 128 / JG times side by side, one PE transpose,
    # 128 / JG strided copies; with 128 slots, two such groups of 64
    vidx = t["vidx"]
    JG = min(JP, _MAX_GATHER_SLOTS)
    rep = _PMAX // JG
    row = 4 * CH
    for g in range(JP // JG):
        Xr = X
        if JP > JG:
            Xr = t["Xr"]
            nisa.tensor_copy(dst=Xr[0:_PMAX, 0:JG], src=X[0:_PMAX, g * JG : (g + 1) * JG], engine=nisa.engine.scalar)
        for h in range(1, rep):
            nisa.tensor_copy(dst=Xr[0:_PMAX, h * JG : (h + 1) * JG], src=Xr[0:_PMAX, 0:JG], engine=nisa.engine.scalar)
        ps = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=ps[0:_PMAX, 0:_PMAX], data=Xr[0:_PMAX, 0:_PMAX], engine=nisa.engine.tensor)
        for h in range(rep):
            nisa.tensor_copy(
                dst=vidx[h * JG : (h + 1) * JG, g * JG : (g + 1) * JG],
                src=ps[h * JG : (h + 1) * JG, 0:_PMAX].slice(dim=1, start=h, end=_PMAX, step=rep),
                engine=nisa.engine.scalar,
            )
        nisa.dma_copy(
            dst=t["rows"][0:_PMAX, g * JG * row : (g + 1) * JG * row],
            src=scratch.ap(
                pattern=[[row, _PMAX * JG], [1, row]],
                offset=0,
                vector_offset=vidx[0:_PMAX, g * JG : (g + 1) * JG],
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.swdge,
        )


def _stage_b(ctx, item, t, qt):
    """Blend one work item's gathered rows into the query tile's sum; after the last level, store it."""
    _, _, _, output, geo, _, _, _, _, prod_t, red = ctx
    b, q0, nq, li, _ = item
    NQ, NH, NL, NP, CH = geo.NQ, geo.NH, geo.NL, geo.NP, geo.CH
    E = NH * CH
    G = geo.plan.groups
    last = li == NL - 1
    if NL == 1:
        dst = qt["res"]
    elif li == 0:
        dst = qt["acc"]
    else:
        dst = red
    # per group of G heads: prod[h, c, (p, k)] = rows[(h, p), k, c] * w4[(h, p), k], summed over (p, k)
    for h0 in range(0, NH, G):
        g = min(G, NH - h0)
        rows = t["rows"][0:_PMAX, h0 * NP * 4 * CH : (h0 + g) * NP * 4 * CH]
        rows = rows.reshape_dim(1, (g, NP * 4, CH)).permute((0, 1, 3, 2))
        w = t["w4"][0:_PMAX, h0 * NP * 4 : (h0 + g) * NP * 4].reshape_dim(1, (g, NP * 4)).expand_dim(2).broadcast(2, CH)
        prod = prod_t[0:_PMAX, 0 : g * CH * NP * 4]
        nisa.tensor_tensor(dst=prod.reshape_dim(1, (g, CH, NP * 4)), data1=rows, data2=w, op=nl.multiply)
        nisa.tensor_reduce(
            dst=dst[0:_PMAX, h0 * CH : (h0 + g) * CH], op=nl.add, data=prod.reshape_dim(1, (g * CH, NP * 4)), axis=(2,)
        )
    if NL > 1 and li > 0:
        sum_dst = qt["acc"]
        if last:
            sum_dst = qt["res"]
        nisa.tensor_tensor(dst=sum_dst[0:_PMAX, 0:E], data1=qt["acc"][0:_PMAX, 0:E], data2=red[0:_PMAX, 0:E], op=nl.add)
    if last:
        of = output.reshape((geo.B * NQ, E))
        nisa.dma_copy(dst=of[b * NQ + q0 : b * NQ + q0 + nq, 0:E], src=qt["res"][0:nq, 0:E])
