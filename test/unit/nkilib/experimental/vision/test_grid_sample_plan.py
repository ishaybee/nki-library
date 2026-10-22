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

"""grid_sample plans and its sampling arithmetic, checked on the host without running a kernel."""

import itertools

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from nkilib_src.nkilib.experimental.vision.grid_sample import (
    _BLEND4_ELEMS,
    _BP_BYTES,
    _BUILD_UNROLL,
    _GATHER_BYTES,
    _LOOP_ROUNDS,
    _MAX_COLS_DEEP,
    _MAX_DEPTH,
    _MAX_EXACT,
    _MAX_SCRATCH_BYTES,
    _MAX_SEG_PIX,
    _MAX_STAGE_BYTES,
    _MIN_LOOP_ITERS,
    _SBUF_SHARE,
    _SCRATCH_ROW_ELEMS,
    _UNROLL_CHUNKS,
    grid_sample_build_ranges,
    grid_sample_exact_rows,
    grid_sample_loop,
    grid_sample_sample_layout,
    grid_sample_sbuf_bytes,
    grid_sample_scratch_width,
    grid_sample_work,
    nki_at_least,
    plan_grid_sample,
)

_PMAX = 128
# nl.tile_size.total_available_sbuf_size while tracing for trn2
_TRN2_SBUF_BYTES = 212984

# (N, C, H, W, Ho, Wo, mode, element bytes): a frame warp's pyramid (992 x 1280, C = 3 .. 128), spatial
# transformers, point sampling, resizes, 16-bit inputs, wide channels (channel-major stores, partial and many blocks
# of 128 channels), the edges of the shape range, PyTorch's GridSampler benchmark set, FlagGems' grid_sample
# benchmark shapes, model call sites (Super SloMo, RAFT's correlation lookup, PointRend, deformable DETR, EG3D's
# triplane, MVSNet's homography warp, multi-view point sampling), output sizes without a fitting divisor,
# staging buffers at their cap, and scratch builds past the unroll budget (pipelined in both layouts, a device loop
# for images of over 128 rows).
_CALLS = [
    (1, 3, 992, 1280, 992, 1280, "bilinear", 4),
    (1, 16, 496, 640, 496, 640, "bilinear", 4),
    (1, 128, 62, 80, 62, 80, "bilinear", 4),
    (1, 3, 1728, 4096, 1728, 4096, "bilinear", 4),
    (32, 3, 224, 224, 224, 224, "bilinear", 4),
    (256, 1, 28, 28, 28, 28, "bilinear", 4),
    (4, 64, 128, 128, 1, 16384, "bilinear", 4),
    (1, 32, 512, 512, 256, 256, "bilinear", 4),
    (1, 3, 992, 1280, 992, 1280, "nearest", 4),
    (1, 16, 496, 640, 496, 640, "bilinear", 2),
    (8, 32, 128, 128, 128, 128, "bilinear", 4),
    (2, 1024, 16, 16, 16, 16, "bilinear", 4),
    (1, 2048, 8, 8, 8, 8, "nearest", 2),
    (1, 1, 1, 1, 1, 1, "bilinear", 4),
    (3, 2, 17, 19, 13, 11, "bilinear", 4),
    (2, 3, 5, 5, 3, 4, "nearest", 4),
    (1, 2, 16, 16, 5, 7, "bilinear", 4),
    (5, 7, 1, 300, 1, 129, "bilinear", 4),
    (1, 3, 4000, 4000, 4000, 4000, "bilinear", 4),
    (10, 5, 20, 20, 15, 15, "bilinear", 4),
    (500, 5, 20, 20, 15, 15, "bilinear", 4),
    (50, 10, 100, 100, 100, 100, "bilinear", 4),
    (500, 10, 50, 50, 50, 50, "bilinear", 4),
    (50, 100, 50, 50, 50, 50, "bilinear", 4),
    (50, 10, 500, 500, 80, 80, "bilinear", 4),
    (50, 10, 80, 80, 500, 500, "bilinear", 4),
    (4, 256, 16, 16, 16, 16, "nearest", 2),
    (8, 3, 64, 64, 64, 64, "bilinear", 2),
    (6, 3, 352, 352, 352, 352, "bilinear", 4),
    (7040, 1, 55, 128, 9, 9, "bilinear", 4),
    (7040, 1, 6, 16, 9, 9, "bilinear", 4),
    (2, 256, 200, 336, 19600, 1, "bilinear", 4),
    (8, 32, 100, 134, 17821, 4, "bilinear", 4),
    (3, 32, 256, 256, 1, 393216, "bilinear", 4),
    (1, 32, 128, 160, 24576, 160, "bilinear", 4),
    (64, 3, 97, 97, 97, 97, "bilinear", 4),
    (40, 3, 97, 97, 97, 97, "bilinear", 4),
    (70, 2, 10, 10, 97, 97, "bilinear", 4),
    (2, 256, 20, 34, 1960, 1, "bilinear", 4),
    (3, 160, 20, 20, 30, 30, "bilinear", 2),
    (1, 300, 9, 9, 9, 9, "nearest", 4),
    (5, 100, 50, 50, 50, 50, "bilinear", 4),
    (1, 8192, 4, 4, 4, 4, "nearest", 4),
    (600, 256, 16, 16, 64, 64, "bilinear", 4),
    (3, 1, 100, 130, 9, 9, "bilinear", 4),
    (2, 1, 1, 300, 1, 4, "bilinear", 4),
    (2, 1, 300, 1, 4, 1, "bilinear", 4),
    (4, 1, 64, 64, 5, 5, "bilinear", 2),
    (300, 8, 40, 40, 6, 6, "bilinear", 4),
    (227 * 31 + 5, 1, 28, 64, 9, 9, "bilinear", 4),
    (1, 3, 2048, 2304, 2048, 2304, "bilinear", 4),
    (12, 3, 64, 64, 600, 600, "bilinear", 4),
    (1, 40, 64, 80, 64, 80, "bilinear", 4),
    (1, 64, 48, 64, 96, 128, "bilinear", 4),
    (1, 16, 124, 160, 124, 160, "bilinear", 4),
    (64, 4, 16, 16, 16, 16, "bilinear", 2),
    (200, 2, 12, 12, 7, 7, "nearest", 4),
    (130, 16, 70, 70, 40, 40, "bilinear", 4),
    (130, 64, 62, 62, 9, 9, "bilinear", 4),
    (130, 8, 130, 130, 48, 48, "bilinear", 4),
    (24, 4, 2160, 3840, 72000 * 9, 1, "bilinear", 2),
    (48, 4, 2160, 3840, 72000 * 9, 1, "bilinear", 2),
    (24, 4, 540, 960, 72000 * 9, 1, "bilinear", 2),
    (1, 4, 2160, 3840, 1280000 * 9, 1, "bilinear", 2),
    (1, 4, 540, 960, 1280000 * 9, 1, "bilinear", 2),
    # past 2^24 scratch rows per batch (exact plans): a strip of rows of a 6K warp, a 6K flow warp (an output
    # past 2^24 pixels as well), two 6K batches, 8K UHD and DCI 8K in 16 bits, few samples from a 6K image (the plain
    # layout), one-channel inputs (nearest, and bilinear without the direct read); an output past 2^24 pixels
    (1, 3, 3456, 6144, 432, 6144, "bilinear", 4),
    (1, 3, 3456, 6144, 3456, 6144, "bilinear", 4),
    (2, 3, 3456, 6144, 216, 6144, "bilinear", 2),
    (1, 3, 4320, 7680, 4320, 7680, "bilinear", 2),
    (1, 3, 4320, 8192, 270, 8192, "bilinear", 2),
    (1, 3, 3456, 6144, 300, 400, "bilinear", 4),
    (1, 1, 4096, 4100, 64, 64, "nearest", 4),
    (1, 1, 4096, 4100, 64, 64, "bilinear", 4),
    (1, 1, 64, 64, 4201, 4101, "bilinear", 4),
]


def _check_scratch_shape(plan, hs, ws):
    """The plan's scratch has hs rows of ws entries per batch, wider past 2^24 rows (an exact plan, whose width
    rounds up to keep its row bases exact in fp32), and stage A computes every scratch row exactly."""
    assert plan.hs == hs and plan.exact == (hs * ws >= _MAX_EXACT)
    if plan.exact:
        assert ws <= plan.ws < 2 * ws and not plan.direct
        assert all(n1 - n0 == 1 for n0, n1 in plan.segments)
    else:
        assert plan.ws == ws
    assert grid_sample_exact_rows(plan)


@pytest.mark.parametrize("N, C, H, W, Ho, Wo, mode, eb", _CALLS)
def test_plan_covers_every_output_pixel(N, C, H, W, Ho, Wo, mode, eb):
    """
    The segments partition the batches; a multi-batch segment keeps its fp32 arithmetic on scratch rows and pixels
    exact and (several channels) its staging buffer within the cap. With LNC 1 and 2, every program's chunks lie in
    its share of a segment, a batched gather (several indices per partition) uses all 128 partitions, and the
    programs together cover every output pixel. A chunk's gathered rows fit their SBUF share, a scratch-build tile
    fits its width, the scratch fits its cap, and the kernel's tiles fit trn2's SBUF, with as many chunks in flight
    as fit.
    """
    plan = plan_grid_sample(N, C, H, W, Ho, Wo, mode, eb)
    T = Ho * Wo
    assert plan.cols >= 1
    # wide chunks only for quads spanning batches; quad chunks of few output elements blend in 4 instructions
    if not plan.quad or len(plan.segments) == N:
        assert plan.cols <= _MAX_COLS_DEEP
    assert plan.blend4 == (plan.quad and C * plan.cols <= _BLEND4_ELEMS)
    assert plan.taps * plan.cols * plan.row * eb <= max(_GATHER_BYTES, plan.taps * plan.row * eb)
    if plan.quad:
        assert mode == "bilinear" and plan.taps == 1 and plan.row == 4 * C and not plan.direct
        _check_scratch_shape(plan, H + 2, W + 2)
        assert N * plan.hs * plan.ws * plan.row * eb <= _MAX_SCRATCH_BYTES
    elif plan.direct:
        # no scratch: the gathers read the one-channel input itself, below 2^24 pixels
        assert mode == "bilinear" and C == 1 and plan.taps == 4 and plan.row == 1 and (plan.hs, plan.ws) == (H, W)
        assert not plan.pipelined and not plan.exact and H * W < _MAX_EXACT
    elif mode == "bilinear":
        assert plan.taps == 4 and plan.row == C
        _check_scratch_shape(plan, H + 3, W + 3)
    else:
        assert plan.taps == 1 and plan.row == C
        _check_scratch_shape(plan, H + 2, W + 2)
    assert 1 <= plan.bk <= N and 1 <= plan.seg <= plan.ws
    assert plan.bk * plan.seg * plan.row <= max(_SCRATCH_ROW_ELEMS, plan.row)
    if plan.bk > 1:
        assert plan.seg == plan.ws
    if plan.pipelined:
        # the pipelined build: more groups of batches than are unrolled, each a single build tile
        assert N // plan.bk > _BUILD_UNROLL and plan.hs <= _PMAX and plan.seg == plan.ws
        assert plan.gp == 1 and not plan.bp
    # groups of batches stacked along a build tile's partitions fill at most all of them
    assert plan.gp >= 1 and (plan.gp == 1 or plan.gp * plan.hs <= _PMAX)
    if plan.gp > 1:
        assert plan.seg == plan.ws and not plan.direct and not plan.bp
    if plan.bp:
        # one batch per partition: many batches of tiny images
        assert not plan.direct and N >= 32 and plan.gp == 1
        assert (C * H * W + plan.hs * plan.ws * plan.row) * eb <= _BP_BYTES
    batches = []
    for n0, n1 in plan.segments:
        batches.extend(range(n0, n1))
        if n1 - n0 > 1:
            assert (n1 - n0) * plan.hs * plan.ws < _MAX_EXACT and (n1 - n0) * T <= _MAX_SEG_PIX
            assert C == 1 or C * (n1 - n0) * T * eb <= _MAX_STAGE_BYTES
    assert batches == list(range(N))
    for n_prgs in (1, 2):
        covered = np.zeros(N * T, dtype=np.uint8)
        for prg_id in range(n_prgs):
            work = grid_sample_work(plan, T, n_prgs, prg_id)
            assert len(work) == len(plan.segments)
            for (s0, s1), (n0, n1, parts, cols, n_full, tail, first, count) in zip(plan.segments, work, strict=True):
                assert s0 <= n0 < n1 <= s1
                S = (n1 - n0) * T
                chunk = parts * cols
                assert 1 <= cols <= plan.cols and 1 <= parts <= _PMAX and chunk <= S
                if cols > 1:
                    assert parts == _PMAX
                starts = [i * chunk for i in range(n_full)]
                if tail >= 0:
                    assert 0 < S - n_full * chunk < chunk and tail == S - chunk
                    starts.append(tail)
                else:
                    assert n_full * chunk == S
                assert first >= 0 and count >= 1 and first + count <= len(starts)
                for st in starts[first : first + count]:
                    covered[n0 * T + st : n0 * T + st + chunk] += 1
        assert covered.min() >= 1
    assert grid_sample_sbuf_bytes(plan, C, eb) <= _TRN2_SBUF_BYTES
    # as many chunks in flight as fit: the next depth would not
    assert 2 <= plan.depth <= _MAX_DEPTH
    if plan.depth < _MAX_DEPTH and grid_sample_sbuf_bytes(plan, C, eb) <= _SBUF_SHARE * _TRN2_SBUF_BYTES:
        plan.depth = plan.depth + 1
        assert grid_sample_sbuf_bytes(plan, C, eb) > _SBUF_SHARE * _TRN2_SBUF_BYTES


@pytest.mark.parametrize("N, C, H, W, Ho, Wo, mode, eb", _CALLS)
def test_programs_build_the_scratch_their_chunks_read(N, C, H, W, Ho, Wo, mode, eb):
    """
    Each program builds the scratch of every batch its chunks read (a segment split by batch: its share only). With
    one program that is every batch once; with LNC=2 both programs get ranges of the same sizes (one control flow),
    and a program's ranges never overlap.
    """
    plan = plan_grid_sample(N, C, H, W, Ho, Wo, mode, eb)
    T = Ho * Wo
    assert grid_sample_build_ranges(plan, 1, 0) == [(0, N)]
    sizes = None
    for prg_id in range(2):
        ranges = grid_sample_build_ranges(plan, 2, prg_id)
        built = np.zeros(N, dtype=np.int64)
        for r0, r1 in ranges:
            assert 0 <= r0 < r1 <= N
            built[r0:r1] += 1
        assert built.max() <= 1
        for n0, n1, _, _, _, _, _, _ in grid_sample_work(plan, T, 2, prg_id):
            assert built[n0:n1].min() == 1
        if sizes is None:
            sizes = [r1 - r0 for r0, r1 in ranges]
        assert [r1 - r0 for r0, r1 in ranges] == sizes


def test_plan_prefers_the_quad_scratch():
    """Bilinear calls take the quad scratch (one gathered row per sample) when it fits the cap and the input is
    not much larger than the output, the plain padded image otherwise; nearest calls always gather one row of C."""
    assert plan_grid_sample(1, 3, 992, 1280, 992, 1280, "bilinear", 4).quad
    assert plan_grid_sample(7040, 1, 6, 16, 9, 9, "bilinear", 4).quad
    big = plan_grid_sample(64, 64, 256, 256, 256, 256, "bilinear", 4)
    assert not big.quad and big.taps == 4
    # few samples from a large input: building the quad scratch would cost more than the gathers it saves
    assert not plan_grid_sample(2, 256, 200, 336, 19600, 1, "bilinear", 4).quad
    assert not plan_grid_sample(7040, 1, 55, 128, 9, 9, "bilinear", 4).quad
    assert not plan_grid_sample(1, 3, 992, 1280, 992, 1280, "nearest", 4).quad


def test_plan_groups_batches():
    """Many small batches share segments (whose chunks span batches, output sizes without a fitting divisor
    included) and scratch-build tiles; a few batches whose own chunks are nearly full run one per segment; staging
    buffers past their cap split a segment."""
    raft = plan_grid_sample(7040, 1, 55, 128, 9, 9, "bilinear", 4)
    assert len(raft.segments) <= 8 and raft.bk > 1
    gist = plan_grid_sample(500, 5, 20, 20, 15, 15, "bilinear", 4)
    assert len(gist.segments) == 1 and gist.bk > 1
    poor = plan_grid_sample(70, 2, 10, 10, 97, 97, "bilinear", 4)
    assert len(poor.segments) == 1 and poor.cols == 64
    big_out = plan_grid_sample(50, 10, 80, 80, 500, 500, "bilinear", 4)
    assert all(n1 - n0 == 1 for n0, n1 in big_out.segments)
    prime = plan_grid_sample(8, 32, 100, 134, 17821, 4, "bilinear", 4)
    assert len(prime.segments) == 8
    capped = plan_grid_sample(600, 256, 16, 16, 64, 64, "bilinear", 4)
    assert len(capped.segments) == 3 and max(n1 - n0 for n0, n1 in capped.segments) == 256


def test_plan_samples_many_views_at_clustered_points():
    """
    Multi-view point sampling reads the 4-channel float16 maps of many views (one batch per view) at clusters of 9
    points. The UHD maps (2160 x 3840) gather 4 rows of 8 bytes per sample from the plain layout once 9 or more views
    would take the quad scratch past 2 GiB (48 views: a plain scratch past 2 GiB as well), the quarter-resolution
    maps one row per sample from quads. A call per view (1,280,000 clusters) gathers both from quads. All run one
    segment per view, 8 chunks deep, past the unroll budget.
    """
    T, V = 72000 * 9, 1280000 * 9
    assert plan_grid_sample(8, 4, 2160, 3840, T, 1, "bilinear", 2).quad
    for N, H, W, Ho, quad in (
        (9, 2160, 3840, T, False),
        (24, 2160, 3840, T, False),
        (48, 2160, 3840, T, False),
        (24, 540, 960, T, True),
        (1, 2160, 3840, V, True),
        (1, 540, 960, V, True),
    ):
        plan = plan_grid_sample(N, 4, H, W, Ho, 1, "bilinear", 2)
        assert plan.quad == quad and not plan.direct and plan.taps == (1 if quad else 4)
        assert plan.segments == [(n, n + 1) for n in range(N)] and plan.depth == _MAX_DEPTH
        for n_prgs in (1, 2):
            assert sum(w[7] for w in grid_sample_work(plan, Ho, n_prgs, 0)) > _UNROLL_CHUNKS
    plan = plan_grid_sample(48, 4, 2160, 3840, T, 1, "bilinear", 2)
    assert 48 * plan.hs * plan.ws * plan.row * 2 > _MAX_SCRATCH_BYTES


def test_plan_reads_one_channel_inputs_directly():
    """A one-channel input that would get the plain layout (few samples from a large input, as RAFT's correlation
    lookup at full resolution) needs no scratch; quads, several channels and nearest keep theirs."""
    raft = plan_grid_sample(7040, 1, 55, 128, 9, 9, "bilinear", 4)
    assert raft.direct and not raft.quad and (raft.hs, raft.ws) == (55, 128)
    assert not plan_grid_sample(7040, 1, 6, 16, 9, 9, "bilinear", 4).direct
    assert not plan_grid_sample(2, 2, 100, 168, 49, 1, "bilinear", 4).direct
    assert not plan_grid_sample(1, 1, 300, 300, 10, 10, "nearest", 4).direct


def test_plan_indexes_scratches_past_2_24_rows_exactly():
    """
    Past 2^24 scratch rows per batch (6K images and up) fp32 no longer holds every scratch row, so the plan is exact:
    stage A adds row bases y * ws, which fp32 holds exactly, and columns in uint32. The scratch width rounds up to the
    next multiple of 2^k that keeps every row base exact: 6K needs none, DCI 8K two columns, a plain 6K scratch one.
    Every batch is its own segment, exact quads gather in the sample layout, and a one-channel bilinear input takes
    the plain layout instead of the direct read. Scratches of 2^24 rows or more are rejected.
    """
    # (call, quad, scratch rows, scratch width)
    for args, quad, hs, ws in (
        ((1, 3, 3456, 6144, 432, 6144, "bilinear", 4), True, 3458, 6146),
        ((1, 3, 4320, 7680, 4320, 7680, "bilinear", 2), True, 4322, 7682),
        ((1, 3, 4320, 8192, 270, 8192, "bilinear", 2), True, 4322, 8196),
        ((1, 3, 3456, 6144, 300, 400, "bilinear", 4), False, 3459, 6148),
        ((1, 1, 4096, 4100, 64, 64, "nearest", 4), False, 4098, 4102),
        ((1, 1, 4096, 4100, 64, 64, "bilinear", 4), False, 4099, 4104),
    ):
        plan = plan_grid_sample(*args)
        assert plan.exact and plan.quad == quad and not plan.direct and (plan.hs, plan.ws) == (hs, ws)
        assert grid_sample_exact_rows(plan)
        # every row base exact in fp32, while the largest scratch row is not
        ys = np.arange(plan.hs, dtype=np.int64)
        bases = ys.astype(np.float32) * np.float32(plan.ws)
        assert np.array_equal(bases.astype(np.int64), ys * plan.ws)
        assert int(np.float32(plan.hs * plan.ws - 1)) != plan.hs * plan.ws - 1
        assert plan.hs * plan.ws < 1 << 32
        if quad:
            for n_prgs in (1, 2):
                for prg_id in range(n_prgs):
                    for n0, n1, parts, cols, _, _, _, _ in grid_sample_work(plan, args[4] * args[5], n_prgs, prg_id):
                        assert grid_sample_sample_layout(plan, n1 - n0, parts, cols)
    # the least power of two that keeps the row bases exact
    for hs in (3458, 4099, 4322, 8194, 60000):
        for ws in range(8190, 8200):
            width = grid_sample_scratch_width(hs, ws)
            if hs * ws < _MAX_EXACT:
                assert width == ws
                continue
            p = 1
            while hs * (-(-ws // p)) >= _MAX_EXACT:
                p = p * 2
            assert width == -(-ws // p) * p
    # below 2^24 rows the plans are unchanged; a one-channel input below 2^24 pixels keeps the direct read
    assert not plan_grid_sample(1, 3, 4000, 4000, 4000, 4000, "bilinear", 4).exact
    assert plan_grid_sample(1, 1, 4096, 4095, 64, 64, "bilinear", 4).direct
    # 2^24 rows or more cannot keep the row bases exact
    assert not grid_sample_exact_rows(plan_grid_sample(1, 1, 1 << 24, 1, 1, 1, "nearest", 4))


def test_plan_pipelines_the_build_of_many_build_tiles():
    """More build tiles than are unrolled, each one group of batches (images of at most 128 rows that do not stack),
    build their scratch in a software pipeline, in both layouts; fewer tiles unroll, and taller images keep the
    per-tile build (in a device loop past the unroll budget)."""
    for args, quad in (((130, 16, 70, 70, 40, 40), True), ((130, 64, 62, 62, 9, 9), False)):
        plan = plan_grid_sample(*args, "bilinear", 4)
        assert plan.pipelined and plan.gp == 1 and not plan.bp and plan.quad == quad
    for args in (
        (500, 10, 50, 50, 50, 50),
        (6, 3, 352, 352, 352, 352),
        (50, 10, 500, 500, 80, 80),
        (130, 8, 130, 130, 48, 48),
    ):
        assert not plan_grid_sample(*args, "bilinear", 4).pipelined
    tall = plan_grid_sample(130, 8, 130, 130, 48, 48, "bilinear", 4)
    assert tall.hs > _PMAX and tall.bk == 1 and 130 > _BUILD_UNROLL


def test_plan_stacks_small_batches_along_the_partitions():
    """Batches whose scratch rows fill at most half the partitions build in tiles of as many groups of bk batches
    as fit stacked along the partitions, in both layouts; larger images, direct plans and many tiny images (one
    batch per partition) keep one group per tile."""
    for args, gp in (
        ((500, 5, 20, 20, 15, 15), 5),
        ((500, 10, 50, 50, 50, 50), 2),
        ((50, 100, 50, 50, 50, 50), 2),
        ((300, 8, 40, 40, 6, 6), 2),
    ):
        plan = plan_grid_sample(*args, "bilinear", 4)
        assert plan.gp == gp and plan.gp * plan.hs <= _PMAX < (plan.gp + 1) * plan.hs
    for args in ((50, 10, 100, 100, 100, 100), (1, 3, 992, 1280, 992, 1280), (7040, 1, 55, 128, 9, 9)):
        assert plan_grid_sample(*args, "bilinear", 4).gp == 1
    assert plan_grid_sample(7040, 1, 6, 16, 9, 9, "bilinear", 4).gp == 1


def test_plan_builds_tiny_images_one_batch_per_partition():
    """Many batches (32 or more) of tiny images, whose image and scratch fit _BP_BYTES, build their scratch one
    batch per partition (RAFT's level-3 lookup, MNIST-sized spatial transformers, 10 x 10 inputs; both layouts and
    16 bits); few batches, larger images and direct plans do not."""
    for args in (
        (7040, 1, 6, 16, 9, 9, "bilinear", 4),
        (256, 1, 28, 28, 28, 28, "bilinear", 4),
        (70, 2, 10, 10, 97, 97, "bilinear", 4),
        (64, 4, 16, 16, 16, 16, "bilinear", 2),
        (200, 2, 12, 12, 7, 7, "nearest", 4),
    ):
        plan = plan_grid_sample(*args)
        assert plan.bp and not plan.pipelined and plan.gp == 1
    for args in ((10, 5, 20, 20, 15, 15), (500, 5, 20, 20, 15, 15), (7040, 1, 55, 128, 9, 9)):
        assert not plan_grid_sample(*args, "bilinear", 4).bp


def test_plan_stores_wide_chunks_channel_major():
    """Chunks with more channels than gathered rows per partition store channel-major through the Tensor Engine
    (C long runs instead of 128 C short ones); few-channel calls store straight from the gathered layout."""
    assert plan_grid_sample(2, 1024, 16, 16, 16, 16, "bilinear", 4).tstore
    assert plan_grid_sample(2, 256, 200, 336, 19600, 1, "bilinear", 4).tstore
    assert plan_grid_sample(4, 256, 16, 16, 16, 16, "nearest", 2).tstore
    assert plan_grid_sample(1, 8192, 4, 4, 4, 4, "nearest", 4).tstore
    assert not plan_grid_sample(1, 3, 992, 1280, 992, 1280, "bilinear", 4).tstore
    assert not plan_grid_sample(1, 32, 128, 160, 24576, 160, "bilinear", 4).tstore


def test_plan_keeps_eight_chunks_in_flight_where_sbuf_allows():
    """Single-image warps (a 992 x 1280 frame, RIFE's 4K frame) and many-batch calls of small rows run 8 chunks deep;
    wide rows and channel-major stores, whose tiles fill SBUF, keep fewer."""
    for args in ((1, 3, 992, 1280, 992, 1280), (1, 3, 2176, 3840, 2176, 3840), (70, 2, 10, 10, 97, 97)):
        assert plan_grid_sample(*args, "bilinear", 4).depth == _MAX_DEPTH
    assert plan_grid_sample(1, 32, 128, 160, 24576, 160, "bilinear", 4).depth < _MAX_DEPTH
    assert plan_grid_sample(2, 256, 200, 336, 19600, 1, "bilinear", 4).depth == 2


def test_single_image_warps_gather_in_the_sample_layout():
    """One-batch quad warps (a 992 x 1280 frame, RIFE's 4K frame, a 16-channel warp, a 6-batch Super SloMo call) take
    the sample layout's stage A in every work item with LNC 1 and 2; quads spanning batches, the plain layout and
    chunks whose width is not a multiple of 32 do not."""
    for args in (
        (1, 3, 992, 1280, 992, 1280),
        (1, 3, 2176, 3840, 2176, 3840),
        (1, 16, 1088, 1920, 1088, 1920),
        (6, 3, 352, 352, 352, 352),
    ):
        plan = plan_grid_sample(*args, "bilinear", 4)
        for n_prgs in (1, 2):
            for prg_id in range(n_prgs):
                for n0, n1, parts, cols, _, _, _, _ in grid_sample_work(plan, args[4] * args[5], n_prgs, prg_id):
                    assert grid_sample_sample_layout(plan, n1 - n0, parts, cols)
    for args in ((7040, 1, 6, 16, 9, 9), (2, 256, 200, 336, 19600, 1), (1, 64, 48, 64, 96, 128)):
        plan = plan_grid_sample(*args, "bilinear", 4)
        for n0, n1, parts, cols, _, _, _, _ in grid_sample_work(plan, args[4] * args[5], 1, 0):
            assert not grid_sample_sample_layout(plan, n1 - n0, parts, cols)


def test_loop_schedule_runs_few_long_iterations():
    """
    A segment past the unroll budget loops over the most rounds of depth chunks (at most _LOOP_ROUNDS) that still
    iterate _MIN_LOOP_ITERS times, leaves the pipeline's leading stage A's and its last chunk (static) out of the
    loop, and unrolls at most (_LOOP_ROUNDS + 1) * depth chunks; under the budget nothing loops.
    """
    for depth in range(2, _MAX_DEPTH + 1):
        for count in range(depth, 3000):
            assert grid_sample_loop(count, depth, True) == (0, 0)
            per_it, n_loop = grid_sample_loop(count, depth, False)
            if per_it == 0:
                assert n_loop == 0 and count - depth < _MIN_LOOP_ITERS * depth
            else:
                assert per_it % depth == 0 and per_it <= _LOOP_ROUNDS * depth and n_loop >= _MIN_LOOP_ITERS
                assert n_loop * per_it <= count - depth
                if per_it < _LOOP_ROUNDS * depth:
                    assert (count - depth) // (per_it + depth) < _MIN_LOOP_ITERS
            assert count - n_loop * per_it <= (_LOOP_ROUNDS + 1) * depth


def _source(g, size, align_corners, padding_mode):
    """The kernel's coordinate arithmetic in numpy (float32 like the vector engine): unnormalize in ATen's order,
    apply the padding mode, clamp."""
    g = g.astype(np.float32)
    if align_corners:
        x = (g + np.float32(1.0)) * np.float32(0.5) * np.float32(size - 1)
    else:
        x = ((g + np.float32(1.0)) * np.float32(size) - np.float32(1.0)) * np.float32(0.5)
    if padding_mode == "reflection":
        lo, hi = (0.0, 2.0 * (size - 1)) if align_corners else (-1.0, 2.0 * size - 1)
        if lo == hi:
            x = np.zeros_like(x)
        else:
            mn = np.float32(lo / 2)
            span = np.float32((hi - lo) / 2)
            t = np.minimum(np.abs(x - mn), np.float32(1 << 23))
            flips = np.floor(t * np.float32(1.0 / span))
            extra = t - flips * span
            fix = extra < 0
            flips = flips - fix
            extra = extra + fix * span
            fix = extra >= span
            flips = flips + fix
            extra = extra - fix * span
            odd = flips - 2 * np.floor(flips * np.float32(0.5))
            x = extra + mn + odd * (span - 2 * extra)
    if padding_mode == "zeros":
        return np.clip(x, -1.0, float(size)).astype(np.float32)
    return np.clip(x, 0.0, float(size - 1)).astype(np.float32)


def _emulate(x, grid, mode, padding_mode, align_corners):
    """grid_sample the way the kernel does it: taps from a zero-padded image (pad 1 before, 2 after), floor or
    round-half-even of the transformed coordinates, bilinear weights from their fractions."""
    N, C, H, W = x.shape
    out = np.zeros((N, C) + grid.shape[1:3])
    for n in range(N):
        pad = np.zeros((C, H + 3, W + 3))
        pad[:, 1 : H + 1, 1 : W + 1] = x[n]
        ix = _source(grid[n, :, :, 0], W, align_corners, padding_mode).astype(np.float64)
        iy = _source(grid[n, :, :, 1], H, align_corners, padding_mode).astype(np.float64)
        if mode == "nearest":
            xn = np.rint(ix).astype(np.int64)
            yn = np.rint(iy).astype(np.int64)
            out[n] = pad[:, yn + 1, xn + 1]
            continue
        x0 = np.floor(ix)
        y0 = np.floor(iy)
        fx = ix - x0
        fy = iy - y0
        xi = x0.astype(np.int64) + 1
        yi = y0.astype(np.int64) + 1
        out[n] = (
            pad[:, yi, xi] * (1 - fx) * (1 - fy)
            + pad[:, yi, xi + 1] * fx * (1 - fy)
            + pad[:, yi + 1, xi] * (1 - fx) * fy
            + pad[:, yi + 1, xi + 1] * fx * fy
        )
    return out


@pytest.mark.parametrize(
    "mode, padding_mode, align_corners",
    list(itertools.product(("bilinear", "nearest"), ("zeros", "border", "reflection"), (False, True))),
)
def test_sampling_arithmetic_matches_torch(mode, padding_mode, align_corners):
    """
    The kernel's arithmetic (clamping zeros-padded coordinates to [-1, size], reflecting with a reciprocal and a
    corrected floor, taps from a zero border) agrees with torch's grid_sample in float64: coordinates well outside
    [-1, 1], exactly on the edges, on pixel centers, and in tiny (1 x 1, 1 x 5) inputs.
    """
    rng = np.random.default_rng(hash((mode, padding_mode, align_corners)) % (1 << 31))
    for N, C, H, W, Ho, Wo in ((2, 3, 7, 9, 5, 6), (1, 2, 1, 1, 3, 3), (1, 1, 1, 5, 2, 8), (1, 4, 6, 1, 4, 4)):
        x = rng.standard_normal((N, C, H, W))
        grid = rng.uniform(-3.0, 3.0, (N, Ho, Wo, 2))
        grid.reshape(-1, 2)[:8] = [[-1, -1], [1, 1], [-1, 1], [1, -1], [0, 0], [-1.0000001, 0.5], [3e5, -2e6], [0.5, 1]]
        if mode == "nearest":
            # keep off rounding ties, where float32 and float64 coordinates may round differently
            for s, d in ((W, 0), (H, 1)):
                pix = _source(grid[..., d], s, align_corners, "zeros")
                tie = np.abs(pix - np.floor(pix) - 0.5) < 1e-3
                grid[..., d] = np.where(tie, grid[..., d] + 0.37 / max(s, 1), grid[..., d])
        grid = grid.astype(np.float32).astype(np.float64)
        ref = F.grid_sample(
            torch.from_numpy(x),
            torch.from_numpy(grid),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        ).numpy()
        got = _emulate(x, grid, mode, padding_mode, align_corners)
        np.testing.assert_allclose(got, ref, rtol=0, atol=2e-5 * max(1.0, float(np.abs(x).max())) * max(H, W))


@pytest.mark.parametrize(
    "version, expected",
    [
        ("0.6.0+31049202112.g85070674", False),
        ("0.6.9", False),
        ("0.7.0+37713544100.gfede04bb", True),
        ("0.7.1", True),
        ("1.0.0", True),
        ("unknown", True),
    ],
)
def test_nki_at_least(monkeypatch, version, expected):
    """The nki version gate: releases before 0.7.0 fail it, local builds compare by their release, and a version it
    cannot parse passes (the compiler then reports for itself)."""
    import nki

    monkeypatch.setattr(nki, "__version__", version)
    assert nki_at_least(0, 7, 0) is expected
