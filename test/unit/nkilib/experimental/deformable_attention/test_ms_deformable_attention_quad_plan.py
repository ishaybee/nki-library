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

"""ms_deformable_attention_quad plans and work split, checked on the host without running a kernel."""

import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.deformable_attention.ms_deformable_attention_quad import (
    _SBUF_SHARE,
    _TRN2_SBUF_BYTES,
    msda_quad_sbuf_bytes,
    msda_quad_work,
    nki_at_least,
    plan_ms_deformable_attention_quad,
)
from nkilib_src.nkilib.experimental.deformable_attention.ms_deformable_attention_quad_torch import (
    ms_deformable_attention_f64,
)

_PMAX = 128

# (N_h, C_h, N_l, N_p, value bytes, location / weight bytes): the library tests' heads, the BEVFormer levels, wide
# heads, float32 values, 16 and 32 points, 16 heads.
_CALLS = [
    (8, 32, 4, 8, 2, 2),
    (8, 32, 1, 8, 2, 2),
    (8, 32, 1, 4, 2, 2),
    (4, 256, 4, 4, 2, 2),
    (4, 144, 4, 4, 2, 2),
    (8, 32, 4, 8, 4, 4),
    (8, 32, 4, 8, 2, 2),
    (8, 32, 4, 16, 2, 2),
    (16, 32, 4, 8, 2, 2),
    (1, 256, 1, 1, 2, 2),
]


@pytest.mark.parametrize("NH, CH, NL, NP, vb, lb", _CALLS)
def test_plan_fits_sbuf_with_the_most_heads_per_blend(NH, CH, NL, NP, vb, lb):
    """Gather slots cover N_h * N_p (32, 64 or 128), the sampling tiles fit their share of SBUF, the pipeline is as
    deep as fits (2 or 3), a blend takes as many heads as fit, and the build tiles hold at most 8192 elements."""
    plan = plan_ms_deformable_attention_quad(NH, CH, NL, NP, vb, lb, lb)
    assert plan.ok
    J = NH * NP
    assert plan.jp in (32, 64, 128) and plan.jp >= J and (plan.jp == 32 or plan.jp // 2 < J)
    assert 1 <= plan.groups <= NH and plan.depth in (2, 3)
    budget = _SBUF_SHARE * _TRN2_SBUF_BYTES
    assert msda_quad_sbuf_bytes(NH, CH, NL, NP, plan.jp, plan.groups, plan.depth, vb, lb, lb) <= budget
    if plan.groups < NH:
        assert msda_quad_sbuf_bytes(NH, CH, NL, NP, plan.jp, plan.groups + 1, plan.depth, vb, lb, lb) > budget
    if plan.depth < 3:
        assert msda_quad_sbuf_bytes(NH, CH, NL, NP, plan.jp, 1, 3, vb, lb, lb) > budget
    assert 1 <= plan.seg <= 8 and plan.seg * 4 * NH * CH <= max(8192, 4 * NH * CH)


def test_plan_takes_wide_heads_a_few_at_a_time():
    """The library's narrow heads blend all at once, three items deep; 256-channel heads blend one at a time, two
    items deep; more than 128 samples per query and level are rejected."""
    narrow = plan_ms_deformable_attention_quad(8, 32, 4, 8, 2, 2, 2)
    assert (narrow.jp, narrow.groups, narrow.depth) == (64, 8, 3)
    wide = plan_ms_deformable_attention_quad(4, 256, 4, 4, 2, 2, 2)
    assert (wide.jp, wide.groups, wide.depth) == (32, 1, 2)
    assert not plan_ms_deformable_attention_quad(8, 32, 4, 32, 2, 2, 2).ok


@pytest.mark.parametrize("B, NQ", [(1, 256), (2, 256), (1, 263), (1, 501), (1, 100), (2, 100), (3, 900), (1, 40000)])
def test_work_covers_every_query_once_per_program_count(B, NQ):
    """Query tiles cover every (batch, query); with LNC 1 one program takes them all, with LNC 2 both programs take
    the same number of consecutive tiles (one control flow) and together cover every query."""
    for n_prgs in (1, 2):
        covered = np.zeros((B, NQ), dtype=np.int64)
        counts = []
        for prg_id in range(n_prgs):
            tiles = msda_quad_work(B, NQ, n_prgs, prg_id)
            counts.append(len(tiles))
            for b, q0, nq in tiles:
                assert 0 <= b < B and 0 <= q0 and q0 + nq <= NQ and nq == min(_PMAX, NQ)
                covered[b, q0 : q0 + nq] += 1
        assert covered.min() >= 1
        assert len(set(counts)) == 1
        if n_prgs == 1:
            assert counts[0] == B * ((NQ + _PMAX - 1) // _PMAX)


@pytest.mark.parametrize("padding_mode", ["zeros", "border"])
@pytest.mark.parametrize("align_corners", [False, True])
def test_reference_matches_the_library_semantics(padding_mode, align_corners):
    """The vectorized float64 reference equals the library's per-sample reference (corner masks for zeros, clamped
    corners for border), here on a small call with samples outside the levels."""
    from nkilib_src.nkilib.experimental.deformable_attention.ms_deformable_attention_torch import (
        ms_deformable_attention_torch_ref,
    )

    rng = np.random.default_rng(3)
    shapes = ((5, 7), (3, 4))
    starts = (0, 35)
    value = torch.from_numpy(rng.standard_normal((1, 47, 2, 3)))
    loc = torch.from_numpy(rng.uniform(-0.3, 1.3, (1, 6, 2, 2, 3, 2)))
    aw = torch.from_numpy(rng.uniform(0.1, 1.0, (1, 6, 2, 2, 3)))
    ours = ms_deformable_attention_f64(value, shapes, starts, loc, aw, "BLNC", "BQHLP2", align_corners, padding_mode)
    lib = ms_deformable_attention_torch_ref(
        value, shapes, starts, loc, aw, "BLNC", "BQHLP2", align_corners, padding_mode
    )
    assert torch.allclose(ours, lib.double().reshape(ours.shape), atol=1e-12)


@pytest.mark.parametrize(
    "version, expected",
    [("0.6.0+31049202112.g85070674", False), ("0.7.0+37713544100.gfede04bb", True), ("1.0.0", True), ("dev", True)],
)
def test_nki_at_least(monkeypatch, version, expected):
    """The nki version gate: releases before 0.7.0 fail it, and a version it cannot parse passes."""
    import nki

    monkeypatch.setattr(nki, "__version__", version)
    assert nki_at_least(0, 7, 0) is expected
