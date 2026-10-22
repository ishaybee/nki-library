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

"""Tests for the quad-scratch multi-scale deformable attention kernel using UnitTestFramework."""

from typing import Any, final

import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
import torch
from nkilib_src.nkilib.experimental.deformable_attention.ms_deformable_attention_quad import (
    MSDA_QUAD_NKI_SUPPORTED,
    ms_deformable_attention_quad,
)
from nkilib_src.nkilib.experimental.deformable_attention.ms_deformable_attention_quad_torch import (
    ms_deformable_attention_f64,
    ms_deformable_attention_quad_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, CustomValidator, CustomValidatorWithOutputTensorData, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Error allowed relative to max(1, max |reference|): fp32 coordinates and sums, and the rounding of a 16-bit output
# (half an ulp is 2^-8 of the value for bfloat16, 2^-11 for float16).
_TOL = {np.dtype(np.float32): 1e-5, np.dtype(nl.bfloat16): 2.0**-7, np.dtype(nl.float16): 2.0**-10}

_SMALL = [(50, 45), (25, 21), (13, 9), (5, 7)]

# The kernel rejects every target but trn2 while tracing.
_UNSUPPORTED_PLATFORMS = [platform for platform in Platforms if platform != Platforms.TRN2]


def generate_msda_inputs(
    batch,
    n_queries,
    n_heads,
    c_head,
    n_levels,
    n_points,
    spatial_shapes,
    dtype,
    value_layout,
    sampling_locations_layout,
    margin,
    seed,
):
    """
    Gaussian value, sampling locations uniform in [-margin, 1 + margin] and attention weights normalised over
    (levels, points), as the library test's generator (16-bit locations and weights in float16 unless the value is
    float32).
    """
    rng = np.random.default_rng(seed)
    L = sum(h * w for h, w in spatial_shapes)
    starts = []
    s = 0
    for h, w in spatial_shapes:
        starts.append(s)
        s += h * w
    vshape = (batch, L, n_heads, c_head) if value_layout == "BLNC" else (batch, n_heads, L, c_head)
    value = rng.standard_normal(vshape).astype(np.float32).astype(dtype)
    sdt = np.float32 if dtype == nl.float32 else np.float16
    if sampling_locations_layout == "BQHLP2":
        lshape = (batch, n_queries, n_heads, n_levels, n_points, 2)
    else:
        lshape = (batch, 2, n_queries, n_heads, n_levels, n_points)
    loc = rng.uniform(-margin, 1.0 + margin, size=lshape).astype(sdt)
    aw = rng.uniform(0.1, 1.0, size=(batch, n_queries, n_heads, n_levels, n_points))
    aw = (aw / aw.sum(axis=(3, 4), keepdims=True)).astype(sdt)
    return value, tuple(tuple(x) for x in spatial_shapes), tuple(starts), loc, aw


def _as_array(raw, dtype, shape):
    """A kernel output handed to a custom validator (raw bytes) as an array."""
    return np.frombuffer(np.asarray(raw).tobytes(), dtype=dtype).reshape(shape)


def _msda_comparator(inputs):
    """The output against the float64 reference on the same (16-bit rounded) inputs, within the dtype's tolerance
    relative to max(1, max |reference|)."""

    def comparator(golden, output_tensors):
        def f64(name):
            return torch.from_numpy(inputs[name].astype(np.float64))

        dtype = inputs["dtype"]
        ref = ms_deformable_attention_f64(
            f64("value"),
            inputs["spatial_shapes"],
            inputs["level_start_index"],
            f64("sampling_locations"),
            f64("attention_weights"),
            value_layout=inputs["value_layout"],
            sampling_locations_layout=inputs["sampling_locations_layout"],
            align_corners=inputs["align_corners"],
            padding_mode=inputs["padding_mode"],
        ).numpy()
        bar = _TOL[dtype] * max(1.0, float(np.abs(ref).max()))

        class _Out(CustomValidator):
            def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                out = _as_array(inference_output, dtype, ref.shape).astype(np.float64)
                finite = bool(np.isfinite(out).all())
                err = float(np.abs(out - ref).max())
                self._print_with_log(f"output: max |out - ref| = {err:.3e} (bar {bar:.3e}), finite = {finite}")
                return finite and err <= bar

        return {"output": CustomValidatorWithOutputTensorData(validator=_Out, output_ndarray=output_tensors["output"])}

    return comparator


# fmt: off
MSDA_QUAD_PARAM_NAMES = (
    "batch, n_queries, n_heads, c_head, n_levels, n_points, spatial_shapes, dtype, value_layout, "
    "sampling_locations_layout, align_corners, padding_mode, margin, lnc"
)
MSDA_QUAD_PARAMS = [
    # the library test's basic set: both layouts of each tensor, border padding, align_corners, wide heads
    # (blended a few heads at a time), N_q not a multiple of 128 (the last query tile overlaps the one before it)
    pytest.param(2, 256, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BLNC", "B2QHLP", False, "zeros", 0.0, 1, marks=pytest.mark.fast),
    (2, 256, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BLNC", "BQHLP2", False, "zeros", 0.0, 1),
    (1, 256, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BNLC", "B2QHLP", False, "zeros", 0.0, 1),
    (1, 256, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BNLC", "BQHLP2", False, "zeros", 0.0, 1),
    (1, 256, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BLNC", "B2QHLP", False, "border", 0.0, 1),
    (1, 512, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BLNC", "BQHLP2", False, "border", 0.0, 1),
    (1, 256, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BLNC", "B2QHLP", True, "zeros", 0.0, 1),
    (1, 256, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BLNC", "BQHLP2", True, "border", 0.0, 1),
    pytest.param(1, 256, 4, 256, 4, 4, _SMALL, nl.bfloat16, "BLNC", "B2QHLP", False, "zeros", 0.0, 1, marks=pytest.mark.fast),
    (1, 256, 4, 144, 4, 4, _SMALL, nl.bfloat16, "BNLC", "BQHLP2", False, "zeros", 0.0, 1),
    (1, 263, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BLNC", "B2QHLP", False, "zeros", 0.0, 1),
    (1, 501, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BNLC", "B2QHLP", False, "zeros", 0.0, 1),
    # BEVFormer: two batches at the fast shape, and the small model's SCA, TSA and decoder (one level each)
    (2, 512, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BLNC", "B2QHLP", False, "zeros", 0.0, 1),
    (1, 13500, 8, 32, 1, 8, [(113, 200)], nl.bfloat16, "BLNC", "B2QHLP", False, "zeros", 0.0, 1),
    (1, 22500, 8, 32, 1, 4, [(150, 150)], nl.bfloat16, "BLNC", "B2QHLP", False, "zeros", 0.0, 1),
    (1, 900, 8, 32, 1, 4, [(150, 150)], nl.bfloat16, "BLNC", "B2QHLP", False, "zeros", 0.0, 1),
    # samples outside the levels (clamped coordinates: zeros, border), one short query tile, float32 and float16
    # values, 16 points per level (128 gather slots per query)
    pytest.param(1, 256, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BLNC", "BQHLP2", False, "zeros", 0.3, 1, marks=pytest.mark.fast),
    (1, 256, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BNLC", "B2QHLP", True, "border", 0.3, 1),
    (1, 100, 8, 32, 4, 4, _SMALL, nl.bfloat16, "BLNC", "BQHLP2", False, "zeros", 0.0, 1),
    (1, 256, 8, 32, 4, 8, _SMALL, nl.float32, "BLNC", "BQHLP2", False, "zeros", 0.1, 1),
    (1, 256, 8, 32, 4, 8, _SMALL, nl.float16, "BNLC", "B2QHLP", True, "border", 0.1, 1),
    (1, 256, 8, 32, 4, 16, _SMALL, nl.bfloat16, "BLNC", "B2QHLP", False, "zeros", 0.0, 1),
    # LNC=2: an even and an odd number of query tiles, one tile both programs take, two batches, wide heads
    pytest.param(2, 256, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BLNC", "B2QHLP", False, "zeros", 0.0, 2, marks=pytest.mark.fast),
    (1, 263, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BNLC", "BQHLP2", False, "border", 0.1, 2),
    (1, 100, 8, 32, 4, 4, _SMALL, nl.bfloat16, "BLNC", "B2QHLP", True, "zeros", 0.0, 2),
    (1, 256, 4, 256, 4, 4, _SMALL, nl.bfloat16, "BLNC", "B2QHLP", False, "zeros", 0.0, 2),
]

# Calls the kernel rejects while tracing: (..., expected error message fragment).
MSDA_QUAD_VALIDATION_ERROR_PARAMS = [
    (1, 256, 8, 32, 4, 32, _SMALL, nl.bfloat16, "BLNC", "B2QHLP", False, "zeros", 0.0, 1, "exceeds 128"),
    (1, 256, 8, 32, 4, 8, _SMALL, nl.bfloat16, "BLNC", "B2QHLP", False, "reflection", 0.0, 1, "padding_mode must be"),
]
# fmt: on


@pytest_test_metadata(name="MS Deformable Attention Quad")
@pytest_marks(["deformable_attention", "ms_deform_attn"])
@pytest.mark.platforms(exclude=_UNSUPPORTED_PLATFORMS)
@pytest.mark.skipif(
    not MSDA_QUAD_NKI_SUPPORTED,
    reason="ms_deformable_attention_quad needs nki 0.7.0 or later (Neuron SDK 2.33): earlier releases reject its "
    "batched indirect DMA gathers",
)
@final
class TestMSDeformableAttentionQuad:
    """Test class for the quad-scratch multi-scale deformable attention kernel."""

    def _run(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_queries,
        n_heads,
        c_head,
        n_levels,
        n_points,
        spatial_shapes,
        dtype,
        value_layout,
        sampling_locations_layout,
        align_corners,
        padding_mode,
        margin,
        lnc,
    ) -> None:
        """Run one call against the float64 reference."""
        inputs = {}

        def input_generator(test_config):
            value, shapes, starts, loc, aw = generate_msda_inputs(
                batch,
                n_queries,
                n_heads,
                c_head,
                n_levels,
                n_points,
                spatial_shapes,
                dtype,
                value_layout,
                sampling_locations_layout,
                margin,
                seed=batch * 131 + n_queries * 7 + n_points,
            )
            args = {
                "value": value,
                "spatial_shapes": shapes,
                "level_start_index": starts,
                "sampling_locations": loc,
                "attention_weights": aw,
                "value_layout": value_layout,
                "sampling_locations_layout": sampling_locations_layout,
                "align_corners": align_corners,
                "padding_mode": padding_mode,
            }
            inputs.update(args, dtype=np.dtype(dtype))
            return args

        def output_tensors(kernel_input):
            return {"output": np.zeros((batch, n_queries, n_heads * c_head), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=ms_deformable_attention_quad,
            torch_ref=torch_ref_wrapper(ms_deformable_attention_quad_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            custom_comparator=_msda_comparator(inputs),
        )

    @pytest.mark.parametrize(MSDA_QUAD_PARAM_NAMES, MSDA_QUAD_PARAMS)
    def test_ms_deformable_attention_quad(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_queries,
        n_heads,
        c_head,
        n_levels,
        n_points,
        spatial_shapes,
        dtype,
        value_layout,
        sampling_locations_layout,
        align_corners,
        padding_mode,
        margin,
        lnc,
    ) -> None:
        """The library test's shapes, BEVFormer calls and the kernel's own paths against the float64 reference."""
        self._run(
            test_manager,
            platform_target,
            batch,
            n_queries,
            n_heads,
            c_head,
            n_levels,
            n_points,
            spatial_shapes,
            dtype,
            value_layout,
            sampling_locations_layout,
            align_corners,
            padding_mode,
            margin,
            lnc,
        )

    @pytest.mark.parametrize(MSDA_QUAD_PARAM_NAMES + ", error_match", MSDA_QUAD_VALIDATION_ERROR_PARAMS)
    def test_ms_deformable_attention_quad_validation_errors(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_queries,
        n_heads,
        c_head,
        n_levels,
        n_points,
        spatial_shapes,
        dtype,
        value_layout,
        sampling_locations_layout,
        align_corners,
        padding_mode,
        margin,
        lnc,
        error_match,
    ) -> None:
        """Unsupported calls are rejected by input validation while tracing (re-raised by the framework)."""
        with pytest.raises(Exception, match=error_match):
            self._run(
                test_manager,
                platform_target,
                batch,
                n_queries,
                n_heads,
                c_head,
                n_levels,
                n_points,
                spatial_shapes,
                dtype,
                value_layout,
                sampling_locations_layout,
                align_corners,
                padding_mode,
                margin,
                lnc,
            )
