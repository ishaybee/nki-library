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

"""Tests for the grid sample kernel using UnitTestFramework."""

from typing import Any, final

import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
import torch
import torch.nn.functional as F
from nkilib_src.nkilib.experimental.vision.grid_sample import GRID_SAMPLE_NKI_SUPPORTED, grid_sample
from nkilib_src.nkilib.experimental.vision.grid_sample_torch import grid_sample_torch_ref

from test.utils.common_dataclasses import CompilerArgs, CustomValidator, CustomValidatorWithOutputTensorData, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Error allowed on top of torch's own fp32 error, relative to max(1, max |reference|): fp32 accumulation, and the
# rounding of a 16-bit output (half an ulp is 2^-8 of the value for bfloat16, 2^-11 for float16).
_TOL = {np.dtype(np.float32): 1e-5, np.dtype(nl.bfloat16): 2.0**-7, np.dtype(nl.float16): 2.0**-10}

# The 9-point cluster of _cluster_points, as multiples of its two axes: the centre, the ends of each axis, then the
# four corners between them.
_CLUSTER_A = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0])
_CLUSTER_B = np.array([0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


def _cluster_points(rng, N, H, W, Ho, Wo, s):
    """
    Pixel coordinates [N, Ho, Wo] of clustered multi-view point sampling, as when per-view feature maps are read
    around the projections of 3-D primitives: Ho Wo / 9 clusters (consecutive in the flat grid), each the 9-point
    pattern of a random anisotropic footprint, at s times the resolution the footprints are drawn at (1: a full-size
    map, 0.25: a quarter-resolution one). Two thirds of the centres lie in a window of a fifth to a half of the image
    per axis, around a point up to a tenth of the image outside it; the others (points the caller discards) anywhere
    within half an image of it. The axis lengths are log-uniform in [0.5, 40] pixels (the minor one at least 0.5) at
    any orientation, and each offset coordinate is clamped to 50 pixels.
    """
    n = Ho * Wo // 9
    fw = rng.uniform(0.2, 0.5, (N, 1)) * W
    fh = rng.uniform(0.2, 0.5, (N, 1)) * H
    cx = rng.uniform(-0.1, 1.1, (N, 1)) * W + (rng.uniform(0.0, 1.0, (N, n)) - 0.5) * fw
    cy = rng.uniform(-0.1, 1.1, (N, 1)) * H + (rng.uniform(0.0, 1.0, (N, n)) - 0.5) * fh
    unseen = rng.uniform(0.0, 1.0, (N, n)) < 1.0 / 3.0
    cx = np.where(unseen, rng.uniform(-0.5, 1.5, (N, n)) * W, cx)
    cy = np.where(unseen, rng.uniform(-0.5, 1.5, (N, n)) * H, cy)
    s1 = np.exp(rng.uniform(np.log(0.5), np.log(40.0), (N, n)))
    s2 = np.maximum(s1 * rng.uniform(0.05, 1.0, (N, n)), 0.5)
    th = rng.uniform(0.0, np.pi, (N, n))
    ox = _CLUSTER_A * (s1 * np.cos(th))[..., None] - _CLUSTER_B * (s2 * np.sin(th))[..., None]
    oy = _CLUSTER_A * (s1 * np.sin(th))[..., None] + _CLUSTER_B * (s2 * np.cos(th))[..., None]
    px = cx[..., None] + np.clip(ox, -50.0, 50.0) * s
    py = cy[..., None] + np.clip(oy, -50.0, 50.0) * s
    return px.reshape(N, Ho, Wo), py.reshape(N, Ho, Wo)


def _grid(rng, N, H, W, Ho, Wo, kind, align_corners):
    """
    Normalized grid [N, Ho, Wo, 2] (x, y). kind: flowD (the identity plus an independent random pixel flow of up to
    D pixels per axis, as in a warp), random (uniform in [-1.3, 1.3]: a margin outside the input), randn (normal),
    far (uniform in [-50, 50]: almost every sample outside), affine (a random rotation, scale and shift, as a spatial
    transformer), identity (a resize to Ho x Wo), window (a Ho x Wo lattice of whole-pixel offsets around one random
    point per batch, as RAFT's correlation lookup), clusterS (clusters of 9 points around random footprints, at S
    times the resolution of the footprints, as multi-view point sampling; see _cluster_points).
    """
    if kind == "random":
        return rng.uniform(-1.3, 1.3, (N, Ho, Wo, 2))
    if kind == "randn":
        return rng.standard_normal((N, Ho, Wo, 2))
    if kind == "far":
        return rng.uniform(-50.0, 50.0, (N, Ho, Wo, 2))
    if kind in ("affine", "identity"):
        theta = np.zeros((N, 2, 3))
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        if kind == "affine":
            ang = rng.uniform(-0.6, 0.6, N)
            sc = rng.uniform(0.75, 1.25, N)
            theta[:, 0, 0] = sc * np.cos(ang)
            theta[:, 0, 1] = -sc * np.sin(ang)
            theta[:, 1, 0] = sc * np.sin(ang)
            theta[:, 1, 1] = sc * np.cos(ang)
            theta[:, :, 2] = rng.uniform(-0.2, 0.2, (N, 2))
        return F.affine_grid(torch.from_numpy(theta), [N, 1, Ho, Wo], align_corners=align_corners).numpy()
    if kind == "window":
        cx = rng.uniform(-4, W + 4, (N, 1, 1))
        cy = rng.uniform(-4, H + 4, (N, 1, 1))
        dy, dx = np.meshgrid(np.arange(Ho) - Ho // 2, np.arange(Wo) - Wo // 2, indexing="ij")
        px = cx + dx[None]
        py = cy + dy[None]
    elif kind.startswith("cluster"):
        px, py = _cluster_points(rng, N, H, W, Ho, Wo, float(kind[7:]))
    else:
        d = float(kind[4:])
        py, px = np.meshgrid(np.arange(Ho, dtype=np.float64), np.arange(Wo, dtype=np.float64), indexing="ij")
        px = px[None] * (W - 1) / max(Wo - 1, 1) + rng.uniform(-d, d, (N, Ho, Wo))
        py = py[None] * (H - 1) / max(Ho - 1, 1) + rng.uniform(-d, d, (N, Ho, Wo))
    if align_corners:
        return np.stack([px * 2.0 / max(W - 1, 1) - 1.0, py * 2.0 / max(H - 1, 1) - 1.0], -1)
    return np.stack([(px * 2.0 + 1.0) / W - 1.0, (py * 2.0 + 1.0) / H - 1.0], -1)


def _source_coords(g, size, align_corners):
    """Unnormalized fp32 coordinates, ATen's operation order (the padding mode does not move a tie)."""
    g = g.astype(np.float32)
    if align_corners:
        return (g + np.float32(1.0)) * np.float32(0.5) * np.float32(size - 1)
    return ((g + np.float32(1.0)) * np.float32(size) - np.float32(1.0)) * np.float32(0.5)


def generate_grid_sample_inputs(N, C, H, W, Ho, Wo, kind, mode, align_corners, dtype, seed):
    """Input [N, C, H, W] (normal) and grid [N, Ho, Wo, 2] in dtype. Nearest grids keep clear of rounding ties,
    where float32 and float64 coordinates may round to different pixels."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, C, H, W)).astype(np.float32).astype(dtype)
    g = _grid(rng, N, H, W, Ho, Wo, kind, align_corners).astype(np.float32).astype(dtype).astype(np.float64)
    if mode == "nearest":
        for d, size in ((0, W), (1, H)):
            for _ in range(4):
                pix = _source_coords(g[..., d], size, align_corners).astype(np.float64)
                tie = np.abs(pix - np.floor(pix) - 0.5) < 0.02
                g[..., d] = np.where(tie, g[..., d] + 0.37 / max(size, 1), g[..., d])
                g = g.astype(np.float32).astype(dtype).astype(np.float64)
    return np.ascontiguousarray(x), np.ascontiguousarray(g.astype(np.float32).astype(dtype))


def _as_array(raw, dtype, shape):
    """A kernel output handed to a custom validator (raw bytes) as an array."""
    return np.frombuffer(np.asarray(raw).tobytes(), dtype=dtype).reshape(shape)


def _grid_sample_comparator(inputs, mode, padding_mode, align_corners):
    """
    The output against torch's grid_sample in float64 on the same (16-bit rounded) values: the error may not
    exceed twice torch's own float32 error, nor the dtype's tolerance relative to max(1, max |reference|).
    """

    def comparator(golden, output_tensors):
        x = torch.from_numpy(inputs["input"].astype(np.float64))
        g = torch.from_numpy(inputs["grid"].astype(np.float64))
        dtype = inputs["dtype"]
        ref = F.grid_sample(x, g, mode=mode, padding_mode=padding_mode, align_corners=align_corners).numpy()
        r32 = F.grid_sample(
            x.float(), g.float(), mode=mode, padding_mode=padding_mode, align_corners=align_corners
        ).double()
        err32 = float(np.abs(r32.numpy() - ref).max())
        bar = max(2.0 * err32, _TOL[np.dtype(dtype)] * max(1.0, float(np.abs(ref).max())))

        class _Out(CustomValidator):
            def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                out = _as_array(inference_output, dtype, ref.shape).astype(np.float64)
                finite = bool(np.isfinite(out).all())
                err = float(np.abs(out - ref).max())
                self._print_with_log(
                    f"output: max |out - ref| = {err:.3e} (bar {bar:.3e}, torch fp32 {err32:.3e}), finite = {finite}"
                )
                return finite and err <= bar

        return {"output": CustomValidatorWithOutputTensorData(validator=_Out, output_ndarray=output_tensors["output"])}

    return comparator


# The kernel rejects every target but trn2 while tracing.
_UNSUPPORTED_PLATFORMS = [platform for platform in Platforms if platform != Platforms.TRN2]

# fmt: off
GRID_SAMPLE_PARAM_NAMES = "batch, channels, height, width, out_height, out_width, kind, mode, padding_mode, align_corners, dtype, lnc"
GRID_SAMPLE_PARAMS = [
    # a frame warp (a bounded pixel flow, border, align_corners) at a pyramid level and at the full 992 x 1280 frame
    pytest.param(1, 3, 248, 320, 248, 320, "flow8", "bilinear", "border", True, nl.float32, 1, marks=pytest.mark.fast),
    (1, 3, 992, 1280, 992, 1280, "flow30", "bilinear", "border", True, nl.float32, 2),
    (1, 128, 62, 80, 62, 80, "flow2", "bilinear", "border", True, nl.float32, 1),
    # one image past the unroll budget (its steady state in a device loop) and many one-batch segments past it
    (1, 3, 2048, 2304, 2048, 2304, "flow30", "bilinear", "border", True, nl.float32, 1),
    (12, 3, 64, 64, 600, 600, "randn", "bilinear", "zeros", False, nl.float32, 1),
    # one-batch quads gathering in the sample layout: with a channel-major store, with 16 channels (the 4-instruction
    # blend), and one whose chunk width (24 pixels) is not a multiple of 32 (the index layout)
    (1, 40, 64, 80, 64, 80, "flow8", "bilinear", "border", True, nl.float32, 1),
    (1, 16, 124, 160, 124, 160, "flow8", "bilinear", "border", True, nl.float32, 2),
    (1, 64, 48, 64, 96, 128, "random", "bilinear", "zeros", False, nl.float32, 2),
    # every padding mode and corner convention with samples outside the input; unbounded and far grids
    pytest.param(2, 5, 40, 50, 30, 70, "random", "bilinear", "zeros", False, nl.float32, 1, marks=pytest.mark.fast),
    (2, 5, 40, 50, 30, 70, "random", "bilinear", "zeros", True, nl.float32, 2),
    (1, 3, 64, 96, 64, 96, "flow8", "bilinear", "reflection", False, nl.float32, 1),
    pytest.param(1, 3, 64, 96, 64, 96, "random", "bilinear", "reflection", True, nl.float32, 2, marks=pytest.mark.fast),
    (1, 4, 33, 47, 40, 40, "randn", "bilinear", "border", False, nl.float32, 1),
    (2, 3, 20, 30, 16, 16, "far", "bilinear", "zeros", False, nl.float32, 1),
    (2, 3, 20, 30, 16, 16, "far", "bilinear", "border", True, nl.float32, 1),
    (2, 3, 20, 30, 16, 16, "far", "bilinear", "reflection", False, nl.float32, 2),
    # nearest (round half to even), each padding mode
    pytest.param(2, 5, 40, 50, 30, 70, "random", "nearest", "zeros", False, nl.float32, 1, marks=pytest.mark.fast),
    (1, 4, 33, 47, 20, 20, "random", "nearest", "border", True, nl.float32, 2),
    (2, 3, 5, 5, 3, 4, "random", "nearest", "reflection", True, nl.float32, 1),
    (1, 3, 992, 1280, 992, 1280, "flow30", "nearest", "zeros", False, nl.float32, 2),
    # 16-bit inputs
    pytest.param(1, 8, 48, 64, 48, 64, "flow8", "bilinear", "border", True, nl.bfloat16, 1, marks=pytest.mark.fast),
    (64, 4, 16, 16, 16, 16, "random", "bilinear", "reflection", False, nl.bfloat16, 2),
    (2, 64, 32, 32, 32, 32, "random", "bilinear", "zeros", False, nl.float16, 1),
    (4, 256, 16, 16, 16, 16, "random", "nearest", "zeros", False, nl.bfloat16, 1),
    # batches: spatial transformers (one segment of many batches: chunks span batches and store through a staging
    # buffer), PyTorch's GridSampler benchmark shapes (scratch-build tiles of batches stacked along the partitions),
    # RAFT's correlation lookup (one channel: shared chunks store straight into the output; at full resolution no
    # scratch, at level 3 a scratch built one batch per partition), large outputs (one batch per segment), and
    # output sizes without a fitting divisor (one batch per segment while its chunks are nearly full, else shared
    # chunks)
    pytest.param(8, 4, 28, 28, 28, 28, "affine", "bilinear", "zeros", False, nl.float32, 1, marks=pytest.mark.fast),
    (32, 3, 224, 224, 224, 224, "affine", "bilinear", "zeros", False, nl.float32, 2),
    (500, 5, 20, 20, 15, 15, "randn", "bilinear", "zeros", False, nl.float32, 1),
    (7040, 1, 55, 128, 9, 9, "window", "bilinear", "zeros", True, nl.float32, 2),
    (7040, 1, 6, 16, 9, 9, "window", "bilinear", "zeros", True, nl.float32, 1),
    (10, 3, 80, 80, 500, 500, "randn", "bilinear", "zeros", False, nl.float32, 2),
    (3, 4, 50, 60, 97, 97, "random", "bilinear", "border", False, nl.float32, 1),
    (70, 2, 10, 10, 97, 97, "random", "bilinear", "zeros", True, nl.float32, 2),
    (200, 2, 12, 12, 7, 7, "random", "nearest", "zeros", False, nl.float32, 1),
    # one channel and few samples from a large input: no scratch, the gathers read the input and zero the taps
    # outside it (each padding mode, a single row, 16-bit); small batches stacked along the build tiles' partitions
    # (quads with shared chunks at LNC=2, the plain layout); and more build tiles than are unrolled, through the
    # pipelined build (quads at LNC=2, the plain layout) and, for images of more than 128 rows, a device loop
    (3, 1, 100, 130, 9, 9, "random", "bilinear", "zeros", False, nl.float32, 1),
    (3, 1, 100, 130, 9, 9, "random", "bilinear", "border", True, nl.float32, 2),
    (3, 1, 100, 130, 9, 9, "random", "bilinear", "reflection", False, nl.float32, 1),
    (2, 1, 1, 300, 1, 4, "random", "bilinear", "zeros", False, nl.float32, 1),
    (4, 1, 64, 64, 5, 5, "random", "bilinear", "zeros", False, nl.bfloat16, 2),
    (500, 10, 50, 50, 50, 50, "randn", "bilinear", "zeros", False, nl.float32, 2),
    (300, 8, 40, 40, 6, 6, "random", "bilinear", "border", False, nl.float32, 1),
    (130, 16, 70, 70, 40, 40, "random", "bilinear", "zeros", False, nl.float32, 2),
    (130, 64, 62, 62, 9, 9, "random", "bilinear", "zeros", False, nl.float32, 1),
    (130, 8, 130, 130, 48, 48, "random", "bilinear", "zeros", False, nl.float32, 1),
    # many channels, stored channel-major through the Tensor Engine: blocks of 128 channels (8, and a partial one),
    # 16-bit casts while evicting, nearest straight from the gathered rows, one batch and shared chunks, 8192
    # channels (64 blocks through a 16-region tile)
    (2, 1024, 16, 16, 16, 16, "random", "bilinear", "zeros", False, nl.float32, 1),
    (2, 256, 20, 34, 1960, 1, "random", "bilinear", "zeros", False, nl.float32, 2),
    (3, 160, 20, 20, 30, 30, "random", "bilinear", "border", True, nl.float16, 1),
    (1, 300, 9, 9, 9, 9, "random", "nearest", "reflection", False, nl.float32, 2),
    (5, 100, 50, 50, 50, 50, "randn", "bilinear", "zeros", False, nl.float32, 2),
    (1, 8192, 4, 4, 4, 4, "random", "nearest", "zeros", False, nl.float32, 1),
    # point sampling from a large input (the plain layout: 4 gathered rows per sample), points, a resize
    (2, 64, 100, 168, 4900, 1, "random", "bilinear", "zeros", False, nl.float32, 1),
    (4, 64, 128, 128, 1, 16384, "random", "bilinear", "zeros", False, nl.float32, 2),
    (1, 3, 135, 240, 270, 480, "identity", "bilinear", "border", False, nl.float32, 1),
    # multi-view point sampling (see _cluster_points): 72,000 clusters of 9 points in the 4-channel float16 maps of 24
    # UHD views (the plain layout, as quads would pass 2 GiB; one segment per view in a device loop) and of their
    # quarter-resolution maps (quads); 48 views, whose plain scratch passes 2 GiB as well; and one view's maps at
    # 1,280,000 clusters (quads at both resolutions)
    (24, 4, 2160, 3840, 648000, 1, "cluster1", "bilinear", "zeros", False, nl.float16, 2),
    (24, 4, 540, 960, 648000, 1, "cluster0.25", "bilinear", "zeros", False, nl.float16, 1),
    (48, 4, 2160, 3840, 648000, 1, "cluster1", "bilinear", "zeros", False, nl.float16, 1),
    (1, 4, 2160, 3840, 11520000, 1, "cluster1", "bilinear", "zeros", False, nl.float16, 1),
    (1, 4, 540, 960, 11520000, 1, "cluster0.25", "bilinear", "zeros", False, nl.float16, 2),
    # past 2^24 scratch rows per batch (6K images and up), where stage A adds the scratch rows in integer arithmetic:
    # strips of rows of a 6K frame's warp (quads in the sample layout; 16 bits at LNC=2), a DCI 8K strip (the quad
    # scratch two columns wider), few samples from a 6K image (the plain layout, one column wider) and one-channel
    # inputs (nearest, and bilinear in the plain layout instead of the direct read); an output past 2^24 pixels,
    # whose last chunk starts at an odd pixel
    (1, 3, 3456, 6144, 432, 6144, "flow64", "bilinear", "border", True, nl.float32, 1),
    (1, 3, 3456, 6144, 216, 6144, "flow64", "bilinear", "zeros", False, nl.bfloat16, 2),
    (1, 3, 4320, 8192, 270, 8192, "flow64", "bilinear", "border", True, nl.float16, 1),
    (1, 3, 3456, 6144, 300, 400, "random", "bilinear", "zeros", False, nl.float32, 1),
    (1, 1, 4096, 4100, 64, 64, "random", "nearest", "zeros", False, nl.float32, 1),
    (1, 1, 4096, 4100, 64, 64, "random", "bilinear", "border", True, nl.float32, 2),
    (1, 1, 64, 64, 4201, 4101, "random", "bilinear", "zeros", False, nl.float32, 1),
    # edges of the shape range: 1 x 1 inputs, single rows, tiny outputs
    (1, 1, 1, 1, 1, 1, "random", "bilinear", "zeros", False, nl.float32, 1),
    (1, 2, 1, 1, 3, 3, "random", "bilinear", "reflection", True, nl.float32, 1),
    (5, 7, 1, 300, 1, 129, "random", "bilinear", "border", False, nl.float32, 2),
    (1, 2, 16, 16, 5, 7, "random", "bilinear", "zeros", False, nl.float32, 1),
]

# Calls the kernel rejects while tracing: (..., expected error message fragment).
GRID_SAMPLE_VALIDATION_ERROR_PARAMS = [
    (1, 16384, 4, 4, 4, 4, "random", "nearest", "zeros", False, nl.float32, 1, "do not fit in SBUF"),
    (1, 1, 1 << 24, 1, 1, 1, "random", "nearest", "zeros", False, nl.float32, 1, "past exact index arithmetic"),
]

# A small valid call, for the target check (the arguments of _run_grid_sample after the platform).
_TARGET_CHECK_CALL = (1, 3, 8, 8, 8, 8, "random", "bilinear", "zeros", False, nl.float32, 1)
# fmt: on


@pytest_test_metadata(name="GridSample")
@pytest_marks(["vision", "grid_sample"])
@pytest.mark.platforms(exclude=_UNSUPPORTED_PLATFORMS)
@pytest.mark.skipif(
    not GRID_SAMPLE_NKI_SUPPORTED,
    reason="grid_sample needs nki 0.7.0 or later (Neuron SDK 2.33): earlier releases reject its batched indirect DMA "
    "gathers",
)
@final
class TestGridSampleKernel:
    """Test class for the grid sample kernel using UnitTestFramework."""

    def _run_grid_sample(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        N: int,
        C: int,
        H: int,
        W: int,
        Ho: int,
        Wo: int,
        kind: str,
        mode: str,
        padding_mode: str,
        align_corners: bool,
        dtype,
        lnc: int,
        overrides: dict | None = None,
    ) -> None:
        """Run a grid_sample test against the fp64 torch reference (overrides replace kernel arguments)."""
        inputs = {}

        def input_generator(test_config):
            x, g = generate_grid_sample_inputs(
                N, C, H, W, Ho, Wo, kind, mode, align_corners, dtype, seed=N * 131 + H * 7 + Wo
            )
            inputs.update(input=x.astype(np.float32), grid=g.astype(np.float32), dtype=np.dtype(dtype))
            args = {"input": x, "grid": g, "mode": mode, "padding_mode": padding_mode, "align_corners": align_corners}
            args.update(overrides or {})
            return args

        def output_tensors(kernel_input):
            return {"output": np.zeros((N, C, Ho, Wo), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=grid_sample,
            torch_ref=torch_ref_wrapper(grid_sample_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            custom_comparator=_grid_sample_comparator(inputs, mode, padding_mode, align_corners),
        )

    @pytest.mark.parametrize(GRID_SAMPLE_PARAM_NAMES, GRID_SAMPLE_PARAMS)
    def test_grid_sample(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        channels: int,
        height: int,
        width: int,
        out_height: int,
        out_width: int,
        kind: str,
        mode: str,
        padding_mode: str,
        align_corners: bool,
        dtype,
        lnc: int,
    ) -> None:
        """grid_sample of warps, transformers, point samples and resizes against the fp64 reference."""
        self._run_grid_sample(
            test_manager,
            platform_target,
            batch,
            channels,
            height,
            width,
            out_height,
            out_width,
            kind,
            mode,
            padding_mode,
            align_corners,
            dtype,
            lnc,
        )

    @pytest.mark.parametrize(GRID_SAMPLE_PARAM_NAMES + ", error_match", GRID_SAMPLE_VALIDATION_ERROR_PARAMS)
    def test_grid_sample_validation_errors(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        channels: int,
        height: int,
        width: int,
        out_height: int,
        out_width: int,
        kind: str,
        mode: str,
        padding_mode: str,
        align_corners: bool,
        dtype,
        lnc: int,
        error_match: str,
    ) -> None:
        """
        Unsupported calls are rejected by input validation. The kernel_assert fires during tracing, which the test
        framework re-raises wrapped in a CompilationException, so this matches on the message.
        """
        with pytest.raises(Exception, match=error_match):
            self._run_grid_sample(
                test_manager,
                platform_target,
                batch,
                channels,
                height,
                width,
                out_height,
                out_width,
                kind,
                mode,
                padding_mode,
                align_corners,
                dtype,
                lnc,
            )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "overrides, error_match",
        [({"mode": "bicubic"}, "mode must be one of"), ({"padding_mode": "circular"}, "padding_mode must be one of")],
    )
    def test_grid_sample_rejects_options(
        self, test_manager: Orchestrator, platform_target: Platforms, overrides: dict, error_match: str
    ) -> None:
        """Modes and padding modes outside the supported set are rejected while tracing."""
        with pytest.raises(Exception, match=error_match):
            self._run_grid_sample(test_manager, platform_target, *_TARGET_CHECK_CALL, overrides=overrides)

    @pytest.mark.fast
    @pytest.mark.platforms(exclude=[Platforms.TRN2])
    def test_grid_sample_rejects_other_targets(self, test_manager: Orchestrator, platform_target: Platforms) -> None:
        """The kernel is validated on trn2 only; any other target is rejected while tracing."""
        with pytest.raises(Exception, match="supports trn2"):
            self._run_grid_sample(test_manager, platform_target, *_TARGET_CHECK_CALL)
