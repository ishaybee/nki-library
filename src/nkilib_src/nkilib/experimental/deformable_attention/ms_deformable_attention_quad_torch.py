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

"""Vectorized PyTorch reference for the quad-scratch multi-scale deformable attention kernel."""

import torch


def ms_deformable_attention_f64(
    value: torch.Tensor,
    spatial_shapes,
    level_start_index,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    value_layout: str = "BLNC",
    sampling_locations_layout: str = "BQHLP2",
    align_corners: bool = False,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """
    Multi-scale deformable attention forward in float64, the semantics of ms_deformable_attention_torch_ref
    (zeros: out-of-range corners dropped; border: corners clamped to the level), vectorized over queries, heads and
    points.

    Returns:
        torch.Tensor: (B, N_q, N_h * C_h) float64.
    """
    v = value.double()
    if value_layout == "BNLC":
        v = v.permute(0, 2, 1, 3)
    loc = sampling_locations.double()
    if sampling_locations_layout == "B2QHLP":
        loc = loc.permute(0, 2, 3, 4, 5, 1)
    aw = attention_weights.double()
    B, _, NH, CH = v.shape
    NQ = loc.shape[1]
    out = torch.zeros(B, NQ, NH, CH, dtype=torch.float64)
    bi = torch.arange(B).view(B, 1, 1, 1)
    hi = torch.arange(NH).view(1, 1, NH, 1)
    for li, (H, W) in enumerate(spatial_shapes):
        start = level_start_index[li]
        vl = v[:, start : start + H * W]
        if align_corners:
            x = loc[:, :, :, li, :, 0] * (W - 1)
            y = loc[:, :, :, li, :, 1] * (H - 1)
        else:
            x = loc[:, :, :, li, :, 0] * W - 0.5
            y = loc[:, :, :, li, :, 1] * H - 0.5
        x0 = torch.floor(x)
        y0 = torch.floor(y)
        fx = x - x0
        fy = y - y0
        for dy in range(2):
            for dx in range(2):
                xi = (x0 + dx).long()
                yi = (y0 + dy).long()
                w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
                if padding_mode == "zeros":
                    w = w * ((xi >= 0) & (xi < W) & (yi >= 0) & (yi < H))
                pos = yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1)
                out += (vl[bi, pos, hi] * (w * aw[:, :, :, li, :])[..., None]).sum(3)
    return out.reshape(B, NQ, NH * CH)


def ms_deformable_attention_quad_torch_ref(
    value: torch.Tensor,
    spatial_shapes,
    level_start_index,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    value_layout: str = "BLNC",
    sampling_locations_layout: str = "BQHLP2",
    align_corners: bool = False,
    padding_mode: str = "zeros",
) -> dict[str, torch.Tensor]:
    """
    PyTorch reference of ms_deformable_attention_quad: ms_deformable_attention_f64 on the call's (possibly 16-bit)
    values, returned in value's dtype.

    Args:
        value (torch.Tensor): (B, L, N_h, C_h) ("BLNC") or (B, N_h, L, C_h) ("BNLC").
        spatial_shapes: ((H_l, W_l), ...) per level.
        level_start_index: First row of each level in L.
        sampling_locations (torch.Tensor): (B, N_q, N_h, N_l, N_p, 2) ("BQHLP2") or (B, 2, N_q, N_h, N_l, N_p).
        attention_weights (torch.Tensor): (B, N_q, N_h, N_l, N_p).
        value_layout, sampling_locations_layout, align_corners, padding_mode: as the kernel.

    Returns:
        dict[str, torch.Tensor]: ``{"output": (B, N_q, N_h * C_h)}``.
    """
    out = ms_deformable_attention_f64(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        value_layout,
        sampling_locations_layout,
        align_corners,
        padding_mode,
    )
    return {"output": out.to(value.dtype)}
