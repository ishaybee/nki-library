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

"""PyTorch reference implementation for the grid sample kernel."""

import torch
import torch.nn.functional as F


def grid_sample_torch_ref(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> dict[str, torch.Tensor]:
    """
    PyTorch reference of grid_sample: torch.nn.functional.grid_sample evaluated in float64 on the call's (possibly
    16-bit) input and grid values, returned in the input's dtype.

    Args:
        input (torch.Tensor): [N, C, H, W] input.
        grid (torch.Tensor): [N, Ho, Wo, 2] normalized (x, y) sample coordinates.
        mode (str): "bilinear" or "nearest".
        padding_mode (str): "zeros", "border" or "reflection".
        align_corners (bool): As torch.nn.functional.grid_sample.

    Returns:
        dict[str, torch.Tensor]: ``{"output": [N, C, Ho, Wo]}``.
    """
    out = F.grid_sample(
        input.double(), grid.double(), mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )
    return {"output": out.to(input.dtype)}
