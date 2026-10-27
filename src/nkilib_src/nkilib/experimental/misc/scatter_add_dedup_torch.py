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
"""PyTorch reference implementation for the scatter_add_dedup kernel."""

import torch


def scatter_add_dedup_torch_ref(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    unique_indices: bool = False,
) -> dict[str, torch.Tensor]:
    """
    PyTorch reference implementation of the scatter_add_dedup kernel.

    Equivalent to ``input.index_add(0, index, src)``: ``output[index[k], :] = input[index[k], :] + src[k, :]`` summed
    over every k, with no restriction on repeated indices.

    Called by ``torch_ref_wrapper``, which converts numpy arrays to torch tensors before invoking this function.

    Args:
        input (torch.Tensor): [N, D], Destination values to accumulate into.
        dim (int): Dimension along which to scatter. Must be 0 to match the kernel contract.
        index (torch.Tensor): [K], 1D integer tensor of row indices into ``input``.
        src (torch.Tensor): [K, D], Source values to scatter-add.
        unique_indices (bool): The kernel's promise that the indices of every 128-index tile are distinct. The result
            does not depend on it, so the reference ignores it.

    Returns:
        dict[str, torch.Tensor]: ``{"output_0": output}`` where ``output`` has shape [N, D].

    Notes:
        - The sum is accumulated in float32 and rounded to the input dtype once at the end, so the reference does not
          carry the per-update rounding of a low-precision kernel.
        - Only ``dim == 0`` is supported, matching the kernel's restriction.
    """
    if dim != 0:
        raise ValueError(f"scatter_add_dedup currently only supports dim=0, got dim={dim}")
    total = input.float().index_add(0, index.long(), src.float())
    return {"output_0": total.to(input.dtype)}
