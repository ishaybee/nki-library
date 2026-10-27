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
"""Integration tests for scatter_add_dedup kernel."""

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.misc.scatter_add_dedup import scatter_add_dedup
from nkilib_src.nkilib.experimental.misc.scatter_add_dedup_torch import scatter_add_dedup_torch_ref

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _make_index(pattern, n_rows, k_size):
    """Index patterns that stress different collision structures (all values in [0, n_rows))."""
    if pattern == "distinct_tiles":
        # each 128-index tile is a slice of a permutation, as scatter_add's unique_indices=True requires
        tiles = [np.random.permutation(n_rows)[: min(128, k_size - s)] for s in range(0, k_size, 128)]
        return np.concatenate(tiles)
    if pattern == "token_major":
        # the top_k rows of a token are consecutive: repeats inside every tile
        return np.repeat(np.arange(n_rows), -(-k_size // n_rows))[:k_size]
    if pattern == "sorted_runs":
        return np.sort(np.random.randint(0, n_rows, size=k_size))
    if pattern == "all_equal":
        return np.full(k_size, n_rows // 2)
    return np.random.randint(0, n_rows, size=k_size)  # "random"


def _generate_inputs(n_rows, d_size, k_size, pattern, dtype):
    np.random.seed(7)
    return {
        "input": np.random.randn(n_rows, d_size).astype(dtype),
        "dim": 0,
        "index": _make_index(pattern, n_rows, k_size).astype(np.int32),
        "src": np.random.randn(k_size, d_size).astype(dtype),
    }


def _output_tensors(kernel_input):
    return {"output_0": np.zeros_like(kernel_input["input"])}


PARAM_NAMES = "n_rows, d_size, k_size, pattern, dtype, lnc"
TEST_PARAMS = [
    # small tables: one-hot matmuls on TensorE (plan_scatter_add_dedup's PATH_ONE_HOT)
    (256, 64, 1024, "random", nl.float32, 1),
    (256, 64, 8192, "random", nl.float32, 2),  # LNC=2: enough tiles to pay for the fold of the two cores' sums
    (200, 3, 500, "random", nl.float32, 1),  # K is not a multiple of 128, tiny D
    (200, 3, 5000, "random", nl.float32, 2),
    (64, 80, 1024, "all_equal", nl.float32, 1),  # every source hits one row
    (64, 80, 8192, "all_equal", nl.float32, 2),
    (16, 64, 4096, "random", nl.float32, 2),  # fewer rows than a tile: heavy collisions
    (64, 16, 256, "token_major", nl.float32, 1),
    (1000, 1, 20000, "random", nl.bfloat16, 1),  # a histogram over eight row blocks
    (300, 128, 8000, "random", nl.bfloat16, 2),  # a partial row block
    (512, 1024, 2048, "random", nl.bfloat16, 1),  # two 512-column chunks
    (512, 1024, 2048, "random", nl.bfloat16, 2),
    (256, 2048, 2048, "token_major", nl.bfloat16, 2),  # top_k = 8
    (512, 1024, 1024, "random", nl.float32, 2),  # few fp32 tiles at LNC=2: two buffers, not one-hot
    (128, 2880, 512, "random", nl.bfloat16, 1),  # D not a multiple of 512
    (128, 2880, 4096, "random", nl.bfloat16, 2),
    # narrow rows: read-modify-writes into float32 buffers, index tiles split over the cores at LNC=2
    (4096, 80, 1024, "all_equal", nl.float32, 1),  # two buffers
    (4096, 80, 1024, "all_equal", nl.float32, 2),  # one buffer per core
    (256, 64, 1024, "random", nl.float32, 2),  # few tiles: the one-hot fold would cost more
    (64, 80, 1024, "all_equal", nl.float32, 2),
    (100, 16, 100, "random", nl.float32, 2),  # one tile
    (128, 1025, 300, "random", nl.float32, 2),
    (5000, 3, 1500, "random", nl.float32, 1),  # K is not a multiple of 128, tiny D
    (5000, 3, 1500, "random", nl.float32, 2),
    (2048, 16, 4096, "token_major", nl.float32, 1),
    (1000, 65, 3000, "sorted_runs", nl.float32, 2),  # odd D
    (4096, 128, 8192, "distinct_tiles", nl.float32, 2),  # distinct indices within every tile
    (1500, 256, 3000, "random", nl.float32, 2),
    (4096, 1025, 600, "random", nl.float32, 2),  # wide but odd D: index tiles split over the cores
    (100000, 64, 1024, "random", nl.float32, 1),  # few tiles over many rows: in place, columns split over the cores
    (100000, 64, 1024, "random", nl.float32, 2),
    (100000, 64, 1024, "random", nl.bfloat16, 2),
    (90000, 32, 700, "random", nl.float32, 2),  # K is not a multiple of 128
    # wide rows: accumulated in the input dtype, columns split over the cores at LNC=2
    (512, 1024, 2048, "random", nl.float32, 1),  # two buffers
    (2048, 1024, 4096, "random", nl.bfloat16, 1),
    (2048, 1024, 4096, "random", nl.bfloat16, 2),
    (1024, 4096, 2048, "distinct_tiles", nl.bfloat16, 2),
    # the plan keeps one private buffer with scratch rows: more than 2048 columns per core, where in place takes one
    # tile at a time
    (1000, 4608, 1920, "random", nl.float32, 2),
    (1000, 2304, 1920, "random", nl.float32, 1),
    (16384, 512, 512, "random", nl.float32, 1),  # few tiles over many rows: in place, gather + add + scatter
    (16384, 512, 512, "random", nl.float32, 2),
    (8192, 512, 1024, "all_equal", nl.float32, 2),
    (32768, 1024, 1024, "random", nl.bfloat16, 2),
    (2048, 2880, 1024, "token_major", nl.bfloat16, 2),
    # in place, groups of tiles: repeats across the tiles of a group, short last groups and partial last tiles
    (8192, 4096, 640, "sorted_runs", nl.float32, 2),  # 8 KB rows: groups of 2, then 1 tile
    (8192, 4096, 640, "random", nl.float32, 1),  # two column blocks per tile
    (4096, 4096, 300, "all_equal", nl.float32, 2),  # a group of 2, then a partial tile
    (8192, 2048, 900, "token_major", nl.bfloat16, 1),  # groups of 4, the last ending in a 4-index tile
    (100000, 64, 1000, "sorted_runs", nl.float32, 2),
    (100000, 64, 1024, "all_equal", nl.bfloat16, 1),
]

# unique_indices=True, on indices that keep its promise (every 128-index tile a slice of a permutation)
UNIQUE_PARAM_NAMES = "n_rows, d_size, k_size, dtype, lnc"
UNIQUE_TEST_PARAMS = [
    (300, 64, 1000, nl.float32, 2),  # narrow rows, two buffers; K is not a multiple of 128
    (256, 1, 1024, nl.float32, 1),  # one-hot, 4-byte rows
    (512, 256, 2048, nl.bfloat16, 2),  # one-hot
    (128, 512, 256, nl.bfloat16, 2),  # the smallest scatter_add test shape: one buffer per core
    (512, 1024, 2048, nl.float32, 2),  # wide rows, two buffers
    (128, 2880, 512, nl.bfloat16, 2),  # one-hot, D not a multiple of 512
    (128, 1025, 300, nl.float32, 2),  # wide but odd D: index tiles split over the cores
    (4096, 128, 8192, nl.float32, 1),  # narrow rows, two buffers
    (4096, 128, 8192, nl.float32, 2),
    (3000, 64, 1000, nl.float32, 2),  # narrow rows, one buffer per core
    (65536, 64, 512, nl.float32, 2),  # narrow rows, few tiles: in place, columns split over the cores
    (512, 1024, 2048, nl.float32, 1),  # wide rows, two buffers
    (2048, 1024, 8192, nl.float32, 2),
    (1024, 4096, 2048, nl.bfloat16, 1),  # wide rows, one buffer: the output itself
    (1024, 4096, 2048, nl.bfloat16, 2),
    (8192, 1024, 256, nl.float32, 1),
    (8192, 1024, 256, nl.float32, 2),
]


@pytest_test_metadata(name="ScatterAddDedup")
@pytest_marks(["scatter_add_dedup"])
class TestScatterAddDedupKernel:
    """Test class for scatter_add_dedup kernel."""

    @pytest.mark.fast
    @pytest_parametrize(PARAM_NAMES, TEST_PARAMS)
    def test_scatter_add_dedup(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_rows,
        d_size,
        k_size,
        pattern,
        dtype,
        lnc,
    ):
        is_bf16 = dtype == nl.bfloat16

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=scatter_add_dedup,
            torch_ref=torch_ref_wrapper(scatter_add_dedup_torch_ref),
            kernel_input_generator=lambda _: _generate_inputs(n_rows, d_size, k_size, pattern, dtype),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                platform_target=platform_target,
                dump_after_lowering=False,
                logical_nc_config=lnc,
            ),
            atol=2.5e-1 if is_bf16 else 1e-4,
            rtol=5e-2 if is_bf16 else 1e-5,
        )

    @pytest.mark.fast
    @pytest_parametrize(UNIQUE_PARAM_NAMES, UNIQUE_TEST_PARAMS)
    def test_scatter_add_dedup_unique_indices(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_rows,
        d_size,
        k_size,
        dtype,
        lnc,
    ):
        is_bf16 = dtype == nl.bfloat16

        def generate_inputs(_):
            kernel_input = _generate_inputs(n_rows, d_size, k_size, "distinct_tiles", dtype)
            kernel_input["unique_indices"] = True
            return kernel_input

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=scatter_add_dedup,
            torch_ref=torch_ref_wrapper(scatter_add_dedup_torch_ref),
            kernel_input_generator=generate_inputs,
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                platform_target=platform_target,
                dump_after_lowering=False,
                logical_nc_config=lnc,
            ),
            atol=2.5e-1 if is_bf16 else 1e-4,
            rtol=5e-2 if is_bf16 else 1e-5,
        )
