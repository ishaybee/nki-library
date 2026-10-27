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

"""scatter_add_dedup's plan (work path, split over cores, buffers), checked on the host without running a kernel."""

import pytest
from nkilib_src.nkilib.experimental.misc.scatter_add_dedup import (
    _CHAINS,
    _D_BLK,
    _GATHER_SCATTER_EXTRA_NS,
    _GROUP_GATHER_SCATTER_EXTRA_NS,
    _MAX_FOLD_SOURCES,
    _PSUM_BANKS,
    _RMW_OVERLAP_BYTES,
    _RMW_OVERLAP_NS,
    _WIDE_ROW_BYTES,
    PATH_COLUMNS,
    PATH_ONE_HOT,
    PATH_TILES,
    _gather_scatter_extra_ns,
    _interp_ns,
    plan_scatter_add_chains,
    plan_scatter_add_dedup,
)


def _option(n_rows, d_size, k_size, itemsize, lnc, unique):
    """The plan of one call, by the name of the measured option it corresponds to."""
    path, _, chains, in_place = plan_scatter_add_dedup(n_rows, d_size, k_size, itemsize, lnc, unique)
    if path == PATH_ONE_HOT:
        return "one_hot"
    if chains == 2:
        return "two"
    if in_place:
        return "in_place"
    return "one"


@pytest.mark.parametrize(
    "d_size, itemsize, num_shards, expected",
    [
        (512, 4, 1, (PATH_COLUMNS, 512)),  # 2048 bytes: the first wide size
        (511, 4, 1, (PATH_TILES, 511)),
        (512, 4, 2, (PATH_COLUMNS, 256)),
        (1024, 2, 2, (PATH_COLUMNS, 512)),
        (1023, 2, 1, (PATH_TILES, 1023)),
        (1025, 4, 2, (PATH_TILES, 1025)),  # wide but odd: columns cannot be split evenly
        (2880, 2, 2, (PATH_COLUMNS, 1440)),
        (64, 4, 2, (PATH_TILES, 64)),
        (3, 4, 1, (PATH_TILES, 3)),
    ],
)
def test_plan_sharding_choice(d_size, itemsize, num_shards, expected):
    # 100,000 rows: too many for the one-hot path; a million sources: too many tiles for narrow rows in place
    assert plan_scatter_add_dedup(100000, d_size, 1 << 20, itemsize, num_shards)[:2] == expected


def test_wide_threshold_is_in_bytes():
    assert _WIDE_ROW_BYTES == 2048
    assert plan_scatter_add_dedup(100000, _WIDE_ROW_BYTES // 2, 1 << 20, 2, 1)[0] == PATH_COLUMNS
    assert plan_scatter_add_dedup(100000, _WIDE_ROW_BYTES // 2 - 1, 1 << 20, 2, 1)[0] == PATH_TILES


@pytest.mark.parametrize("num_shards", [1, 2])
def test_fold_fits_one_dma_compute(num_shards):
    # the final reduction adds the input and every buffer of every core in one instruction
    assert num_shards * _CHAINS + 1 <= _MAX_FOLD_SOURCES


# (N, D, K, bytes per element, LNC, unique_indices, {option: ms}), trn2, 20 back-to-back calls, best of 3, with the path
# forced: one_hot = TensorE one-hot matmuls, two = two buffers, one = one buffer (a private copy with scratch rows, or
# one float32 buffer per core for narrow rows), in_place = the output itself, columns split over the cores
# (read-modify-writes with unique_indices, groups of gathers, adds and plain scatters with repeats). unique_indices rows
# use indices distinct within every tile, the others uniform random rows (or the named workload's own pattern).
_MEASURED = [
    (16384, 80, 235200, 4, 1, False, {"one": 8.0478, "two": 4.2332}),  # bev_lss_g128_c80
    (50257, 768, 16384, 4, 1, False, {"in_place": 1.4761, "one": 1.9010, "two": 2.2993}),  # cr_gpt2_emb
    (128256, 4096, 16384, 2, 1, False, {"in_place": 5.4114, "one": 8.5651, "two": 12.1906}),  # cr_llama_emb_bf16
    (100000, 64, 1024, 4, 1, False, {"in_place": 0.1460, "one": 0.2501, "two": 0.3654}),  # cr_n100k_d64_k1k
    (100000, 64, 1024, 2, 1, False, {"in_place": 0.1083, "one": 0.2182, "two": 0.3294}),  # cr_n100k_d64_k1k_bf16
    (16384, 1024, 16384, 4, 1, False, {"in_place": 1.3560, "one": 1.4905, "two": 1.4659}),  # cr_n16k_d1024_k16k
    (1048576, 64, 131072, 4, 1, False, {"in_place": 6.2522, "one": 6.4238, "two": 5.5423}),  # cr_n1m_d64_k131k
    (1048576, 64, 8192, 4, 1, False, {"in_place": 1.2018, "one": 2.2074, "two": 3.3636}),  # cr_n1m_d64_k8k
    (262144, 128, 16384, 4, 1, False, {"in_place": 1.1525, "one": 1.5853, "two": 1.9546}),  # cr_n256k_d128_k16k
    (262144, 128, 4096, 4, 1, False, {"in_place": 0.6149, "one": 1.1350, "two": 1.7157}),  # cr_n256k_d128_k4k
    (262144, 128, 65536, 4, 1, False, {"in_place": 3.2039, "one": 3.3774, "two": 2.8696}),  # cr_n256k_d128_k64k
    (32768, 4096, 16384, 4, 1, False, {"in_place": 4.4564, "one": 6.3951, "two": 7.8917}),  # cr_n32k_d4096_k16k
    (4096, 2048, 8192, 2, 1, False, {"in_place": 0.6046, "one": 0.6639, "two": 0.5850}),  # cr_n4k_d2048_k8k_bf16
    (4096, 4096, 8192, 4, 1, False, {"in_place": 1.6435, "one": 1.9586, "two": 1.9685}),  # cr_n4k_d4096_k8k
    (50000, 256, 2048, 4, 1, False, {"in_place": 0.2761, "one": 0.4795, "two": 0.6906}),  # cr_n50k_d256_k2k
    (65536, 1024, 16384, 2, 1, False, {"in_place": 1.2733, "one": 1.6534, "two": 1.9919}),  # cr_n64k_d1024_k16k_bf16
    (8192, 2048, 131072, 4, 1, False, {"in_place": 13.0603, "one": 13.0102, "two": 11.0311}),  # cr_n8k_d2048_k131k
    (32768, 4096, 16384, 4, 1, True, {"in_place": 3.9613, "two": 7.2873}),  # cu_n32k_d4096_k16k
    (65536, 1024, 16384, 2, 1, True, {"in_place": 1.1623, "two": 1.8273}),  # cu_n64k_d1024_k16k_bf16
    (65536, 64, 512, 4, 1, True, {"in_place": 0.0917, "one": 0.1659, "two": 0.2386}),  # cu_n65k_d64_k512
    (8192, 1024, 2048, 4, 1, True, {"in_place": 0.2514, "two": 0.4387}),  # cu_n8k_d1024_k2k
    (8192, 2048, 131072, 4, 1, True, {"in_place": 11.6564, "two": 8.2659}),  # cu_n8k_d2048_k131k
    (16384, 1024, 65536, 4, 1, False, {"in_place": 5.9210, "one": 4.7085, "two": 3.7537}),  # gnn_rand_n16k_d1024
    (65536, 64, 524288, 4, 1, False, {"one": 17.9798, "two": 9.3448}),  # gnn_rand_n64k_d64
    (1024, 4096, 2048, 2, 1, True, {"in_place": 0.2359, "two": 0.2628}),  # moe_t1024_d4096_k2
    (1024, 4096, 2048, 2, 1, False, {"in_place": 0.3115, "one": 0.3324, "two": 0.3096}),  # moe_t1024_d4096_k2
    (1024, 4096, 2048, 4, 1, True, {"in_place": 0.3634, "two": 0.4364}),  # moe_t1024_d4096_k2_f32
    (1024, 4096, 2048, 4, 1, False, {"in_place": 0.4111, "one": 0.4926, "two": 0.5283}),  # moe_t1024_d4096_k2_f32
    # moe_t128_d512_k2
    (128, 512, 256, 2, 1, True, {"in_place": 0.0885, "one": 0.0895, "one_hot": 0.0836, "two": 0.0935}),
    # moe_t128_d512_k2
    (128, 512, 256, 2, 1, False, {"in_place": 0.0837, "one": 0.0864, "one_hot": 0.0818, "two": 0.0829}),
    (128, 512, 512, 4, 1, True, {"in_place": 0.0926, "one_hot": 0.0847, "two": 0.0911}),  # moe_t128_d512_k4_f32
    # moe_t128_d512_k4_f32
    (128, 512, 512, 4, 1, False, {"in_place": 0.0834, "one": 0.0840, "one_hot": 0.0816, "two": 0.0793}),
    (256, 2048, 2048, 2, 1, True, {"in_place": 0.1508, "one_hot": 0.0899, "two": 0.1104}),  # moe_t256_d2048_k8
    # moe_t256_d2048_k8
    (256, 2048, 2048, 2, 1, False, {"in_place": 0.1572, "one": 0.1655, "one_hot": 0.0853, "two": 0.1318}),
    (256, 2048, 2048, 4, 1, True, {"in_place": 0.2056, "one_hot": 0.1471, "two": 0.1694}),  # moe_t256_d2048_k8_f32
    # moe_t256_d2048_k8_f32
    (256, 2048, 2048, 4, 1, False, {"in_place": 0.2373, "one": 0.2235, "one_hot": 0.1465, "two": 0.2188}),
    (4096, 2880, 16384, 2, 1, True, {"in_place": 1.2367, "two": 1.2777}),  # moe_t4096_d2880_k4
    (4096, 2880, 16384, 4, 1, True, {"in_place": 1.9298, "two": 2.0236}),  # moe_t4096_d2880_k4_f32
    (4096, 2880, 16384, 4, 1, False, {"in_place": 2.5503, "one": 2.7551, "two": 2.3468}),  # moe_t4096_d2880_k4_f32
    (512, 1024, 2048, 2, 1, True, {"in_place": 0.1219, "one_hot": 0.0849, "two": 0.0926}),  # moe_t512_d1024_k4
    # moe_t512_d1024_k4
    (512, 1024, 2048, 2, 1, False, {"in_place": 0.1295, "one": 0.1357, "one_hot": 0.0834, "two": 0.1111}),
    (512, 1024, 2048, 4, 1, True, {"in_place": 0.1615, "one_hot": 0.1428, "two": 0.1226}),  # moe_t512_d1024_k4_f32
    # moe_t512_d1024_k4_f32
    (512, 1024, 2048, 4, 1, False, {"in_place": 0.1687, "one": 0.1730, "one_hot": 0.1441, "two": 0.1391}),
    (4096, 2880, 16384, 2, 1, False, {"in_place": 1.9332, "one": 1.9568, "two": 1.5152}),  # moe_tok_t4096_d2880_k4
    (1024, 128, 65536, 2, 1, False, {"one_hot": 0.3355, "two": 1.2473}),  # sn_n1024_d128_bf16
    (1024, 128, 65536, 4, 1, False, {"one_hot": 0.9724, "two": 1.2643}),  # sn_n1024_d128_f32
    (1024, 1, 65536, 2, 1, False, {"one_hot": 0.3251, "two": 1.2277}),  # sn_n1024_d1_bf16
    (1024, 1, 65536, 4, 1, False, {"one_hot": 1.8000, "two": 1.2088}),  # sn_n1024_d1_f32
    (1024, 256, 65536, 2, 1, False, {"one_hot": 0.4875, "two": 1.6538}),  # sn_n1024_d256_bf16
    (1024, 256, 65536, 4, 1, False, {"one_hot": 1.8320, "two": 1.5616}),  # sn_n1024_d256_f32
    (1024, 32, 65536, 2, 1, False, {"one_hot": 0.3271, "two": 1.1961}),  # sn_n1024_d32_bf16
    (1024, 32, 65536, 4, 1, False, {"one_hot": 1.7999, "two": 1.1864}),  # sn_n1024_d32_f32
    (1024, 512, 65536, 2, 1, False, {"one_hot": 0.9293, "two": 1.9443}),  # sn_n1024_d512_bf16
    (1024, 512, 65536, 4, 1, False, {"one_hot": 3.5849, "two": 2.3034}),  # sn_n1024_d512_f32
    (1024, 64, 65536, 2, 1, False, {"one_hot": 0.3307, "two": 1.1974}),  # sn_n1024_d64_bf16
    (1024, 64, 65536, 4, 1, False, {"one_hot": 1.7989, "two": 1.2214}),  # sn_n1024_d64_f32
    (256, 128, 65536, 2, 1, False, {"one_hot": 0.2261, "two": 1.2274}),  # sn_n256_d128_bf16
    (256, 128, 65536, 4, 1, False, {"one_hot": 0.2888, "two": 1.2659}),  # sn_n256_d128_f32
    (256, 1, 65536, 2, 1, False, {"one_hot": 0.2371, "two": 1.5115}),  # sn_n256_d1_bf16
    (256, 1, 65536, 4, 1, False, {"one_hot": 0.4773, "two": 1.5267}),  # sn_n256_d1_f32
    (256, 256, 65536, 2, 1, False, {"one_hot": 0.2710, "two": 1.6481}),  # sn_n256_d256_bf16
    (256, 256, 65536, 4, 1, False, {"one_hot": 0.4797, "two": 1.5483}),  # sn_n256_d256_f32
    (256, 32, 65536, 2, 1, False, {"one_hot": 0.2262, "two": 1.2247}),  # sn_n256_d32_bf16
    (256, 32, 65536, 4, 1, False, {"one_hot": 0.4727, "two": 1.2076}),  # sn_n256_d32_f32
    (256, 512, 65536, 2, 1, False, {"one_hot": 0.3088, "two": 1.9424}),  # sn_n256_d512_bf16
    (256, 512, 65536, 4, 1, False, {"one_hot": 0.9276, "two": 2.2904}),  # sn_n256_d512_f32
    (256, 64, 65536, 2, 1, False, {"one_hot": 0.2196, "two": 1.2068}),  # sn_n256_d64_bf16
    (256, 64, 65536, 4, 1, False, {"one_hot": 0.4726, "two": 1.2155}),  # sn_n256_d64_f32
    (512, 128, 65536, 2, 1, False, {"one_hot": 0.2867, "two": 1.2315}),  # sn_n512_d128_bf16
    (512, 128, 65536, 4, 1, False, {"one_hot": 0.4949, "two": 1.2607}),  # sn_n512_d128_f32
    (512, 1, 65536, 2, 1, False, {"one_hot": 0.2807, "two": 1.2630}),  # sn_n512_d1_bf16
    (512, 1, 65536, 4, 1, False, {"one_hot": 0.9307, "two": 1.2727}),  # sn_n512_d1_f32
    (512, 256, 65536, 2, 1, False, {"one_hot": 0.3527, "two": 1.6541}),  # sn_n512_d256_bf16
    (512, 256, 65536, 4, 1, False, {"one_hot": 0.9303, "two": 1.5504}),  # sn_n512_d256_f32
    (512, 32, 65536, 2, 1, False, {"one_hot": 0.2686, "two": 1.1976}),  # sn_n512_d32_bf16
    (512, 32, 65536, 4, 1, False, {"one_hot": 0.9214, "two": 1.1891}),  # sn_n512_d32_f32
    (512, 512, 65536, 2, 1, False, {"one_hot": 0.4821, "two": 1.9499}),  # sn_n512_d512_bf16
    (512, 512, 65536, 4, 1, False, {"one_hot": 1.8109, "two": 2.2928}),  # sn_n512_d512_f32
    (512, 64, 65536, 2, 1, False, {"one_hot": 0.2697, "two": 1.1991}),  # sn_n512_d64_bf16
    (512, 64, 65536, 4, 1, False, {"one_hot": 0.9254, "two": 1.2266}),  # sn_n512_d64_f32
    (64, 128, 65536, 2, 1, False, {"one_hot": 0.2170, "two": 1.2176}),  # sn_n64_d128_bf16
    (64, 128, 65536, 4, 1, False, {"one_hot": 0.1959, "two": 1.2479}),  # sn_n64_d128_f32
    (64, 1, 65536, 2, 1, False, {"one_hot": 0.2351, "two": 2.2772}),  # sn_n64_d1_bf16
    (64, 1, 65536, 4, 1, False, {"one_hot": 0.2166, "two": 2.2629}),  # sn_n64_d1_f32
    (64, 256, 65536, 2, 1, False, {"one_hot": 0.1939, "two": 1.6412}),  # sn_n64_d256_bf16
    (64, 256, 65536, 4, 1, False, {"one_hot": 0.2761, "two": 1.5402}),  # sn_n64_d256_f32
    (64, 32, 65536, 2, 1, False, {"one_hot": 0.2224, "two": 1.2167}),  # sn_n64_d32_bf16
    (64, 32, 65536, 4, 1, False, {"one_hot": 0.2004, "two": 1.2014}),  # sn_n64_d32_f32
    (64, 512, 65536, 2, 1, False, {"one_hot": 0.2724, "two": 1.9268}),  # sn_n64_d512_bf16
    (64, 512, 65536, 4, 1, False, {"one_hot": 0.4814, "two": 2.2747}),  # sn_n64_d512_f32
    (64, 64, 65536, 2, 1, False, {"one_hot": 0.2131, "two": 1.2125}),  # sn_n64_d64_bf16
    (64, 64, 65536, 4, 1, False, {"one_hot": 0.2115, "two": 1.2192}),  # sn_n64_d64_f32
    (16384, 80, 235200, 4, 2, False, {"one": 4.1046, "two": 2.2225}),  # bev_lss_g128_c80
    (50257, 768, 16384, 4, 2, False, {"in_place": 1.3038, "one": 1.7736, "two": 2.0572}),  # cr_gpt2_emb
    (128256, 4096, 16384, 2, 2, False, {"in_place": 4.4310, "one": 7.7138, "two": 10.7219}),  # cr_llama_emb_bf16
    (100000, 64, 1024, 4, 2, False, {"in_place": 0.1505, "one": 0.2797, "two": 0.4233}),  # cr_n100k_d64_k1k
    (100000, 64, 1024, 2, 2, False, {"in_place": 0.1136, "one": 0.2421, "two": 0.3902}),  # cr_n100k_d64_k1k_bf16
    (16384, 1024, 16384, 4, 2, False, {"in_place": 1.0953, "one": 1.3020, "two": 1.2703}),  # cr_n16k_d1024_k16k
    (1048576, 64, 131072, 4, 2, False, {"in_place": 6.1946, "one": 4.8509, "two": 5.6365}),  # cr_n1m_d64_k131k
    (1048576, 64, 8192, 4, 2, False, {"in_place": 1.2198, "one": 2.6973, "two": 4.4768}),  # cr_n1m_d64_k8k
    (262144, 128, 16384, 4, 2, False, {"in_place": 1.1358, "one": 1.5934, "two": 2.3221}),  # cr_n256k_d128_k16k
    (262144, 128, 4096, 4, 2, False, {"in_place": 0.6152, "one": 1.3564, "two": 2.2798}),  # cr_n256k_d128_k4k
    (262144, 128, 65536, 4, 2, False, {"in_place": 3.1669, "one": 2.4974, "two": 2.8103}),  # cr_n256k_d128_k64k
    (32768, 4096, 16384, 4, 2, False, {"in_place": 3.3882, "one": 5.0711, "two": 6.4015}),  # cr_n32k_d4096_k16k
    (4096, 2048, 8192, 2, 2, False, {"in_place": 0.5003, "one": 0.5517, "two": 0.4937}),  # cr_n4k_d2048_k8k_bf16
    (4096, 4096, 8192, 4, 2, False, {"in_place": 1.1002, "one": 1.2872, "two": 1.3870}),  # cr_n4k_d4096_k8k
    (50000, 256, 2048, 4, 2, False, {"in_place": 0.2736, "one": 0.5344, "two": 0.8175}),  # cr_n50k_d256_k2k
    (65536, 1024, 16384, 2, 2, False, {"in_place": 1.2016, "one": 1.5727, "two": 1.7629}),  # cr_n64k_d1024_k16k_bf16
    (8192, 2048, 131072, 4, 2, False, {"in_place": 9.3828, "one": 9.7539, "two": 7.1480}),  # cr_n8k_d2048_k131k
    (32768, 4096, 16384, 4, 2, True, {"in_place": 3.2840, "two": 6.2467}),  # cu_n32k_d4096_k16k
    (65536, 1024, 16384, 2, 2, True, {"in_place": 1.0421, "two": 1.6856}),  # cu_n64k_d1024_k16k_bf16
    (65536, 64, 512, 4, 2, True, {"in_place": 0.0926, "one": 0.1946, "two": 0.2982}),  # cu_n65k_d64_k512
    (8192, 1024, 2048, 4, 2, True, {"in_place": 0.2275, "two": 0.4087}),  # cu_n8k_d1024_k2k
    (8192, 2048, 131072, 4, 2, True, {"in_place": 9.0992, "two": 5.8587}),  # cu_n8k_d2048_k131k
    (16384, 1024, 65536, 4, 2, False, {"in_place": 4.5278, "one": 3.6780, "two": 3.0615}),  # gnn_rand_n16k_d1024
    (65536, 64, 524288, 4, 2, False, {"one": 9.1657, "two": 4.9258}),  # gnn_rand_n64k_d64
    (1024, 4096, 2048, 2, 2, True, {"in_place": 0.1856, "two": 0.1956}),  # moe_t1024_d4096_k2
    (1024, 4096, 2048, 2, 2, False, {"in_place": 0.1894, "one": 0.2261, "two": 0.2166}),  # moe_t1024_d4096_k2
    (1024, 4096, 2048, 4, 2, True, {"in_place": 0.2710, "two": 0.3403}),  # moe_t1024_d4096_k2_f32
    (1024, 4096, 2048, 4, 2, False, {"in_place": 0.2989, "one": 0.3370, "two": 0.3801}),  # moe_t1024_d4096_k2_f32
    # moe_t128_d512_k2
    (128, 512, 256, 2, 2, True, {"in_place": 0.0875, "one": 0.0816, "one_hot": 0.0897, "two": 0.0913}),
    # moe_t128_d512_k2
    (128, 512, 256, 2, 2, False, {"in_place": 0.0826, "one": 0.0925, "one_hot": 0.0806, "two": 0.0795}),
    (128, 512, 512, 4, 2, True, {"in_place": 0.0803, "one_hot": 0.0983, "two": 0.0911}),  # moe_t128_d512_k4_f32
    # moe_t128_d512_k4_f32
    (128, 512, 512, 4, 2, False, {"in_place": 0.0801, "one": 0.0911, "one_hot": 0.0811, "two": 0.0805}),
    (256, 2048, 2048, 2, 2, True, {"in_place": 0.1272, "one_hot": 0.0839, "two": 0.0966}),  # moe_t256_d2048_k8
    # moe_t256_d2048_k8
    (256, 2048, 2048, 2, 2, False, {"in_place": 0.1373, "one": 0.1401, "one_hot": 0.0812, "two": 0.1149}),
    (256, 2048, 2048, 4, 2, True, {"in_place": 0.1661, "one_hot": 0.1134, "two": 0.1356}),  # moe_t256_d2048_k8_f32
    # moe_t256_d2048_k8_f32
    (256, 2048, 2048, 4, 2, False, {"in_place": 0.1756, "one": 0.1806, "one_hot": 0.1121, "two": 0.1494}),
    (4096, 2880, 16384, 2, 2, True, {"in_place": 0.9656, "two": 0.7824}),  # moe_t4096_d2880_k4
    (4096, 2880, 16384, 4, 2, True, {"in_place": 1.4390, "two": 1.3748}),  # moe_t4096_d2880_k4_f32
    (4096, 2880, 16384, 4, 2, False, {"in_place": 1.6651, "one": 1.6405, "two": 1.5767}),  # moe_t4096_d2880_k4_f32
    (512, 1024, 2048, 2, 2, True, {"in_place": 0.1069, "one_hot": 0.0827, "two": 0.0917}),  # moe_t512_d1024_k4
    # moe_t512_d1024_k4
    (512, 1024, 2048, 2, 2, False, {"in_place": 0.1238, "one": 0.1216, "one_hot": 0.0895, "two": 0.0973}),
    (512, 1024, 2048, 4, 2, True, {"in_place": 0.1312, "one_hot": 0.1045, "two": 0.1095}),  # moe_t512_d1024_k4_f32
    # moe_t512_d1024_k4_f32
    (512, 1024, 2048, 4, 2, False, {"in_place": 0.1387, "one": 0.1507, "one_hot": 0.1044, "two": 0.1280}),
    (4096, 2880, 16384, 2, 2, False, {"in_place": 1.0525, "one": 1.1218, "two": 0.9313}),  # moe_tok_t4096_d2880_k4
    (1024, 128, 65536, 2, 2, False, {"one_hot": 0.1913, "two": 0.6533}),  # sn_n1024_d128_bf16
    (1024, 128, 65536, 4, 2, False, {"one_hot": 0.5071, "two": 0.6656}),  # sn_n1024_d128_f32
    (1024, 1, 65536, 2, 2, False, {"one_hot": 0.1832, "two": 0.7051}),  # sn_n1024_d1_bf16
    (1024, 1, 65536, 4, 2, False, {"one_hot": 0.9254, "two": 0.6998}),  # sn_n1024_d1_f32
    (1024, 256, 65536, 2, 2, False, {"one_hot": 0.2692, "two": 0.8484}),  # sn_n1024_d256_bf16
    (1024, 256, 65536, 4, 2, False, {"one_hot": 0.9479, "two": 0.8097}),  # sn_n1024_d256_f32
    (1024, 32, 65536, 2, 2, False, {"one_hot": 0.1821, "two": 0.6346}),  # sn_n1024_d32_bf16
    (1024, 32, 65536, 4, 2, False, {"one_hot": 0.9211, "two": 0.6452}),  # sn_n1024_d32_f32
    (1024, 512, 65536, 2, 2, False, {"one_hot": 0.4948, "two": 1.1255}),  # sn_n1024_d512_bf16
    (1024, 512, 65536, 4, 2, False, {"one_hot": 1.8343, "two": 1.7344}),  # sn_n1024_d512_f32
    (1024, 64, 65536, 2, 2, False, {"one_hot": 0.2111, "two": 0.6326}),  # sn_n1024_d64_bf16
    (1024, 64, 65536, 4, 2, False, {"one_hot": 0.9222, "two": 0.6594}),  # sn_n1024_d64_f32
    (256, 128, 65536, 2, 2, False, {"one_hot": 0.1512, "two": 0.6474}),  # sn_n256_d128_bf16
    (256, 128, 65536, 4, 2, False, {"one_hot": 0.1768, "two": 0.6614}),  # sn_n256_d128_f32
    (256, 1, 65536, 2, 2, False, {"one_hot": 0.1496, "two": 1.0355}),  # sn_n256_d1_bf16
    (256, 1, 65536, 4, 2, False, {"one_hot": 0.2624, "two": 1.0333}),  # sn_n256_d1_f32
    (256, 256, 65536, 2, 2, False, {"one_hot": 0.1849, "two": 0.8379}),  # sn_n256_d256_bf16
    (256, 256, 65536, 4, 2, False, {"one_hot": 0.2687, "two": 0.8021}),  # sn_n256_d256_f32
    (256, 32, 65536, 2, 2, False, {"one_hot": 0.1418, "two": 0.6707}),  # sn_n256_d32_bf16
    (256, 32, 65536, 4, 2, False, {"one_hot": 0.2557, "two": 0.6631}),  # sn_n256_d32_f32
    (256, 512, 65536, 2, 2, False, {"one_hot": 0.2097, "two": 1.0974}),  # sn_n256_d512_bf16
    (256, 512, 65536, 4, 2, False, {"one_hot": 0.4856, "two": 1.7304}),  # sn_n256_d512_f32
    (256, 64, 65536, 2, 2, False, {"one_hot": 0.1513, "two": 0.6312}),  # sn_n256_d64_bf16
    (256, 64, 65536, 4, 2, False, {"one_hot": 0.2569, "two": 0.6483}),  # sn_n256_d64_f32
    (512, 128, 65536, 2, 2, False, {"one_hot": 0.2263, "two": 0.6559}),  # sn_n512_d128_bf16
    (512, 128, 65536, 4, 2, False, {"one_hot": 0.2685, "two": 0.6633}),  # sn_n512_d128_f32
    (512, 1, 65536, 2, 2, False, {"one_hot": 0.1602, "two": 0.7602}),  # sn_n512_d1_bf16
    (512, 1, 65536, 4, 2, False, {"one_hot": 0.4824, "two": 0.7684}),  # sn_n512_d1_f32
    (512, 256, 65536, 2, 2, False, {"one_hot": 0.2077, "two": 0.8410}),  # sn_n512_d256_bf16
    (512, 256, 65536, 4, 2, False, {"one_hot": 0.4852, "two": 0.8076}),  # sn_n512_d256_f32
    (512, 32, 65536, 2, 2, False, {"one_hot": 0.1593, "two": 0.6320}),  # sn_n512_d32_bf16
    (512, 32, 65536, 4, 2, False, {"one_hot": 0.4772, "two": 0.6426}),  # sn_n512_d32_f32
    (512, 512, 65536, 2, 2, False, {"one_hot": 0.2671, "two": 1.1044}),  # sn_n512_d512_bf16
    (512, 512, 65536, 4, 2, False, {"one_hot": 0.9327, "two": 1.7142}),  # sn_n512_d512_f32
    (512, 64, 65536, 2, 2, False, {"one_hot": 0.1575, "two": 0.6294}),  # sn_n512_d64_bf16
    (512, 64, 65536, 4, 2, False, {"one_hot": 0.4791, "two": 0.6483}),  # sn_n512_d64_f32
    (64, 128, 65536, 2, 2, False, {"one_hot": 0.1403, "two": 0.6447}),  # sn_n64_d128_bf16
    (64, 128, 65536, 4, 2, False, {"one_hot": 0.1405, "two": 0.6589}),  # sn_n64_d128_f32
    (64, 1, 65536, 2, 2, False, {"one_hot": 0.1449, "two": 1.8957}),  # sn_n64_d1_bf16
    (64, 1, 65536, 4, 2, False, {"one_hot": 0.1459, "two": 1.9072}),  # sn_n64_d1_f32
    (64, 256, 65536, 2, 2, False, {"one_hot": 0.1392, "two": 0.8312}),  # sn_n64_d256_bf16
    (64, 256, 65536, 4, 2, False, {"one_hot": 0.1699, "two": 0.7968}),  # sn_n64_d256_f32
    (64, 32, 65536, 2, 2, False, {"one_hot": 0.1384, "two": 0.6667}),  # sn_n64_d32_bf16
    (64, 32, 65536, 4, 2, False, {"one_hot": 0.1387, "two": 0.6654}),  # sn_n64_d32_f32
    (64, 512, 65536, 2, 2, False, {"one_hot": 0.1726, "two": 1.0924}),  # sn_n64_d512_bf16
    (64, 512, 65536, 4, 2, False, {"one_hot": 0.2604, "two": 1.7175}),  # sn_n64_d512_f32
    (64, 64, 65536, 2, 2, False, {"one_hot": 0.1404, "two": 0.6544}),  # sn_n64_d64_bf16
    (64, 64, 65536, 4, 2, False, {"one_hot": 0.1397, "two": 0.6539}),  # sn_n64_d64_f32
]


# picks that lose more than 5 %, and what they lose at most: the one-hot cost model passes over float32 512 x 1024 at
# LNC=2 (0.128 against 0.104 ms), and 128 x 512 bf16 at LNC=2, where every option takes 0.08 to 0.09 ms
_KNOWN_MISSES = {
    (512, 1024, 2048, 4, 2, False): 1.23,
    (128, 512, 256, 2, 2, False): 1.17,
}


@pytest.mark.parametrize("n_rows, d_size, k_size, itemsize, lnc, unique, times", _MEASURED)
def test_plan_is_close_to_the_fastest_measured_option(n_rows, d_size, k_size, itemsize, lnc, unique, times):
    chosen = times[_option(n_rows, d_size, k_size, itemsize, lnc, unique)]
    best = min(times.values())
    # seven picks are not the fastest; five of them lose 0.5 % to 4.8 %
    limit = _KNOWN_MISSES.get((n_rows, d_size, k_size, itemsize, lnc, unique), 1.05)
    assert chosen <= limit * best, (chosen, times)


def test_plan_picks_the_fastest_option_on_all_but_seven_measurements():
    misses = [row[:6] for row in _MEASURED if row[6][_option(*row[:6])] > min(row[6].values())]
    assert len(misses) == 7, misses


def test_grouped_in_place_loop_has_its_own_cost():
    # up to _D_BLK columns per core the tiles go in groups whose gathers and scatters overlap; beyond, one at a time
    for itemsize in (2, 4):
        block_bytes = _D_BLK * itemsize
        assert _gather_scatter_extra_ns(_D_BLK, itemsize) == _interp_ns(block_bytes, _GROUP_GATHER_SCATTER_EXTRA_NS)
        assert _gather_scatter_extra_ns(_D_BLK + 1, itemsize) == _interp_ns(block_bytes, _GATHER_SCATTER_EXTRA_NS)
        assert _gather_scatter_extra_ns(64, itemsize) == _GROUP_GATHER_SCATTER_EXTRA_NS[0]
    # few tiles over 1,024 rows of 4,096 float32 columns at LNC=2 (2,048 per core): the output itself, not two buffers
    assert plan_scatter_add_dedup(1024, 4096, 2048, 4, 2) == (PATH_COLUMNS, 2048, 1, True)


@pytest.mark.parametrize("itemsize", [2, 4])
def test_one_hot_needs_one_psum_bank_per_accumulator(itemsize):
    # 128-row blocks x 512-column chunks must fit the PSUM banks, however many sources there are
    for n_rows, d_size in ((1024, 512), (512, 1024), (256, 2048), (128, 4096)):
        assert (n_rows + 127) // 128 * ((d_size + 511) // 512) == _PSUM_BANKS
        assert plan_scatter_add_dedup(n_rows + 1, d_size, 1 << 22, itemsize, 1)[0] != PATH_ONE_HOT
        assert plan_scatter_add_dedup(n_rows, d_size + 1, 1 << 22, itemsize, 1)[0] != PATH_ONE_HOT
    assert plan_scatter_add_dedup(64, 32, 1 << 20, itemsize, 2)[0] == PATH_ONE_HOT


def test_buffer_plan_extremes():
    # one tile over many rows: a second buffer cannot pay for zeroing and folding it
    assert plan_scatter_add_chains(1 << 20, 64, 4, 1, False, False) == (1, False)
    # many tiles over few rows: the overlap wins
    assert plan_scatter_add_chains(1024, 64, 4, 10000, False, False) == (2, False)
    assert plan_scatter_add_chains(1024, 512, 4, 10000, False, True) == (2, False)
    # few tiles over a large table with split columns: the output itself, with or without repeats
    assert plan_scatter_add_chains(1 << 17, 2048, 2, 64, False, True) == (1, True)
    assert plan_scatter_add_chains(1 << 17, 2048, 2, 64, True, True) == (1, True)
    # narrow rows never accumulate in place
    assert plan_scatter_add_chains(1 << 20, 64, 4, 1, True, False)[1] is False


def test_cost_tables_are_linear_between_measured_widths():
    for table in (_RMW_OVERLAP_NS, _GATHER_SCATTER_EXTRA_NS, _GROUP_GATHER_SCATTER_EXTRA_NS):
        assert _interp_ns(1, table) == table[0]
        assert _interp_ns(1 << 20, table) == table[-1]
        for row_bytes, ns in zip(_RMW_OVERLAP_BYTES, table, strict=True):
            assert _interp_ns(row_bytes, table) == ns
        # between two measured widths the cost stays between their costs (- 1: integer division rounds down)
        knots = list(zip(_RMW_OVERLAP_BYTES, table, strict=True))
        for (b0, n0), (b1, n1) in zip(knots, knots[1:], strict=False):
            assert all(min(n0, n1) - 1 <= _interp_ns(w, table) <= max(n0, n1) for w in range(b0, b1 + 1, 64))
    for table in (_RMW_OVERLAP_NS, _GATHER_SCATTER_EXTRA_NS):
        widths = range(64, 16384, 64)
        assert all(_interp_ns(a, table) <= _interp_ns(b, table) for a, b in zip(widths, widths[1:], strict=False))


def test_narrow_rows_go_in_place_only_for_few_tiles():
    # 1,024 sources into a 256 MB float32 table: copying the columns beats zeroing and folding buffers
    assert plan_scatter_add_dedup(1 << 20, 64, 1024, 4, 2) == (PATH_COLUMNS, 32, 1, True)
    # a million sources: per-tile cost decides, and the tiles path splits the tiles over the cores
    assert plan_scatter_add_dedup(1 << 20, 64, 1 << 20, 4, 2)[0] == PATH_TILES
