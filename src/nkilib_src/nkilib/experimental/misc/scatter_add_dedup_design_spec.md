# Scatter-Add Dedup Kernel Design Spec

## Overview

`scatter_add_dedup` computes `output = input.index_add(0, index, src)` for any index values and returns a new tensor.
`scatter_add` (same package) computes the same sums in place: per 128-index tile it gathers the tile's rows, adds the
sources on VectorE and scatters the rows back, and with `unique_indices=False` it first sums the tile's repeated
indices with an equality-matrix matmul, so that positions that share a row write the same value. Each tile then costs
a gather and a scatter, two indirect DMAs whose descriptors queue on GpSimd, and the next tile's gather waits for the
scatter. This kernel adds each tile with one DMA read-modify-write instead and lets consecutive tiles overlap in two
accumulation buffers. Splatting, BEV pooling, graph aggregation, histograms and MoE combines in token order all repeat
indices inside tiles.

One kernel covers every regime and picks its path per call from cost models measured on trn2
(`plan_scatter_add_dedup`):

- **Small tables**: one-hot matmuls on TensorE accumulate every tile in PSUM, with no DMA read-modify-writes.
- **Otherwise**: the repeats of each tile are removed on TensorE, then one DMA read-modify-write adds the tile into
  one or two accumulation buffers. With `unique_indices=True` the caller promises distinct indices within every
  tile, as with `scatter_add`'s default, and the kernel skips the deduplication.
- **Few tiles over a large table**: the one buffer is the output itself. With repeats the tiles then go in groups of
  up to four: the group's gathers, its sums on TensorE, its plain indirect scatters. That needs no scratch rows.

### Public interface

```python
def scatter_add_dedup(input, dim, index, src, unique_indices=False) -> output
```

| Tensor | Shape | Notes |
|--------|-------|-------|
| `input` | `[N, D]` | float32, bfloat16 or float16; not modified |
| `index` | `[K]` | int32, every value in `[0, N)` (not checked) |
| `src` | `[K, D]` | same dtype as `input` |
| `output` | `[N, D]` | `input` plus every `src[k]` added to row `index[k]` |

`dim` must be 0. `unique_indices=True` means that `index[128 * t : 128 * (t + 1)]` holds distinct values for every
`t`; repeats across tiles are fine. Repeats inside a tile then give wrong sums without an error, as with `scatter_add`.
`N` is at most `2^24 - 2048`, because indices are compared as float32. LNC=1 and LNC=2 are supported.

## Algorithm

Per 128-index tile, on the read-modify-write paths:

1. `E[k, m] = (index[k] == index[m])`: the index column is broadcast along the free dimension, transposed on TensorE
   and compared with the column on VectorE.
2. `E^T @ src` on TensorE gives every position the sum of all the sources of the tile that share its index (fp32
   PSUM, 512 columns per matmul, `is_stationary_onezero` since `E` holds only zeros and ones).
3. Only the first position of each index keeps its row. Later positions are sent to per-partition scratch rows
   `N + m`, so the 128 rows of the tile are distinct.
4. One `nisa.dma_compute` read-modify-write (`dst[rows] += summed`) adds the tile into an accumulation buffer, 2048
   columns per instruction.

With `unique_indices=True`, steps 1 to 3 are skipped: the source tile goes straight to step 4 with `index` as the rows.

When the one buffer is the output itself and indices repeat, steps 3 and 4 work on groups of G tiles: G gathers
(`old_b = out[index_b]`), then `new_b = old_b + sum_a E_ab^T @ src_a` over the group's tiles `a`, with
`E_ab[k, m] = (index_a[k] == index_b[m])`, then G plain indirect scatters `out[index_b] = new_b`. Positions that share a
row, in one tile or in different tiles of the group, read the same old value and carry the same sum, so their writes
are equal and may land in any order, and only consecutive groups are ordered. One tile at a time, each gather waited
for the previous tile's scatter. G is 4, or what fits 16 KB of SBUF per partition for one array of the group's
2048-column blocks (2 for float32 blocks of 2048 columns), and 1 when a core's columns span several blocks: those blocks
already overlap one another, and every extra tile of a group adds a pass of matmuls per block (with 4,096 float32
columns on one core, groups of 2 ran at 0.88x).

The compares and the matmul run on VectorE and TensorE while the DMA queues run the read-modify-writes, so a tile costs
about one read-modify-write whatever the repeats: 2.40, 2.40 and 2.37 ms for 131,072 rows of 64 fp32 at 0 %, 50 % and
95 % in-tile repeats on one core. Index and source tiles are loaded with static DMA descriptors (`dge_mode.none`), which
leaves GpSimd to generate the read-modify-write descriptors.

On the one-hot path each tile becomes `O[k, n] = (index[k] == n)` (one VectorE compare against an iota row) and
`O^T @ src` accumulates in PSUM: one matmul per 128-row block of the table and 512-column chunk, each accumulator in a
bank of its own. The path needs no deduplication, so `unique_indices` does not matter there. Tables of 768 rows or more
compare against a float16 iota row: row numbers stay below 1,024, which float16 holds exactly, and VectorE compares
16-bit data faster. There building the one-hot matrix bounds the loop, and bf16 tables of 768 and 1,024 rows run 1.15x
to 1.40x faster than with a float32 row. Below 768 rows the build and the matmuls take about as long, and float16 ran up
to 8 % slower.

## Work split and paths

`plan_scatter_add_dedup(n_rows, d_size, k_size, itemsize, num_shards, unique_indices)` returns
`(path, columns_per_core, chains, in_place)`.

- **`PATH_ONE_HOT`** when the accumulators fit the 8 PSUM banks and `_one_hot_wins` expects the matmuls to beat the
  read-modify-writes. At LNC=2 each core takes half of the tiles, stores its sums in its own float32 buffer, and the
  buffers are folded after a core barrier.
- **`PATH_COLUMNS`** for wide rows (`D * itemsize >= 2048` bytes and `D` divisible by the number of cores): each core
  owns a column range and walks every tile. The buffers hold the input dtype, as `scatter_add` accumulates. Narrow rows
  take this path too when `_narrow_in_place_wins` prefers accumulating in the output itself (few tiles over a large
  table).
- **`PATH_TILES`** for the other narrow rows: each core walks its share of the tiles into its own float32 buffers. At
  LNC=2 the cores meet at one core barrier per buffer, then each core folds half of the rows. Rows under 64 bytes space
  their scratch rows 64 bytes apart, because read-modify-writes to rows that share a 64-byte line serialize (a D = 1
  histogram ran 3.5x slower at 80 % repeats than at 6 % before the spacing).

### One-hot cost model (`_one_hot_wins`)

Fitted on 96 small tables (64 to 1,024 rows, 1 to 512 columns, fp32 and bf16, 65,536 random sources):

```
one-hot  = tiles per core * blocks * chunks * matmul (+ 15 us at LNC=2: store, barrier, fold)
           matmul: 16-bit 80 ns + 0.25 ns per column; float32 1.7 ns per column from 128 columns, 420 ns below
read-modify-write = its tiles per core * (2.2 us + 4 ns per column beyond 128)
```

Narrow float32 matmuls do not reach the per-column rate: at 1,024 rows, 1 to 64 columns cost 3.4 us per tile and 128
columns 1.7 us. On the 65,536-source grid, float32 tables of 1,024 rows therefore stay on the read-modify-writes
except at 128 columns, while 16-bit tables up to 1,024 rows and every table up to 512 rows take the one-hot path. With
few tiles at LNC=2 the 15 us of the fold keep the read-modify-writes.

### Accumulation buffers (`plan_scatter_add_chains`)

Read-modify-writes into one buffer form a dependent chain. On 128 rows, one read-modify-write takes 3.75, 5.38 and
9.10 us at 256 B, 2 KB and 8 KB rows when every instruction targets the same buffer, and 2.16, 2.72 and 5.41 us when
consecutive instructions alternate between two buffers (four buffers: 2.14 us at 256 B, so two is the most the kernel
uses). A gather, an add and a plain scatter take 6.01, 7.82 and 13.49 us one tile at a time. In groups of tiles (at
most 2048 columns per core) their gathers and scatters overlap: 5.44, 6.76 and 13.08 us per tile at LNC=2, against
6.17, 8.30 and 15.87 us one at a time in the same runs. Costs, against one private buffer (copy in,
read-modify-writes, copy out), in passes over the core's `N x columns_per_core` slice at 3 ps per byte plus 2 us each:

```
two buffers  = 2 passes - tiles per core * overlap(row bytes)        (zeroing, one more fold read)
in place     = -2 passes                                             (unique_indices: read-modify-writes)
             = -2 passes + tiles per core * gather_scatter(row bytes) (repeats: gather, add, scatter)
pick the cheapest of 0, two buffers and in place
```

`overlap()` (1.59, 2.66, 3.69 us) and `gather_scatter()` (over one read-modify-write: 2.26, 2.44, 4.39 us one tile at
a time, 2.5, 0.3, 0.8 us in groups, fitted on the measured calls of the unit tests) are linear between 256 B, 2 KB
and 8 KB, taken at the row bytes of one 2048-column block: the blocks of one tile already overlap one another, so
only consecutive tiles count. In place applies to split columns only; narrow rows compare the float32 buffers of
`PATH_TILES` with the in-place columns plan in `_narrow_in_place_wins`, by the same costs.

| Plan | Buffers per core | Passes over the slice |
|------|------------------|-----------------------|
| two buffers | input copy, zeroed buffer, fold of both | 6 |
| one buffer, deduplicating | private copy with scratch rows, copied out | 4 |
| one buffer, in place | the output's own columns | 2 |

The in-place plans need a core barrier at LNC=2 between the copy of the input and the first update. Without it,
core 1's first read-modify-writes read its columns of the shared output before the copy had landed: every row of
core 1's columns was wrong on a 256 x 2048 probe, while core 0 and LNC=1 were exact. Because of that barrier each core
can copy whole rows (a contiguous range, in descriptors of 2,048 elements) instead of its own columns. For 64 float32
columns at LNC=2 a core's columns are 128-byte segments of every row: profiled on 1,048,576 x 64, that copy moved 94
GB/s per core with the DMA engines idle half of the time, and took 1.45 of the call's 1.83 ms.

Against 184 calls run with every option forced (the unit tests carry them), the plan picks the fastest option on 177,
and the seven misses lose 0.087 ms in total. Five of them lose 0.5 to 4.8 %; the other two are the one-hot cost model
on fp32 512 x 1024 at LNC=2 (0.128 against 0.104 ms) and 128 x 512 bf16 at LNC=2, where every option takes 0.08 to
0.09 ms. The knots of the grouped `gather_scatter()` sit inside the range of the knot sets with the fewest misses.

## Choosing among scatter-add implementations

All exact alternatives were measured on trn2 (ms per call, 20 back-to-back calls, best of 3; see Performance).

| Alternative | Exact for | Measured against this kernel |
|-------------|-----------|------------------------------|
| `scatter_add`, `unique_indices=False` (gather, equality-matrix sums, add, scatter, in place) | any indices | many tiles: 1.42x to 3.36x slower at LNC=1 and 1.52x to 6.16x at LNC=2, where both of its cores walk every tile, each over half of the columns; few tiles over a large table: 1.04x to 1.27x slower; small tables 1.8x to 45x slower, bf16 small tables with errors of 2e-2 to 3e-2 of the largest value, and `D` not divisible by the number of cores is rejected at LNC=2 |
| `scatter_add`, `unique_indices=True` (its default) | distinct indices per tile only | on its ten test shapes this kernel with `unique_indices=True` is 1.00x to 2.42x faster at LNC=1 and 1.00x to 1.86x at LNC=2; the 128 x 512 shapes sit at the launch floor of about 0.08 ms for both |
| the compiler's `torch.index_add` | any indices | 11x to 56x slower on colliding workloads at LNC=2; token-embedding backward with Zipf token ids, Llama-3 (128,256 x 4,096, 8k tokens) 7.47 against 3.87 ms in bf16, GPT-2 (50,257 x 768, 16k tokens) 12.02 against 1.31 ms in fp32 |
| a one-hot matrix and a matmul in PyTorch | any indices | small tables (64 to 1,024 rows): 1.52x to 13.7x slower at LNC=1 and 1.32x to 11.1x at LNC=2 |
| one copy of the target per lane (the `ms_deformable_attention_bwd` scheme) | any indices | BEV pooling 10.79 against 4.23 ms (LNC=1); SimpleBEV lift backward 58.6 against 17.6 ms, max-pool backward 248.3 against 85.7 ms (LNC=2); 21 GiB of copies for a 256-wide graph layer on ogbn-arxiv |
| `nisa.dma_compute(unique_indices=False)` | would be any indices | rejected by nki 0.6.0 at trace time with `vector_offset` |

Use `unique_indices=True` where the indices of every tile are distinct (for example one MoE expert block of token ids
per call) and the default everywhere else. `scatter_add` stays the choice when the input must be updated in place.

## Numerics

Narrow rows on `PATH_TILES` sum in float32 and round once, and so do small tables (fp32 PSUM). Wide rows add each tile's
summed repeats into the input dtype, as `scatter_add` adds each tile: the sums leave PSUM in the input dtype and the
read-modify-write rounds again, so a bf16 tile is rounded twice where `scatter_add(unique_indices=False)` rounds once.
Rows accumulated in the output itself round once per group of tiles. Against float64, as a fraction of the largest
value:

| Call | `scatter_add` | this kernel |
|------|--------------:|------------:|
| 256 x 32 bf16 table, 65,536 sources | 3.2e-2 | 2.0e-3 |
| 1024 x 128 bf16 table, 65,536 sources | 2.1e-2 | 3.3e-3 |
| GPT-2 embedding backward, bf16, 16k Zipf tokens | 2.3e-2 | 9.9e-3 |
| Llama-3 embedding backward, bf16, 8k Zipf tokens, LNC=1 / LNC=2 | 1.6e-2 / 1.6e-2 | 1.6e-2 / 8.9e-3 |
| MoE 4096 x 2880 bf16, distinct tiles (`unique_indices=True` for both) | 1.1e-2 | 7.3e-3 |
| MoE 4096 x 2880 bf16, token order | 2.7e-3 | 5.4e-3 |

float32 calls stay within 1.4e-6 for both. `torch.index_add` reaches 4.2e-2 on the Llama-3 call.

## Performance

trn2, ms per call of a standalone NEFF (20 back-to-back calls, best of 3), with `NEURON_RT_IO_RING_CACHE_SIZE=64`
(see Runtime setting below). Pairs are LNC=1 / LNC=2. `scatter_add` is the faster of a call in place and a call on a
copy of the input; `index_add` is the compiler's lowering of `torch.index_add`, the faster of its LNC=1 and LNC=2
times.

Many tiles with repeated indices (N x D, sources, in-tile repeats), against `scatter_add(unique_indices=False)`:

| Workload | index_add | scatter_add | this kernel | gain |
|---|---:|---:|---:|---:|
| BEV pooling 16384 x 80, 235k, 61 % | 125.3 | 11.8 / 11.9 | 4.24 / 2.23 | 2.78x / 5.33x |
| forward splat 107520 x 4, 430k, 47 % | 53.0 | 26.2 / 27.3 | 7.81 / 4.43 | 3.36x / 6.16x |
| graph 65536 x 64, 524k, 5 % | 94.2 | 25.7 / 25.9 | 9.36 / 4.92 | 2.75x / 5.26x |
| graph by target 16384 x 128, 262k, 94 % | 55.8 | 13.0 / 12.9 | 4.99 / 2.62 | 2.60x / 4.94x |
| graph 16384 x 512, 131k, 9 % | 54.0 | 8.36 / 7.36 | 4.88 / 3.71 | 1.71x / 1.98x |
| MoE in token order 4096 x 2880 bf16, 16k, 75 % | 10.4 | 2.15 / 1.41 | 1.51 / 0.928 | 1.42x / 1.52x |
| 16384 x 64, 131k, 50 % | 19.0 | 6.51 / 6.71 | 2.39 / 1.32 | 2.72x / 5.07x |

The 16384 x 64 calls take 2.39, 2.39 and 2.38 ms at 0, 50 and 95 % in-tile repeats on one core (`scatter_add`: 6.52,
6.51 and 6.59 ms).

Few tiles over large tables, against `scatter_add(unique_indices=False)`:

| Workload | index_add | scatter_add | this kernel | gain |
|---|---:|---:|---:|---:|
| Llama-3 embedding bwd 128256 x 4096 bf16, 8k Zipf | 7.47 | 4.73 / 4.04 | 4.35 / 3.87 | 1.09x / 1.04x |
| GPT-2 embedding bwd 50257 x 768 bf16, 16k Zipf | 11.31 | 1.28 / 1.16 | 1.05 / 0.994 | 1.22x / 1.17x |
| GPT-2 embedding bwd 50257 x 768 fp32, 16k Zipf | 12.02 | 1.72 / 1.51 | 1.47 / 1.31 | 1.17x / 1.16x |
| 32768 x 4096 fp32, 16k random | 10.79 | 5.67 / 3.73 | 4.45 / 3.39 | 1.27x / 1.10x |
| 1048576 x 64 fp32, 8k random | 2.04 | 1.29 / 1.31 | 1.21 / 1.21 | 1.06x / 1.08x |
| 262144 x 128 fp32, 4k random | 1.31 | 0.640 / 0.660 | 0.614 / 0.617 | 1.04x / 1.07x |
| 100000 x 64 fp32, 1k random | 0.260 | 0.157 / 0.163 | 0.145 / 0.152 | 1.09x / 1.07x |
| 100000 x 64 bf16, 1k random | 0.205 | 0.118 / 0.124 | 0.106 / 0.114 | 1.11x / 1.09x |

Small tables, 65,536 random sources; the gain is over the faster of the two alternatives. `scatter_add` rejects
`D = 1` at LNC=2 (`D` must be divisible by the number of cores):

| Workload | scatter_add | one-hot matmul in PyTorch | this kernel | gain |
|---|---:|---:|---:|---:|
| 64 x 1 fp32 | 9.78 / rejected | 0.370 / 0.208 | 0.217 / 0.158 | 1.71x / 1.32x |
| 64 x 512 fp32 | 4.15 / 3.65 | 0.941 / 0.480 | 0.481 / 0.262 | 1.96x / 1.83x |
| 256 x 1 bf16 | 6.08 / rejected | 1.07 / 0.593 | 0.237 / 0.148 | 4.51x / 4.00x |
| 256 x 32 bf16 | 3.25 / 3.31 | 1.14 / 0.618 | 0.225 / 0.144 | 5.07x / 4.28x |
| 256 x 512 fp32 | 4.14 / 3.63 | 1.41 / 0.776 | 0.929 / 0.487 | 1.52x / 1.59x |
| 1024 x 128 bf16 | 3.33 / 3.34 | 4.60 / 2.37 | 0.335 / 0.212 | 10.0x / 11.1x |
| 1024 x 512 fp32 | 4.17 / 3.63 | 7.03 / 3.63 | 2.31 / 1.75 | 1.81x / 2.08x |

Distinct indices per tile (`scatter_add`'s test shapes, tokens x hidden, top-k), this kernel with
`unique_indices=True` against `scatter_add`'s default. The 128 x 512 shapes sit at the launch floor of about 0.08 ms:

| Workload | scatter_add | this kernel | gain |
|---|---:|---:|---:|
| 128 x 512, 4, fp32 | 0.078 / 0.091 | 0.078 / 0.089 | 1.00x / 1.03x |
| 128 x 512, 2, bf16 | 0.079 / 0.089 | 0.078 / 0.089 | 1.02x / 1.00x |
| 256 x 2048, 8, bf16 | 0.194 / 0.171 | 0.080 / 0.098 | 2.42x / 1.73x |
| 256 x 2048, 8, fp32 | 0.260 / 0.209 | 0.146 / 0.112 | 1.79x / 1.86x |
| 512 x 1024, 4, bf16 | 0.161 / 0.146 | 0.091 / 0.091 | 1.77x / 1.61x |
| 512 x 1024, 4, fp32 | 0.198 / 0.168 | 0.123 / 0.110 | 1.61x / 1.53x |
| 1024 x 4096, 2, bf16 | 0.352 / 0.228 | 0.238 / 0.185 | 1.48x / 1.23x |
| 1024 x 4096, 2, fp32 | 0.506 / 0.323 | 0.363 / 0.273 | 1.39x / 1.18x |
| 4096 x 2880, 4, bf16 | 2.07 / 1.31 | 1.29 / 0.790 | 1.61x / 1.66x |
| 4096 x 2880, 4, fp32 | 2.91 / 1.79 | 1.92 / 1.37 | 1.51x / 1.30x |

### Runtime setting

The index and source tiles use static DMA descriptors (`dge_mode.none`), which the runtime prepares before an
execution. For a standalone call, whose `index`, `src`, `input` and output are inputs and outputs of the NEFF, the
times therefore depend on `NEURON_RT_IO_RING_CACHE_SIZE`. With the default runtime settings, against the fastest
alternative measured with the same settings (`scatter_add` in place, or the one-hot matmul in PyTorch on small
tables), LNC=1 / LNC=2:

| Call | fastest alternative | this kernel, default | this kernel, cache 64 |
|---|---:|---:|---:|
| BEV pooling 16384 x 80, 235k | 11.8 / 11.8 | 4.39 / 3.29 | 4.24 / 2.23 |
| graph by target 16384 x 128, 262k | 13.0 / 13.0 | 5.15 / 3.55 | 4.99 / 2.62 |
| 16384 x 64, 131k, 50 % | 6.52 / 6.59 | 2.48 / 2.11 | 2.39 / 1.32 |
| MoE in token order 4096 x 2880 bf16, 16k | 2.62 / 1.42 | 1.56 / 1.33 | 1.51 / 0.928 |
| GPT-2 embedding bwd bf16, 16k Zipf | 1.28 / 1.16 | 1.08 / 1.16 | 1.05 / 0.994 |
| 100000 x 64 bf16, 1k random | 0.119 / 0.124 | 0.168 / 0.209 | 0.106 / 0.114 |
| 256 x 32 bf16 table, 65,536 sources | 1.14 / 0.618 | 0.223 / 0.635 | 0.225 / 0.144 |
| 1024 x 128 bf16 table, 65,536 sources | 3.34 / 2.38 | 0.335 / 1.34 | 0.335 / 0.212 |
| distinct tiles 512 x 1024, 4, fp32 | 0.199 / 0.168 | 0.246 / 0.789 | 0.123 / 0.110 |
| distinct tiles 4096 x 2880, 4, bf16 | 2.41 / 1.31 | 1.32 / 1.51 | 1.29 / 0.790 |

`scatter_add` in place times the same with and without the cache. Loading the tiles with HWDGE (`dge_mode.hwdge`)
instead makes the read-modify-write paths at LNC=2 independent of the setting (BEV pooling 2.35 ms with the default
settings, graph by target 2.89 ms), but with the cache it runs up to 13 % slower at LNC=1 and 1.1x to 4.4x slower on
small tables, so the kernel keeps static descriptors.

## Testing

- `test/unit/nkilib/experimental/misc/test_scatter_add_dedup_plan.py` (203 tests) checks the work split, the plan
  against the 184 measured calls (every pick within 5 % of the fastest option but two listed misses, and seven misses
  in all), the PSUM limit of the one-hot path, the cost tables and the plan's extremes on the host.
- `test/integration/nkilib/experimental/misc/test_scatter_add_dedup.py` runs 67 cases (23 at LNC=1, 44 at LNC=2)
  against an fp32 `index_add` reference: narrow and wide rows, fp32 and bf16, K not a multiple of 128, every index
  equal, fewer rows than a tile, token order, sorted runs, distinct tiles, `unique_indices=True`, every path and buffer
  plan, and the in-place groups of 1, 2 and 4 tiles with repeats across a group's tiles, short last groups and partial
  last tiles.
