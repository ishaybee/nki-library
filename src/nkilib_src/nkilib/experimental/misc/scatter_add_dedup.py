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
"""
Scatter-add kernel built on DMA read-modify-writes: ``output = input.index_add(0, index, src)`` for any index values.

``scatter_add`` (same package) adds each 128-index tile in place with an indirect gather, an add and an indirect
scatter; with ``unique_indices=False`` it first sums the repeated indices of the tile with an equality-matrix matmul,
so that positions sharing a row write the same value. This kernel returns a new tensor and adds each tile with one
``nisa.dma_compute`` read-modify-write in place of the gather and the scatter, which lets consecutive tiles overlap
in two accumulation buffers. Splatting, BEV pooling, graph aggregation, histograms and token-major MoE combines all
repeat indices inside a tile.

Per 128-index tile:
    1. Equality matrix E[k, m] = (index[k] == index[m]) from the index column and its TensorE transpose.
    2. ``E^T @ src`` on TensorE sums, onto every position, all the sources of the tile that share its index.
    3. Only the first position of each index keeps its row; every later position is redirected to a per-partition
       scratch row (N + position), so the 128 rows of the tile are distinct.
    4. One ``nisa.dma_compute`` read-modify-write adds the summed rows into the accumulation buffer.
Read-modify-writes to one buffer form a dependent chain, so tiles are dealt round robin to two independent buffers
that are summed with the input at the end, unless ``plan_scatter_add_chains`` finds the zeroing and the fold of the
second buffer dearer than the overlap it buys (few tiles over many rows). Index and source tiles are loaded with
static DMA (no GpSimd descriptor generation), which leaves the GpSimd queue to the read-modify-writes.

With ``unique_indices=True`` the caller guarantees that the indices of every tile are distinct, as with
``scatter_add``'s default, and steps 1 to 3 are skipped: each source tile goes straight to its read-modify-write.
Without repeats no scratch rows are needed, so a single buffer can be the output itself.

``plan_scatter_add_dedup`` also picks two other paths:
    - Small tables (one PSUM bank per 128-row block and 512-column chunk): no read-modify-writes at all. Each tile
      becomes a one-hot matrix O[k, n] = (index[k] == n) and O^T @ src accumulates in PSUM on TensorE.
    - Few tiles over a large table: the one buffer is the output itself, also with repeats. Tiles are then taken
      in groups of up to four: the group's gathers, the sums over the whole group on TensorE, the group's plain
      indirect scatters. Positions that share a row read the same old value and carry the same sum, so their
      writes are equal and need no scratch rows, and the transfers of a group overlap.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_program_sharding_info

_PT = 128  # nl.tile_size.pmax: sources per tile
_D_BLK = 2048  # columns per read-modify-write (the d tile of scatter_add)
_MM_TILE = 512  # columns per matmul: one fp32 PSUM bank
_CHAINS = 2  # most independent accumulation buffers per core (four ran at 2.14 against 2.16 us per read-modify-write)
_WIDE_ROW_BYTES = 2048  # rows at least this wide are sharded over columns at LNC=2, narrower ones over index tiles
_MIN_ROW_BYTES = 64  # scratch rows of the float32 slabs start at least this many bytes apart (see _scratch_pitch)
_MAX_ROWS = (1 << 24) - _PT * (_MIN_ROW_BYTES // 4)  # indices are compared as fp32; the scratch rows must stay exact
_MAX_FOLD_SOURCES = 16  # nisa.dma_compute takes at most 16 sources
# plan_scatter_add_chains, measured on trn2 at three row widths, per 128-row instruction: what a second buffer saves on
# a read-modify-write (one buffer 3.75 / 5.38 / 9.10 us, two 2.16 / 2.72 / 5.41 us), what a gather, add and plain
# scatter costs over one read-modify-write (6.01 / 7.82 / 13.49 us), and the cost of one streaming pass per byte
_RMW_OVERLAP_BYTES = (256, 2048, 8192)
_RMW_OVERLAP_NS = (1590, 2660, 3690)
_GATHER_SCATTER_EXTRA_NS = (2260, 2440, 4390)
_RMW_ONE_BUFFER_NS = (3750, 5380, 9100)
# the same over-cost when the tiles go in groups (_group_size > 1: up to 2048 columns per core), whose gathers and
# scatters overlap; fitted on the measured calls of the unit tests. Per tile at LNC=2 the groups take 5.44 / 6.76 /
# 13.08 us at 256 B / 2 KB / 8 KB, one tile at a time 6.17 / 8.30 / 15.87 us
_GROUP_GATHER_SCATTER_EXTRA_NS = (2500, 300, 800)
_GROUP_SBUF_BYTES = 16384  # in place with repeats: SBUF bytes per partition for one tile array of a group of tiles
_MAX_GROUP = 4  # tiles per group; 2, 3 and 4 were measured, and with 8 KB rows 4 ran slower than 2
_PASS_PS_PER_BYTE = 3
_PASS_FIXED_NS = 2000  # and its fixed cost (DMA latency)
# plan_scatter_add_dedup's work paths
PATH_ONE_HOT = 0  # small tables: one-hot matmuls on TensorE, no read-modify-writes
PATH_COLUMNS = 1  # wide rows: each core owns a column range
PATH_TILES = 2  # narrow rows: each core walks a share of the index tiles into float32 buffers
_PSUM_BANKS = 8  # one-hot accumulators, one 128-row x 512-column bank each
# _one_hot_wins, fitted on trn2 (see the function): ns per one-hot matmul of a 128-row block, and per deduplicating
# read-modify-write tile with two buffers
_ONE_HOT_FP32_NARROW_NS = 420  # float32, fewer than 128 columns
_ONE_HOT_FP32_PS_PER_COL = 1700  # float32, 128 columns or more
_ONE_HOT_16BIT_NS = 80
_ONE_HOT_16BIT_PS_PER_COL = 250
_RMW_TILE_NS = 2200
_RMW_TILE_PS_PER_COL = 4000  # beyond 128 columns
_ONE_HOT_LNC2_NS = 15000  # LNC=2: storing the sums, the core barrier and the fold
_ONE_HOT_FP16_IOTA_ROWS = 768  # one-hot tables of at least this many rows compare against a float16 iota row


def plan_scatter_add_dedup(
    n_rows: int, d_size: int, k_size: int, itemsize: int, num_shards: int, unique_indices: bool = False
) -> tuple[int, int, int, bool]:
    """
    Choose the work path, how the work is split across the cores of a logical NeuronCore, and the buffers per core.

    Args:
        n_rows (int): N, rows of ``input``.
        d_size (int): D, columns of ``input``.
        k_size (int): K, source rows.
        itemsize (int): bytes per element of ``input``.
        num_shards (int): number of cores (1, or 2 at LNC=2).
        unique_indices (bool): the kernel's ``unique_indices``.

    Returns:
        tuple[int, int, int, bool]: ``(path, columns_per_core, chains, in_place)``.
            ``PATH_ONE_HOT`` for small tables where one-hot matmuls beat the read-modify-writes (_one_hot_wins).
            ``PATH_COLUMNS`` for wide rows (``D * itemsize >= 2048`` bytes and ``D`` divisible by ``num_shards``):
            each core owns a column range and walks every index tile. ``PATH_TILES`` otherwise: each core walks its
            share of the index tiles into its own fp32 buffers, summed after a core barrier. ``chains`` (1 or 2) and
            ``in_place`` (the one buffer is the output itself) come from plan_scatter_add_chains.
    """
    num_tiles = div_ceil(k_size, _PT)
    tiles_per_core = div_ceil(num_tiles, num_shards)
    columns = d_size * itemsize >= _WIDE_ROW_BYTES and d_size % num_shards == 0
    if _one_hot_wins(n_rows, d_size, itemsize, num_shards, num_tiles, columns):
        return PATH_ONE_HOT, d_size, 1, False
    if columns:
        d_local = d_size // num_shards
        chains, in_place = plan_scatter_add_chains(n_rows, d_local, itemsize, num_tiles, unique_indices, True)
        return PATH_COLUMNS, d_local, chains, in_place
    chains, in_place = plan_scatter_add_chains(n_rows, d_size, 4, tiles_per_core, unique_indices, False)
    if d_size % num_shards == 0:
        if _narrow_in_place_wins(n_rows, d_size, itemsize, num_tiles, num_shards, unique_indices, chains):
            return PATH_COLUMNS, d_size // num_shards, 1, True
    return PATH_TILES, d_size, chains, in_place


def plan_scatter_add_chains(
    n_rows: int, d_local: int, itemsize: int, tiles_per_core: int, unique_indices: bool, columns: bool
) -> tuple[int, bool]:
    """
    Choose one or two accumulation buffers per core, and whether the one buffer is the output itself.

    Tiles alternate between two buffers so that consecutive read-modify-writes overlap; the second buffer costs two
    streaming passes over this core's N x d_local slice (zeroing it, and one more read in the fold). With column-split
    rows the one buffer can be the output's own columns, which saves the copy in and out of a private buffer: with
    ``unique_indices`` at no other cost, and with repeats (no scratch rows to send them to) at the price of a gather,
    an add and a plain scatter per tile instead of one read-modify-write (_gather_scatter_extra_ns).

    Args:
        n_rows (int): N, rows of ``input``.
        d_local (int): columns this core accumulates.
        itemsize (int): bytes per element of the accumulation buffers.
        tiles_per_core (int): 128-index tiles this core adds.
        unique_indices (bool): the kernel's ``unique_indices``.
        columns (bool): the rows are split over columns (PATH_COLUMNS), so a buffer can be the output itself.

    Returns:
        tuple[int, bool]: ``(chains, in_place)``, the cheapest of the plans by the cost model.
    """
    row_bytes = min(d_local, _D_BLK) * itemsize
    pass_ps = n_rows * d_local * itemsize * _PASS_PS_PER_BYTE + 1000 * _PASS_FIXED_NS
    # costs against one private buffer (copy in, read-modify-writes, copy out); the column blocks of one tile already
    # overlap one another, so a second buffer only overlaps consecutive tiles
    two_buffers = 2 * pass_ps - 1000 * tiles_per_core * _interp_ns(row_bytes, _RMW_OVERLAP_NS)
    if not columns:
        if two_buffers < 0:
            return 2, False
        return 1, False
    in_place = -2 * pass_ps
    if not unique_indices:
        in_place = in_place + 1000 * tiles_per_core * _gather_scatter_extra_ns(d_local, itemsize)
    best = min(0, two_buffers, in_place)
    if best == two_buffers:
        return 2, False
    if best == in_place:
        return 1, True
    return 1, False


def _narrow_in_place_wins(n_rows, d_size, itemsize, num_tiles, num_shards, unique_indices, chains):
    """
    Narrow rows: True when splitting the columns and accumulating in the output itself beats the float32 buffers of
    PATH_TILES. A core of the tiles path zeroes its chains buffers and folds (2 * chains passes over N x D float32,
    plus reading the input and writing the output for its rows) and adds its share of the tiles. In place, a core
    copies its columns of the input into the output but adds every tile, with a gather, an add and a scatter when
    indices repeat. Few tiles over a large table favour in place. Costs are scaled by num_shards to stay in integers.
    """
    table_ps = n_rows * d_size * _PASS_PS_PER_BYTE
    tiles_per_core = div_ceil(num_tiles, num_shards)
    tile_ps = 1000 * _interp_ns(d_size * 4, _RMW_ONE_BUFFER_NS)
    if chains == 2:
        tile_ps = tile_ps - 1000 * _interp_ns(d_size * 4, _RMW_OVERLAP_NS)
    tiles_cost = (2 * chains * num_shards * 4 + 2 * itemsize) * table_ps + num_shards * tiles_per_core * tile_ps
    local_bytes = d_size // num_shards * itemsize
    in_place_tile_ps = 1000 * _interp_ns(local_bytes, _RMW_ONE_BUFFER_NS)
    if not unique_indices:
        in_place_tile_ps = in_place_tile_ps + 1000 * _gather_scatter_extra_ns(d_size // num_shards, itemsize)
    in_place_cost = 2 * itemsize * table_ps + num_shards * num_tiles * in_place_tile_ps
    return in_place_cost < tiles_cost


def _interp_ns(row_bytes, table):
    """table's ns at row_bytes, linear between the measured row widths of _RMW_OVERLAP_BYTES."""
    b0, b1, b2 = _RMW_OVERLAP_BYTES
    n0, n1, n2 = table
    if row_bytes <= b0:
        return n0
    if row_bytes <= b1:
        return n0 + (n1 - n0) * (row_bytes - b0) // (b1 - b0)
    if row_bytes <= b2:
        return n1 + (n2 - n1) * (row_bytes - b1) // (b2 - b1)
    return n2


def _gather_scatter_extra_ns(num_cols, itemsize):
    """
    What one tile of the in-place loop with repeats costs over a read-modify-write, for num_cols columns per core: the
    grouped cost when its tiles go in groups (_group_size), the one-tile cost when a tile spans several column blocks.
    """
    table = _GATHER_SCATTER_EXTRA_NS
    if _group_size(num_cols, itemsize) > 1:
        table = _GROUP_GATHER_SCATTER_EXTRA_NS
    return _interp_ns(min(num_cols, _D_BLK) * itemsize, table)


def _one_hot_wins(n_rows, d_size, itemsize, num_shards, num_tiles, columns):
    """
    True when one-hot matmuls on TensorE are expected to beat the read-modify-writes for this call: the accumulators
    fit the PSUM banks (one per 128-row block and 512-column chunk), and the matmuls of a core's tiles take less time
    than its read-modify-write tiles.

    Per tile, the one-hot path runs one matmul per block and chunk. Fitted on 96 small tables (64 to 1,024 rows, 1 to
    512 columns, fp32 and bf16, 65,536 random sources): a 16-bit matmul costs 80 ns + 0.25 ns per column; a float32
    one costs 1.7 ns per column from 128 columns and 420 ns below that (the narrow float32 matmuls do not reach the
    per-column rate). A deduplicating read-modify-write tile with two buffers costs 2.2 us up to 128 columns and
    4 ns per column beyond. The one-hot path splits the tiles over the cores and at LNC=2 pays about 15 us to store
    its sums, meet at a core barrier and fold; the read-modify-write path splits the tiles for narrow rows and the
    columns for wide ones.
    """
    n_blocks = div_ceil(n_rows, _PT)
    n_chunks = div_ceil(d_size, _MM_TILE)
    if n_blocks * n_chunks > _PSUM_BANKS:
        return False
    cw = min(d_size, _MM_TILE)
    block_ps = 1000 * _ONE_HOT_16BIT_NS + cw * _ONE_HOT_16BIT_PS_PER_COL
    if itemsize == 4:
        block_ps = cw * _ONE_HOT_FP32_PS_PER_COL
        if cw < _PT:
            block_ps = 1000 * _ONE_HOT_FP32_NARROW_NS
    one_hot_ps = div_ceil(num_tiles, num_shards) * n_blocks * n_chunks * block_ps
    if num_shards > 1:
        one_hot_ps = one_hot_ps + 1000 * _ONE_HOT_LNC2_NS
    rmw_cols = d_size
    rmw_tiles = div_ceil(num_tiles, num_shards)
    if columns:
        rmw_cols = d_size // num_shards
        rmw_tiles = num_tiles
    rmw_ps = rmw_tiles * (1000 * _RMW_TILE_NS + max(0, rmw_cols - _PT) * _RMW_TILE_PS_PER_COL)
    return one_hot_ps < rmw_ps


@nki.jit
def scatter_add_dedup(
    input: nl.NkiTensor, dim: int, index: nl.NkiTensor, src: nl.NkiTensor, unique_indices: bool = False
) -> nl.NkiTensor:
    """
    Scatter-add ``src`` into a copy of ``input`` for arbitrary (repeated) indices.

    Equivalent to PyTorch's ``input.index_add(0, index, src)``: ``output[index[k], :] = input[index[k], :] + src[k, :]``
    summed over every k, including all the k that share an index, in any positions.

    Dimensions:
        N: Number of rows in input tensor
        D: Feature dimension size
        K: Number of source rows / indices

    Args:
        input (nl.NkiTensor): [N, D], Destination values to accumulate into (not modified), float32, bfloat16 or
            float16.
        dim (int): Dimension along which to scatter (must be 0).
        index (nl.NkiTensor): [K], 1D int32 row indices into input, all in [0, N).
        src (nl.NkiTensor): [K, D], Source values, same dtype as input.
        unique_indices (bool): The caller guarantees that the indices of every 128-index tile,
            ``index[128 * t : 128 * (t + 1)]``, are distinct, as with scatter_add's ``unique_indices=True``
            (repeats across tiles are fine). The per-tile deduplication is then skipped. Repeats inside a tile give
            wrong sums, without an error, when this is set. Default False.

    Returns:
        output (nl.NkiTensor): [N, D], A new tensor holding input plus the scattered sources.

    Notes:
        - Any index pattern is exact (unless unique_indices is set): repeats inside a 128-index tile, across tiles, or
          all indices equal.
        - Indices must lie in [0, N); they are not checked.
        - N must be at most 2^24 - 2048, because indices are compared as float32.
        - Narrow rows (under 2 KB) accumulate in float32; wide rows, and rows accumulated in the output itself,
          accumulate in the input dtype as scatter_add does, with the summed duplicates of a tile (or of a group of
          tiles) rounded to that dtype once. Small tables sum in fp32 PSUM and round once.
        - LNC=2: wide rows are split over columns, narrow rows and small tables over index tiles (each core sums into
          its own float32 buffers, then the cores reduce half of the rows each after a core barrier).
        - The path and the buffers come from cost models measured on trn2 (plan_scatter_add_dedup): one-hot matmuls
          for small tables, one or two accumulation buffers per core otherwise, and for few tiles over a large table
          the output itself as the one buffer (no zeroing and no fold).
        - Static DMA descriptors are used for the index and source loads, so the time of a standalone call (index
          and src are inputs of the NEFF) depends on NEURON_RT_IO_RING_CACHE_SIZE, mostly at LNC=2; the design spec
          has the measurements.
        - Cost is about one read-modify-write per 128 sources (2.3 us on one core for rows up to 256 bytes), whatever
          the number of repeats.

    Pseudocode (read-modify-write paths; small tables use one-hot matmuls instead):
        for tile in tiles(K):                      # 128 indices, tiles dealt to 1 or 2 buffers
            if unique_indices:
                summed, row = src[tile], index[tile]
            else:
                E = (index[tile] == index[tile]^T)
                summed = E^T @ src[tile]           # TensorE, fp32 accumulate
                row = index[tile] where first occurrence else N + position
            buffer[tile % chains][row] += summed   # one dma_compute read-modify-write
        output = input + sum(buffers)              # or output is the one buffer (in place; with repeats the
                                                   # tiles then go in groups of gathers, adds and plain scatters)
    """
    n_rows, d_size = input.shape
    k_size = src.shape[0]
    _, num_shards, shard_id = get_program_sharding_info()
    _validate_scatter_add_dedup_inputs(input, dim, index, src, num_shards)

    itemsize = 4 if input.dtype == nl.float32 else 2
    path, d_local, chains, in_place = plan_scatter_add_dedup(
        n_rows, d_size, k_size, itemsize, num_shards, unique_indices
    )
    if path == PATH_ONE_HOT:
        return _scatter_one_hot(input, index, src, n_rows, d_size, k_size, num_shards, shard_id)
    if path == PATH_COLUMNS:
        return _scatter_sharded_columns(
            input, index, src, n_rows, d_size, k_size, num_shards, shard_id, d_local, unique_indices, chains, in_place
        )
    return _scatter_sharded_tiles(
        input, index, src, n_rows, d_size, k_size, num_shards, shard_id, unique_indices, chains
    )


def _validate_scatter_add_dedup_inputs(input, dim, index, src, num_shards):
    """Validate inputs for scatter_add_dedup."""
    kernel_assert(len(input.shape) == 2, f"scatter_add_dedup only supports 2D tensors, got input shape {input.shape}")
    kernel_assert(len(index.shape) == 1, f"scatter_add_dedup expects 1D index tensor, got index shape {index.shape}")
    kernel_assert(len(src.shape) == 2, f"scatter_add_dedup only supports 2D tensors, got src shape {src.shape}")
    kernel_assert(dim == 0, f"scatter_add_dedup currently only supports dim=0, got dim={dim}")
    kernel_assert(
        input.shape[1] == src.shape[1], f"Dimension mismatch: input has {input.shape[1]}, src has {src.shape[1]}"
    )
    kernel_assert(
        index.shape[0] == src.shape[0],
        f"Index and src must have same dimension 0, got index {index.shape[0]}, src {src.shape[0]}",
    )
    kernel_assert(
        input.dtype == src.dtype, f"input and src must have the same dtype, got {input.dtype} and {src.dtype}"
    )
    kernel_assert(src.shape[0] >= 1, "scatter_add_dedup needs at least one source row")
    kernel_assert(
        input.shape[0] <= _MAX_ROWS, f"scatter_add_dedup supports at most {_MAX_ROWS} rows, got {input.shape[0]}"
    )
    kernel_assert(
        num_shards * _CHAINS + 1 <= _MAX_FOLD_SOURCES,
        f"num_shards={num_shards} needs more than {_MAX_FOLD_SOURCES} fold sources",
    )
    kernel_assert(nl.tile_size.pmax == _PT, f"expected {_PT} partitions, got {nl.tile_size.pmax}")


def _load_static(dst, src):
    """DMA with descriptors prepared before the run, so no engine spends time generating them."""
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.none)


def _zero_flat(buf, rows, cols, dtype):
    """Zero buf[0:rows, 0:cols] (contiguous rows) with large flat DMAs from one zeroed SBUF tile."""
    free = _D_BLK
    zero_tile = nl.ndarray((_PT, free), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(dst=zero_tile, value=0.0)
    total = rows * cols
    block = _PT * free
    num_blocks = total // block
    if num_blocks > 0:
        for b in nl.affine_range(num_blocks):
            nisa.dma_copy(dst=buf.ap(pattern=[[free, _PT], [1, free]], offset=b * block), src=zero_tile)
    base = num_blocks * block
    rem = total - base
    if rem // free > 0:
        full_rows = rem // free
        nisa.dma_copy(
            dst=buf.ap(pattern=[[free, full_rows], [1, free]], offset=base), src=zero_tile[0:full_rows, 0:free]
        )
        base += full_rows * free
        rem -= full_rows * free
    if rem > 0:
        nisa.dma_copy(dst=buf.ap(pattern=[[rem, 1], [1, rem]], offset=base), src=zero_tile[0:1, 0:rem])


def _scratch_pitch(d_size):
    """
    Rows between consecutive scratch rows of a float32 slab: 1, or enough rows that scratch rows start 64 bytes apart.

    Heavily repeated tiles send most of their 128 positions to the scratch rows, and with rows of a few bytes
    (histograms, D = 1) consecutive scratch rows would share 64-byte lines, which the read-modify-write serializes:
    a D = 1 histogram ran 3.5x slower at 80% repeats than at 6%. Spaced scratch rows keep the cost flat in repeats.
    """
    return max(1, _MIN_ROW_BYTES // (4 * d_size))


def _tile_constants(n_rows, scratch_pitch=1):
    """ones, strictly-lower-triangular mask low[m, n] = (n < m), and the per-partition scratch row N + m (float32)."""
    ones = nl.ndarray((_PT, _PT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    diff_i = nl.ndarray((_PT, _PT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=diff_i, pattern=[[-1, _PT]], offset=0, channel_multiplier=1)
    diff = nl.ndarray((_PT, _PT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=diff, src=diff_i, engine=nisa.engine.vector)
    low = nl.ndarray((_PT, _PT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=low, data=diff, op0=nl.greater, operand0=0.0)
    scratch_i = nl.ndarray((_PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=scratch_i, pattern=[[0, 1]], offset=n_rows, channel_multiplier=scratch_pitch)
    scratch = nl.ndarray((_PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=scratch, src=scratch_i, engine=nisa.engine.vector)
    return ones, low, scratch


def _tile_index(index, k0, kv):
    """The kv indices index[k0:k0+kv] as an int32 column, one per partition."""
    idx_i = nl.ndarray((kv, 1), dtype=nl.int32, buffer=nl.sbuf)
    _load_static(idx_i, index.ap(pattern=[[1, kv], [1, 1]], offset=k0))
    return idx_i


def _tile_eq(index, k0, kv, ones, eq_dtype):
    """For the kv indices index[k0:k0+kv]: eq[k, m] = (index[k] == index[m]) in eq_dtype, and the index column."""
    idx_i = _tile_index(index, k0, kv)
    idx_f = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idx_f, src=idx_i, engine=nisa.engine.vector)
    # row_bcast[m, j] = index[m]; its transpose has index[j] in every column j of every partition.
    row_bcast = nl.ndarray((kv, kv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=row_bcast, data=ones[0:kv, 0:kv], op0=nl.multiply, operand0=idx_f[0:kv, 0:1])
    row_t = nl.ndarray((kv, kv), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=row_t, data=row_bcast)
    eq = nl.ndarray((kv, kv), dtype=eq_dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=eq, data=row_t, op0=nl.equal, operand0=idx_f[0:kv, 0:1])
    return eq, idx_i, idx_f


def _tile_rows(index, k0, kv, ones, low, scratch, eq_dtype):
    """
    For the kv indices index[k0:k0+kv]:
        eq[k, m] = (index[k] == index[m]) in eq_dtype, and
        rows[m]  = index[m] if no earlier position of the tile holds the same index, else N + m (int32).
    """
    eq, idx_i, idx_f = _tile_eq(index, k0, kv, ones, eq_dtype)
    # dup[m] = 1 if an earlier position holds the same index
    earlier = nl.ndarray((kv, kv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=earlier, data1=eq, data2=low[0:kv, 0:kv], op=nl.multiply)
    dup = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=dup, op=nl.max, data=earlier, axis=1, keepdims=True)
    # rows = index + dup * (scratch - index)
    gap = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gap, data1=scratch[0:kv, 0:1], data2=idx_f, op=nl.subtract)
    rows_f = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=rows_f, data=dup, op0=nl.multiply, operand0=gap[0:kv, 0:1], op1=nl.add, operand1=idx_f
    )
    rows = nl.ndarray((kv, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rows, src=rows_f, engine=nisa.engine.vector)
    return eq, rows


def _accumulate_tile(
    src, k0, kv, d_size, col0, num_cols, eq, rows, target, target_stride, target_col0, eq_dtype, stage_dtype
):
    """
    target[rows[p], target_col0 : target_col0 + num_cols] += sum over the tile of src[., col0 : col0 + num_cols]
    sharing the row, in 2048-column blocks. eq None (distinct rows): each source row is added as it is, no matmul.
    """
    for blk in nl.affine_range(div_ceil(num_cols, _D_BLK)):
        c = blk * _D_BLK
        dv = min(_D_BLK, num_cols - c)
        s = nl.ndarray((kv, dv), dtype=eq_dtype, buffer=nl.sbuf)
        _load_static(s, src.ap(pattern=[[d_size, kv], [1, dv]], offset=k0 * d_size + col0 + c))
        if eq is None:
            summed = s
        else:
            summed_psum = _tile_sums(eq, s, kv, dv)
            summed = nl.ndarray((kv, dv), dtype=stage_dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=summed, src=summed_psum)
        ap = target.ap(
            pattern=[[target_stride, kv], [1, dv]], offset=target_col0 + c, vector_offset=rows, indirect_dim=0
        )
        nisa.dma_compute(dst=ap, srcs=[ap, summed], reduce_op=nl.add, unique_indices=True)


def _tile_sums(eq, s, kv, dv):
    """eq^T @ s in fp32 PSUM: every position gets the sum of the tile's sources that share its index."""
    summed_psum = nl.ndarray((kv, dv), dtype=nl.float32, buffer=nl.psum)
    for ch in range(div_ceil(dv, _MM_TILE)):
        cv = min(_MM_TILE, dv - ch * _MM_TILE)
        nisa.nc_matmul(
            dst=summed_psum[0:kv, ch * _MM_TILE : ch * _MM_TILE + cv],
            stationary=eq,
            moving=s[0:kv, ch * _MM_TILE : ch * _MM_TILE + cv],
            is_stationary_onezero=True,
        )
    return summed_psum


def _group_size(num_cols, itemsize):
    """
    Tiles per group of the in-place loop with repeats: as many as fit _GROUP_SBUF_BYTES per tile array, and one when
    a tile has several column blocks. Those blocks already overlap one another's transfers, and every extra tile of
    a group adds one pass of matmuls per block: with 4,096 float32 columns on one core, groups of 2 ran 0.88x.
    """
    if num_cols > _D_BLK:
        return 1
    return max(1, min(_MAX_GROUP, _GROUP_SBUF_BYTES // (num_cols * itemsize)))


def _gather_add_scatter_group(index, src, k0s, kvs, d_size, col0, num_cols, ones, target, target_stride, target_col0):
    """
    target[index[k], target_col0 : target_col0 + num_cols] += src[k, col0 : col0 + num_cols] for the G tiles
    index[k0s[b] : k0s[b] + kvs[b]], without scratch rows: the G gathers first, then every position gets its old row
    plus the sum of all the group's sources with its index (E_ab^T @ src_a over the group's tiles a, on TensorE),
    then the G plain indirect scatters. Positions that share a row, in one tile or in different tiles of the group,
    read the same old value and write the same sum, so a group's gathers overlap one another, its scatters overlap
    one another, and only consecutive groups are ordered.
    """
    g = len(k0s)
    rows = []
    idx_fs = []
    row_ts = []
    for b in range(g):
        idx_i = _tile_index(index, k0s[b], kvs[b])
        idx_f = nl.ndarray((kvs[b], 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idx_f, src=idx_i, engine=nisa.engine.vector)
        # row_t[k, m] = index_b[m] on all 128 partitions, so tiles of any size can be compared with it
        row_bcast = nl.ndarray((kvs[b], _PT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=row_bcast, data=ones[0 : kvs[b], 0:_PT], op0=nl.multiply, operand0=idx_f[0 : kvs[b], 0:1]
        )
        row_t = nl.ndarray((_PT, kvs[b]), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=row_t, data=row_bcast)
        rows.append(idx_i)
        idx_fs.append(idx_f)
        row_ts.append(row_t)
    # eqs[a][b][k, m] = (index_a[k] == index_b[m])
    eqs = []
    for a in range(g):
        eq_a = []
        for b in range(g):
            e = nl.ndarray((kvs[a], kvs[b]), dtype=src.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=e, data=row_ts[b][0 : kvs[a], 0 : kvs[b]], op0=nl.equal, operand0=idx_fs[a][0 : kvs[a], 0:1]
            )
            eq_a.append(e)
        eqs.append(eq_a)
    for blk in nl.affine_range(div_ceil(num_cols, _D_BLK)):
        c = blk * _D_BLK
        dv = min(_D_BLK, num_cols - c)
        srcs = []
        olds = []
        aps = []
        for b in range(g):
            s = nl.ndarray((kvs[b], dv), dtype=src.dtype, buffer=nl.sbuf)
            _load_static(s, src.ap(pattern=[[d_size, kvs[b]], [1, dv]], offset=k0s[b] * d_size + col0 + c))
            ap = target.ap(
                pattern=[[target_stride, kvs[b]], [1, dv]],
                offset=target_col0 + c,
                vector_offset=rows[b],
                indirect_dim=0,
            )
            old = nl.ndarray((kvs[b], dv), dtype=target.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=old, src=ap)
            srcs.append(s)
            olds.append(old)
            aps.append(ap)
        news = []
        for b in range(g):
            new = nl.ndarray((kvs[b], dv), dtype=target.dtype, buffer=nl.sbuf)
            for ch in range(div_ceil(dv, _MM_TILE)):
                c0 = ch * _MM_TILE
                cv = min(_MM_TILE, dv - c0)
                summed = nl.ndarray((kvs[b], cv), dtype=nl.float32, buffer=nl.psum)
                for a in range(g):
                    nisa.nc_matmul(
                        dst=summed,
                        stationary=eqs[a][b],
                        moving=srcs[a][0 : kvs[a], c0 : c0 + cv],
                        is_stationary_onezero=True,
                    )
                nisa.tensor_tensor(
                    dst=new[0 : kvs[b], c0 : c0 + cv], data1=summed, data2=olds[b][0 : kvs[b], c0 : c0 + cv], op=nl.add
                )
            news.append(new)
        for b in range(g):
            nisa.dma_copy(dst=aps[b], src=news[b])


def _gather_add_scatter_tiles(index, src, k_size, d_size, col0, num_cols, ones, target, target_stride, target_col0):
    """Every index tile through _gather_add_scatter_group, _group_size tiles per group, the last group shorter."""
    g = _group_size(num_cols, 2 if target.dtype != nl.float32 else 4)
    num_tiles = div_ceil(k_size, _PT)
    n_groups = (k_size // _PT) // g
    if n_groups > 0:
        for gi in nl.sequential_range(n_groups):
            k0s = []
            for b in range(g):
                k0s.append((gi * g + b) * _PT)
            _gather_add_scatter_group(
                index, src, k0s, [_PT] * g, d_size, col0, num_cols, ones, target, target_stride, target_col0
            )
    k0s = []
    kvs = []
    for t in range(n_groups * g, num_tiles):
        k0s.append(t * _PT)
        kvs.append(min(_PT, k_size - t * _PT))
    if len(k0s) > 0:
        _gather_add_scatter_group(
            index, src, k0s, kvs, d_size, col0, num_cols, ones, target, target_stride, target_col0
        )


def _copy_columns(dst, dst_col0, src, src_col0, n_rows, num_cols):
    """dst[0:n_rows, dst_col0 : dst_col0 + num_cols] = src[0:n_rows, src_col0 : src_col0 + num_cols]."""
    copy_rows = 8192
    for r in nl.affine_range(div_ceil(n_rows, copy_rows)):
        r0 = r * copy_rows
        rv = min(copy_rows, n_rows - r0)
        nisa.dma_copy(
            dst=dst[nl.ds(r0, rv), nl.ds(dst_col0, num_cols)], src=src[nl.ds(r0, rv), nl.ds(src_col0, num_cols)]
        )


def _copy_rows(dst, src, n_rows, d_size, num_shards, shard_id):
    """
    dst = src for this core's share of the rows of two [n_rows, d_size] tensors. Whole rows are one contiguous
    range, copied with descriptors of _D_BLK elements whatever the row width: copying a core's columns instead moves
    one short segment per row (128 bytes for 64 float32 columns at LNC=2), at half the bandwidth or less.
    """
    rows_per_core = div_ceil(n_rows, num_shards)
    r_lo = shard_id * rows_per_core
    r_hi = min(n_rows, r_lo + rows_per_core)
    if r_hi <= r_lo:
        return
    e0 = r_lo * d_size
    count = (r_hi - r_lo) * d_size
    free = _D_BLK
    blk = 8192
    full = count // free
    n_big = full // blk
    if n_big > 0:
        for r in nl.affine_range(n_big):
            off = e0 + r * blk * free
            nisa.dma_copy(
                dst=dst.ap(pattern=[[free, blk], [1, free]], offset=off),
                src=src.ap(pattern=[[free, blk], [1, free]], offset=off),
            )
    left = full - n_big * blk
    base = e0 + n_big * blk * free
    if left > 0:
        nisa.dma_copy(
            dst=dst.ap(pattern=[[free, left], [1, free]], offset=base),
            src=src.ap(pattern=[[free, left], [1, free]], offset=base),
        )
    base = base + left * free
    rem = e0 + count - base
    if rem > 0:
        nisa.dma_copy(
            dst=dst.ap(pattern=[[rem, 1], [1, rem]], offset=base), src=src.ap(pattern=[[rem, 1], [1, rem]], offset=base)
        )


def _scatter_sharded_columns(
    input, index, src, n_rows, d_size, k_size, num_shards, shard_id, d_local, unique_indices, chains, in_place
):
    """
    Wide rows: this core owns columns [d0, d0 + d_local) and walks every index tile. in_place: one buffer, the
    output's own columns; with repeats the tiles go in groups through gathers, adds and plain scatters
    (_gather_add_scatter_group) instead of read-modify-writes into scratch rows.
    """
    output = nl.ndarray((n_rows, d_size), dtype=input.dtype, buffer=nl.shared_hbm)
    d0 = shard_id * d_local
    work = []
    if in_place:
        # no scratch rows are needed, so the one buffer can be this core's columns of the output. Each core copies
        # whole rows (contiguous); the core barrier below orders both cores' copies before any update.
        _copy_rows(output, input, n_rows, d_size, num_shards, shard_id)
        if num_shards > 1:
            # without it, core 1's first read-modify-writes ran before its copy had landed in the shared output:
            # every row of core 1's columns was wrong on a 256 x 2048 probe, while core 0 and LNC=1 were exact
            nisa.core_barrier(output, (0, 1))
        work.append(output)
        target_stride, target_col0 = d_size, d0
    else:
        for g in range(chains):
            work.append(nl.ndarray((n_rows + _PT, d_local), dtype=input.dtype, buffer=nl.private_hbm))
        _copy_columns(work[0], 0, input, d0, n_rows, d_local)
        for g in range(1, chains):
            _zero_flat(work[g], n_rows + _PT, d_local, input.dtype)
        target_stride, target_col0 = d_local, 0

    ones, low, scratch = None, None, None
    if not unique_indices:
        ones, low, scratch = _tile_constants(n_rows)
    if in_place and not unique_indices:
        _gather_add_scatter_tiles(index, src, k_size, d_size, d0, d_local, ones, output, d_size, d0)
    else:
        for t in nl.sequential_range(div_ceil(k_size, _PT)):
            k0 = t * _PT
            kv = min(_PT, k_size - k0)
            if unique_indices:
                eq, rows = None, _tile_index(index, k0, kv)
            else:
                eq, rows = _tile_rows(index, k0, kv, ones, low, scratch, src.dtype)
            _accumulate_tile(
                src,
                k0,
                kv,
                d_size,
                d0,
                d_local,
                eq,
                rows,
                work[t % chains],
                target_stride,
                target_col0,
                src.dtype,
                src.dtype,
            )

    if not in_place:
        if chains == 1:
            _copy_columns(output, d0, work[0], 0, n_rows, d_local)
        else:
            copy_rows = 8192
            for r in nl.affine_range(div_ceil(n_rows, copy_rows)):
                r0 = r * copy_rows
                rv = min(copy_rows, n_rows - r0)
                sources = []
                for g in range(chains):
                    sources.append(work[g][nl.ds(r0, rv), 0:d_local])
                nisa.dma_compute(dst=output[nl.ds(r0, rv), nl.ds(d0, d_local)], srcs=sources, reduce_op=nl.add)
    return output


def _scatter_sharded_tiles(input, index, src, n_rows, d_size, k_size, num_shards, shard_id, unique_indices, chains):
    """Narrow rows: this core walks its share of the index tiles into its own float32 buffers."""
    output = nl.ndarray((n_rows, d_size), dtype=input.dtype, buffer=nl.shared_hbm)
    pitch = _scratch_pitch(d_size)
    buffer_kind = nl.shared_hbm if num_shards > 1 else nl.private_hbm
    slabs = []
    for c in range(num_shards):
        per_core = []
        for g in range(chains):
            per_core.append(
                nl.ndarray(
                    (n_rows + _PT * pitch, d_size),
                    dtype=nl.float32,
                    buffer=buffer_kind,
                    name=f"scatter_add_dedup_slab{c}_{g}",
                )
            )
        slabs.append(per_core)
    mine = slabs[shard_id]
    for g in range(chains):
        _zero_flat(mine[g], n_rows + _PT * pitch, d_size, nl.float32)

    ones, low, scratch = None, None, None
    if not unique_indices:
        ones, low, scratch = _tile_constants(n_rows, pitch)
    num_tiles = div_ceil(k_size, _PT)
    tiles_per_core = div_ceil(num_tiles, num_shards)
    first_tile = shard_id * tiles_per_core
    last_tile = min(num_tiles, first_tile + tiles_per_core)
    if last_tile > first_tile:
        for t in nl.sequential_range(last_tile - first_tile):
            k0 = (first_tile + t) * _PT
            kv = min(_PT, k_size - k0)
            if unique_indices:
                eq, rows = None, _tile_index(index, k0, kv)
            else:
                eq, rows = _tile_rows(index, k0, kv, ones, low, scratch, src.dtype)
            _accumulate_tile(
                src, k0, kv, d_size, 0, d_size, eq, rows, mine[t % chains], d_size, 0, src.dtype, nl.float32
            )

    _fold_rows(output, input, slabs, n_rows, d_size, num_shards, shard_id)
    return output


def _fold_rows(output, input, slabs, n_rows, d_size, num_shards, shard_id):
    """output = input + every buffer of every core (slabs[core][buffer]); each core folds its share of the rows."""
    if num_shards > 1:
        for c in range(num_shards):
            for g in range(len(slabs[c])):
                nisa.core_barrier(slabs[c][g], (0, 1))
    rows_per_core = div_ceil(n_rows, num_shards)
    row_lo = shard_id * rows_per_core
    row_hi = min(n_rows, row_lo + rows_per_core)
    block_rows = max(_PT, (1 << 20) // d_size)
    for r in nl.affine_range(div_ceil(row_hi - row_lo, block_rows)):
        r0 = row_lo + r * block_rows
        rv = min(block_rows, row_hi - r0)
        sources = [input[nl.ds(r0, rv), 0:d_size]]
        for c in range(num_shards):
            for g in range(len(slabs[c])):
                sources.append(slabs[c][g][nl.ds(r0, rv), 0:d_size])
        nisa.dma_compute(dst=output[nl.ds(r0, rv), 0:d_size], srcs=sources, reduce_op=nl.add)


def _scatter_one_hot(input, index, src, n_rows, d_size, k_size, num_shards, shard_id):
    """
    Small tables: no read-modify-writes. Each tile of 128 indices becomes a one-hot matrix O[k, n] = (index[k] == n),
    and O^T @ src accumulates in fp32 PSUM, one 128-row x 512-column accumulator per bank. At LNC=2 each core takes
    half of the tiles, stores its sums in its own float32 buffer, and the buffers are folded after a core barrier.
    """
    output = nl.ndarray((n_rows, d_size), dtype=input.dtype, buffer=nl.shared_hbm)
    n_blocks = div_ceil(n_rows, _PT)
    n_chunks = div_ceil(d_size, _MM_TILE)
    width = n_blocks * _PT
    iota_i = nl.ndarray((_PT, width), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=iota_i, pattern=[[1, width]], offset=0, channel_multiplier=0)
    # Row numbers stay below 1024 (at most _PSUM_BANKS row blocks), exact in float16, and VectorE builds the one-hot
    # matrix from 16-bit data faster: 1.16x to 1.40x on tables of 768 and 1,024 rows, where that build bounds the
    # loop. Below 768 rows the build and the matmuls take about as long, and float16 ran up to 8 % slower.
    iota_dtype = nl.float32
    if width >= _ONE_HOT_FP16_IOTA_ROWS:
        iota_dtype = nl.float16
    iota = nl.ndarray((_PT, width), dtype=iota_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=iota, src=iota_i, engine=nisa.engine.vector)
    # one full bank per accumulator: accumulation groups that share a bank gave wrong sums in another kernel, and
    # small accumulators packed into one bank would serialize their matmuls
    acc = []
    for b in range(n_blocks):
        for c in range(n_chunks):
            acc.append(
                nl.ndarray((_PT, _MM_TILE), dtype=nl.float32, buffer=nl.psum, name=f"scatter_add_one_hot_{b}_{c}")
            )
    num_tiles = div_ceil(k_size, _PT)
    tiles_per_core = div_ceil(num_tiles, num_shards)
    first_tile = shard_id * tiles_per_core
    last_tile = min(num_tiles, first_tile + tiles_per_core)
    if last_tile > first_tile:
        for t in nl.sequential_range(last_tile - first_tile):
            k0 = (first_tile + t) * _PT
            kv = min(_PT, k_size - k0)
            idx_i = _tile_index(index, k0, kv)
            idx_f = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=idx_f, src=idx_i, engine=nisa.engine.vector)
            one_hot = nl.ndarray((kv, width), dtype=src.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=one_hot, data=iota[0:kv, 0:width], op0=nl.equal, operand0=idx_f[0:kv, 0:1])
            s = nl.ndarray((kv, d_size), dtype=src.dtype, buffer=nl.sbuf)
            _load_static(s, src.ap(pattern=[[d_size, kv], [1, d_size]], offset=k0 * d_size))
            for b in range(n_blocks):
                rb = min(_PT, n_rows - b * _PT)
                for c in range(n_chunks):
                    cw = min(_MM_TILE, d_size - c * _MM_TILE)
                    nisa.nc_matmul(
                        dst=acc[b * n_chunks + c][0:rb, 0:cw],
                        stationary=one_hot[0:kv, b * _PT : b * _PT + rb],
                        moving=s[0:kv, c * _MM_TILE : c * _MM_TILE + cw],
                        is_stationary_onezero=True,
                    )
    if num_shards == 1:
        for b in range(n_blocks):
            rb = min(_PT, n_rows - b * _PT)
            for c in range(n_chunks):
                cw = min(_MM_TILE, d_size - c * _MM_TILE)
                inp = nl.ndarray((rb, cw), dtype=input.dtype, buffer=nl.sbuf)
                _load_static(inp, input[nl.ds(b * _PT, rb), nl.ds(c * _MM_TILE, cw)])
                res = nl.ndarray((rb, cw), dtype=input.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=res, data1=acc[b * n_chunks + c][0:rb, 0:cw], data2=inp, op=nl.add)
                nisa.dma_copy(dst=output[nl.ds(b * _PT, rb), nl.ds(c * _MM_TILE, cw)], src=res)
        return output
    slabs = []
    for c in range(num_shards):
        slabs.append(
            [nl.ndarray((n_rows, d_size), dtype=nl.float32, buffer=nl.shared_hbm, name=f"scatter_add_one_hot_slab{c}")]
        )
    mine = slabs[shard_id][0]
    if last_tile > first_tile:
        for b in range(n_blocks):
            rb = min(_PT, n_rows - b * _PT)
            for c in range(n_chunks):
                cw = min(_MM_TILE, d_size - c * _MM_TILE)
                part = nl.ndarray((rb, cw), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=part, src=acc[b * n_chunks + c][0:rb, 0:cw])
                nisa.dma_copy(dst=mine[nl.ds(b * _PT, rb), nl.ds(c * _MM_TILE, cw)], src=part)
    else:
        _zero_flat(mine, n_rows, d_size, nl.float32)
    _fold_rows(output, input, slabs, n_rows, d_size, num_shards, shard_id)
    return output
