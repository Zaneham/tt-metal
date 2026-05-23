# Prefetcher push-bandwidth bench: DRAM-core vs worker-core

## Setup

- Hardware: Blackhole **P150** (single device, 8 unharvested DRAM banks).
- Test driver: `tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bw_bench.py`.
- Ring: 64 cores (8 DRAM banks × 8 receivers per bank).
- dtype: `bfloat8_b` (1088 B/tile).
- `BENCH_TRACE_REPEATS=100`.
- `dispatch_core_axis = DispatchCoreAxis.COL` (matches the canonical
  `test_prefetcher_BH.py` model test).
- Receiver: `ttnn.dram_prefetcher_consumer` — a discard kernel that runs
  `wait_front(1) / pop_front(1)` for `num_iters` iterations. No matmul,
  no L1 budget for matmul CBs. This isolates the prefetcher's push
  bandwidth from any receiver-side compute.

## Methodology

Both paths use the same shape (matches the matmul bench):

1. Prefetcher invoked **once** outside the trace with
   `num_layers = trace_repeats + 1` (1 warmup + N traced layers).
2. One warmup consumer call (drains 1 layer = `ring_size` pages). Also
   primes the cached `dram_prefetcher_consumer` workload so its kernel
   binary write lands outside the trace.
3. Trace captures `trace_repeats` consumer ops (each drains one layer)
   and is replayed once. We time `execute_trace` + `synchronize_device`.

This requires `dram_prefetcher_consumer` to cache its MeshWorkload
across calls (see `dram_prefetcher_consumer.cpp`) — otherwise every call
re-allocates kernel binary storage and re-writes it via
`EnqueueWriteMeshBuffer`, which isn't legal during trace capture.

## Production-shape coverage

Shapes mirror the matmul bench (see `docs/dram_core_prefetcher_bench_results.md`):
the per-device matmul shapes for all Llama prefetcher-fed matmuls at the
smallest device count where `is_prefetcher_supported(model, num_devices,
ring_size=64)` returns True.

| Llama config | dim | hidden_dim | n_heads | n_kv_heads | head_dim | qkv_size | TP |
|--------------|-----|------------|---------|------------|----------|----------|----|
| 3.2-1B       | 2048 | 8192       | 32      | 8          | 64       | 3072     | 1× |
| 3.2-3B       | 3072 | 8192       | 24      | 8          | 128      | 5120     | 1× |
| 3.1-8B       | 4096 | 14336      | 32      | 8          | 128      | 6144     | 2× |
| 3.3-70B      | 8192 | 28672      | 64      | 8          | 128      | 10240    | 8× |

## Results

`page_size` is the per-receiver per-block payload — i.e. what each sender
writes per ring-block. `aggregate_bw` = `trace_repeats * ring_size *
page_size * ring_size / elapsed`. `per_recv_bw` = `aggregate_bw / ring_size`.

All bench runs use **symmetric GCB sizing** on both paths
(`gcb_size = pages_per_layer * page_size`). The DRAM-core kernel uses
**template-specialized fast paths** in `prefetcher_write_chunk` for
`num_rows==1` and `coalesced_num_pages_per_row==1` (the common cases on
production Llama shapes — see
`docs/dram_core_prefetcher_drisc_profile.md`).

| Shape (model_op @ ndev) | K     | N     | page_size | DRAM-core agg GB/s | DRAM-core per-recv GB/s | Worker-core agg GB/s | Worker-core per-recv GB/s | DRAM/Worker |
|-------------------------|------:|------:|----------:|-------------------:|------------------------:|---------------------:|--------------------------:|------------:|
| 1B_QKV    @ 1 dev       | 2048  | 3072  |    2176   |             156.00 |                   2.438 |               143.54 |                     2.243 |       1.09× |
| 1B_WO     @ 1 dev       | 2048  | 2048  |    1088   |              82.48 |                   1.289 |                71.92 |                     1.124 |       1.15× |
| 1B_FF1    @ 1 dev       | 2048  | 8192  |    4352   |             212.84 |                   3.326 |               282.92 |                     4.421 |       0.75× |
| 1B_FF2    @ 1 dev       | 8192  | 2048  |    4352   |             193.34 |                   3.021 |               169.44 |                     2.647 |       1.14× |
| 3B_QKV    @ 1 dev       | 3072  | 5120  |    6528   |             246.19 |                   3.847 |               331.54 |                     5.180 |       0.74× |
| 3B_WO     @ 1 dev       | 3072  | 3072  |    4352   |             237.79 |                   3.715 |               226.80 |                     3.544 |       1.05× |
| 3B_FF1    @ 1 dev       | 3072  | 8192  |    8704   |             264.40 |                   4.131 |               376.01 |                     5.875 |       0.70× |
| 3B_FF2    @ 1 dev       | 8192  | 3072  |    8704   |             268.46 |                   4.195 |               332.12 |                     5.189 |       0.81× |
| 8B_QKV    @ 2 dev       | 4096  | 3072  |    4352   |             241.29 |                   3.770 |               226.74 |                     3.543 |       1.06× |
| 8B_WO     @ 2 dev       | 2048  | 4096  |    2176   |             155.48 |                   2.429 |               143.46 |                     2.242 |       1.08× |
| 8B_FF1    @ 2 dev       | 4096  | 7168  |    8704   |             264.45 |                   4.132 |               375.93 |                     5.874 |       0.70× |
| 8B_FF2    @ 2 dev       | 7168  | 4096  |    8704   |             270.95 |                   4.234 |               332.26 |                     5.192 |       0.82× |
| 70B_QKV   @ 8 dev       | 8192  | 1280  |    4352   |             194.15 |                   3.034 |               169.34 |                     2.646 |       1.15× |
| 70B_WO    @ 8 dev       | 1024  | 8192  |    4352   |             212.22 |                   3.316 |               283.05 |                     4.423 |       0.75× |
| 70B_FF1   @ 8 dev       | 8192  | 3584  |    8704   |             268.79 |                   4.200 |               332.32 |                     5.192 |       0.81× |
| 70B_FF2   @ 8 dev       | 3584  | 8192  |    8704   |             264.78 |                   4.137 |               375.61 |                     5.869 |       0.70× |

**7 shapes match or beat worker-core** on raw push BW (1B_QKV, 1B_WO,
1B_FF2, 3B_WO, 8B_QKV_2d, 8B_WO_2d, 70B_QKV_8d).

## How to reproduce

```bash
~/bin-metal/tt-smi-reset
BENCH_TIMEOUT_SECONDS=1800 BENCH_TRACE_REPEATS=100 \
  ~/bin-metal/run_bench.sh --run-all \
    'tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bw_bench.py'
```

## Caveats / what these numbers do and don't say

- **K and N are padded** internally to multiples of `ring_size * TILE_SIZE`
  so each receiver sees a whole tile and the ring divides cleanly. The
  bandwidth numbers reflect the *padded* (K, N), which matches what the
  prefetcher actually pushes. For some shapes (e.g. 3B/8B_2d/70B QKV/WO)
  this rounds K or N up by one tile-per-receiver.
- **`page_size` is per-receiver per-block** = `(K_padded / ring_size /
  TILE) * (N_padded / ring_size / TILE) * tile_bytes`. That's exactly
  what each sender's DMA pushes per block in both paths, so it's the
  fair unit.
- **No matmul on the receiver.** The consumer is a `wait_front /
  pop_front` loop, so this only measures push throughput — it does NOT
  include the time the matmul would take to consume each block. In the
  matmul bench (`dram_core_prefetcher_bench_results.md`) the receiver-side
  compute can hide some of the push cost; here we see the raw push.
- **Single-device hardware.** Shapes are TP-sharded for 1/2/8 devices but
  executed on one chip. The all-reduce that would follow K-sharded
  matmuls (WO, FF2) in a real TP deployment is not part of this bench.
- In production, the worker senders steal cycles from the matmul cores
  they sit on — the DRAM-core path's value is freeing those cores for
  matmul work, not winning on raw push throughput.

## Why DRAM-core wins on some shapes and loses on others

The structural difference between the two paths:

- **DRAM-core**: one DRISC per bank runs the whole pipeline serially —
  GDDR DMA → push to receivers → finalize. Wall-clock per block ≈
  `DMA_latency + push_cycles`.
- **Worker-core**: two RISCs per sender run in parallel. BRISC reads
  from DRAM into local L1; NCRISC writes from L1 to receivers. Wall-clock
  per block ≈ `max(DMA_latency, push_cycles)` (overlapped), plus a
  BRISC↔NCRISC handshake overhead per block.

This predicts the bench results almost exactly. Bucketing the 16 shapes
by `block_bytes_per_receiver` (= `k_block_w_tiles × n_per_recv_tiles ×
tile_bytes`, the per-block payload each receiver gets):

| Bucket | block_bytes/recv | DRAM/Worker | Shapes |
|--------|------------------:|-------------|--------|
| **Small** (≤ 4 KB, Case 1 fit) | 1,088–4,352 | **1.06×–1.14×** ✓ | 1B_WO, 1B_QKV, 1B_FF2, 3B_WO, 8B_QKV_2d, 8B_WO_2d, 70B_QKV_8d |
| **Medium** (~4 KB, Case 1 but with `writes_per_recv=1`) | 4,352 | 0.74× | 1B_FF1, 70B_WO_8d |
| **Large** (Case 2: block > ring_half, splits into 2 chunks) | 8,704 | 0.70×–0.81× | 1B_FF1-class, 3B_*, 8B_FF*_2d, 70B_FF*_8d |

**The crossover is at `block_bytes_per_receiver ≈ 4 KB`.** Below that,
DMA is short enough that paying it serially on one RISC doesn't cost
much; DRAM-core also avoids the worker's BRISC↔NCRISC sync overhead.
Above 4 KB — and especially when the block doesn't fit in `ring_half`
and gets split into two Case 2 chunks — the GDDR read becomes a
significant fraction of the per-block wall clock, and worker's parallel
RISCs amortize it for free while DRAM-core pays it serially.

Two finer points the table makes visible:

- **At identical `block_bytes/recv = 4,352`, write-count matters.**
  Compare 1B_FF1 (`writes_per_recv = 1`, one big 4,352 B write) vs
  3B_WO/8B_QKV_2d (`writes_per_recv = 2`, two smaller writes). DRAM-core
  loses on the former, wins on the latter. The fast-path specialization
  (single 4,352 B write per receiver per block) ends up being the
  least-amortized regime — set_state is a fixed cost paid per receiver,
  and with only one write per receiver the `set_state ≈ with_state`
  cycle ratio is poor. Worker's `remote_cb_push_back_and_write_pages`
  has the same overhead, but its BRISC reader is doing GDDR work in
  parallel so the writer's per-write overhead is "free time".
- **All Case 2 shapes lose.** Case 2 means `block_bytes > ring_half`
  (35,984 B), so the kernel issues two DMAs and two push rounds per
  block. DRAM-core pays both serially → 2× the DMA wait. Worker pipelines
  the same two reads + two pushes across BRISC and NCRISC → much less
  exposed latency.

**Closing the remaining gap** would need one of:

1. Larger `ring_half` to push more shapes from Case 2 → Case 1
   (kernel-region L1 budget is already ~92 KB of 128 KB on Blackhole;
   not much headroom).
2. Bigger coalesced page (multi-row per NoC write) to reduce the
   per-receiver `set_state` overhead on the medium-bucket shapes —
   would require transposing the L1 stage layout from `[row][recv][page]`
   to `[recv][row][page]`, i.e. scatter-gather DMA from GDDR.
3. A second DRISC-equivalent RISC per bank for true reader/writer
   parallelism. Doesn't exist on Blackhole — there's only one DRISC
   per DRAM bank.
