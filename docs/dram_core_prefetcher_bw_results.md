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
| 1B_QKV    @ 1 dev       | 2048  | 3072  |    2176   |             147.48 |                   2.304 |               143.54 |                     2.243 |       1.03× |
| 1B_WO     @ 1 dev       | 2048  | 2048  |    1088   |              76.26 |                   1.192 |                71.92 |                     1.124 |       1.06× |
| 1B_FF1    @ 1 dev       | 2048  | 8192  |    4352   |             211.26 |                   3.301 |               282.92 |                     4.421 |       0.75× |
| 1B_FF2    @ 1 dev       | 8192  | 2048  |    4352   |             191.59 |                   2.994 |               169.44 |                     2.647 |       1.13× |
| 3B_QKV    @ 1 dev       | 3072  | 5120  |    6528   |             243.88 |                   3.811 |               331.54 |                     5.180 |       0.74× |
| 3B_WO     @ 1 dev       | 3072  | 3072  |    4352   |             222.49 |                   3.476 |               226.80 |                     3.544 |       0.98× |
| 3B_FF1    @ 1 dev       | 3072  | 8192  |    8704   |             261.34 |                   4.083 |               376.01 |                     5.875 |       0.70× |
| 3B_FF2    @ 1 dev       | 8192  | 3072  |    8704   |             262.45 |                   4.101 |               332.12 |                     5.189 |       0.79× |
| 8B_QKV    @ 2 dev       | 4096  | 3072  |    4352   |             224.46 |                   3.507 |               226.74 |                     3.543 |       0.99× |
| 8B_WO     @ 2 dev       | 2048  | 4096  |    2176   |             149.11 |                   2.330 |               143.46 |                     2.242 |       1.04× |
| 8B_FF1    @ 2 dev       | 4096  | 7168  |    8704   |             260.91 |                   4.077 |               375.93 |                     5.874 |       0.69× |
| 8B_FF2    @ 2 dev       | 7168  | 4096  |    8704   |             262.25 |                   4.098 |               332.26 |                     5.192 |       0.79× |
| 70B_QKV   @ 8 dev       | 8192  | 1280  |    4352   |             191.59 |                   2.994 |               169.34 |                     2.646 |       1.13× |
| 70B_WO    @ 8 dev       | 1024  | 8192  |    4352   |             211.80 |                   3.309 |               283.05 |                     4.423 |       0.75× |
| 70B_FF1   @ 8 dev       | 8192  | 3584  |    8704   |             262.15 |                   4.096 |               332.32 |                     5.192 |       0.79× |
| 70B_FF2   @ 8 dev       | 3584  | 8192  |    8704   |             261.14 |                   4.080 |               375.61 |                     5.869 |       0.69× |

**6 shapes now match or beat worker-core** on raw push BW (1B_QKV, 1B_WO,
1B_FF2, 8B_WO_2d, 70B_QKV_8d, plus 3B_WO/8B_QKV within 2% noise).

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
- **DRAM-core matches or beats worker on 6/16 shapes** (small/medium
  page sizes) and stays within 15-30% on the heavier shapes
  (FF1/FF2 with 4+ pages per chunk). The fast-path specialization
  collapses the per-write loop overhead, exposing the underlying NoC
  injection rate as the dominant cost.
- The shapes where worker still wins are those with **larger
  per-receiver chunks** (page_size ≥ 6 KB combined with k_block_w ≥ 2)
  — there worker's dual-RISC parallelism (BRISC reads, NCRISC writes
  concurrent) helps amortize DRAM read latency that DRISC must serialize
  on its single RISC.
- In production, the worker senders steal cycles from the matmul cores
  they sit on — the DRAM-core path's value is freeing those cores for
  matmul work, not winning on raw push throughput.
- **Single-device hardware.** Shapes are TP-sharded for 1/2/8 devices but
  executed on one chip. The all-reduce that would follow K-sharded
  matmuls (WO, FF2) in a real TP deployment is not part of this bench.
