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

| Shape (model_op @ ndev) | K     | N     | page_size | DRAM-core agg GB/s | DRAM-core per-recv GB/s | Worker-core agg GB/s | Worker-core per-recv GB/s | DRAM/Worker |
|-------------------------|------:|------:|----------:|-------------------:|------------------------:|---------------------:|--------------------------:|------------:|
| 1B_QKV    @ 1 dev       | 2048  | 3072  |    2176   |             119.92 |                   1.874 |               142.66 |                     2.229 |       0.84× |
| 1B_WO     @ 1 dev       | 2048  | 2048  |    1088   |              60.44 |                   0.944 |                71.54 |                     1.118 |       0.85× |
| 1B_FF1    @ 1 dev       | 2048  | 8192  |    4352   |             207.95 |                   3.249 |               282.88 |                     4.420 |       0.74× |
| 1B_FF2    @ 1 dev       | 8192  | 2048  |    4352   |             148.96 |                   2.327 |               169.44 |                     2.648 |       0.88× |
| 3B_QKV    @ 1 dev       | 3072  | 5120  |    6528   |             222.18 |                   3.472 |               331.71 |                     5.183 |       0.67× |
| 3B_WO     @ 1 dev       | 3072  | 3072  |    4352   |             196.42 |                   3.069 |               226.79 |                     3.544 |       0.87× |
| 3B_FF1    @ 1 dev       | 3072  | 8192  |    8704   |             264.76 |                   4.137 |               375.63 |                     5.869 |       0.70× |
| 3B_FF2    @ 1 dev       | 8192  | 3072  |    8704   |             232.52 |                   3.633 |               332.47 |                     5.195 |       0.70× |
| 8B_QKV    @ 2 dev       | 4096  | 3072  |    4352   |             195.40 |                   3.053 |               226.60 |                     3.541 |       0.86× |
| 8B_WO     @ 2 dev       | 2048  | 4096  |    2176   |             119.93 |                   1.874 |               143.63 |                     2.244 |       0.83× |
| 8B_FF1    @ 2 dev       | 4096  | 7168  |    8704   |             260.51 |                   4.070 |               374.68 |                     5.854 |       0.70× |
| 8B_FF2    @ 2 dev       | 7168  | 4096  |    8704   |             232.51 |                   3.633 |               332.04 |                     5.188 |       0.70× |
| 70B_QKV   @ 8 dev       | 8192  | 1280  |    4352   |             148.93 |                   2.327 |               169.36 |                     2.646 |       0.88× |
| 70B_WO    @ 8 dev       | 1024  | 8192  |    4352   |             203.77 |                   3.184 |               283.13 |                     4.424 |       0.72× |
| 70B_FF1   @ 8 dev       | 8192  | 3584  |    8704   |             232.66 |                   3.635 |               332.38 |                     5.194 |       0.70× |
| 70B_FF2   @ 8 dev       | 3584  | 8192  |    8704   |             265.31 |                   4.145 |               375.76 |                     5.871 |       0.70× |

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
- **Worker-core path beats DRAM-core on every row** (~0.7×–0.88× ratio).
  Worker senders sit physically closer to the receivers on the NoC; DRAM
  senders are at the edges. For the bench shapes here both paths are
  push-bound, so the worker's NoC-distance advantage shows up directly.
  In production the worker senders are stealing cycles from the matmul
  cores they sit on — the DRAM-core path's value is freeing those cores,
  not winning on raw push throughput.
- **Single-device hardware.** Shapes are TP-sharded for 1/2/8 devices but
  executed on one chip. The all-reduce that would follow K-sharded
  matmuls (WO, FF2) in a real TP deployment is not part of this bench.
