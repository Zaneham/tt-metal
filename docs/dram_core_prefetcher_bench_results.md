# Prefetcher matmul benchmark: DRAM-core vs worker-core

## Setup

- Hardware: Blackhole **P150** (single device, 8 unharvested DRAM banks).
- Test driver: `tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bench.py`.
- Ring: 64 cores (8 DRAM banks × 8 receivers per bank).
- dtype: `bfloat8_b` (1088 B/tile).
- M (batch / activation height): 32, decode-style.
- `BENCH_TRACE_REPEATS=100` for both paths (matches the traced-shape default).
- `dispatch_core_axis = DispatchCoreAxis.COL` (matches the canonical
  `test_prefetcher_BH.py` model test).

## Production-shape coverage

Shapes are the production prefetcher-fed matmul shapes from `models/tt_transformers/tt/model_config.py`,
sharded for tensor parallelism using the smallest device count where
`is_prefetcher_supported(model, num_devices, ring_size=64)` returns True:

- **Llama-3.2-1B** (dim=2048, hidden_dim=8192, n_heads=32, n_kv_heads=8, head_dim=64,
  qkv_size=3072) — runs at 1 device.
- **Llama-3.2-3B** (dim=3072, hidden_dim=8192, n_heads=24, n_kv_heads=8, head_dim=128,
  qkv_size=5120) — runs at 1 device.
- **Llama-3.1-8B** (dim=4096, hidden_dim=14336, n_heads=32, n_kv_heads=8, head_dim=128,
  qkv_size=6144) — runs at 2 devices.
- **Llama-3.3-70B** (dim=8192, hidden_dim=28672, n_heads=64, n_kv_heads=8, head_dim=128,
  qkv_size=10240) — runs at 8 devices.

Per-op shape conventions: QKV / FF1 are N-sharded across devices; WO / FF2 are
K-sharded. The K and N values below are per-device (after TP sharding).

## Results

Symmetric trace shape between the two paths: prefetcher is dispatched once
ahead of the trace (`start_dram_core_prefetcher(num_layers=N+1)` /
`dram_prefetcher(num_layers=N+1)`), a warmup matmul consumes 1 layer, then the
trace replays N `ttnn.linear` launches that consume the remaining layers. The
only difference is the prefetcher implementation; matmul kernels, receiver
layout, sub-device pinning, and GCB structure are identical.

| Shape (model_op @ ndev) | K     | N     | DRAM-core µs | DRAM-core TFLOP/s | Worker-core µs | Worker-core TFLOP/s | Worker/DRAM |
|-------------------------|------:|------:|-------------:|------------------:|---------------:|--------------------:|------------:|
| 1B_QKV   @ 1 dev        | 2048  | 3072  |          215 |              1.87 |            201 |               2.00  | 0.94×       |
| 1B_WO    @ 1 dev        | 2048  | 2048  |          169 |              1.58 |            209 |               1.28  | 1.24×       |
| 1B_FF1   @ 1 dev        | 2048  | 8192  |          225 |              4.78 |            219 |               4.89  | 0.97×       |
| 1B_FF2   @ 1 dev        | 8192  | 2048  |          231 |              4.64 |            270 |               3.98  | 1.17×       |
| 3B_QKV   @ 1 dev        | 3072  | 5120  |          240 |              4.20 |            162 |               6.20  | 0.68×       |
| 3B_WO    @ 1 dev        | 3072  | 3072  |          180 |              3.35 |            216 |               2.80  | 1.20×       |
| 3B_FF1   @ 1 dev        | 3072  | 8192  |          234 |              6.88 |            254 |               6.33  | 1.09×       |
| 3B_FF2   @ 1 dev        | 8192  | 3072  |          281 |              5.73 |            212 |               7.60  | 0.75×       |
| 8B_QKV   @ 2 dev        | 4096  | 3072  |          143 |              5.64 |            225 |               3.58  | 1.57×       |
| 8B_WO    @ 2 dev        | 2048  | 4096  |          178 |              3.01 |            128 |               4.20  | 0.72×       |
| 8B_FF1   @ 2 dev        | 4096  | 7168  |          254 |              7.39 |            177 |              10.62  | 0.70×       |
| 8B_FF2   @ 2 dev        | 7168  | 4096  |          293 |              6.41 |            209 |               8.97  | 0.71×       |
| 70B_QKV  @ 8 dev        | 8192  | 1280  |          184 |              3.64 |            193 |               3.49  | 1.05×       |
| 70B_WO   @ 8 dev        | 1024  | 8192  |          238 |              2.26 |            204 |               2.63  | 0.86×       |
| 70B_FF1  @ 8 dev        | 8192  | 3584  |          262 |              7.18 |            282 |               6.67  | 1.08×       |
| 70B_FF2  @ 8 dev        | 3584  | 8192  |          241 |              7.80 |            176 |              10.69  | 0.73×       |

PCC ≥ 0.99996 on every row. Run on 2026-05-24 (Blackhole P150, 8 unharvested
DRAM banks, `BENCH_TRACE_REPEATS=100`).

## How to reproduce

```bash
# Resets device + runs the full parametrized sweep with --run-all so a single
# failure doesn't stop the rest.
~/bin-metal/tt-smi-reset
BENCH_TIMEOUT_SECONDS=900 \
  ~/bin-metal/run_bench.sh --run-all \
    'tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bench.py'
```

Single-shape ad-hoc run (override the parametrize values via env):

```bash
BENCH_K=4096 BENCH_N=14336 BENCH_DTYPE=bfloat8_b BENCH_RECV_PER_BANK=8 \
  ~/bin-metal/run_bench.sh \
    'tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bench.py::test_bench_dram_core_repeats[8B_FF1_2d-device_params={'\''dispatch_core_axis'\'': DispatchCoreAxis.COL, '\''trace_region_size'\'': 23887872}]'
```

## Caveats / what these numbers do and don't say

- **Both paths now trace N discrete matmul launches.** The prefetcher is
  dispatched once ahead of the trace (out-of-band on both sides); the trace
  itself contains only the N `ttnn.linear` launches. Production-faithful —
  each decoder layer is a separate dispatch on both paths.
- **TFLOP/s formula uses the unpadded (K, N)** on both paths. K and N are padded
  internally to a multiple of `ring_size * TILE_SIZE` for ring divisibility, but
  the padded zeros aren't counted as useful flops.
- **Receiver layout** (shared by both paths): senders and receivers from
  `models/tt_transformers/tt/prefetcher/prefetcher_config.yaml` (col-0 / col-7
  workers as the worker-sender path's senders; matmul receivers scattered per
  the production placement). The matmul is pinned to the receiver set via a
  sub-device and `sub_device_id=` on `ttnn.linear`. DRAM bank `i` is paired
  with the receivers at the `i`-th sorted y-row, so the gather_in0 matmul's
  bank-to-receivers ordering assertion is satisfied.
- **The 70B shapes are 8-device per-device shapes**; 2-device and 4-device
  configs fail `is_prefetcher_supported`'s 1 MB/core L1 bound (`bytes_per_core`
  too high). The 70B numbers are still valid as a benchmark — they reflect what
  the matmul looks like on a single chip of an 8-chip TP deployment.
- **Single-device hardware**: this bench runs on one P150. The shapes are
  computed for 1/2/8-device TP but executed on one chip. The all-reduce that
  would follow K-sharded matmuls (WO, FF2) in a real TP deployment is not part
  of this bench.

## Earlier asymmetric measurements

A previous version of this bench captured the worker-core path's
`dram_prefetcher` op *inside* the trace alongside the N matmul launches (while
the DRAM-core path used out-of-band `start_dram_core_prefetcher`, which has no
in-trace analog). That asymmetry gave the worker-core path a ~2-3× headline
advantage (67-133 µs vs DRAM-core 150-300 µs).

When the worker-core prefetcher was hoisted out of the trace to match the
DRAM-core structure, worker-core regressed to the 130-280 µs range while
DRAM-core was unchanged — closing the gap. The mechanism behind the
in-trace advantage isn't fully understood (the prefetcher kernel is dispatched
before any matmul kernel begins running in both cases, and the device-side
pipeline is credit-gated regardless of when the dispatch command was issued);
hypotheses worth investigating include whether `ttnn.dram_prefetcher` is
synchronous on dispatch (which would make the out-of-trace structure a
serialization point that doesn't exist with in-trace capture), and whether
`set_sub_device_stall_group([receiver])` actually excludes the sender
sub-device from `ttnn.to_torch` / `ttnn.synchronize_device` host waits.

What the current numbers do say: with structurally identical trace shape, the
two implementations have comparable per-matmul cost on these decode-style
shapes. No consistent winner per-shape.

For an apples-to-apples comparison of just the push throughput, see the BW
bench: `tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bw_bench.py`.
