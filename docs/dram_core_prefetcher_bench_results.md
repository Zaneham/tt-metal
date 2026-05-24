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

Re-measured after the worker-core path was moved from a device-side
`num_kernel_repeats=N` matmul loop to N discrete `ttnn.linear` launches inside
the trace (preceded by one `dram_prefetcher(num_layers=N)`). Both paths now
carry the same per-launch dispatch cost.

| Shape (model_op @ ndev) | K     | N     | DRAM-core µs | DRAM-core TFLOP/s | Worker-core µs | Worker-core TFLOP/s | Worker/DRAM |
|-------------------------|------:|------:|-------------:|------------------:|---------------:|--------------------:|------------:|
| 1B_QKV   @ 1 dev        | 2048  | 3072  |          155 |              2.60 |             69 |               5.85  | 0.44×       |
| 1B_WO    @ 1 dev        | 2048  | 2048  |          141 |              1.91 |             67 |               3.99  | 0.48×       |
| 1B_FF1   @ 1 dev        | 2048  | 8192  |          213 |              5.04 |             75 |              14.25  | 0.35×       |
| 1B_FF2   @ 1 dev        | 8192  | 2048  |          170 |              6.33 |            119 |               9.03  | 0.70×       |
| 3B_QKV   @ 1 dev        | 3072  | 5120  |          214 |              4.70 |             98 |              10.24  | 0.46×       |
| 3B_WO    @ 1 dev        | 3072  | 3072  |          192 |              3.14 |             92 |               6.56  | 0.48×       |
| 3B_FF1   @ 1 dev        | 3072  | 8192  |          272 |              5.92 |            115 |              13.96  | 0.42×       |
| 3B_FF2   @ 1 dev        | 8192  | 3072  |          282 |              5.72 |            133 |              12.11  | 0.47×       |
| 8B_QKV   @ 2 dev        | 4096  | 3072  |          211 |              3.82 |             92 |               8.77  | 0.44×       |
| 8B_WO    @ 2 dev        | 2048  | 4096  |          150 |              3.57 |             69 |               7.80  | 0.46×       |
| 8B_FF1   @ 2 dev        | 4096  | 7168  |          302 |              6.22 |            115 |              16.29  | 0.38×       |
| 8B_FF2   @ 2 dev        | 7168  | 4096  |          225 |              8.35 |            133 |              14.16  | 0.59×       |
| 70B_QKV  @ 8 dev        | 8192  | 1280  |          173 |              3.87 |            119 |               5.65  | 0.69×       |
| 70B_WO   @ 8 dev        | 1024  | 8192  |          162 |              3.32 |             75 |               7.11  | 0.47×       |
| 70B_FF1  @ 8 dev        | 8192  | 3584  |          289 |              6.49 |            133 |              14.17  | 0.46×       |
| 70B_FF2  @ 8 dev        | 3584  | 8192  |          219 |              8.60 |            116 |              16.27  | 0.53×       |

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

- **Both paths now trace N discrete matmul launches.** The DRAM-core trace is
  N single-matmul launches fed by an out-of-band DRISC prefetcher (1 warmup +
  N traced layers). The worker-core trace is `dram_prefetcher(num_layers=N)`
  followed by N `ttnn.linear` launches inside the same trace. Both paths now
  carry the same per-launch host dispatch overhead — the comparison is
  production-faithful (each decoder layer is a separate dispatch on both paths).
- **TFLOP/s formula uses the unpadded (K, N)** on both paths. K and N are padded
  internally to a multiple of `ring_size * TILE_SIZE` for ring divisibility, but
  the padded zeros aren't counted as useful flops.
- **Worker-core layout**: senders and receivers from
  `models/tt_transformers/tt/prefetcher/prefetcher_config.yaml` (col-0 / col-7
  senders, scattered receivers per the production placement). The matmul is
  pinned to the receiver set via a sub-device and `sub_device_id=` on
  `ttnn.linear`. Without this the matmul lands on the (0..7, 0..7) logical
  rectangle which doesn't overlap the production receivers and hangs at the
  dispatch cores.
- **The 70B shapes are 8-device per-device shapes**; 2-device and 4-device
  configs fail `is_prefetcher_supported`'s 1 MB/core L1 bound (`bytes_per_core`
  too high). The 70B numbers are still valid as a benchmark — they reflect what
  the matmul looks like on a single chip of an 8-chip TP deployment.
- **Single-device hardware**: this bench runs on one P150. The shapes are
  computed for 1/2/8-device TP but executed on one chip. The all-reduce that
  would follow K-sharded matmuls (WO, FF2) in a real TP deployment is not part
  of this bench.

## Why per-op cost matters here

The DRAM-core path was originally designed to win on shapes where push
bandwidth bottlenecks the matmul (large N with many ring positions); on those
shapes the per-block DMA dominates and dispatch overhead amortizes. The bench
numbers here are decode-style (M=32) matmuls where neither push BW nor compute
dominates.

Worth noting: the worker-core µs column is essentially unchanged from the
previous device-side `num_kernel_repeats=N` measurements (e.g. 1B_QKV 68→69,
8B_FF1 116→115, 70B_FF1 132→133). FD trace-replay dispatch overhead per
launch is ~zero on this hardware, so the N matmul launches inside the trace
amortize to the same per-op cost as a single op with an N-iteration kernel
loop. The DRAM-core µs column then reflects something else (per-launch DRISC
prefetcher hand-off, sub-device handshake, smaller in-flight pipeline depth)
rather than raw dispatch overhead.

For an apples-to-apples comparison of just the push throughput, see the BW
bench: `tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bw_bench.py`.
