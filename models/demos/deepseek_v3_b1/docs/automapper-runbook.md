# DeepSeek V3 B1 — Auto-Mapper Run Guide

How to run DeepSeek V3 B1 multihost tests using `--mesh-graph-descriptor` (auto-mapper) instead of manual rank-binding files.

---

## Prerequisites

- `tt-metal` built and installed on all worker hosts under `/home/user/tt-metal`
- `ttnn` wheel installed on all workers
- Models mounted at `/mnt/models/deepseek-ai/`
- Reference data at `/mnt/models/ref_data/`
- Hostfile at `/ci/ttrun/hostfile` (one hostname per line)
- `ttrun.py` at `/ci/tt-metal/ttnn/ttnn/distributed/ttrun.py`

---

## Step 1 — Reset Cards

Run on all hosts before each test. Performs a GLX reset and waits for links to settle.

```bash
mpirun --pernode tt-smi -glx_reset
sleep 5
```

---

## Step 2 — Validate Cluster

Runs fabric traffic to confirm all inter-galaxy links are healthy.

```bash
mpirun --pernode --tag-output \
  /home/user/tt-metal/build/tools/scaleout/run_cluster_validation \
  --send-traffic \
  --hard-fail \
  --cabling-descriptor-path /data/scaleout_configs/bh_glx_exabox/cabling_descriptor.textproto \
  --deployment-descriptor-path /data/scaleout_configs/bh_glx_exabox/deployment_descriptor.textproto \
  --num-iterations 5
```

If this fails, repeat Steps 1–2 (up to 10 times). Do not proceed until validation passes.

---

## Step 3 — Run the Test

Set the working directory to `/home/user/tt-metal` for all runs below.

```bash
cd /home/user/tt-metal
```

### Build the host list

```bash
PIPELINE_HOSTS=$(awk '{printf "%s,", $1}' /ci/ttrun/hostfile | sed 's/,$//')
TTRUN_PY=/ci/tt-metal/ttnn/ttnn/distributed/ttrun.py
```

### Available mesh graph descriptors

| Configuration | Descriptor file |
|---|---|
| Single galaxy (1×) | `models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_single_galaxy_mesh_graph_descriptor.textproto` |
| Single pod / SP1 (4 galaxies) | `models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_single_pod_mesh_graph_descriptor.textproto` |
| Dual pod (8 galaxies) | `models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_mesh_graph_descriptor_dual_pod.textproto` |
| Superpod / SP4 (16 galaxies) | `models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_mesh_graph_descriptor_superpod.textproto` |

---

### SP1 — Stress test (4 galaxies, `demo.cli`)

```bash
TT_METAL_ALLOCATOR_MODE_HYBRID=1 TT_METAL_SLOW_DISPATCH_MODE=1 python3 "${TTRUN_PY}" \
  --tcp-interface ens5f0np0 \
  --hosts $PIPELINE_HOSTS \
  --mpi-args "--oversubscribe --tag-output --wdir /home/user/tt-metal" \
  --mesh-graph-descriptor models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_single_pod_mesh_graph_descriptor.textproto \
  python3 -m models.demos.deepseek_v3_b1.demo.cli \
    --max-new-tokens 128000 \
    --weights real \
    --cache-path /mnt/models/deepseek-ai/cache-2026-03-22 \
    --prompt "Your prompt here" \
    --model-path /mnt/models/deepseek-ai/DeepSeek-R1-0528-dequantized
```

---

### SP4 — Teacher forcing test (16 galaxies, `demo.teacher_forced`)

```bash
TT_METAL_ALLOCATOR_MODE_HYBRID=1 TT_METAL_SLOW_DISPATCH_MODE=1 python3 "${TTRUN_PY}" \
  --tcp-interface ens5f0np0 \
  --hosts $PIPELINE_HOSTS \
  --mpi-args "--oversubscribe --tag-output --wdir /home/user/tt-metal" \
  --mesh-graph-descriptor models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_mesh_graph_descriptor_superpod.textproto \
  python3 -m models.demos.deepseek_v3_b1.demo.teacher_forced \
    --weights real \
    --cache-path /mnt/models/deepseek-ai/cache-2026-03-22 \
    --model-path /mnt/models/deepseek-ai/DeepSeek-R1-0528-dequantized \
    --prompt "Your prompt here" \
    --reference-file /mnt/models/ref_data/output1_reference.txt \
    --output-file output1_teacher_forcing_results.json \
    --chunk-accuracy 256

jq . output1_teacher_forcing_results.json
```

---

## Key differences from rank-binding approach

| | Old (rank-binding) | New (auto-mapper) |
|---|---|---|
| Host slots | `host:4` per entry in `--host` | plain `host` — no slot count |
| MPI args | `--map-by rankfile:file=... --bind-to hwt:overload-allowed` | `--oversubscribe` |
| Extra files | `rev_c_rank_binding_*.yaml` + `rev_c_rank_file_*` | none |
| ttrun flag | `--rank-binding <file>` | `--mesh-graph-descriptor <file>` |

---

## Troubleshooting

- *Reset fails repeatedly*: check `tt-smi` is accessible on all hosts and driver version is consistent across nodes.
- *Validation fails after reset*: wait 30 s and retry; fabric links need time to train after GLX reset.
- *`ttrun.py` import error*: make sure `TT_METAL_HOME` or `PYTHONPATH` points to the correct tt-metal checkout.
- *OOM*: confirm `TT_METAL_ALLOCATOR_MODE_HYBRID=1` is set; this enables host+device hybrid allocation required for B1.
