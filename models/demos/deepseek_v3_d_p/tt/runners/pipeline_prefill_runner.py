#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Pipeline-parallel prefill runner for DeepSeek V3 (bringup).

Splits the 61-layer prefill across 4 ranks under tt-run. Each rank owns a
contiguous layer range and an 8x4 mesh; activations between ranks flow over
D2D sockets handled inside the kernels (no Python-side socket plumbing).

Layer assignment (hard-coded):
    rank 0: layers 0..15  + embedding
    rank 1: layers 16..31
    rank 2: layers 32..47
    rank 3: layers 48..60

Scope: standalone JSON input, no SHM bridge, no KV migration, no LM head.
KV is the prefill output and is handled by the kernel-level downstream
socket; the Python runner has nothing to read back.

Design doc: docs/plans/pipeline_prefill_runner_design.md
"""

import json
import os
import signal
import time
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reorder_tensor_chunks
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import (
    DEFAULT_TTNN_CACHE,
    load_hf_config,
    open_mesh_device,
    resolve_weight_cache_path,
    setdefault_cache_env,
)
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 3200 * _sp))
IS_BALANCED = os.environ.get("PREFILL_IS_BALANCED", "1") == "1"
CAPACITY_FACTOR = int(os.environ.get("PREFILL_CAPACITY_FACTOR", 8))
_gate_mode_name = os.environ.get("PREFILL_GATE_FALLBACK_MODE", "DEVICE_FP32")

setdefault_cache_env()

PIPELINE_NUM_RANKS = 4

# inclusive ranges
RANK_TO_LAYER_RANGE: dict[int, tuple[int, int]] = {
    0: (0, 15),
    1: (16, 31),
    2: (32, 47),
    3: (48, 60),
}

# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


# ---------------------------------------------------------------------------
# Config dump
# ---------------------------------------------------------------------------


def _print_config(local_rank: int, start_layer: int, end_layer: int) -> None:
    UNSET = "<NOT SET>"
    rows = [
        ("LOCAL_RANK", str(local_rank), False),
        ("LAYER_RANGE", f"[{start_layer}, {end_layer}]", False),
        ("DEEPSEEK_V3_HF_MODEL", os.environ.get("DEEPSEEK_V3_HF_MODEL", UNSET), True),
        ("TT_DS_PREFILL_TTNN_CACHE", os.environ.get("TT_DS_PREFILL_TTNN_CACHE", DEFAULT_TTNN_CACHE), False),
        ("PREFILL_SP", str(_sp), False),
        ("PREFILL_TP", str(_tp), False),
        ("PREFILL_NUM_LAYERS", str(NUM_LAYERS), False),
        ("PREFILL_MAX_SEQ_LEN", str(MAX_SEQ_LEN), False),
        ("PREFILL_IS_BALANCED", str(IS_BALANCED), False),
        ("PREFILL_CAPACITY_FACTOR", str(CAPACITY_FACTOR), False),
        ("PREFILL_GATE_FALLBACK_MODE", _gate_mode_name, False),
        ("PREFILL_STANDALONE_INPUT", os.environ.get("PREFILL_STANDALONE_INPUT", "<default>"), False),
        ("PREFILL_STANDALONE_ITERS", os.environ.get("PREFILL_STANDALONE_ITERS", "5"), False),
    ]
    missing = [label for label, val, req in rows if req and val == UNSET]
    sep = "=" * 70
    lines = [sep, f"pipeline_prefill_runner configuration (rank {local_rank})", sep]
    for label, val, req in rows:
        flag = " [REQUIRED]" if req else ""
        warn = " *** MISSING ***" if val == UNSET and req else ""
        lines.append(f"  {label:<35} = {val}{flag}{warn}")
    lines.append(sep)
    logger.info("\n" + "\n".join(lines))
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _build_embedding(mesh_device: ttnn.MeshDevice, hf_config, weight_cache_path: Optional[Path]):
    logger.info("Building TtParallelEmbedding (rank 0 only)")
    return TtParallelEmbedding(
        mesh_device=mesh_device,
        vocab_size=hf_config.vocab_size,
        emb_dim=hf_config.hidden_size,
        torch_weight=None,
        sp_axis=0,
        tp_axis=1,
        weight_cache_path=weight_cache_path,
    )


def _build_layers(
    mesh_device: ttnn.MeshDevice,
    hf_config,
    start_layer: int,
    end_layer: int,
    weight_cache_path: Optional[Path],
) -> list[TtPrefillBlock]:
    layers = []
    for i in range(start_layer, end_layer + 1):
        logger.info(f"Building TtPrefillBlock layer_idx={i}")
        layer = TtPrefillBlock(
            mesh_device=mesh_device,
            config=hf_config,
            state_dict={},
            layer_idx=i,
            seq_len=MAX_SEQ_LEN,
            dispatch_buffer_capacity_factor=CAPACITY_FACTOR,
            num_links=2,
            topology=ttnn.Topology.Linear,
            sp_axis=0,
            tp_axis=1,
            is_balanced=IS_BALANCED,
            gate_fallback_mode=GateComputeMode[_gate_mode_name],
            weight_cache_path=weight_cache_path,
        )
        layers.append(layer)
    return layers


def _allocate_placeholder_activation(mesh_device: ttnn.MeshDevice, hf_config) -> ttnn.Tensor:
    """Allocate a zero-init activation tensor for ranks 1..3.

    Shape matches the embedding output: [1, 1, seq_per_chip, emb_dim/tp],
    TILE_LAYOUT, DRAM. The first layer's attn_norm kernel reads its real
    input from the upstream D2D socket and overwrites this placeholder.
    """
    sp, tp = GLOBAL_MESH_SHAPE
    seq_per_chip = MAX_SEQ_LEN // sp
    emb_per_tp = hf_config.hidden_size // tp
    torch_zeros = torch.zeros(1, 1, seq_per_chip, emb_per_tp, dtype=torch.bfloat16)
    return ttnn.from_torch(
        torch_zeros,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


# ---------------------------------------------------------------------------
# Input prep
# ---------------------------------------------------------------------------


def _prepare_input_tensor(mesh_device: ttnn.MeshDevice, token_ids: list[int]) -> ttnn.Tensor:
    """SP-shard token IDs across mesh rows for embedding lookup.

    Copied from TtDeepSeekPrefillPipeline._prepare_input_tensor — we need the
    same balanced-chunk reordering so any layer-by-layer PCC comparison with
    the single-rank runner is meaningful.
    """
    sp, _ = GLOBAL_MESH_SHAPE
    isl_per_chip = len(token_ids) // sp

    if IS_BALANCED:
        chunk_order = create_balanced_chunk_order(sp)
        t = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        t = reorder_tensor_chunks(t, chunk_order, seq_dim=2)
        token_ids_sharded = t.squeeze(0).squeeze(-1).reshape(sp, 1, isl_per_chip)
    else:
        token_ids_sharded = torch.tensor(token_ids, dtype=torch.int64).reshape(sp, 1, isl_per_chip)

    return ttnn.from_torch(
        token_ids_sharded,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=GLOBAL_MESH_SHAPE,
            dims=(0, None),
        ),
    )


def _load_standalone_input() -> tuple[int, list[int], int]:
    """Read the standalone JSON. All ranks read it (deterministic, file-only).

    Returns (task_id, padded_token_ids, actual_isl).
    """
    default_path = Path(__file__).parent / "standalone_input.json"
    input_path = Path(os.environ.get("PREFILL_STANDALONE_INPUT", default_path))
    logger.info(f"[standalone] Reading input from {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    task_id = data["task_id"]
    token_ids = list(data["token_ids"])

    if len(token_ids) > MAX_SEQ_LEN:
        tail_preview = token_ids[MAX_SEQ_LEN : MAX_SEQ_LEN + 10]
        tail_suffix = "..." if len(token_ids) > MAX_SEQ_LEN + 10 else ""
        raise ValueError(
            f"task_id={task_id} prompt has {len(token_ids)} tokens but "
            f"MAX_SEQ_LEN={MAX_SEQ_LEN}. Bump PREFILL_MAX_SEQ_LEN. "
            f"Dropped tail tokens would have been: {tail_preview}{tail_suffix}"
        )

    actual_isl = len(token_ids)
    if len(token_ids) < MAX_SEQ_LEN:
        token_ids = token_ids + [1] * (MAX_SEQ_LEN - len(token_ids))
    return task_id, token_ids, actual_isl


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _forward(
    local_rank: int,
    mesh_device: ttnn.MeshDevice,
    embed: Optional[TtParallelEmbedding],
    layers: list[TtPrefillBlock],
    kvpe_cache: ttnn.Tensor,
    rope_setup: RotarySetup,
    start_layer: int,
    token_ids_tt: Optional[ttnn.Tensor],
    actual_isl: int,
    placeholder_activation: Optional[ttnn.Tensor],
) -> None:
    rope_tensors = rope_setup.get_rope_tensors(MAX_SEQ_LEN)

    if local_rank == 0:
        assert token_ids_tt is not None and embed is not None
        h = embed(token_ids_tt)
        h = ttnn.unsqueeze_to_4D(h)
    else:
        assert placeholder_activation is not None
        h = placeholder_activation

    for i, layer in enumerate(layers):
        global_layer_idx = start_layer + i
        signpost(f"forward_layer_{global_layer_idx}_start")
        h, _ = layer(
            h,
            rope_tensors,
            kvpe_cache,
            cache_layer_idx=i,
            actual_isl=actual_isl,
        )
        signpost(f"forward_layer_{global_layer_idx}_end")

    ttnn.synchronize_device(mesh_device)


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------


def _run_loop(
    local_rank: int,
    mesh_device: ttnn.MeshDevice,
    hf_config,
    embed: Optional[TtParallelEmbedding],
    layers: list[TtPrefillBlock],
    kvpe_cache: ttnn.Tensor,
    rope_setup: RotarySetup,
    start_layer: int,
) -> None:
    """One unified loop for all ranks. Rank 0 prepares input from JSON;
    ranks 1..3 supply a placeholder activation that the upstream-socket
    kernel will overwrite. Same iteration count on every rank so they stay
    in lockstep with the kernel-level socket transport.
    """
    task_id, padded_token_ids, actual_isl = _load_standalone_input()
    num_iterations = int(os.environ.get("PREFILL_STANDALONE_ITERS", "5"))

    if local_rank == 0:
        token_ids_tt = _prepare_input_tensor(mesh_device, padded_token_ids)
        placeholder = None
    else:
        token_ids_tt = None
        placeholder = _allocate_placeholder_activation(mesh_device, hf_config)

    logger.info(f"[rank {local_rank}] task_id={task_id} actual_isl={actual_isl} " f"num_iterations={num_iterations}")

    iter_times_ms: list[float] = []
    for i in range(num_iterations):
        if _shutdown:
            logger.info(f"[rank {local_rank}] shutdown requested, breaking loop")
            break
        t0 = time.perf_counter()
        _forward(
            local_rank=local_rank,
            mesh_device=mesh_device,
            embed=embed,
            layers=layers,
            kvpe_cache=kvpe_cache,
            rope_setup=rope_setup,
            start_layer=start_layer,
            token_ids_tt=token_ids_tt,
            actual_isl=actual_isl,
            placeholder_activation=placeholder,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        iter_times_ms.append(dt_ms)
        logger.info(
            f"[rank {local_rank}] [prefill timing] task_id={task_id} iter={i} "
            f"num_tokens={MAX_SEQ_LEN} _forward() = {dt_ms:.2f} ms"
        )

        ttnn.distributed_context_barrier()

    logger.info(f"[rank {local_rank}] [iter timing summary] per-iter ms = " f"{[round(t, 2) for t in iter_times_ms]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    # Distributed context: tt-run pre-initializes MPI; we just bind the Python
    # handle. Reuse migration_setup's helpers — they handle the
    # is-already-initialized case gracefully.
    from models.demos.deepseek_v3_d_p.tt.runners.migration_setup import ensure_distributed_context, get_distributed_info

    ensure_distributed_context()
    subctx_id, local_rank, local_size, world_rank, world_size = get_distributed_info()

    if local_size != PIPELINE_NUM_RANKS:
        raise RuntimeError(
            f"pipeline_prefill_runner expects local_size={PIPELINE_NUM_RANKS}, got {local_size}. "
            f"Check the tt-run --rank-bindings-mapping for this sub-context."
        )
    if local_rank not in RANK_TO_LAYER_RANGE:
        raise RuntimeError(
            f"pipeline_prefill_runner has no layer range for local_rank={local_rank}. "
            f"Known ranks: {sorted(RANK_TO_LAYER_RANGE.keys())}"
        )

    start_layer, end_layer = RANK_TO_LAYER_RANGE[local_rank]
    num_my_layers = end_layer - start_layer + 1

    _print_config(local_rank, start_layer, end_layer)
    logger.info(
        f"pipeline_prefill_runner subctx={subctx_id} local={local_rank}/{local_size} "
        f"world={world_rank}/{world_size} mesh={GLOBAL_MESH_SHAPE} "
        f"layers={start_layer}..{end_layer}"
    )

    mesh_device = open_mesh_device(GLOBAL_MESH_SHAPE)

    hf_config = load_hf_config()
    hf_config.max_seq_len = MAX_SEQ_LEN

    cache_path = resolve_weight_cache_path(GLOBAL_MESH_SHAPE)

    embed = _build_embedding(mesh_device, hf_config, cache_path) if local_rank == 0 else None
    layers = _build_layers(mesh_device, hf_config, start_layer, end_layer, cache_path)

    # KVPE cache sized for this rank's layers only; cache_layer_idx is local.
    kvpe_head_dim = hf_config.qk_rope_head_dim + hf_config.kv_lora_rank
    kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_head_dim,
        mesh_device=mesh_device,
        seq_len=MAX_SEQ_LEN,
        mesh_shape=list(GLOBAL_MESH_SHAPE),
        sp_axis=0,
        num_kvpe_cache_layers=num_my_layers,
    )

    rope_setup = RotarySetup(hf_config, mesh_device, sp_axis=0, is_balanced=IS_BALANCED)

    # Warmup: synthetic [0]*MAX_SEQ_LEN forward to compile kernels.
    logger.info(f"[rank {local_rank}] warming up with {MAX_SEQ_LEN} tokens")
    warmup_token_ids = [0] * MAX_SEQ_LEN
    if local_rank == 0:
        warmup_tt = _prepare_input_tensor(mesh_device, warmup_token_ids)
        warmup_placeholder = None
    else:
        warmup_tt = None
        warmup_placeholder = _allocate_placeholder_activation(mesh_device, hf_config)

    t0 = time.perf_counter()
    _forward(
        local_rank=local_rank,
        mesh_device=mesh_device,
        embed=embed,
        layers=layers,
        kvpe_cache=kvpe_cache,
        rope_setup=rope_setup,
        start_layer=start_layer,
        token_ids_tt=warmup_tt,
        actual_isl=MAX_SEQ_LEN,
        placeholder_activation=warmup_placeholder,
    )
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(f"[rank {local_rank}] [prefill timing] task_id=WARMUP _forward() = {warmup_ms:.2f} ms")

    ttnn.distributed_context_barrier()

    logger.info(f"[rank {local_rank}] entering run loop")
    _run_loop(
        local_rank=local_rank,
        mesh_device=mesh_device,
        hf_config=hf_config,
        embed=embed,
        layers=layers,
        kvpe_cache=kvpe_cache,
        rope_setup=rope_setup,
        start_layer=start_layer,
    )

    ttnn.distributed_context_barrier()
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    logger.info(f"[rank {local_rank}] shutdown complete")


if __name__ == "__main__":
    main()
