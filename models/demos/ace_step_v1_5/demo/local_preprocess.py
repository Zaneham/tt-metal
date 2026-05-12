"""
Local conditioning pipeline — replaces ``AceStepHandler`` + ``official_lm_preprocess.py``.

Loads HuggingFace models directly, runs the local 5 Hz LM, then calls the HF
model's ``prepare_condition()`` to produce the conditioning tensors needed by
the TTNN diffusion loop.  No ACE-Step repo on ``sys.path`` required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from models.demos.ace_step_v1_5.demo.local_lm import generate_metadata_and_codes, load_lm, parse_audio_codes

_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
_CODES_PER_SECOND = 5


def _build_text_prompt(caption: str, metadata: dict[str, Any]) -> str:
    return f"""# Instruction
{_DIT_INSTRUCTION}

# Caption
{caption}

# Metas
{metadata}<|endoftext|>
"""


def _null_condition_emb(ace_model: torch.nn.Module) -> torch.Tensor:
    return ace_model.null_condition_emb.detach().clone()


def prepare_conditioning(
    *,
    prompt: str,
    lyrics: str,
    duration_sec: float,
    ckpt_dir: str | Path,
    variant: str,
    lm_variant: str,
    seed: int,
    thinking: bool = True,
    temperature: float = 0.85,
    top_p: float | None = 0.9,
    torch_dev: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, dict[str, Any]]:
    """
    Run the full conditioning pipeline locally (no ACE repo needed).

    Returns
    -------
    enc_hs : Tensor  – encoder hidden states
    enc_mask : Tensor  – encoder attention mask
    ctx_lat : Tensor  – context latents
    frames : int  – number of latent frames (duration_sec * 25)
    null_emb : Tensor  – null condition embedding for CFG
    metadata : dict  – parsed LM metadata (bpm, caption, duration, keyscale, …)
    """
    ckpt_dir = Path(ckpt_dir)
    if torch_dev is None:
        torch_dev = torch.device("cpu")

    model_dir = ckpt_dir / variant
    text_model_dir = ckpt_dir / "Qwen3-Embedding-0.6B"
    lm_dir = ckpt_dir / lm_variant
    silence_latent_path = model_dir / "silence_latent.pt"

    # ── 1. Run local LLM (CoT metadata + audio codes) ──
    print("[local_preprocess] Loading 5 Hz LM …", flush=True)
    lm_model, lm_tokenizer = load_lm(str(lm_dir), device=str(torch_dev))
    lm_result = generate_metadata_and_codes(
        lm_model,
        lm_tokenizer,
        caption=prompt,
        lyrics=lyrics or "[Instrumental]",
        duration_sec=duration_sec,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    del lm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not lm_result["success"]:
        raise RuntimeError(f"LM generation failed: {lm_result['error']}")

    metadata = lm_result["metadata"]
    audio_codes_str = lm_result["audio_codes"]

    if metadata.get("caption"):
        prompt = metadata["caption"]
    if metadata.get("duration"):
        try:
            duration_sec = float(metadata["duration"])
        except (ValueError, TypeError):
            pass

    frames = int(round(duration_sec * 25.0))
    if frames <= 0:
        raise ValueError("duration_sec must be > 0")

    # ── 2. Load text encoder (Qwen3-Embedding) ──
    print("[local_preprocess] Loading Qwen3 text encoder …", flush=True)
    tok = AutoTokenizer.from_pretrained(str(text_model_dir))
    txt_model = AutoModel.from_pretrained(str(text_model_dir)).eval().to(torch_dev)

    metas = {"caption": prompt, "duration": float(duration_sec), "language": metadata.get("language", "en")}
    text_prompt = _build_text_prompt(prompt, metas)
    tokens = tok(text_prompt, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids = tokens["input_ids"].to(torch_dev)
    attn_mask = tokens["attention_mask"].to(torch_dev).to(torch.bool)
    with torch.inference_mode():
        text_out = txt_model(input_ids=input_ids, attention_mask=attn_mask)
        text_hidden_states = text_out.last_hidden_state
    del txt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 3. Load ACE-Step DiT model (for prepare_condition + detokenize) ──
    print("[local_preprocess] Loading ACE-Step DiT model …", flush=True)
    ace = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True).eval().to(torch_dev)

    # ── 4. Build precomputed_lm_hints_25Hz from audio codes ──
    precomputed_lm_hints_25Hz: torch.Tensor | None = None
    code_ids = parse_audio_codes(audio_codes_str)
    if code_ids:
        codes_tensor = torch.tensor(code_ids, dtype=torch.long, device=torch_dev).unsqueeze(0)
        with torch.inference_mode():
            lm_hints_5Hz = ace.tokenize.quantizer.get_output_from_indices(codes_tensor)
            precomputed_lm_hints_25Hz = ace.detokenize(lm_hints_5Hz)

    # ── 5. Build remaining tensors ──
    silence = torch.load(str(silence_latent_path), map_location="cpu").to(torch.float32)
    if silence.ndim != 3:
        raise RuntimeError(f"Unexpected silence_latent rank: {tuple(silence.shape)}")
    if int(silence.shape[-1]) == 64:
        pass
    elif int(silence.shape[1]) == 64:
        silence = silence.transpose(1, 2).contiguous()
    else:
        raise RuntimeError(f"Unexpected silence_latent shape: {tuple(silence.shape)}")

    src_latents = silence[:, :frames, :].contiguous()
    if src_latents.shape[1] < frames:
        rep = (frames + src_latents.shape[1] - 1) // src_latents.shape[1]
        src_latents = src_latents.repeat(1, rep, 1)[:, :frames, :].contiguous()

    B = 1
    lyric_dim = int(text_hidden_states.shape[-1])
    lyric_hidden_states = torch.zeros((B, 1, lyric_dim), dtype=torch.float32, device=torch_dev)
    lyric_attention_mask = torch.ones((B, 1), dtype=torch.bool, device=torch_dev)
    refer_audio_packed = torch.zeros((B, 1, 64), dtype=torch.float32, device=torch_dev)
    refer_audio_order_mask = torch.zeros((B,), dtype=torch.long, device=torch_dev)
    latent_attention_mask = torch.ones((B, frames), dtype=torch.float32, device=torch_dev)
    chunk_masks = torch.ones((B, frames, 64), dtype=torch.float32)

    # ── 6. Call prepare_condition ──
    with torch.inference_mode():
        enc_hs, enc_mask, ctx_lat = ace.prepare_condition(
            text_hidden_states=text_hidden_states.to(dtype=torch.float32),
            text_attention_mask=attn_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=src_latents.to(device=torch_dev, dtype=torch.float32),
            attention_mask=latent_attention_mask,
            silence_latent=silence.to(device=torch_dev, dtype=torch.float32),
            src_latents=src_latents.to(device=torch_dev, dtype=torch.float32),
            chunk_masks=chunk_masks.to(device=torch_dev, dtype=torch.float32),
            is_covers=torch.zeros((B,), dtype=torch.bool, device=torch_dev),
            precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
        )

    null_emb = _null_condition_emb(ace)
    del ace
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    enc_hs = enc_hs.float().cpu()
    enc_mask = enc_mask.float().cpu()
    ctx_lat = ctx_lat.float().cpu()
    null_emb = null_emb.float().cpu()
    return enc_hs, enc_mask, ctx_lat, frames, null_emb, metadata
