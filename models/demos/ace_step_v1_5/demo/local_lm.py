"""
Local 5 Hz LM inference — replaces ``acestep.llm_inference.LLMHandler``.

Loads the Qwen3-based 5 Hz LM via ``AutoModelForCausalLM`` and runs two-phase
generation (CoT metadata + audio codes) without any ACE-Step repo dependency.
"""

from __future__ import annotations

import re
import time
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

_LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"
_MAX_AUDIO_CODE = 63999
_CODES_PER_SECOND = 5


def load_lm(
    checkpoint_path: str,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if dtype is None:
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        .eval()
        .to(device)
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def _build_prompt(tokenizer: AutoTokenizer, caption: str, lyrics: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": f"# Instruction\n{_LM_INSTRUCTION}\n\n"},
            {"role": "user", "content": f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _build_prompt_with_cot(
    tokenizer: AutoTokenizer,
    caption: str,
    lyrics: str,
    cot_text: str,
) -> str:
    base = _build_prompt(tokenizer, caption, lyrics)
    return base + cot_text + "\n\n"


def _format_metadata_as_cot(metadata: dict[str, Any]) -> str:
    cot_items: dict[str, Any] = {}
    for key in ("bpm", "caption", "duration", "keyscale", "language", "timesignature"):
        val = metadata.get(key)
        if val is None:
            continue
        if key == "timesignature" and isinstance(val, str) and val.endswith("/4"):
            val = val.split("/")[0]
        if isinstance(val, str) and val.isdigit():
            val = int(val)
        cot_items[key] = val
    cot_yaml = yaml.dump(cot_items, allow_unicode=True, sort_keys=True).strip() if cot_items else ""
    return f"<think>\n{cot_yaml}\n</think>"


def _parse_metadata(text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if not match:
        return metadata
    reasoning = match.group(1).strip()
    current_key: str | None = None
    current_lines: list[str] = []

    def _save():
        nonlocal current_key, current_lines
        if not current_key or not current_lines:
            current_key = None
            current_lines = []
            return
        value = "\n".join(current_lines).strip()
        if current_key == "bpm":
            try:
                metadata["bpm"] = int(value)
            except (ValueError, TypeError):
                metadata["bpm"] = value
        elif current_key == "caption":
            metadata["caption"] = value
        elif current_key == "duration":
            try:
                metadata["duration"] = int(value)
            except (ValueError, TypeError):
                metadata["duration"] = value
        elif current_key == "genres":
            metadata["genres"] = value
        elif current_key == "keyscale":
            metadata["keyscale"] = value
        elif current_key == "language":
            metadata["language"] = value
        elif current_key == "timesignature":
            metadata["timesignature"] = value
        current_key = None
        current_lines = []

    for line in reasoning.split("\n"):
        if line.strip().startswith("<"):
            continue
        if line and not line[0].isspace() and ":" in line:
            _save()
            parts = line.split(":", 1)
            current_key = parts[0].strip().lower()
            rest = parts[1]
            if rest.strip():
                current_lines.append(rest)
        elif current_key is not None:
            current_lines.append(line)
    _save()
    return metadata


def parse_audio_codes(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"<\|audio_code_(\d+)\|>", text)]


def _build_audio_code_mask(
    tokenizer: AutoTokenizer,
    full_vocab_size: int | None = None,
) -> tuple[set[int], torch.Tensor]:
    """Build set of valid audio-code token IDs and a logit mask that blocks everything else.

    *full_vocab_size* should be the model's output logit dimension (which may
    exceed ``tokenizer.vocab_size`` when special tokens like ``<|audio_code_N|>``
    have been added).  If not given, ``len(tokenizer)`` is used.
    """
    if full_vocab_size is None:
        full_vocab_size = len(tokenizer)
    pattern = re.compile(r"^<\|audio_code_(\d+)\|>$")
    code_ids: set[int] = set()
    for tid in range(full_vocab_size):
        try:
            tok_text = tokenizer.decode([tid])
            m = pattern.match(tok_text)
            if m and 0 <= int(m.group(1)) <= _MAX_AUDIO_CODE:
                code_ids.add(tid)
        except Exception:
            continue
    mask = torch.full((1, full_vocab_size), float("-inf"), dtype=torch.float32)
    for tid in code_ids:
        mask[0, tid] = 0.0
    eos = tokenizer.eos_token_id
    if eos is not None and eos < mask.shape[1]:
        mask[0, eos] = 0.0
    return code_ids, mask


@torch.inference_mode()
def _generate_until_stop(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    stop_token_ids: set[int] | None = None,
    stop_string: str | None = None,
    tokenizer: AutoTokenizer | None = None,
    logit_mask: torch.Tensor | None = None,
    block_eos_until: int = 0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Simple autoregressive generation loop with optional logit masking and stop conditions."""
    device = input_ids.device
    generated: list[int] = []
    past_key_values = None
    cur_ids = input_ids

    if logit_mask is not None:
        logit_mask = logit_mask.to(device)

    for _ in range(max_new_tokens):
        out = model(input_ids=cur_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
        logits = out.logits[:, -1, :].float()
        past_key_values = out.past_key_values

        if logit_mask is not None:
            logits = logits + logit_mask

        if block_eos_until > 0 and eos_token_id is not None and len(generated) < block_eos_until:
            logits[:, eos_token_id] = float("-inf")

        if temperature > 0:
            logits = logits / temperature
        if top_k is not None and top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k, dim=-1)
            logits[logits < topk_vals[:, -1:]] = float("-inf")
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            mask = cumprobs - sorted_logits.softmax(dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = logits.softmax(dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tid = int(next_token.item())
        generated.append(tid)

        if stop_token_ids and tid in stop_token_ids:
            break
        if eos_token_id is not None and tid == eos_token_id and len(generated) >= block_eos_until:
            break
        if stop_string and tokenizer is not None:
            tail = tokenizer.decode(generated[-20:], skip_special_tokens=False)
            if stop_string in tail:
                break

        cur_ids = next_token
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)], dim=1
            )

    return torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)


def generate_metadata_and_codes(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    caption: str,
    lyrics: str,
    duration_sec: float,
    *,
    temperature: float = 0.85,
    top_k: int | None = None,
    top_p: float | None = 0.9,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Two-phase generation mirroring ``LLMHandler.generate_with_stop_condition``.

    Returns dict with keys: metadata, audio_codes, success, error, extra_outputs.
    """
    device = next(model.parameters()).device

    if seed is not None:
        torch.manual_seed(seed)

    # ── Phase 1: CoT metadata ──
    t0 = time.time()
    prompt_text = _build_prompt(tokenizer, caption, lyrics)
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    gen_ids = _generate_until_stop(
        model,
        input_ids,
        attn_mask,
        max_new_tokens=1024,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_string="</think>",
        tokenizer=tokenizer,
    )
    phase1_text = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
    phase1_time = time.time() - t0
    print(f"[local_lm] Phase 1 done in {phase1_time:.1f}s", flush=True)

    if "</think>" not in phase1_text:
        phase1_text += "\n</think>"
    metadata = _parse_metadata(phase1_text)

    if (duration_sec is None or duration_sec <= 0) and metadata.get("duration"):
        try:
            d = float(metadata["duration"])
            if d > 0:
                duration_sec = d
        except (ValueError, TypeError):
            pass
    if duration_sec is None or duration_sec <= 0:
        duration_sec = 10.0

    # ── Phase 2: audio codes ──
    t1 = time.time()
    cot_text = _format_metadata_as_cot(metadata)
    codes_prompt = _build_prompt_with_cot(tokenizer, caption, lyrics, cot_text)
    inputs2 = tokenizer(codes_prompt, return_tensors="pt", padding=False, truncation=True)
    input_ids2 = inputs2["input_ids"].to(device)
    attn_mask2 = inputs2.get("attention_mask")
    if attn_mask2 is not None:
        attn_mask2 = attn_mask2.to(device)

    target_codes = int(round(duration_sec * _CODES_PER_SECOND))
    model_vocab_size = model.config.vocab_size
    _, codes_mask = _build_audio_code_mask(tokenizer, full_vocab_size=model_vocab_size)

    gen_ids2 = _generate_until_stop(
        model,
        input_ids2,
        attn_mask2,
        max_new_tokens=target_codes + 32,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        logit_mask=codes_mask,
        block_eos_until=target_codes,
        eos_token_id=tokenizer.eos_token_id,
    )
    phase2_text = tokenizer.decode(gen_ids2[0], skip_special_tokens=False)
    phase2_time = time.time() - t1
    codes_list = parse_audio_codes(phase2_text)
    audio_codes_str = "".join(f"<|audio_code_{c}|>" for c in codes_list)
    print(f"[local_lm] Phase 2 done in {phase2_time:.1f}s, {len(codes_list)} codes", flush=True)

    return {
        "metadata": metadata,
        "audio_codes": audio_codes_str,
        "success": True,
        "error": None,
        "extra_outputs": {
            "time_costs": {
                "phase1_time": phase1_time,
                "phase2_time": phase2_time,
                "total_time": phase1_time + phase2_time,
            },
            "codes_count": len(codes_list),
        },
    }
