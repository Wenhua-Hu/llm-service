"""Measure KV-cache memory and cached-versus-recomputed logits on a fixed model."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from experiments.common import (
    ExperimentError,
    append_jsonl,
    run_with_manifest,
    sha256_json,
    stop_project_service_if_running,
    sync_cuda,
    write_json,
)


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LENGTHS = (128, 512)
DECODE_TOKENS = 32
REPEATS = 3
WARMUP = 1
ATOL = 0.001
RTOL = 0.02


def _prompt_ids(tokenizer: Any, length: int) -> list[int]:
    text = "A deterministic language-model cache measurement record. " * (length * 3)
    ids = tokenizer.encode(text, add_special_tokens=True)
    if len(ids) < length:
        raise ExperimentError(f"tokenizer generated only {len(ids)} tokens for length {length}")
    return ids[:length]


def _generate_cached(torch: Any, model: Any, prompt: Any) -> tuple[list[int], list[Any], float, int, int]:
    torch.cuda.reset_peak_memory_stats()
    sync_cuda(torch)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(input_ids=prompt, use_cache=True)
        past = output.past_key_values
        next_logits = output.logits[:, -1, :]
        generated: list[int] = []
        logits: list[Any] = []
        for index in range(DECODE_TOKENS):
            logits.append(next_logits[0].float().cpu())
            token = int(next_logits.argmax(dim=-1).item())
            generated.append(token)
            if index + 1 < DECODE_TOKENS:
                token_tensor = torch.tensor([[token]], device=prompt.device, dtype=prompt.dtype)
                output = model(input_ids=token_tensor, past_key_values=past, use_cache=True)
                past = output.past_key_values
                next_logits = output.logits[:, -1, :]
    sync_cuda(torch)
    elapsed = (time.perf_counter() - started) * 1000
    allocated, reserved = int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())
    return generated, logits, elapsed, allocated, reserved


def _recompute(
    torch: Any, model: Any, prompt: Any, generated: list[int], cached_logits: list[Any]
) -> tuple[float, int, int, float, float]:
    torch.cuda.reset_peak_memory_stats()
    sync_cuda(torch)
    started = time.perf_counter()
    maximum_error = 0.0
    agreements = 0
    with torch.inference_mode():
        for index in range(DECODE_TOKENS):
            if index:
                suffix = torch.tensor([generated[:index]], device=prompt.device, dtype=prompt.dtype)
                tokens = torch.cat((prompt, suffix), dim=1)
            else:
                tokens = prompt
            logits = model(input_ids=tokens, use_cache=False).logits[0, -1, :].float().cpu()
            difference = (cached_logits[index] - logits).abs()
            maximum_error = max(maximum_error, float(difference.max().item()))
            agreements += int(int(cached_logits[index].argmax().item()) == int(logits.argmax().item()))
    sync_cuda(torch)
    elapsed = (time.perf_counter() - started) * 1000
    allocated, reserved = int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())
    return elapsed, allocated, reserved, maximum_error, agreements / DECODE_TOKENS


def run(out: Path, manifest: dict[str, Any]) -> bool:
    stop_project_service_if_running()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise ExperimentError("CUDA is required for the KV-cache experiment")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261018)
    torch.cuda.manual_seed_all(20261018)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()

    config = model.config
    layers = int(config.num_hidden_layers)
    kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    dtype_bytes = 4
    bytes_per_token = layers * 2 * kv_heads * head_dim * dtype_bytes
    manifest["model"] = MODEL
    manifest["revision"] = REVISION
    manifest["parameters"].update(
        {"lengths": list(LENGTHS), "decode_tokens": DECODE_TOKENS,
         "warmup": WARMUP, "repeats": REPEATS, "dtype": "float32",
         "tf32": False, "kv_layers": layers, "kv_heads": kv_heads,
         "head_dim": head_dim, "dtype_bytes": dtype_bytes}
    )

    raw_path = out / "raw.jsonl"
    measurements: list[dict[str, Any]] = []
    passed = True
    for length in LENGTHS:
        ids = _prompt_ids(tokenizer, length)
        input_hash = sha256_json(ids)
        prompt = torch.tensor([ids], device="cuda", dtype=torch.long)
        for _ in range(WARMUP):
            with torch.inference_mode():
                model(input_ids=prompt, use_cache=True)
                model(input_ids=prompt, use_cache=False)
        sync_cuda(torch)

        for repeat in range(REPEATS):
            generated, cached_logits, on_ms, on_allocated, on_reserved = _generate_cached(
                torch, model, prompt
            )
            off_ms, off_allocated, off_reserved, max_error, top1 = _recompute(
                torch, model, prompt, generated, cached_logits
            )
            continuation_hash = sha256_json(generated)
            for variant, elapsed, allocated, reserved, error in (
                ("cache_on", on_ms, on_allocated, on_reserved, 0.0),
                ("cache_off", off_ms, off_allocated, off_reserved, max_error),
            ):
                row = {
                    "variant": variant,
                    "length": length,
                    "repeat": repeat,
                    "input_hash": input_hash,
                    "continuation_hash": continuation_hash,
                    "decode_tokens": DECODE_TOKENS,
                    "dtype": "float32",
                    "elapsed_ms": elapsed,
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                    "max_abs_error": error,
                    "atol": ATOL,
                    "rtol": RTOL,
                    "top1_agreement": 1.0 if variant == "cache_on" else top1,
                }
                append_jsonl(raw_path, row)
                measurements.append(row)
            passed = passed and max_error <= ATOL and top1 == 1.0

    summary = {
        "case": "kv",
        "layers": layers,
        "KV_heads": kv_heads,
        "head_dim": head_dim,
        "dtype_bytes": dtype_bytes,
        "KV_bytes_per_token": bytes_per_token,
        "measurements": [
            {"length": length, "KV_bytes_one_request": bytes_per_token * (length + DECODE_TOKENS),
             "KV_bytes_eight_requests": bytes_per_token * (length + DECODE_TOKENS) * 8}
            for length in LENGTHS
        ],
        "rows": len(measurements),
        "passed": passed,
    }
    write_json(out / "summary.json", summary)
    return passed and len(measurements) == 12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return run_with_manifest(
        "kv", args.out, MODEL, REVISION,
        {"lengths": list(LENGTHS), "decode_tokens": DECODE_TOKENS,
         "warmup": WARMUP, "repeats": REPEATS, "dtype": "float32"}, run
    )


if __name__ == "__main__":
    raise SystemExit(main())
