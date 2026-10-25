"""Math versus CUDA FlashAttention using deterministic synthetic tensors."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from experiments.common import (
    ExperimentError,
    append_jsonl,
    run_with_manifest,
    stop_project_service_if_running,
    sync_cuda,
    tensor_sha256,
    write_json,
)


LENGTHS = (128, 512, 1024)
REPEATS = 3
WARMUP = 5
ITERATIONS = 10
ATOL = 0.02
RTOL = 0.02


def _attention(torch: Any, q: Any, k: Any, v: Any, backend: Any) -> Any:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    selected = SDPBackend.MATH if backend == "math" else SDPBackend.FLASH_ATTENTION
    with sdpa_kernel(selected):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )


def _profile(
    torch: Any, q: Any, k: Any, v: Any, backend: str, out: Path, length: int
) -> list[str]:
    from torch.profiler import ProfilerActivity, profile

    sync_cuda(torch)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _attention(torch, q, k, v, backend)
        sync_cuda(torch)
    names = sorted({event.key for event in prof.key_averages()})
    if backend == "flash" and not any("flash" in name.lower() for name in names):
        raise ExperimentError(f"FlashAttention kernel was not recorded for length {length}")
    if backend == "math" and not any(
        "attention_math" in name.lower() or "bmm" in name.lower() for name in names
    ):
        raise ExperimentError(f"math attention operators were not recorded for length {length}")
    write_json(out / f"{backend}-{length}-operators.json", names)
    prof.export_chrome_trace(str(out / f"{backend}-{length}-trace.json"))
    return names


def run(out: Path, manifest: dict[str, Any]) -> bool:
    stop_project_service_if_running()
    import torch

    if not torch.cuda.is_available():
        raise ExperimentError("CUDA is required for the attention experiment")

    torch.manual_seed(20261018)
    torch.cuda.manual_seed_all(20261018)
    manifest["parameters"].update(
        {"lengths": list(LENGTHS), "warmup": WARMUP, "groups": len(LENGTHS),
         "iterations": ITERATIONS, "dtype": "bfloat16", "causal": True,
         "dropout": 0.0, "batch": 1, "heads": 8, "head_dim": 64}
    )
    raw_path = out / "raw.jsonl"
    rows: list[dict[str, Any]] = []
    passed = True

    for length in LENGTHS:
        generator = torch.Generator(device="cuda").manual_seed(20261018 + length)
        q, k, v = [
            torch.randn((1, 8, length, 64), generator=generator, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        hashes = {name: tensor_sha256(value) for name, value in (("q", q), ("k", k), ("v", v))}
        reference = _attention(torch, q, k, v, "math")
        _profile(torch, q, k, v, "math", out, length)
        flash = _attention(torch, q, k, v, "flash")
        _profile(torch, q, k, v, "flash", out, length)
        max_error = float((reference.float() - flash.float()).abs().max().item())
        passed = passed and max_error <= ATOL

        for variant in ("math", "flash"):
            for _ in range(WARMUP):
                _attention(torch, q, k, v, variant)
            sync_cuda(torch)
            for repeat in range(REPEATS):
                torch.cuda.reset_peak_memory_stats()
                sync_cuda(torch)
                start = time.perf_counter()
                for _iteration in range(ITERATIONS):
                    _attention(torch, q, k, v, variant)
                sync_cuda(torch)
                elapsed_ms = (time.perf_counter() - start) * 1000 / ITERATIONS
                row = {
                    "variant": variant,
                    "length": length,
                    "repeat": repeat,
                    "shape": [1, 8, length, 64],
                    "output_shape": list(reference.shape),
                    "input_hashes": hashes,
                    "elapsed_ms": elapsed_ms,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "max_abs_error": 0.0 if variant == "math" else max_error,
                    "atol": ATOL,
                    "rtol": RTOL,
                }
                append_jsonl(raw_path, row)
                rows.append(row)

    write_json(
        out / "summary.json",
        {"case": "attention", "rows": len(rows), "passed": passed,
         "max_abs_error": max((row["max_abs_error"] for row in rows), default=None),
         "atol": ATOL, "rtol": RTOL},
    )
    return passed and len(rows) == 18


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    parameters: dict[str, Any] = {}
    return run_with_manifest("attention", args.out, "synthetic-QKV", None, parameters, run)


if __name__ == "__main__":
    raise SystemExit(main())
