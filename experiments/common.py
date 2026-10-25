"""Shared evidence handling for the three GPU mechanism experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


class ExperimentError(RuntimeError):
    """An experiment could not produce complete, trustworthy evidence."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True).encode())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_identity() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    value = result.stdout.rstrip("\r\n")
    if not value:
        raise ExperimentError("nvidia-smi returned no GPU identity")
    return value


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5
    )
    value = result.stdout.strip()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ExperimentError("could not identify a full 40-character Git commit")
    return value


def new_output_dir(value: str) -> Path:
    out = Path(value).expanduser()
    if out.exists() or out.is_symlink():
        raise ExperimentError(f"output directory already exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def base_manifest(case: str, model: str, revision: str | None, parameters: dict[str, Any]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "case": case,
        "time": utc_now(),
        "commit": git_commit(),
        "model": model,
        "revision": revision,
        "gpu": gpu_identity(),
        "software": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "sglang": package_version("sglang"),
        },
        "cuda": None,
        "complete": False,
        "parameters": parameters,
    }
    try:
        import torch

        manifest["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return manifest


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def sync_cuda(torch: Any) -> None:
    torch.cuda.synchronize()


def reset_cuda_peaks(torch: Any) -> None:
    torch.cuda.reset_peak_memory_stats()


def cuda_peaks(torch: Any) -> tuple[int, int]:
    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().contiguous().view(__import__("torch").uint8).cpu().numpy().tobytes()
    return sha256_bytes(value)


def stop_project_service_if_running() -> None:
    """Release this project's model and gateway before a GPU experiment."""
    try:
        with urlopen("http://127.0.0.1:30000/health", timeout=1) as response:
            if not 200 <= response.status < 300:
                return
    except (OSError, URLError, TimeoutError):
        return

    if not os.environ.get("DATA_DIR"):
        raise ExperimentError(
            "the project SGLang engine is running, but DATA_DIR is unset; "
            "refusing to stop an untracked process or load a second model"
        )
    result = subprocess.run(
        ["bash", "setup.sh", "--stop"],
        cwd=Path(__file__).resolve().parents[1],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    if result.returncode != 0:
        raise ExperimentError(f"setup.sh --stop failed: {(result.stderr or result.stdout)[-2000:]}")
    try:
        with urlopen("http://127.0.0.1:30000/health", timeout=1) as response:
            if 200 <= response.status < 300:
                raise ExperimentError("setup.sh --stop returned success, but SGLang is still healthy")
    except (OSError, URLError, TimeoutError):
        pass


def run_with_manifest(
    case: str,
    out_value: str,
    model: str,
    revision: str | None,
    parameters: dict[str, Any],
    runner: Any,
) -> int:
    out: Path | None = None
    manifest: dict[str, Any] | None = None
    previous_handler: Any = None
    try:
        if hasattr(signal, "SIGTERM"):
            previous_handler = signal.getsignal(signal.SIGTERM)

            def interrupted(signum: int, frame: Any) -> None:
                raise ExperimentError(f"experiment interrupted by signal {signum}")

            signal.signal(signal.SIGTERM, interrupted)
        out = new_output_dir(out_value)
        manifest = base_manifest(case, model, revision, parameters)
        passed = runner(out, manifest)
        manifest["complete"] = True
        write_json(out / "manifest.json", manifest)
        return 0 if passed else 1
    except Exception as exc:
        if out is not None and manifest is not None:
            manifest["complete"] = False
            manifest["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            try:
                write_json(out / "manifest.json", manifest)
            except Exception as write_error:
                print(f"could not write manifest: {write_error}", file=sys.stderr)
        print(f"{case} experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
