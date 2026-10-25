"""Measure SGLang RadixAttention reuse with identical, recorded token IDs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from experiments.common import (
    ExperimentError,
    append_jsonl,
    run_with_manifest,
    sha256_json,
    stop_project_service_if_running,
    write_json,
)


PREFIX_TOKENS = 512
REQUESTS = 8
REPEATS = 3
INPUT_TOKENS = 1024
OUTPUT_TOKENS = 32
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def _tokenizer_inputs(tokenizer: Any) -> dict[str, list[list[int]]]:
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], tokenize=True, add_generation_prompt=True
    )
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    try:
        end_index = template.index(end_id)
    except ValueError as exc:
        raise ExperimentError("could not locate the user message boundary in the model chat template") from exc
    skeleton_prefix, skeleton_suffix = template[:end_index], template[end_index:]
    content_length = INPUT_TOKENS - len(skeleton_prefix) - len(skeleton_suffix)
    if content_length <= PREFIX_TOKENS:
        raise ExperimentError("chat template leaves too few content tokens for the shared prefix")

    split = PREFIX_TOKENS - len(skeleton_prefix)
    if split < 0:
        raise ExperimentError("chat template is longer than the measured prefix")

    def make_content(text: str, marker: str = "") -> list[int]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        marker_ids = tokenizer.encode(marker, add_special_tokens=False) if marker else []
        filler = tokenizer.encode(" repeated", add_special_tokens=False)
        if not ids or not filler:
            raise ExperimentError("tokenizer failed to encode deterministic prefix text")
        shared_content = list(ids)
        while len(shared_content) < split:
            shared_content.extend(filler)
        content = shared_content[:split] + marker_ids
        while len(content) < content_length:
            content.extend(filler)
        content = content[:content_length]
        return skeleton_prefix + content + skeleton_suffix

    shared_text = "Respond with exactly OK. Shared experiment context."
    shared = [
        make_content(shared_text, f" Request index {index}.") for index in range(REQUESTS)
    ]
    unique = [
        make_content(f"Respond with exactly OK. Unique prefix group {index}. Experiment context.")
        for index in range(REQUESTS)
    ]
    if any(len(ids) != INPUT_TOKENS for ids in shared + unique):
        raise ExperimentError("generated prefix prompts do not have equal token lengths")
    if len({tuple(ids[:PREFIX_TOKENS]) for ids in shared}) != 1:
        raise ExperimentError("shared prompts do not share exactly the required token prefix")
    if len({tuple(ids[:PREFIX_TOKENS]) for ids in unique}) != REQUESTS:
        raise ExperimentError("the eight unique prompt prefixes are not distinct")
    return {"shared": shared, "unique": unique}


def _start_server(model: str, revision: str, enabled: bool, log_path: Path) -> subprocess.Popen:
    port = int(os.environ.get("PREFIX_SERVER_PORT", "30000"))
    command = [
        sys.executable,
        "-m", "sglang.launch_server",
        "--model-path", model,
        "--revision", revision,
        "--served-model-name", model,
        "--dtype", "bfloat16",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--max-running-requests", "8",
        "--mem-fraction-static", "0.8",
    ]
    if not enabled:
        command.append("--disable-radix-cache")
    env = os.environ.copy()
    env.pop("ENGINE_BASE_URL", None)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    deadline = time.monotonic() + float(os.environ.get("PREFIX_START_TIMEOUT", "180"))
    url = f"http://127.0.0.1:{port}/health"
    import httpx

    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(errors="replace")[-4000:]
            raise ExperimentError(
                f"SGLang exited before becoming ready (exit {process.returncode}); log tail: {tail}"
            )
        try:
            response = httpx.get(url, timeout=2)
            if response.is_success:
                return process
        except httpx.HTTPError:
            pass
        time.sleep(1)
    _stop_server(process)
    tail = log_path.read_text(errors="replace")[-4000:]
    raise ExperimentError(f"SGLang did not become ready before the startup timeout; log tail: {tail}")


def _stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _flush(client: Any, out: Path, variant: str, group: str, repeat: int, base_url: str) -> None:
    response = client.post(f"{base_url}/flush_cache", timeout=30)
    evidence = {"status": response.status_code, "body": response.text[:2000]}
    write_json(out / f"{variant}-{group}-{repeat}-flush.json", evidence)
    response.raise_for_status()


def _request(client: Any, base_url: str, input_ids: list[int]) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": OUTPUT_TOKENS,
        },
        "stream": True,
    }
    started = time.perf_counter()
    ttft_ms: float | None = None
    output = ""
    meta_info: dict[str, Any] = {}
    with client.stream("POST", f"{base_url}/generate", json=payload, timeout=120) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data_line = line[5:].strip() if line.startswith("data:") else line.strip()
            if data_line == "[DONE]":
                continue
            try:
                chunk = json.loads(data_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text")
            if isinstance(text, str):
                if text and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1000
                # SGLang streaming responses may contain either deltas or cumulative text.
                output = text if text.startswith(output) else output + text
            info = chunk.get("meta_info")
            if isinstance(info, dict):
                meta_info.update(info)
    e2e_ms = (time.perf_counter() - started) * 1000
    if ttft_ms is None:
        raise ExperimentError("stream returned no text chunk; TTFT is unavailable")
    cached_tokens = meta_info.get("cached_tokens")
    if not isinstance(cached_tokens, int):
        raise ExperimentError("SGLang response did not report integer meta_info.cached_tokens")
    return {"output": output, "ttft_ms": ttft_ms, "e2e_ms": e2e_ms,
            "cached_tokens": cached_tokens, "meta_info": meta_info}


def run(out: Path, manifest: dict[str, Any]) -> bool:
    stop_project_service_if_running()
    from transformers import AutoTokenizer
    import httpx

    model = os.environ.get("MODEL_ID", DEFAULT_MODEL)
    revision = os.environ.get("MODEL_REVISION", DEFAULT_REVISION)
    if model != DEFAULT_MODEL or revision != DEFAULT_REVISION:
        raise ExperimentError("prefix experiment requires the taskbook's frozen 7B model and revision")
    manifest["model"] = model
    manifest["revision"] = revision
    manifest["parameters"].update(
        {"prefix_tokens": PREFIX_TOKENS, "requests": REQUESTS, "repeats": REPEATS,
         "input_tokens": INPUT_TOKENS, "max_new_tokens": OUTPUT_TOKENS,
         "dtype": "bfloat16", "cache_variants": ["on", "off"],
         "groups": ["shared", "unique"]}
    )
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    inputs = _tokenizer_inputs(tokenizer)
    write_json(out / "inputs.json", inputs)

    raw_path = out / "raw.jsonl"
    rows: list[dict[str, Any]] = []
    passed = True
    port = int(os.environ.get("PREFIX_SERVER_PORT", "30000"))
    base_url = f"http://127.0.0.1:{port}"
    group_elapsed: dict[str, float] = {}
    for variant in ("on", "off"):
        process: subprocess.Popen | None = None
        try:
            process = _start_server(
                model, revision, enabled=(variant == "on"),
                log_path=out / f"sglang-{variant}.log",
            )
            with httpx.Client(timeout=120) as client:
                for group in ("shared", "unique"):
                    for repeat in range(REPEATS):
                        group_started = time.perf_counter()
                        _flush(client, out, variant, group, repeat, base_url)
                        for index, token_ids in enumerate(inputs[group]):
                            result = _request(client, base_url, token_ids)
                            cache_state = (
                                "cold" if index == 0 else
                                "hot" if variant == "on" and group == "shared" else
                                "unique" if group == "unique" else "uncached"
                            )
                            prefix_hash = sha256_json(token_ids[:PREFIX_TOKENS])
                            row = {
                                "variant": variant,
                                "group": group,
                                "repeat": repeat,
                                "index": index,
                                "input_hash": sha256_json(token_ids),
                                "prefix_hash": prefix_hash,
                                "prefix_length": PREFIX_TOKENS,
                                "input_tokens": len(token_ids),
                                "cached_tokens": result["cached_tokens"],
                                "cache_state": cache_state,
                                "output": result["output"],
                                "transport_ok": True,
                                "ttft_ms": result["ttft_ms"],
                                "e2e_ms": result["e2e_ms"],
                            }
                            append_jsonl(raw_path, row)
                            rows.append(row)
                            if result["output"].strip() != "OK":
                                passed = False
                            if variant == "off" and result["cached_tokens"] != 0:
                                passed = False
                            if index == 0 and result["cached_tokens"] >= PREFIX_TOKENS:
                                passed = False
                            if group == "shared" and variant == "on" and index > 0:
                                if result["cached_tokens"] < PREFIX_TOKENS:
                                    passed = False
                            if group == "unique" and index > 0 and result["cached_tokens"] >= PREFIX_TOKENS:
                                passed = False
                        group_elapsed[f"{variant}-{group}"] = (
                            group_elapsed.get(f"{variant}-{group}", 0.0)
                            + time.perf_counter() - group_started
                        )
        finally:
            _stop_server(process)

    write_json(
        out / "summary.json",
        {"case": "prefix", "rows": len(rows), "passed": passed,
         "groups": {
             f"{variant}-{group}": {
                 "requests": sum(1 for row in rows if row["variant"] == variant and row["group"] == group),
                 "success_rate": sum(1 for row in rows if row["variant"] == variant and row["group"] == group and row["transport_ok"]) / (REPEATS * REQUESTS),
                 "quality_goodput_per_s": (
                     sum(1 for row in rows if row["variant"] == variant and row["group"] == group and row["output"].strip() == "OK")
                     / group_elapsed[f"{variant}-{group}"]
                     if group_elapsed.get(f"{variant}-{group}", 0.0) > 0 else 0.0
                 ),
             }
             for variant in ("on", "off") for group in ("shared", "unique")
         }},
    )
    return passed and len(rows) == 96


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model = os.environ.get("MODEL_ID", DEFAULT_MODEL)
    revision = os.environ.get("MODEL_REVISION", DEFAULT_REVISION)
    return run_with_manifest(
        "prefix", args.out, model, revision,
        {"prefix_tokens": PREFIX_TOKENS, "requests": REQUESTS, "repeats": REPEATS}, run
    )


if __name__ == "__main__":
    raise SystemExit(main())
