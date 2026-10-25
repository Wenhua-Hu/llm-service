# llm-service

An OpenAI-compatible model service gateway for the OpenLabs LLM-server project.

## Current implementation status

Implemented locally so far:

- `GET /healthz`, authenticated `GET /info`, and authenticated `GET /v1/models`
- bearer-key authentication backed by a SQLite WAL identity store
- `POST /v1/chat/completions` forwarding to the configured engine without rewriting sampling parameters
- incremental SSE forwarding, explicit stream errors when `[DONE]` is missing, and best-effort engine abort on an incomplete stream
- an async bounded admission gate configured with `GATEWAY_SLOTS` and `GATEWAY_QUEUE`
- authenticated `/metrics`, an admin console connected to tenant/key/credit/pricing/usage/audit APIs, and `/agent/run` planner → `lookup_order` → final-answer execution
- `setup.sh` starts a configured SGLang engine and gateway; `--contract-test` starts only the gateway against the supplied CPU fixture, and `--stop` stops project-owned processes

This is an in-progress implementation, not a completed OpenLabs submission. SGLang launch/supervision and the KV, RadixAttention, and Attention experiment entrypoints are implemented locally but have not yet been run on the Daytona GPU. Baseline/optimized profile values, GPU evidence, and official soft/GPU evaluation remain outstanding. Tenant administration, key rotation/revocation, metering, quotas, the working console, and a first Agent runner are implemented locally but still need the official checker and broader concurrency/recovery coverage.

## Project layout mapping

| Taskbook responsibility | Current implementation |
|---|---|
| Gateway and OpenAI API | `server/app.py` |
| Engine HTTP adapter and usage tokenization | `server/engine.py` |
| Admission control | `server/admission.py` |
| SSE event parsing, error events, usage validation | `server/streaming.py` (following the reference project's streaming boundary) |
| Runtime metrics | `server/metrics.py` |
| Process startup, readiness, and stop | `setup.sh` delegates owned engine/gateway process lifecycle to `server/supervisor.py` |
| Tenant, key, quota, billing, usage, migration recovery | `control_plane/storage.py` (these concerns are not yet split into separate modules) |
| Admin web console and schema | `console/`, `migrations/` |
| Agent planning, whitelisted tool, final answer | `agent/runner.py`; `/agent/run` route in `server/app.py` delegates to it |
| Tenant/key/usage/audit/pricing admin routes | `control_plane/api.py`; database operations remain in `control_plane/storage.py` |
| Profile configuration | `configs/baseline.yaml` and `configs/optimized.yaml`; both are marked pending GPU validation and cannot yet launch a measured profile |
| Experimental pricing | `configs/pricing.json`; CPU fixture accounting only, not a commercial price |
| CPU/API and storage verification | `tests/`; public service-level control-plane checker is still pending |
| Performance method and evidence | `docs/performance.md`, `experiments/`; implementations exist, but no GPU measurements or results are recorded yet |

The taskbook layout is the target. This table records current path differences; it does not mark missing modules or measurements as complete.

The repository's `.github/workflows/llm-soft.yml` runs CPU diagnostics on GitHub Actions. The current local workflow change installs a pinned `uv` version because the previous Actions run was blocked before service startup when `uv` was missing. It has not been pushed or rerun yet.

## Local development

Local development uses Python 3.13 and `uv`. The code remains compatible with
the course's Python 3.12 Linux environment.

```bash
uv sync
uv run pytest
export API_KEY=local-development-key
export KEY_HASH_SECRET=local-development-pepper-32-bytes-minimum
export DATA_DIR="$PWD/.local-state"
export MODEL_ID=local-model
export MODEL_REVISION=development
uv run uvicorn server.app:app --reload --port 8000
```

Then check the service:

```bash
curl http://127.0.0.1:8000/healthz
curl -H 'Authorization: Bearer local-development-key' \
  http://127.0.0.1:8000/info
curl -H 'Authorization: Bearer local-development-key' \
  http://127.0.0.1:8000/v1/models
```

## Contract-test entry point

```bash
bash setup.sh --contract-test
# Stop only the gateway PID recorded under DATA_DIR; the ledger is retained.
bash setup.sh --stop
```

Configuration is read from environment variables. `--contract-test` requires the
external CPU fixture URL in `ENGINE_BASE_URL`; it must not start SGLang or download
weights. Normal startup installs the pinned SGLang runtime; the `optimized`
profile also installs the pinned FlashInfer extra. It downloads model weights on
first engine launch. Use a Linux NVIDIA GPU host for this path; the profile values
are still unvalidated until the Daytona run. Set `HOST=0.0.0.0` only when the API
should accept connections from outside the machine. `DATA_DIR` stores logs, PID
identity, and the SQLite ledger and is retained by `--stop`. Secrets must not be
committed to this repository.

For a normal SGLang-backed launch, set the required API key, key-hash secret,
model ID/revision and a dedicated absolute data directory, then run:

```bash
export API_KEY=local-development-key
export KEY_HASH_SECRET=local-development-pepper-32-bytes-minimum
export DATA_DIR="$PWD/.local-state"
export MODEL_ID=Qwen/Qwen2.5-7B-Instruct
export MODEL_REVISION=a09a35458c702b33eeacc393d103063234e8bc28
# Optional: baseline (default) or optimized; expose the API beyond localhost with HOST=0.0.0.0
bash setup.sh
```

The local test suite can be run with:

```bash
uv run --frozen pytest
```
