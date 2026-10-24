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
- `setup.sh --contract-test` and `setup.sh --stop` entry points

This is an in-progress implementation, not a completed OpenLabs submission. SGLang launch/supervision, validated baseline/optimized profiles, benchmark and experiment entrypoints, GPU evidence, and official soft/GPU evaluation are still outstanding. Tenant administration, key rotation/revocation, metering, quotas, the working console, and a first Agent runner are implemented locally but still need the official checker and broader concurrency/recovery coverage.

## Project layout mapping

| Taskbook responsibility | Current implementation |
|---|---|
| Gateway and OpenAI API | `server/app.py` |
| Engine HTTP adapter and usage tokenization | `server/engine.py` |
| Admission control | `server/admission.py` |
| SSE event parsing, error events, usage validation | `server/streaming.py` (following the reference project's streaming boundary) |
| Runtime metrics | `server/metrics.py` |
| Process startup, readiness, and stop | root `setup.sh` (the taskbook's `supervisor.py` responsibility is currently in this script) |
| Tenant, key, quota, billing, usage, migration recovery | `control_plane/storage.py` (these concerns are not yet split into separate modules) |
| Admin web console and schema | `console/`, `migrations/` |
| Agent planning, whitelisted tool, final answer | `agent/runner.py`; `/agent/run` route in `server/app.py` delegates to it |
| Tenant/key/usage/audit/pricing admin routes | `control_plane/api.py`; database operations remain in `control_plane/storage.py` |
| Profile configuration | `configs/baseline.yaml` and `configs/optimized.yaml`; both are marked pending GPU validation and cannot yet launch a measured profile |
| Experimental pricing | `configs/pricing.json`; CPU fixture accounting only, not a commercial price |
| CPU/API and storage verification | `tests/`; public service-level control-plane checker is still pending |
| Performance method and evidence | `docs/performance.md` records pending work; benchmark and experiment tools/results are still missing |

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
weights. `--stop` only considers the recorded gateway PID and refuses to stop a
process it cannot identify as this service. Secrets must not be committed to this
repository.

The local test suite can be run with:

```bash
uv run --frozen pytest
```
