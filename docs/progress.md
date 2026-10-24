# Development record

## 2026-10-24 — local CPU interface and control-plane pass

### Changes

- Pinned `uv==0.11.23` installation in the GitHub Actions CPU workflow. The last observed Actions run stopped before service startup because `uv` was unavailable; this local change has not been pushed or rerun on GitHub.
- Updated `/healthz` to return `{"status":"ready"}` only when the configured engine's `/health` endpoint responds successfully; otherwise it returns HTTP 503.
- Implemented the first persistent SQLite-backed tenant, API-key lifecycle, quota reservation, pricing, usage settlement, recovery, and audit paths; added the working admin page under `/console`.
- Made repeated in-flight idempotency requests return `request_in_progress`; usage above a reserved amount stays pending reconciliation instead of being charged or lost.
- Added date-filterable, cursor-paginated usage records and an admin audit API consumed by the console.
- Made the SSE usage parser handle fragmented CRLF event boundaries and guaranteed gate release through stream cleanup errors.
- Implemented `/agent/run` as two real engine calls around the single `lookup_order` tool, with a zero-cost parent ledger row and separately billed child rows linked by `parent_request_id`.
- `/info` now reads SGLang `/server_info` and reports runtime version/configuration when the engine provides them, rather than claiming optimized settings from profile labels.
- Aligned the tree toward the taskbook: added `.env.example`, `configs/`, `benchmarks/`, `experiments/`, `results/`, `scripts/`, and `agent/`; moved request gating to `server/admission.py`; extracted engine helpers, metrics, and SSE parsing/usage validation into focused `server/` modules. The SSE boundary follows the reference project's separation of stream handling from API routes.
- Moved Agent workflow logic to `agent/runner.py` and management endpoints to `control_plane/api.py`; kept request routing and SQLite transaction ownership in their existing layers.
- Added explicit pending-status baseline/optimized profile files, experimental fixture pricing config, and a performance document that distinguishes frozen targets from unmeasured project results. Removed `results/` from `.gitignore` so real evidence can be committed when produced.

### Local verification

- `UV_CACHE_DIR="$PWD/.venv/.uv-cache" .venv/bin/python -m pytest -q --basetemp=.venv/pytest-tmp`: **43 passed**.
- `UV_CACHE_DIR="$PWD/.venv/.uv-cache" uv lock --check --offline`: **passed**, 30 packages resolved.
- `bash -n setup.sh`: **passed**.
- `git diff --check`: **passed**.
- Pytest emitted one Starlette deprecation warning about its current `httpx` TestClient integration.

### Not yet verified

- The modified commit has not been pushed, so GitHub Actions has not run these changes and no new OpenLabs soft report exists.
- CPU fixture contract tests, network-level SSE cancellation/buffering, the full 15-item credential audit, and the public 18-check workflow remain unverified against the platform fixture.
- `/agent/run` has a controllable-engine integration test; its behavior and answer quality with real SGLang remain unverified.
- No Daytona machine was started. SGLang deployment, optimization experiments, and GPU performance measurements remain pending.
- Official performance gates remain the frozen taskbook values; no project result is claimed against them.
