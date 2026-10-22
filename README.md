# llm-service

An OpenAI-compatible model service gateway for the OpenLabs LLM-server project.

## Current milestone

The first milestone provides a minimal FastAPI service with:

- `GET /healthz` for liveness checks
- `GET /info` for non-secret service metadata
- environment-based configuration
- a `setup.sh --contract-test` entry point
- basic automated tests

The second milestone adds:

- Bearer API Key authentication
- `GET /v1/models` using the OpenAI-compatible list shape
- authenticated `/info` metadata

The third milestone adds:

- SQLite WAL persistence
- a bootstrap tenant and API Key stored only as an HMAC-SHA256 digest
- tenant disable and Key revoke enforcement
- persistent pepper identity checks

Inference, authentication, tenant management, billing, and the web console will be
added in later milestones.

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
```

Configuration is read from environment variables. Secrets must not be committed
to this repository.
