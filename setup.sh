#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

mode="${1:-}"
if [[ -n "${mode}" && "${mode}" != "--contract-test" ]]; then
  echo "usage: bash setup.sh [--contract-test]" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to install and run llm-service" >&2
  exit 1
fi

for required_name in API_KEY KEY_HASH_SECRET DATA_DIR MODEL_ID MODEL_REVISION; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "${required_name} is required" >&2
    exit 1
  fi
done

port="${PORT:-8000}"

if [[ -n "${BOOTSTRAP_PYTHON:-}" ]]; then
  python_bin="${BOOTSTRAP_PYTHON}"
elif command -v python3.13 >/dev/null 2>&1; then
  python_bin="python3.13"
elif command -v python3.12 >/dev/null 2>&1; then
  python_bin="python3.12"
else
  python_bin="python3"
fi

exec uv run --python "${python_bin}" uvicorn server.app:app --host 0.0.0.0 --port "${port}"
