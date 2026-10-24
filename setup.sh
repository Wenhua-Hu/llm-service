#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

mode="${1:-}"
if [[ -n "${mode}" && "${mode}" != "--contract-test" && "${mode}" != "--stop" ]]; then
  echo "usage: bash setup.sh [--contract-test|--stop]" >&2
  exit 2
fi

if [[ "${mode}" == "--stop" ]]; then
  data_dir="${DATA_DIR:-}"
  if [[ -z "${data_dir}" ]]; then
    echo "DATA_DIR is required for --stop" >&2
    exit 1
  fi
  if [[ "${data_dir}" != /* || "${data_dir}" == "/" || "${data_dir}" == "${HOME}" || "${data_dir}" == "${PWD}" ]]; then
    echo "DATA_DIR must be a dedicated absolute subdirectory for --stop" >&2
    exit 1
  fi
  pid_file="${data_dir%/}/gateway.pid"
  if [[ ! -f "${pid_file}" ]]; then
    echo "No recorded gateway process to stop"
    exit 0
  fi
  read -r gateway_pid < "${pid_file}"
  if [[ ! "${gateway_pid}" =~ ^[0-9]+$ ]]; then
    echo "Refusing to stop: invalid PID in ${pid_file}" >&2
    exit 1
  fi
  if ! kill -0 "${gateway_pid}" 2>/dev/null; then
    rm -f -- "${pid_file}"
    echo "Recorded gateway process is already stopped"
    exit 0
  fi
  process_command="$(ps -p "${gateway_pid}" -o command= 2>/dev/null || true)"
  if [[ "${process_command}" != *"-m uvicorn server.app:app"* ]]; then
    echo "Refusing to stop PID ${gateway_pid}: it is not the recorded llm-service gateway" >&2
    exit 1
  fi
  kill "${gateway_pid}"
  for _ in {1..50}; do
    if ! kill -0 "${gateway_pid}" 2>/dev/null; then
      rm -f -- "${pid_file}"
      echo "Stopped llm-service gateway PID ${gateway_pid}"
      exit 0
    fi
    sleep 0.1
  done
  echo "Gateway PID ${gateway_pid} did not stop within 5 seconds" >&2
  exit 1
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

if [[ "${mode}" == "--contract-test" && -z "${ENGINE_BASE_URL:-}" ]]; then
  echo "ENGINE_BASE_URL is required for --contract-test" >&2
  exit 1
fi

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

data_dir="${DATA_DIR%/}"
if [[ -z "${data_dir}" || "${data_dir}" != /* || "${data_dir}" == "/" || "${data_dir}" == "${HOME}" || "${data_dir}" == "${PWD}" ]]; then
  echo "DATA_DIR must be a dedicated absolute subdirectory" >&2
  exit 1
fi
mkdir -p -- "${data_dir}"
pid_file="${data_dir}/gateway.pid"
if [[ -f "${pid_file}" ]]; then
  read -r existing_pid < "${pid_file}"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    existing_command="$(ps -p "${existing_pid}" -o command= 2>/dev/null || true)"
    if [[ "${existing_command}" == *"-m uvicorn server.app:app"* ]]; then
      echo "llm-service gateway already running as PID ${existing_pid}"
      exit 0
    fi
    echo "Refusing to replace ${pid_file}: PID ${existing_pid} belongs to another process" >&2
    exit 1
  fi
  rm -f -- "${pid_file}"
fi

uv sync --frozen --python "${python_bin}"
"${PWD}/.venv/bin/python" -m uvicorn server.app:app \
  --host 127.0.0.1 --port "${port}" \
  >>"${data_dir}/gateway.log" 2>&1 &
gateway_pid=$!
printf '%s\n' "${gateway_pid}" > "${pid_file}"

for _ in {1..100}; do
  if ! kill -0 "${gateway_pid}" 2>/dev/null; then
    echo "Gateway exited during startup; see ${data_dir}/gateway.log" >&2
    rm -f -- "${pid_file}"
    exit 1
  fi
  if command -v curl >/dev/null 2>&1 && curl --silent --fail "http://127.0.0.1:${port}/healthz" >/dev/null; then
    echo "llm-service ready on port ${port} (PID ${gateway_pid})"
    exit 0
  fi
  sleep 0.1
done

echo "Gateway did not become ready within 10 seconds; see ${data_dir}/gateway.log" >&2
kill "${gateway_pid}" 2>/dev/null || true
rm -f -- "${pid_file}"
exit 1
