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
    echo "DATA_DIR is unset; no recorded llm-service process to stop"
    exit 0
  fi
  if [[ "${data_dir}" != /* || "${data_dir}" == "/" || "${data_dir}" == "${HOME}" || "${data_dir}" == "${PWD}" ]]; then
    echo "DATA_DIR must be a dedicated absolute subdirectory for --stop" >&2
    exit 1
  fi
  pid_file="${data_dir%/}/gateway.pid"
  state_file="${data_dir%/}/processes.json"
  if [[ -f "${state_file}" ]]; then
    python_bin="${BOOTSTRAP_PYTHON:-python3}"
    if [[ -x "${PWD}/.venv/bin/python" ]]; then
      python_bin="${PWD}/.venv/bin/python"
    fi
    exec "${python_bin}" -m server.supervisor stop --data-dir "${data_dir}"
  fi
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
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to install uv" >&2
    exit 1
  fi
  uv_install_dir="${UV_INSTALL_DIR:-${HOME}/.local/bin}"
  echo "uv not found; installing it into ${uv_install_dir}"
  curl --fail --show-error --silent --location https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="${uv_install_dir}" UV_NO_MODIFY_PATH=1 sh
  export PATH="${uv_install_dir}:${PATH}"
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv installation completed, but uv is not available in PATH" >&2
    exit 1
  fi
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

if [[ "${mode}" != "--contract-test" && "${BENCH_PROFILE:-baseline}" != "baseline" && "${BENCH_PROFILE:-baseline}" != "optimized" ]]; then
  echo "BENCH_PROFILE must be baseline or optimized" >&2
  exit 1
fi

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
sync_args=(sync --frozen --no-dev --python "${python_bin}")
if [[ "${mode}" != "--contract-test" ]]; then
  sync_args+=(--extra sglang)
  if [[ "${BENCH_PROFILE:-baseline}" == "optimized" ]]; then
    sync_args+=(--extra flashinfer)
  fi
fi
uv "${sync_args[@]}"

if [[ "${mode}" == "--contract-test" ]]; then
  action=contract-test
else
  action=start
fi
exec "${PWD}/.venv/bin/python" -m server.supervisor "${action}" --data-dir "${data_dir}"
