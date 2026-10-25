#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

case_name=""
out_dir=""
while (($#)); do
  case "$1" in
    --case)
      (($# >= 2)) || { echo "--case requires a value" >&2; exit 2; }
      case_name="$2"
      shift 2
      ;;
    --out)
      (($# >= 2)) || { echo "--out requires a directory" >&2; exit 2; }
      out_dir="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage: bash experiments/run.sh --case {kv|prefix|attention} --out DIRECTORY

Run one GPU experiment in a new output directory. Dependencies must already be
installed by setup.sh; this command does not install or upgrade packages.
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$case_name" in
  kv|prefix|attention) ;;
  *) echo "--case must be kv, prefix, or attention" >&2; exit 2 ;;
esac
if [[ -z "$out_dir" ]]; then
  echo "--out is required" >&2
  exit 2
fi

if [[ -x .venv/bin/python ]]; then
  python_bin=".venv/bin/python"
elif command -v python3.13 >/dev/null 2>&1; then
  python_bin="python3.13"
elif command -v python3.12 >/dev/null 2>&1; then
  python_bin="python3.12"
else
  python_bin="python3"
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required to enforce the experiment time budget" >&2
  exit 1
fi
if [[ "$case_name" == "prefix" ]]; then
  time_limit=480
else
  time_limit=300
fi
exec timeout --signal=TERM --kill-after=15s "${time_limit}s" \
  "$python_bin" -m "experiments.${case_name}" --out "$out_dir"
