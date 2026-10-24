"""Start and stop only the SGLang and gateway processes owned by this project."""

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PORT = 30000


def process_identity(pid: int) -> tuple[str, str] | None:
    """Return a live process start identity and command, avoiding reused PIDs."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        if fields[0] == "Z":
            return None
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        return fields[19], command
    except (OSError, IndexError):
        # macOS has no /proc; `ps` still provides a start identity and command.
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        value = result.stdout.strip()
        if result.returncode != 0 or not value:
            return None
        fields = value.split(None, 5)
        if len(fields) < 6:
            return None
        return " ".join(fields[:5]), fields[5]


def _matches(entry: dict[str, object]) -> bool:
    identity = process_identity(int(entry["pid"]))
    return bool(
        identity
        and identity[0] == entry.get("start_ticks")
        and str(entry["command_marker"]) in identity[1]
    )


def _read_state(data_dir: Path) -> dict[str, object]:
    path = data_dir / "processes.json"
    if not path.exists():
        return {"processes": {}}
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("processes"), dict):
        raise RuntimeError(f"Invalid process state file: {path}")
    return value


def _write_state(data_dir: Path, state: dict[str, object]) -> None:
    target = data_dir / "processes.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)


def stop_owned(data_dir: Path, *, timeout: float = 30.0) -> bool:
    state = _read_state(data_dir)
    processes = state["processes"]
    stopped_any = False
    for name in ("gateway", "engine"):
        entry = processes.get(name)
        if not isinstance(entry, dict):
            continue
        pid = int(entry["pid"])
        if not _matches(entry):
            continue
        stopped_any = True
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        deadline = time.monotonic() + timeout
        while _matches(entry) and time.monotonic() < deadline:
            time.sleep(0.1)
        if _matches(entry):
            raise RuntimeError(f"Owned {name} process {pid} did not stop; inspect its log")
    state["processes"] = {}
    _write_state(data_dir, state)
    (data_dir / "gateway.pid").unlink(missing_ok=True)
    (data_dir / "engine.pid").unlink(missing_ok=True)
    (data_dir / "runtime.json").unlink(missing_ok=True)
    return stopped_any


def _healthy(url: str, timeout: float) -> bool:
    try:
        with urlopen(url, timeout=min(timeout, 2.0)) as response:
            return 200 <= response.status < 300
    except (OSError, URLError, TimeoutError):
        return False


def _wait_ready(process: subprocess.Popen, url: str, timeout: float, label: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{label} exited during startup; inspect its log")
        if _healthy(url, timeout=2.0):
            return
        time.sleep(1)
    raise RuntimeError(f"{label} did not become ready within {timeout:g}s")


def _profile_args(profile: dict[str, object], model: str, revision: str) -> list[str]:
    if profile.get("model_id") != model or profile.get("model_revision") != revision:
        raise ValueError("MODEL_ID and MODEL_REVISION must match the selected profile")
    options = [
        "--model-path", model,
        "--revision", revision,
        "--served-model-name", model,
        "--dtype", str(profile["dtype"]),
        "--context-length", str(profile["context_length"]),
        "--max-running-requests", str(profile["max_running_requests"]),
        "--mem-fraction-static", str(profile["mem_fraction_static"]),
        "--attention-backend", str(profile["attention_backend"]),
        "--sampling-defaults", str(profile["sampling_defaults"]),
        "--host", "127.0.0.1",
        "--port", str(ENGINE_PORT),
    ]
    max_total_tokens = profile.get("max_total_tokens")
    if max_total_tokens is not None:
        options.extend(["--max-total-tokens", str(max_total_tokens)])
    if profile.get("prefix_cache") is False:
        options.append("--disable-radix-cache")
    return options


def _spawn(
    name: str,
    command: list[str],
    log_path: Path,
    state: dict[str, object],
    data_dir: Path,
    command_marker: str,
    env: dict[str, str],
) -> subprocess.Popen:
    log = log_path.open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log.close()
    deadline = time.monotonic() + 5
    identity = process_identity(process.pid)
    while identity is None and time.monotonic() < deadline:
        time.sleep(0.05)
        identity = process_identity(process.pid)
    if identity is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        raise RuntimeError(f"Could not record the started {name} process")
    entry = {"pid": process.pid, "start_ticks": identity[0], "command_marker": command_marker}
    state["processes"][name] = entry
    _write_state(data_dir, state)
    if name == "gateway":
        (data_dir / "gateway.pid").write_text(f"{process.pid}\n")
    elif name == "engine":
        (data_dir / "engine.pid").write_text(f"{process.pid}\n")
    return process


def start(data_dir: Path) -> None:
    import yaml

    profile_name = os.environ.get("BENCH_PROFILE", "baseline")
    if profile_name not in {"baseline", "optimized"}:
        raise ValueError("BENCH_PROFILE must be baseline or optimized")
    profile_path = ROOT / "configs" / f"{profile_name}.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    model = os.environ["MODEL_ID"]
    revision = os.environ["MODEL_REVISION"]
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")

    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = _read_state(data_dir)
        if (
            state.get("mode") == "normal"
            and state.get("profile") == profile_name
            and state.get("model_id") == model
            and state.get("model_revision") == revision
            and state.get("host") == host
            and state.get("port") == port
            and all(
                isinstance(state.get("processes", {}).get(name), dict)
                and _matches(state["processes"][name])
                for name in ("engine", "gateway")
            )
        ):
            if _healthy(f"http://127.0.0.1:{ENGINE_PORT}/health", 2) and _healthy(
                f"http://127.0.0.1:{port}/healthz", 2
            ):
                print(f"llm-service {profile_name} ready (PID {state['processes']['gateway']['pid']})")
                return
        stop_owned(data_dir)
        state = {
            "mode": "normal",
            "profile": profile_name,
            "model_id": model,
            "model_revision": revision,
            "host": host,
            "port": port,
            "processes": {},
        }
        _write_state(data_dir, state)
        (data_dir / "runtime.json").unlink(missing_ok=True)

        engine_command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            *_profile_args(profile, model, revision),
        ]
        engine_env = os.environ.copy()
        engine_env.pop("ENGINE_BASE_URL", None)
        engine = None
        try:
            engine = _spawn(
                "engine", engine_command, data_dir / "engine.log", state, data_dir,
                "sglang.launch_server", engine_env,
            )
            _wait_ready(engine, f"http://127.0.0.1:{ENGINE_PORT}/health", 900, "SGLang")

            gateway_env = os.environ.copy()
            gateway_env["ENGINE_BASE_URL"] = f"http://127.0.0.1:{ENGINE_PORT}"
            gateway_env["HOST"] = host
            gateway_env["PORT"] = str(port)
            gateway = _spawn(
                "gateway",
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "server.app:app",
                    "--host",
                    host,
                    "--port",
                    str(port),
                ],
                data_dir / "gateway.log",
                state,
                data_dir,
                "uvicorn server.app:app",
                gateway_env,
            )
            _wait_ready(gateway, f"http://127.0.0.1:{port}/healthz", 60, "Gateway")
        except Exception:
            try:
                stop_owned(data_dir)
            except Exception as stop_error:
                print(f"Cleanup failed: {stop_error}", file=sys.stderr)
            raise

        try:
            engine_version = importlib.metadata.version("sglang")
        except importlib.metadata.PackageNotFoundError:
            engine_version = "unknown"
        (data_dir / "runtime.json").write_text(
            json.dumps(
                {
                    "profile": profile_name,
                    "model_id": model,
                    "model_revision": revision,
                    "sglang_version": engine_version,
                    "engine_port": ENGINE_PORT,
                    "gateway_port": port,
                    "config_status": profile.get("status", "unknown"),
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
        )
        print(f"llm-service {profile_name} ready on port {port} (PID {gateway.pid})")


def start_contract_test(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    with (data_dir / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = _read_state(data_dir)
        gateway_entry = state["processes"].get("gateway")
        if (
            state.get("mode") == "contract-test"
            and state.get("engine_base_url") == os.environ["ENGINE_BASE_URL"]
            and state.get("host") == host
            and state.get("port") == port
            and isinstance(gateway_entry, dict)
            and _matches(gateway_entry)
        ):
            if _healthy(f"http://127.0.0.1:{port}/healthz", 2):
                print(f"llm-service gateway already running as PID {gateway_entry['pid']}")
                return
        stop_owned(data_dir)
        state = {
            "mode": "contract-test",
            "engine_base_url": os.environ["ENGINE_BASE_URL"],
            "host": host,
            "port": port,
            "processes": {},
        }
        _write_state(data_dir, state)
        (data_dir / "runtime.json").unlink(missing_ok=True)
        env = os.environ.copy()
        env["HOST"] = host
        env["PORT"] = str(port)
        gateway = _spawn(
            "gateway",
            [
                sys.executable,
                "-m",
                "uvicorn",
                "server.app:app",
                "--host",
                host,
                "--port",
                str(port),
            ],
            data_dir / "gateway.log",
            state,
            data_dir,
            "uvicorn server.app:app",
            env,
        )
        try:
            _wait_ready(gateway, f"http://127.0.0.1:{port}/healthz", 60, "Gateway")
        except Exception:
            stop_owned(data_dir)
            raise
        print(f"llm-service ready on port {port} (PID {gateway.pid})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "contract-test", "stop"))
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if data_dir in {Path("/"), Path.home(), ROOT}:
        parser.error("DATA_DIR must be a dedicated private directory")
    try:
        if args.action == "stop":
            if stop_owned(data_dir):
                print("Stopped llm-service")
            else:
                print("No recorded gateway process")
        elif args.action == "contract-test":
            start_contract_test(data_dir)
        else:
            start(data_dir)
    except Exception as error:
        print(f"setup: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
