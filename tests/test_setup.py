import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def run_setup(data_dir: Path, *arguments: str, **extra_env: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "DATA_DIR": str(data_dir),
            "API_KEY": "setup-test-key",
            "KEY_HASH_SECRET": "setup-test-pepper-with-more-than-32-bytes",
            "MODEL_ID": "test-model",
            "MODEL_REVISION": "test-revision",
        }
    )
    environment.update(extra_env)
    return subprocess.run(
        ["bash", "setup.sh", *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_stop_is_idempotent_when_no_gateway_pid_exists(tmp_path: Path) -> None:
    result = run_setup(tmp_path / "state", "--stop")

    assert result.returncode == 0
    assert "No recorded gateway process" in result.stdout


def test_stop_refuses_malformed_pid_without_removing_it(tmp_path: Path) -> None:
    data_dir = tmp_path / "state"
    data_dir.mkdir()
    pid_file = data_dir / "gateway.pid"
    pid_file.write_text("not-a-pid\n")

    result = run_setup(data_dir, "--stop")

    assert result.returncode == 1
    assert "invalid PID" in result.stderr
    assert pid_file.read_text() == "not-a-pid\n"


def test_contract_test_requires_external_engine_fixture(tmp_path: Path) -> None:
    result = run_setup(tmp_path / "state", "--contract-test")

    assert result.returncode == 1
    assert "ENGINE_BASE_URL is required" in result.stderr


def test_unknown_setup_mode_is_rejected(tmp_path: Path) -> None:
    result = run_setup(tmp_path / "state", "--not-a-mode")

    assert result.returncode == 2
    assert "usage:" in result.stderr
