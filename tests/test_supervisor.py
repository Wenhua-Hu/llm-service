from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from server.supervisor import _profile_args, process_identity


ROOT = Path(__file__).resolve().parents[1]


def load_profile(name: str) -> dict[str, object]:
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())


def test_process_identity_uses_ps_when_proc_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable_proc(self: Path) -> str:
        raise FileNotFoundError

    monkeypatch.setattr(Path, "read_text", unavailable_proc)
    monkeypatch.setattr(
        "server.supervisor.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Sat Oct 24 12:34:56 2026 /usr/bin/python -m uvicorn server.app:app\n",
        ),
    )

    identity = process_identity(123)

    assert identity == (
        "Sat Oct 24 12:34:56 2026",
        "/usr/bin/python -m uvicorn server.app:app",
    )


def test_baseline_profile_disables_radix_cache_until_enabled_by_profile() -> None:
    args = _profile_args(
        load_profile("baseline"),
        "Qwen/Qwen2.5-7B-Instruct",
        "a09a35458c702b33eeacc393d103063234e8bc28",
    )

    assert "--disable-radix-cache" in args
    assert "--max-total-tokens" not in args
    assert args[args.index("--sampling-defaults") + 1] == "openai"


def test_optimized_profile_enables_radix_cache_and_flashinfer() -> None:
    args = _profile_args(
        load_profile("optimized"),
        "Qwen/Qwen2.5-7B-Instruct",
        "a09a35458c702b33eeacc393d103063234e8bc28",
    )

    assert "--disable-radix-cache" not in args
    assert args[args.index("--attention-backend") + 1] == "flashinfer"
    assert args[args.index("--max-running-requests") + 1] == "16"
    assert "--max-total-tokens" not in args


def test_profile_rejects_a_different_model_or_revision() -> None:
    with pytest.raises(ValueError, match="must match the selected profile"):
        _profile_args(load_profile("baseline"), "another-model", "another-revision")
