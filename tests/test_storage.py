from pathlib import Path

import pytest

from control_plane.storage import LedgerError, Store


PEPPER = "storage-test-pepper-at-least-32-bytes"


def make_store(tmp_path: Path, key: str = "bootstrap-test-key") -> Store:
    return Store(str(tmp_path / "state"), PEPPER, key)


def test_store_enables_wal_and_does_not_persist_raw_key(tmp_path: Path) -> None:
    raw_key = "raw-key-that-must-not-be-persisted"
    store = make_store(tmp_path, raw_key)

    assert store.journal_mode().lower() == "wal"
    assert raw_key.encode() not in store.path.read_bytes()


def test_bootstrap_key_authenticates(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    auth = store.authenticate("bootstrap-test-key")

    assert auth["tenant_id"] == "tenant_benchmark"
    assert auth["key_id"].startswith("key_")


def test_wrong_or_revoked_key_is_rejected(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    key_id = store.authenticate("bootstrap-test-key")["key_id"]

    with pytest.raises(LedgerError, match="invalid api key"):
        store.authenticate("wrong-key")

    store.revoke_key(key_id)
    with pytest.raises(LedgerError, match="invalid api key"):
        store.authenticate("bootstrap-test-key")


def test_disabled_tenant_is_rejected(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_tenant_status("tenant_benchmark", "disabled")

    with pytest.raises(LedgerError, match="tenant disabled") as caught:
        store.authenticate("bootstrap-test-key")

    assert caught.value.status == 403


def test_changing_pepper_for_existing_ledger_is_rejected(tmp_path: Path) -> None:
    state = tmp_path / "state"
    Store(str(state), PEPPER, "bootstrap-test-key")

    with pytest.raises(ValueError, match="KEY_HASH_SECRET differs"):
        Store(
            str(state),
            "different-storage-pepper-at-least-32-bytes",
            "bootstrap-test-key",
        )
