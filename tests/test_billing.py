from pathlib import Path

import pytest

from control_plane.storage import LedgerError, Store


PEPPER = "billing-test-pepper-has-at-least-32-bytes"
BOOTSTRAP_KEY = "billing-bootstrap-key"


def make_store(path: Path) -> Store:
    return Store(str(path), PEPPER, BOOTSTRAP_KEY, "test-model")


def create_tenant_and_auth(store: Store, *, balance: int = 10000) -> dict[str, str]:
    tenant = store.create_tenant(
        name="Billing test",
        balance_micro_usd=balance,
        token_limit=10000,
        rpm_limit=100,
        concurrency_limit=10,
    )
    created_key = store.create_key(tenant["tenant_id"], name="billing test key")
    return store.authenticate(created_key["api_key"])


def reserve(store: Store, auth: dict[str, str], body: dict[str, object], *, idem: str | None = None):
    return store.reserve_request(
        auth=auth,
        route="/v1/chat/completions",
        model="test-model",
        body=body,
        input_tokens=100,
        max_output_tokens=50,
        idempotency_key=idem,
    )


def test_reservation_settles_exact_usage_and_releases_unused_amount(tmp_path: Path) -> None:
    store = make_store(tmp_path / "state")
    auth = create_tenant_and_auth(store)
    reservation = reserve(store, auth, {"messages": []})

    assert reservation["reserved_micro_usd"] == 500
    store.journal_usage(reservation["request_id"], 100, 50)
    settled = store.settle_request(reservation["request_id"])
    usage = store.usage(auth["tenant_id"])

    assert settled["cost_micro_usd"] == 500
    assert usage["summary"]["cost_micro_usd"] == 500
    assert usage["summary"]["input_tokens"] == 100
    assert usage["summary"]["output_tokens"] == 50
    assert usage["summary"]["reserved_micro_usd"] == 0
    assert usage["items"][0]["pricing_version"] == reservation["pricing_version"]
    assert store.settle_request(reservation["request_id"])["status"] == "settled"
    assert store.usage(auth["tenant_id"])["summary"]["cost_micro_usd"] == 500


def test_idempotency_prevents_duplicate_generation_and_body_reuse_conflicts(tmp_path: Path) -> None:
    store = make_store(tmp_path / "state")
    auth = create_tenant_and_auth(store)
    first = reserve(store, auth, {"messages": [{"content": "A"}]}, idem="same-key")

    with pytest.raises(LedgerError) as same_body:
        reserve(store, auth, {"messages": [{"content": "A"}]}, idem="same-key")
    with pytest.raises(LedgerError) as different_body:
        reserve(store, auth, {"messages": [{"content": "B"}]}, idem="same-key")

    assert same_body.value.status == 409
    assert same_body.value.code == "request_in_progress"
    assert different_body.value.status == 409
    assert different_body.value.code == "idempotency_conflict"
    assert len(store.usage(auth["tenant_id"])["items"]) == 1
    store.release_request(first["request_id"], "test_complete")
    with pytest.raises(LedgerError) as completed:
        reserve(store, auth, {"messages": [{"content": "A"}]}, idem="same-key")
    assert completed.value.code == "request_already_completed"


def test_usage_above_reservation_is_not_settled_or_fabricated(tmp_path: Path) -> None:
    store = make_store(tmp_path / "state")
    auth = create_tenant_and_auth(store)
    reservation = reserve(store, auth, {"messages": []})
    store.journal_usage(reservation["request_id"], 100, 51)

    result = store.settle_request(reservation["request_id"])
    usage = store.usage(auth["tenant_id"])

    assert result["status"] == "pending_reconciliation"
    assert result["error_code"] == "usage_exceeds_reservation"
    assert usage["summary"]["cost_micro_usd"] == 0
    assert usage["summary"]["reserved_micro_usd"] == 500
    assert usage["items"][0]["status"] == "pending_reconciliation"


def test_usage_cursor_pages_are_stable_and_do_not_repeat_requests(tmp_path: Path) -> None:
    store = make_store(tmp_path / "state")
    auth = create_tenant_and_auth(store)
    for index in range(3):
        reservation = reserve(store, auth, {"messages": [{"content": str(index)}]})
        store.journal_usage(reservation["request_id"], 100, 50)
        store.settle_request(reservation["request_id"])

    first = store.usage(auth["tenant_id"], limit=2)
    second = store.usage(auth["tenant_id"], limit=2, cursor=first["next_cursor"])

    first_ids = {item["request_id"] for item in first["items"]}
    second_ids = {item["request_id"] for item in second["items"]}
    assert len(first_ids) == 2
    assert len(second_ids) == 1
    assert first_ids.isdisjoint(second_ids)
    assert second["next_cursor"] is None
    assert second["summary"]["cost_micro_usd"] == 1500


def test_pricing_changes_do_not_reprice_existing_request(tmp_path: Path) -> None:
    store = make_store(tmp_path / "state")
    auth = create_tenant_and_auth(store)
    before = reserve(store, auth, {"messages": [{"content": "before"}]})
    store.journal_usage(before["request_id"], 100, 50)
    store.settle_request(before["request_id"])
    store.set_pricing("test-model", input_price=4, output_price=8)
    after = reserve(store, auth, {"messages": [{"content": "after"}]})
    store.journal_usage(after["request_id"], 100, 50)
    store.settle_request(after["request_id"])

    items = store.usage(auth["tenant_id"])["items"]
    assert {item["pricing_version"] for item in items} == {
        before["pricing_version"], after["pricing_version"]
    }
    assert {item["cost_micro_usd"] for item in items} == {500, 800}
    assert store.usage(auth["tenant_id"])["summary"]["cost_micro_usd"] == 1300


def test_crash_recovery_settles_journaled_usage_exactly_once(tmp_path: Path) -> None:
    state = tmp_path / "state"
    store = make_store(state)
    auth = create_tenant_and_auth(store)
    reservation = reserve(store, auth, {"messages": []})
    store.journal_usage(reservation["request_id"], 100, 50)

    restarted = make_store(state)

    assert restarted.usage(auth["tenant_id"])["summary"]["cost_micro_usd"] == 500
    assert restarted.usage(auth["tenant_id"])["summary"]["reserved_micro_usd"] == 0
    restarted.recover_incomplete_requests()
    assert restarted.usage(auth["tenant_id"])["summary"]["cost_micro_usd"] == 500


def test_crash_recovery_releases_old_unknown_usage_with_audit(tmp_path: Path) -> None:
    state = tmp_path / "state"
    store = make_store(state)
    auth = create_tenant_and_auth(store)
    reservation = reserve(store, auth, {"messages": []})
    with store.transaction() as database:
        database.execute(
            "UPDATE requests SET created_epoch=created_epoch-61 WHERE request_id=?",
            (reservation["request_id"],),
        )

    restarted = make_store(state)
    usage = restarted.usage(auth["tenant_id"])

    assert usage["summary"]["reserved_micro_usd"] == 0
    assert usage["items"][0]["status"] == "reconciled_unknown"
    assert usage["items"][0]["input_tokens"] is None
    with restarted.connection() as database:
        adjustment = database.execute(
            "SELECT reason FROM adjustments WHERE idempotency_key=?",
            (f"recovery:{reservation['request_id']}",),
        ).fetchone()
    assert adjustment["reason"] == "recovery_release_unknown_usage"
