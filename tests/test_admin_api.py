from pathlib import Path

from fastapi.testclient import TestClient

from control_plane.storage import Store
from server.app import create_app
from server.config import Settings


ADMIN_KEY = "admin-test-secret-key"
USER_KEY = "user-test-inference-key"


def admin_client(tmp_path: Path) -> tuple[TestClient, Store]:
    settings = Settings(
        service_name="test-service",
        service_version="0.1.0",
        model_id="test-model",
        model_revision="test-revision",
        engine_base_url=None,
        api_key=USER_KEY,
        key_hash_secret="admin-test-pepper-with-more-than-32-bytes",
        data_dir=str(tmp_path / "state"),
        admin_api_key=ADMIN_KEY,
    )
    store = Store(settings.data_dir, settings.key_hash_secret, USER_KEY, settings.model_id)
    return TestClient(create_app(settings, store)), store


def create_tenant(client: TestClient) -> str:
    response = client.post(
        "/admin/tenants",
        headers={"Authorization": f"Bearer {ADMIN_KEY}"},
        json={
            "name": "Alice",
            "balance_micro_usd": 10000,
            "token_limit": 1000,
            "rpm_limit": 10,
            "concurrency_limit": 2,
        },
    )
    assert response.status_code == 201
    return response.json()["tenant_id"]


def test_admin_routes_distinguish_admin_and_inference_keys(tmp_path: Path) -> None:
    client, _ = admin_client(tmp_path)

    assert client.get("/admin/tenants").status_code == 401
    assert client.get("/admin/tenants", headers={"Authorization": f"Bearer {USER_KEY}"}).status_code == 403
    assert client.get("/admin/tenants", headers={"Authorization": f"Bearer {ADMIN_KEY}"}).status_code == 200


def test_key_creation_listing_rotation_and_revocation_do_not_leak_raw_key(tmp_path: Path) -> None:
    client, store = admin_client(tmp_path)
    tenant_id = create_tenant(client)
    admin_headers = {"Authorization": f"Bearer {ADMIN_KEY}"}

    created = client.post(
        f"/admin/tenants/{tenant_id}/keys",
        headers=admin_headers,
        json={"name": "Alice laptop"},
    )
    assert created.status_code == 201
    old_key = created.json()["api_key"]
    old_key_id = created.json()["key_id"]
    listing = client.get(f"/admin/tenants/{tenant_id}/keys", headers=admin_headers)
    assert listing.status_code == 200
    assert old_key not in listing.text
    assert "api_key" not in listing.json()["items"][0]
    assert old_key.encode() not in store.path.read_bytes()

    rotated = client.post(f"/admin/keys/{old_key_id}/rotate", headers=admin_headers, json={})
    assert rotated.status_code == 200
    new_key = rotated.json()["api_key"]
    assert new_key != old_key
    assert rotated.json()["replaced_status"] == "revoked"
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {old_key}"}).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {new_key}"}).status_code == 200

    revoked = client.post(
        f"/admin/keys/{rotated.json()['key_id']}/revoke", headers=admin_headers
    )
    assert revoked.status_code == 200, revoked.text
    assert revoked.json()["status"] == "revoked"
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {new_key}"}).status_code == 401


def test_credit_adjustment_is_idempotent_and_cannot_make_balance_negative(tmp_path: Path) -> None:
    client, _ = admin_client(tmp_path)
    tenant_id = create_tenant(client)
    headers = {"Authorization": f"Bearer {ADMIN_KEY}", "Idempotency-Key": "credit-1"}
    request = {"delta_micro_usd": 500, "reason": "test top-up"}

    first = client.post(f"/admin/tenants/{tenant_id}/credits", headers=headers, json=request)
    replay = client.post(f"/admin/tenants/{tenant_id}/credits", headers=headers, json=request)
    conflict = client.post(
        f"/admin/tenants/{tenant_id}/credits",
        headers=headers,
        json={"delta_micro_usd": 501, "reason": "test top-up"},
    )
    overdraft = client.post(
        f"/admin/tenants/{tenant_id}/credits",
        headers={"Authorization": f"Bearer {ADMIN_KEY}", "Idempotency-Key": "credit-2"},
        json={"delta_micro_usd": -20000, "reason": "invalid withdrawal"},
    )

    assert first.status_code == 200
    assert replay.json()["adjustment_id"] == first.json()["adjustment_id"]
    assert conflict.status_code == 409
    assert overdraft.status_code == 402


def test_admin_audit_shows_tenant_operations_without_exposing_key_material(tmp_path: Path) -> None:
    client, _ = admin_client(tmp_path)
    tenant_id = create_tenant(client)
    headers = {"Authorization": f"Bearer {ADMIN_KEY}"}
    key_response = client.post(
        f"/admin/tenants/{tenant_id}/keys", headers=headers, json={"name": "Audit key"}
    )
    assert key_response.status_code == 201
    raw_key = key_response.json()["api_key"]

    response = client.get(f"/admin/audit?tenant_id={tenant_id}", headers=headers)

    assert response.status_code == 200
    assert {item["event"] for item in response.json()["items"]} >= {
        "tenant_created", "key_created"
    }
    assert raw_key not in response.text


def test_pricing_versions_are_immutable_and_tenant_can_be_disabled(tmp_path: Path) -> None:
    client, store = admin_client(tmp_path)
    tenant_id = create_tenant(client)
    headers = {"Authorization": f"Bearer {ADMIN_KEY}"}
    initial = client.get("/admin/pricing", headers=headers).json()
    updated = client.put(
        "/admin/pricing",
        headers=headers,
        json={
            "model": "test-model",
            "input_micro_usd_per_token": 4,
            "output_micro_usd_per_token": 8,
        },
    )

    assert updated.status_code == 200
    assert updated.json()["pricing_version"] != initial["pricing_version"]
    assert client.get("/admin/pricing", headers=headers).json()["input_micro_usd_per_token"] == 4
    with store.connection() as database:
        versions = database.execute(
            "SELECT pricing_version FROM prices WHERE model='test-model'"
        ).fetchall()
    assert len(versions) == 2

    key = client.post(
        f"/admin/tenants/{tenant_id}/keys",
        headers=headers,
        json={"name": "Alice key"},
    ).json()["api_key"]
    disabled = client.patch(
        f"/admin/tenants/{tenant_id}", headers=headers, json={"status": "disabled"}
    )
    assert disabled.status_code == 200
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {key}"}).status_code == 403
