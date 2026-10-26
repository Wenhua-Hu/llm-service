from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_models_rejects_missing_key() -> None:
    response = client.get("/v1/models")

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "code": "invalid_api_key",
            "message": "invalid api key",
        }
    }


def test_models_rejects_wrong_key() -> None:
    response = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer wrong-key"},
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_models_accepts_valid_key() -> None:
    response = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer test-api-key"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [{"id": "test-model", "object": "model"}],
    }


def test_usage_rejects_tenant_override_inputs() -> None:
    query_override = client.get(
        "/v1/usage?tenant_id=another-tenant",
        headers={"Authorization": "Bearer test-api-key"},
    )
    header_override = client.get(
        "/v1/usage",
        headers={
            "Authorization": "Bearer test-api-key",
            "X-Tenant-ID": "another-tenant",
        },
    )

    assert query_override.status_code == 403
    assert header_override.status_code == 403


def test_inference_routes_reject_tenant_override_headers() -> None:
    response = client.get(
        "/v1/models",
        headers={
            "Authorization": "Bearer test-api-key",
            "X-Tenant-ID": "another-tenant",
        },
    )

    assert response.status_code == 403


def test_inference_routes_reject_tenant_override_query_parameters() -> None:
    response = client.get(
        "/v1/models?tenant_id=another-tenant",
        headers={"Authorization": "Bearer test-api-key"},
    )

    assert response.status_code == 403


def test_info_is_not_public() -> None:
    response = client.get("/info")

    assert response.status_code == 401
