from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_healthz_reports_ok() -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_info_contains_public_metadata_only() -> None:
    response = client.get(
        "/info",
        headers={"Authorization": "Bearer test-api-key"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "contract": "llm-lab/1.0",
        "service": "llm-service",
        "version": "0.1.0",
        "model_id": "test-model",
        "model_revision": "test-revision",
        "engine": "sglang",
        "engine_configured": False,
    }
