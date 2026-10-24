from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_healthz_reports_not_ready_without_engine() -> None:
    response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


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
        "engine_version": None,
        "engine_configured": False,
        "profile": "baseline",
        "config": {
            "gateway_slots": 4,
            "gateway_queue": 16,
            "prefix_cache": False,
            "attention_backend": "unset",
            "sampling_defaults": "openai",
            "repetition_penalty": 1.0,
            "top_p": 1.0,
            "top_k": -1,
        },
    }
