import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

import server.app as gateway_module
import server.engine as engine_module
import agent.runner as agent_module
from control_plane.storage import Store
from server.app import SSEUsageParser, app, create_app
from server.config import Settings


client = TestClient(app)
VALID_KEY = {"Authorization": "Bearer test-api-key"}


class FakeEngineResponse:
    def __init__(self, body: bytes, content_type: str = "application/json") -> None:
        self.status_code = 200
        self.headers = {"content-type": content_type}
        self.content = body
        self.body = body

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    async def aread(self) -> bytes:
        return self.body

    def json(self):
        return json.loads(self.body)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("unexpected upstream status")

    async def aiter_bytes(self):
        midpoint = max(1, len(self.body) // 2)
        yield self.body[:midpoint]
        yield self.body[midpoint:]

    async def aclose(self) -> None:
        return None


class FakeEngineClient:
    def __init__(self, *args, calls: list[dict[str, object]], stream_body: bytes | None = None, **kwargs) -> None:
        self.calls = calls
        self.stream_body = stream_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, object], headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if url.endswith("/abort_request"):
            return FakeEngineResponse(b"{}")
        if url.endswith("/tokenize"):
            return FakeEngineResponse(b'{"count":3,"tokens":[1,2,3]}')
        return FakeEngineResponse(
            b'{"id":"engine-response","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}'
        )

    def build_request(self, method: str, url: str, *, json: dict[str, object], headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return {"url": url, "json": json, "headers": headers}

    async def send(self, request: dict[str, object], *, stream: bool):
        return FakeEngineResponse(self.stream_body or b"", "text/event-stream")

    async def aclose(self) -> None:
        return None


def engine_client(tmp_path: Path) -> tuple[TestClient, dict[str, str]]:
    key = "engine-test-api-key"
    settings = Settings(
        service_name="test-service",
        service_version="0.1.0",
        model_id="test-model",
        model_revision="test-revision",
        engine_base_url="http://engine.test",
        api_key=key,
        key_hash_secret="engine-test-secret-with-more-than-32-bytes",
        data_dir=str(tmp_path / "state"),
    )
    store = Store(settings.data_dir, settings.key_hash_secret, key, settings.model_id)
    return TestClient(create_app(settings, store)), {"Authorization": f"Bearer {key}"}


def test_chat_completions_requires_authentication() -> None:
    response = client.post("/v1/chat/completions", json={"model": "test-model"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_healthz_reports_ready_only_when_engine_health_succeeds(monkeypatch, tmp_path) -> None:
    engine_api, _ = engine_client(tmp_path)

    class HealthyEngineClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            return type("HealthyResponse", (), {"is_success": url.endswith("/health")})()

    monkeypatch.setattr(
        gateway_module,
        "httpx",
        type("HttpxWithHealthClient", (), {"AsyncClient": HealthyEngineClient}),
    )
    monkeypatch.setattr(engine_module, "httpx", gateway_module.httpx)

    assert engine_api.get("/healthz").status_code == 200
    assert engine_api.get("/healthz").json() == {"status": "ready"}


def test_info_reflects_runtime_sglang_version_and_optimization_settings(monkeypatch, tmp_path) -> None:
    engine_api, engine_auth = engine_client(tmp_path)

    class RuntimeInfoResponse:
        is_success = True

        def json(self):
            return {
                "version": "test-sglang-version",
                "max_running_requests": 8,
                "max_total_tokens": 4096,
                "dtype": "bfloat16",
                "attention_backend": "flashinfer",
                "disable_radix_cache": False,
            }

    class RuntimeInfoClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            return RuntimeInfoResponse()

    monkeypatch.setattr(
        gateway_module,
        "httpx",
        type("HttpxWithRuntimeInfo", (), {"AsyncClient": RuntimeInfoClient}),
    )
    monkeypatch.setattr(engine_module, "httpx", gateway_module.httpx)

    response = engine_api.get("/info", headers=engine_auth)

    assert response.status_code == 200
    assert response.json()["engine_version"] == "test-sglang-version"
    assert response.json()["config"]["max_running_requests"] == 8
    assert response.json()["config"]["max_total_tokens"] == 4096
    assert response.json()["config"]["attention_backend"] == "flashinfer"
    assert response.json()["config"]["prefix_cache"] is True


def test_chat_completions_reports_missing_engine_without_fabricating_usage() -> None:
    response = client.post(
        "/v1/chat/completions",
        headers=VALID_KEY,
        json={"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "engine_unavailable",
            "message": "No inference engine is configured",
        }
    }


def test_chat_completions_rejects_client_tenant_identity() -> None:
    response = client.post(
        "/v1/chat/completions",
        headers=VALID_KEY,
        json={"model": "test-model", "messages": [], "tenant_id": "another-tenant"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_chat_completions_rejects_unknown_model_before_engine_call() -> None:
    response = client.post(
        "/v1/chat/completions",
        headers=VALID_KEY,
        json={"model": "another-model", "messages": []},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_nonstream_completion_is_forwarded_and_engine_usage_is_preserved(monkeypatch, tmp_path) -> None:
    engine_api, engine_auth = engine_client(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        gateway_module,
        "httpx",
        type("HttpxWithFakeClient", (), {
            "AsyncClient": lambda *args, **kwargs: FakeEngineClient(*args, calls=calls, **kwargs)
        }),
    )
    monkeypatch.setattr(engine_module, "httpx", gateway_module.httpx)
    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0,
        "stream_options": {"include_usage": True},
    }

    response = engine_api.post("/v1/chat/completions", headers=engine_auth, json=request_body)

    assert response.status_code == 200, response.text
    assert response.json()["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}
    assert response.headers["x-request-id"].startswith("req_")
    assert len(calls) == 2
    assert calls[1]["json"]["messages"] == request_body["messages"]
    assert calls[1]["json"]["stream_options"] == {"include_usage": True}
    assert calls[1]["json"]["rid"] == response.headers["x-request-id"]
    usage = engine_api.get("/v1/usage", headers=engine_auth).json()
    assert usage["summary"]["input_tokens"] == 3
    assert usage["summary"]["output_tokens"] == 2
    assert usage["summary"]["cost_micro_usd"] == 18
    assert usage["summary"]["reserved_micro_usd"] == 0


def test_second_key_cannot_bypass_shared_token_quota(monkeypatch, tmp_path) -> None:
    engine_api, _ = engine_client(tmp_path)
    store = engine_api.app.state.store
    tenant = store.create_tenant(
        name="Shared token quota",
        balance_micro_usd=10000,
        token_limit=10,
        rpm_limit=100,
        concurrency_limit=10,
    )
    key_a = store.create_key(tenant["tenant_id"], name="key A")["api_key"]
    key_b = store.create_key(tenant["tenant_id"], name="key B")["api_key"]
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        gateway_module,
        "httpx",
        type("HttpxWithFakeClient", (), {
            "AsyncClient": lambda *args, **kwargs: FakeEngineClient(*args, calls=calls, **kwargs)
        }),
    )
    monkeypatch.setattr(engine_module, "httpx", gateway_module.httpx)
    body = {"model": "test-model", "messages": [], "max_tokens": 5}

    first = engine_api.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key_a}"}, json=body)
    second = engine_api.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key_b}"}, json=body)

    assert first.status_code == 200
    assert second.status_code == 402
    assert second.json()["error"]["code"] == "insufficient_quota"
    usage = engine_api.get("/v1/usage", headers={"Authorization": f"Bearer {key_b}"}).json()
    assert usage["summary"]["total_tokens"] == 5


def test_sse_completion_relays_events_and_done_without_buffer_rewrite(monkeypatch, tmp_path) -> None:
    engine_api, engine_auth = engine_client(tmp_path)
    calls: list[dict[str, object]] = []
    body = (
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
        b"data: [DONE]\n\n"
    )
    monkeypatch.setattr(
        gateway_module,
        "httpx",
        type("HttpxWithFakeClient", (), {
            "AsyncClient": lambda *args, **kwargs: FakeEngineClient(
                *args, calls=calls, stream_body=body, **kwargs
            )
        }),
    )
    monkeypatch.setattr(engine_module, "httpx", gateway_module.httpx)

    response = engine_api.post(
        "/v1/chat/completions",
        headers=engine_auth,
        json={"model": "test-model", "messages": [], "stream": True},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == body
    assert response.headers["x-accel-buffering"] == "no"
    assert len(calls) == 2
    usage = engine_api.get("/v1/usage", headers=engine_auth).json()
    assert usage["summary"]["cost_micro_usd"] == 18


def test_incomplete_sse_requests_engine_abort_for_same_request_id(monkeypatch, tmp_path) -> None:
    engine_api, engine_auth = engine_client(tmp_path)
    calls: list[dict[str, object]] = []
    body = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
    monkeypatch.setattr(
        gateway_module,
        "httpx",
        type("HttpxWithFakeClient", (), {
            "AsyncClient": lambda *args, **kwargs: FakeEngineClient(
                *args, calls=calls, stream_body=body, **kwargs
            )
        }),
    )
    monkeypatch.setattr(engine_module, "httpx", gateway_module.httpx)

    response = engine_api.post(
        "/v1/chat/completions",
        headers=engine_auth,
        json={"model": "test-model", "messages": [], "stream": True},
    )

    assert response.status_code == 200, response.text
    assert b"event: error" in response.content
    assert b"data: [DONE]" not in response.content
    engine_request = calls[1]
    abort_request = calls[2]
    assert abort_request["url"].endswith("/abort_request")
    assert abort_request["json"]["rid"] == engine_request["json"]["rid"]
    metrics = engine_api.get("/metrics", headers=engine_auth).json()
    assert metrics["counters"]["abort_acknowledged"] == 1
    usage = engine_api.get("/v1/usage", headers=engine_auth).json()
    assert usage["items"][0]["status"] == "pending_reconciliation"
    assert usage["items"][0]["cost_micro_usd"] is None


def test_client_disconnect_aborts_and_releases_tenant_slot(monkeypatch, tmp_path) -> None:
    engine_api, _ = engine_client(tmp_path)
    store = engine_api.app.state.store
    tenant = store.create_tenant(
        name="Disconnect",
        balance_micro_usd=10000,
        token_limit=10000,
        rpm_limit=100,
        concurrency_limit=1,
    )
    api_key = store.create_key(tenant["tenant_id"], name="disconnect key")["api_key"]
    calls: list[dict[str, object]] = []
    stream_body = b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
    monkeypatch.setattr(
        gateway_module,
        "httpx",
        type("HttpxWithFakeClient", (), {
            "AsyncClient": lambda *args, **kwargs: FakeEngineClient(
                *args, calls=calls, stream_body=stream_body, **kwargs
            )
        }),
    )
    monkeypatch.setattr(engine_module, "httpx", gateway_module.httpx)
    body = json.dumps({"model": "test-model", "messages": [], "stream": True, "max_tokens": 1}).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"authorization", f"Bearer {api_key}".encode()),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    received_request = False
    response_chunks = 0

    async def receive():
        nonlocal received_request
        if not received_request:
            received_request = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal response_chunks
        if message["type"] == "http.response.body" and message.get("body"):
            response_chunks += 1
            raise OSError("simulated client disconnect")

    async def run_disconnected_request():
        try:
            await engine_api.app(scope, receive, send)
        except Exception as error:
            assert "disconnect" in str(error).lower()

    asyncio.run(run_disconnected_request())

    usage = store.usage(tenant["tenant_id"])
    assert response_chunks == 1
    assert usage["items"][0]["status"] == "cancelled_unknown"
    assert any(call["url"].endswith("/abort_request") for call in calls)

    following = engine_api.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "test-model", "messages": [], "stream": True, "max_tokens": 1},
    )
    assert following.status_code == 200


def test_sse_usage_parser_handles_fragmented_crlf_events() -> None:
    parser = SSEUsageParser()

    assert parser.feed(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,") == []
    assert parser.feed(b"\"completion_tokens\":2}}\r") == []
    assert parser.feed(b"\n\r\ndata: [DONE]\r\n\r\n") == [
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        "[DONE]",
    ]


def test_metrics_requires_authentication_and_reports_nonnegative_counters() -> None:
    assert client.get("/metrics").status_code == 401

    response = client.get("/metrics", headers=VALID_KEY)

    assert response.status_code == 200
    assert response.json()["active"] == 0
    assert response.json()["queued"] == 0
    assert response.json()["counters"]["abort_acknowledged"] == 0


def test_console_returns_non_sensitive_admin_login_page() -> None:
    response = client.get("/console")

    assert response.status_code == 200
    assert "LLM Service Console" in response.text
    assert "Administrator access" in response.text
    assert 'id="adminKey" type="password"' in response.text


def test_agent_rejects_invalid_request_without_calling_engine() -> None:
    response = client.post(
        "/agent/run",
        headers=VALID_KEY,
        json={"task": None, "tool_data": None},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_agent_request"


def test_agent_executes_planner_whitelisted_tool_and_final_answer_with_child_ledger(
    monkeypatch, tmp_path
) -> None:
    engine_api, engine_auth = engine_client(tmp_path)
    generated: list[dict[str, object]] = []
    planner = '{"name":"lookup_order","arguments":{"order_id":"A20417"}}'
    final = '{"order_id":"A20417","status":"shipped","eta_days":3}'

    class AgentEngineClient:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, json, headers=None):
            if url.endswith("/tokenize"):
                return FakeEngineResponse(b'{"count":3,"tokens":[1,2,3]}')
            generated.append({"url": url, "json": json, "headers": headers})
            content = planner if len(generated) == 1 else final
            body = jsonlib.dumps({
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }).encode()
            return FakeEngineResponse(body)

    import json as jsonlib

    monkeypatch.setattr(
        gateway_module,
        "httpx",
        type("HttpxWithAgentClient", (), {"AsyncClient": AgentEngineClient}),
    )
    monkeypatch.setattr(agent_module, "httpx", gateway_module.httpx)
    monkeypatch.setattr(engine_module, "httpx", gateway_module.httpx)
    request = {
        "task": "Where is order A20417? Return order_id, status and eta_days.",
        "tool_data": {"orders": {"A20417": {"status": "shipped", "eta_days": 3}}},
    }
    response = engine_api.post(
        "/agent/run",
        headers={**engine_auth, "Idempotency-Key": "agent-once"},
        json=request,
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "answer": {"order_id": "A20417", "status": "shipped", "eta_days": 3},
        "tool_calls": [{"name": "lookup_order", "arguments": {"order_id": "A20417"}}],
        "usage": {"model_calls": 2, "completion_tokens": 4},
    }
    assert response.headers["x-request-id"].startswith("req_")
    assert len(generated) == 2
    assert generated[0]["json"]["rid"] != generated[1]["json"]["rid"]
    usage = engine_api.get("/v1/usage", headers=engine_auth).json()
    agent_items = [item for item in usage["items"] if item["route"].startswith("/agent/run")]
    parent = next(item for item in agent_items if item["route"] == "/agent/run")
    children = [item for item in agent_items if item["route"] != "/agent/run"]
    assert len(children) == 2
    assert all(item["parent_request_id"] == parent["request_id"] for item in children)
    assert parent["cost_micro_usd"] == 0

    duplicate = engine_api.post(
        "/agent/run",
        headers={**engine_auth, "Idempotency-Key": "agent-once"},
        json=request,
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "request_already_completed"
    assert len(generated) == 2
