"""FastAPI application for the LLM service gateway."""

import asyncio
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from control_plane.storage import LedgerError, Store
from control_plane.api import create_admin_router
from agent.runner import run_agent
from server.auth import install_authentication, ledger_error
from server.config import Settings
from server.admission import GateFull, RequestGate
from server.engine import abort_request, tokenize_request
from server.metrics import ServiceMetrics
from server.streaming import SSEUsageParser, sse_error as _sse_error, usage_counts as _usage_counts


def create_app(
    settings: Settings | None = None,
    store: Store | None = None,
) -> FastAPI:
    """Create an application with explicit, testable configuration."""

    settings = settings or Settings.from_env()
    store = store or Store(
        settings.data_dir,
        settings.key_hash_secret,
        settings.api_key,
        settings.model_id,
    )
    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
    )
    app.state.store = store
    install_authentication(app, store, settings.admin_api_key)
    app.state.gate = RequestGate(settings.gateway_slots, settings.gateway_queue)
    app.state.metrics = ServiceMetrics()

    @app.get("/healthz")
    async def healthz() -> Response:
        """Report readiness only after the configured inference engine is healthy."""

        if not settings.engine_base_url:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{settings.engine_base_url}/health")
            if not response.is_success:
                return JSONResponse(status_code=503, content={"status": "not_ready"})
        except httpx.HTTPError:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return JSONResponse(content={"status": "ready"})

    @app.get("/info")
    async def info() -> dict[str, object]:
        """Return authenticated service metadata used by contract checks."""
        runtime: dict[str, object] = {}
        if settings.engine_base_url:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(f"{settings.engine_base_url}/server_info")
                if response.is_success:
                    data = response.json()
                    if isinstance(data, dict):
                        runtime = data
            except (httpx.HTTPError, ValueError):
                runtime = {}
        config: dict[str, object] = {
            "gateway_slots": settings.gateway_slots,
            "gateway_queue": settings.gateway_queue,
            "prefix_cache": False,
            "attention_backend": "unset",
            "sampling_defaults": "openai",
            "repetition_penalty": 1.0,
            "top_p": 1.0,
            "top_k": -1,
        }
        for field in ("max_running_requests", "context_length", "max_total_tokens", "dtype", "attention_backend"):
            if field in runtime:
                config[field] = runtime[field]
        if "disable_radix_cache" in runtime and isinstance(runtime["disable_radix_cache"], bool):
            config["prefix_cache"] = not runtime["disable_radix_cache"]
        elif "enable_prefix_caching" in runtime and isinstance(runtime["enable_prefix_caching"], bool):
            config["prefix_cache"] = runtime["enable_prefix_caching"]
        return {
            "contract": "llm-lab/1.0",
            "service": settings.service_name,
            "version": settings.service_version,
            "model_id": settings.model_id,
            "model_revision": settings.model_revision,
            "engine": "sglang",
            "engine_version": runtime.get("version"),
            "engine_configured": settings.engine_base_url is not None,
            "profile": settings.profile,
            "config": config,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        """List the model exposed by this gateway."""

        return {
            "object": "list",
            "data": [{"id": settings.model_id, "object": "model"}],
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, object]:
        return await app.state.metrics.snapshot(app.state.gate)

    @app.get("/console", response_class=HTMLResponse)
    async def console() -> HTMLResponse:
        """Serve the browser console; credentials are entered client-side, not embedded."""

        page = Path(__file__).resolve().parents[1] / "console" / "index.html"
        return HTMLResponse(
            content=page.read_text(encoding="utf-8"),
            headers={"Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"},
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        """Forward an OpenAI-compatible completion request to the configured engine."""

        try:
            payload = await request.json()
        except (ValueError, UnicodeDecodeError):
            return _error_response(400, "invalid_json", "Request body must be valid JSON")
        if not isinstance(payload, dict):
            return _error_response(400, "invalid_request", "Request body must be a JSON object")
        if any(field in payload for field in ("tenant_id", "user_id", "key_id")):
            return _error_response(400, "invalid_request", "Tenant identity comes from the API key")
        if payload.get("model") not in {None, settings.model_id}:
            return _error_response(404, "model_not_found", "Requested model is not available")
        if not settings.engine_base_url:
            return _error_response(503, "engine_unavailable", "No inference engine is configured")

        max_output_tokens = payload.get("max_completion_tokens", payload.get("max_tokens", 256))
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or max_output_tokens < 1:
            return _error_response(400, "invalid_request", "max_tokens must be a positive integer")
        try:
            input_tokens = await tokenize_request(settings.engine_base_url, settings.model_id, payload)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return _error_response(502, "tokenization_failed", "The inference engine could not tokenize this request")
        try:
            await app.state.gate.acquire()
        except GateFull as error:
            await app.state.metrics.increment("timed_out" if error.reason == "timeout" else "rejected")
            return JSONResponse(
                status_code=429,
                content={"error": {"code": "overloaded", "message": "Inference queue is full"}},
                headers={"Retry-After": "1"},
            )
        try:
            reservation = store.reserve_request(
                auth=request.state.auth,
                route="/v1/chat/completions",
                model=settings.model_id,
                body=payload,
                input_tokens=input_tokens,
                max_output_tokens=max_output_tokens,
                idempotency_key=request.headers.get("Idempotency-Key"),
            )
        except LedgerError as error:
            await app.state.gate.release()
            await app.state.metrics.increment("rejected")
            if error.status == 429:
                return JSONResponse(
                    status_code=429,
                    content={"error": {"code": error.code, "message": error.message}},
                    headers={"Retry-After": "1"},
                )
            return ledger_error(error)

        request_id = str(reservation["request_id"])
        upstream_payload = dict(payload)
        upstream_payload["model"] = settings.model_id
        upstream_payload["rid"] = request_id
        upstream_url = f"{settings.engine_base_url}/v1/chat/completions"
        headers = {"X-Request-ID": request_id, "Content-Type": "application/json"}
        store.mark_engine_started(request_id)

        if upstream_payload.get("stream") is True:
            client = httpx.AsyncClient(timeout=45.0)
            try:
                upstream_request = client.build_request(
                    "POST", upstream_url, json=upstream_payload, headers=headers
                )
                upstream = await client.send(upstream_request, stream=True)
            except httpx.HTTPError:
                await client.aclose()
                store.mark_pending_reconciliation(request_id, "engine_connection_failed")
                await app.state.gate.release()
                await app.state.metrics.increment("rejected")
                return _error_response(502, "engine_unavailable", "Inference engine request failed")

            if upstream.status_code >= 400:
                await app.state.metrics.increment("rejected")
                content = await upstream.aread()
                status_code = upstream.status_code
                await upstream.aclose()
                await client.aclose()
                if 400 <= status_code < 500:
                    store.release_request(request_id, "engine_rejected_request")
                else:
                    store.mark_pending_reconciliation(request_id, "engine_error_without_usage")
                await app.state.gate.release()
                return Response(
                    content=content,
                    status_code=status_code,
                    media_type=upstream.headers.get("content-type", "application/json"),
                    headers={"X-Request-ID": request_id, "Cache-Control": "no-store"},
                )

            async def relay() -> object:
                completed = False
                client_cancelled = False
                parser = SSEUsageParser()
                usage_recorded = False
                try:
                    async for chunk in upstream.aiter_bytes():
                        for event in parser.feed(chunk):
                            if event == "[DONE]":
                                completed = True
                            elif isinstance(event, dict) and isinstance(event.get("usage"), dict):
                                usage = _usage_counts(event["usage"])
                                if usage is not None:
                                    store.journal_usage(request_id, *usage)
                                    usage_recorded = True
                        yield chunk
                    if not completed:
                        yield _sse_error("engine_stream_incomplete", "Inference stream ended before [DONE]")
                except asyncio.CancelledError:
                    client_cancelled = True
                    raise
                except httpx.HTTPError:
                    yield _sse_error("engine_stream_error", "Inference stream was interrupted")
                finally:
                    try:
                        if usage_recorded:
                            settlement = store.settle_request(request_id)
                            if settlement["status"] == "settled":
                                await app.state.metrics.increment("completed")
                        elif client_cancelled:
                            store.mark_cancelled_unknown(request_id, "client_cancelled_without_usage")
                        else:
                            store.mark_pending_reconciliation(request_id, "stream_ended_without_usage")
                        if not completed:
                            await app.state.metrics.increment("cancelled")
                            await abort_request(settings.engine_base_url, request_id, app.state.metrics)
                    except Exception:
                        store.mark_pending_reconciliation(request_id, "gateway_stream_cleanup_failed")
                        raise
                    finally:
                        try:
                            await upstream.aclose()
                        finally:
                            try:
                                await client.aclose()
                            finally:
                                await app.state.gate.release()

            return StreamingResponse(
                relay(),
                status_code=upstream.status_code,
                media_type="text/event-stream",
                headers={
                    "X-Request-ID": request_id,
                    "Cache-Control": "no-cache, no-store",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                upstream = await client.post(upstream_url, json=upstream_payload, headers=headers)
        except httpx.HTTPError:
            store.mark_pending_reconciliation(request_id, "engine_connection_failed")
            return _error_response(502, "engine_unavailable", "Inference engine request failed")
        finally:
            await app.state.gate.release()
        if upstream.status_code >= 400:
            await app.state.metrics.increment("rejected")
            if 400 <= upstream.status_code < 500:
                store.release_request(request_id, "engine_rejected_request")
            else:
                store.mark_pending_reconciliation(request_id, "engine_error_without_usage")
        else:
            try:
                body = upstream.json()
                usage = body.get("usage") if isinstance(body, dict) else None
                counts = _usage_counts(usage) if isinstance(usage, dict) else None
            except ValueError:
                counts = None
            if counts is None:
                store.mark_pending_reconciliation(request_id, "successful_response_without_usage")
                return _error_response(502, "usage_unavailable", "The engine response did not include billable usage")
            store.journal_usage(request_id, *counts)
            settlement = store.settle_request(request_id)
            if settlement["status"] != "settled":
                return _error_response(
                    502,
                    "usage_exceeds_reservation",
                    "The engine usage exceeded the authorized reservation and requires reconciliation",
                )
            await app.state.metrics.increment("completed")
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
            headers={"X-Request-ID": request_id, "Cache-Control": "no-store"},
        )

    @app.post("/agent/run")
    async def agent_run(request: Request) -> Response:
        return await run_agent(
            request,
            settings=settings,
            store=store,
            gate=app.state.gate,
            metrics=app.state.metrics,
        )

    @app.get("/v1/usage")
    async def own_usage(request: Request) -> Response:
        if "tenant_id" in request.query_params:
            return ledger_error(LedgerError(403, "tenant_override_forbidden"))
        auth = request.state.auth
        result = store.usage(auth["tenant_id"])
        tenant = next(
            item for item in store.list_tenants() if item["tenant_id"] == auth["tenant_id"]
        )
        result["summary"]["balance_micro_usd"] = tenant["balance_micro_usd"]
        return JSONResponse(content=result)

    app.include_router(create_admin_router(store, settings.model_id))
    return app


app = create_app()


def _error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}},
    )
