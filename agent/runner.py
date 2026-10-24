"""Agent workflow executed by the /agent/run HTTP route."""

import json

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response

from control_plane.storage import LedgerError, Store
from server.admission import GateFull, RequestGate
from server.auth import ledger_error
from server.config import Settings
from server.engine import tokenize_request
from server.metrics import ServiceMetrics
from server.streaming import usage_counts


def _error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


async def run_agent(
    request: Request,
    *,
    settings: Settings,
    store: Store,
    gate: RequestGate,
    metrics: ServiceMetrics,
) -> Response:
    """Run planning, the whitelisted lookup tool, and the final model call."""


    try:
        payload = await request.json()
    except (ValueError, UnicodeDecodeError):
        return _error_response(400, "invalid_json", "Request body must be valid JSON")
    if not isinstance(payload, dict) or payload.get("task") is None or payload.get("tool_data") is None:
        return _error_response(422, "invalid_agent_request", "task and tool_data are required")
    task = payload["task"]
    tool_data = payload["tool_data"]
    if not isinstance(task, str) or not task.strip() or len(task) > 8000:
        return _error_response(422, "invalid_agent_request", "task must be a non-empty string up to 8000 characters")
    if not isinstance(tool_data, dict) or not isinstance(tool_data.get("orders"), dict):
        return _error_response(422, "invalid_agent_request", "tool_data.orders must be an object")
    if not settings.engine_base_url:
        return _error_response(503, "engine_unavailable", "No inference engine is configured")

    try:
        parent = store.reserve_request(
            auth=request.state.auth,
            route="/agent/run",
            model=settings.model_id,
            body=payload,
            input_tokens=0,
            max_output_tokens=0,
            idempotency_key=request.headers.get("Idempotency-Key"),
        )
        parent_id = str(parent["request_id"])
        store.journal_usage(parent_id, 0, 0)
        store.settle_request(parent_id)
    except LedgerError as error:
        return ledger_error(error)

    async def model_call(label: str, messages: list[dict[str, str]]) -> tuple[str, tuple[int, int]]:
        model_body: dict[str, object] = {
            "model": settings.model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 512,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        try:
            input_count = await tokenize_request(settings.engine_base_url, settings.model_id, model_body)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise LedgerError(502, "tokenization_failed", "The inference engine could not tokenize the Agent step")
        try:
            await gate.acquire()
        except GateFull as error:
            await metrics.increment("timed_out" if error.reason == "timeout" else "rejected")
            raise LedgerError(429, "overloaded", "Inference queue is full")
        try:
            child = store.reserve_request(
                auth=request.state.auth,
                route=f"/agent/run/{label}",
                model=settings.model_id,
                body=model_body,
                input_tokens=input_count,
                max_output_tokens=512,
                parent_request_id=parent_id,
            )
        except LedgerError:
            await gate.release()
            raise
        child_id = str(child["request_id"])
        model_body["rid"] = child_id
        store.mark_engine_started(child_id)
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                upstream = await client.post(
                    f"{settings.engine_base_url}/v1/chat/completions",
                    json=model_body,
                    headers={"X-Request-ID": child_id},
                )
        except httpx.HTTPError as error:
            store.mark_pending_reconciliation(child_id, "engine_connection_failed")
            raise LedgerError(502, "engine_unavailable", "Agent model call failed") from error
        finally:
            await gate.release()
        if upstream.status_code >= 400:
            if upstream.status_code < 500:
                store.release_request(child_id, "engine_rejected_request")
            else:
                store.mark_pending_reconciliation(child_id, "engine_error_without_usage")
            raise LedgerError(502, "engine_error", "Agent model call failed")
        try:
            result = upstream.json()
            choices = result.get("choices") if isinstance(result, dict) else None
            message = choices[0].get("message") if isinstance(choices, list) and choices else None
            content = message.get("content") if isinstance(message, dict) else None
            usage = result.get("usage") if isinstance(result, dict) else None
            counts = usage_counts(usage) if isinstance(usage, dict) else None
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            content, counts = None, None
        if not isinstance(content, str) or counts is None:
            store.mark_pending_reconciliation(child_id, "successful_response_without_usage")
            raise LedgerError(502, "invalid_engine_response", "Agent model response was incomplete")
        store.journal_usage(child_id, *counts)
        settlement = store.settle_request(child_id)
        if settlement["status"] != "settled":
            raise LedgerError(502, "usage_exceeds_reservation", "Agent model usage requires reconciliation")
        await metrics.increment("completed")
        return content, (counts[0], counts[1])

    try:
        planner_text, planner_usage = await model_call(
            "planner",
            [
                {"role": "system", "content": "Choose exactly one tool call as JSON: {\"name\":\"lookup_order\",\"arguments\":{\"order_id\":\"...\"}}. Never invent tool names."},
                {"role": "user", "content": json.dumps({"task": task, "available_order_ids": list(tool_data["orders"])}, ensure_ascii=False)},
            ],
        )
        plan = json.loads(planner_text)
        if not isinstance(plan, dict) or plan.get("name") != "lookup_order":
            return _error_response(422, "unsupported_tool", "Only lookup_order is allowed")
        arguments = plan.get("arguments")
        order_id = arguments.get("order_id") if isinstance(arguments, dict) else None
        if not isinstance(order_id, str) or order_id not in tool_data["orders"]:
            return _error_response(422, "tool_not_found", "lookup_order did not match a provided order")
        order = tool_data["orders"][order_id]
        if not isinstance(order, dict):
            return _error_response(422, "invalid_tool_data", "The selected order must be an object")
        final_text, final_usage = await model_call(
            "final",
            [
                {"role": "system", "content": "Answer the task using the tool result. Return only a JSON object."},
                {"role": "user", "content": json.dumps({"task": task, "tool_result": {"order_id": order_id, **order}}, ensure_ascii=False)},
            ],
        )
        answer = json.loads(final_text)
        return JSONResponse(
            content={
                "answer": answer,
                "tool_calls": [{"name": "lookup_order", "arguments": {"order_id": order_id}}],
                "usage": {
                    "model_calls": 2,
                    "completion_tokens": planner_usage[1] + final_usage[1],
                },
            },
            headers={"X-Request-ID": parent_id},
        )
    except LedgerError as error:
        return ledger_error(error)
    except (json.JSONDecodeError, TypeError):
        return _error_response(502, "invalid_agent_plan", "The Agent model returned invalid JSON")
