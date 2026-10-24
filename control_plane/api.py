"""Admin API routes for tenant, key, usage, audit, and pricing management."""

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, Response

from control_plane.storage import LedgerError, Store
from server.auth import ledger_error


def create_admin_router(store: Store, model: str) -> APIRouter:
    router = APIRouter()

    @router.post("/admin/tenants", status_code=201)
    async def admin_create_tenant(request: Request) -> Response:
        try:
            payload = await _json_object(request)
            result = store.create_tenant(
                name=payload.get("name"),
                balance_micro_usd=payload.get("balance_micro_usd"),
                token_limit=payload.get("token_limit"),
                rpm_limit=payload.get("rpm_limit"),
                concurrency_limit=payload.get("concurrency_limit"),
            )
            return JSONResponse(status_code=201, content=result)
        except LedgerError as error:
            return ledger_error(error)

    @router.get("/admin/tenants")
    async def admin_list_tenants() -> dict[str, object]:
        return {"items": store.list_tenants()}

    @router.patch("/admin/tenants/{tenant_id}")
    async def admin_update_tenant(tenant_id: str, request: Request) -> Response:
        try:
            return JSONResponse(content=store.update_tenant(tenant_id, await _json_object(request)))
        except LedgerError as error:
            return ledger_error(error)

    @router.post("/admin/tenants/{tenant_id}/credits")
    async def admin_adjust_credits(tenant_id: str, request: Request) -> Response:
        try:
            payload = await _json_object(request)
            return JSONResponse(
                content=store.adjust_credits(
                    tenant_id,
                    delta_micro_usd=payload.get("delta_micro_usd"),
                    reason=payload.get("reason"),
                    idempotency_key=request.headers.get("Idempotency-Key", ""),
                )
            )
        except LedgerError as error:
            return ledger_error(error)

    @router.post("/admin/tenants/{tenant_id}/keys", status_code=201)
    async def admin_create_key(tenant_id: str, request: Request) -> Response:
        try:
            payload = await _json_object(request)
            result = store.create_key(
                tenant_id, name=payload.get("name"), expires_at=payload.get("expires_at")
            )
            return JSONResponse(status_code=201, content=result)
        except LedgerError as error:
            return ledger_error(error)

    @router.get("/admin/tenants/{tenant_id}/keys")
    async def admin_list_keys(tenant_id: str) -> Response:
        try:
            return JSONResponse(content={"items": store.list_keys(tenant_id)})
        except LedgerError as error:
            return ledger_error(error)

    @router.post("/admin/keys/{key_id}/rotate")
    async def admin_rotate_key(key_id: str, request: Request) -> Response:
        try:
            payload = await _json_object(request)
            return JSONResponse(content=store.rotate_key(key_id, expires_at=payload.get("expires_at")))
        except LedgerError as error:
            return ledger_error(error)

    @router.post("/admin/keys/{key_id}/revoke")
    async def admin_revoke_key(key_id: str) -> Response:
        try:
            return JSONResponse(content=store.revoke_key(key_id))
        except LedgerError as error:
            return ledger_error(error)

    @router.get("/admin/usage")
    async def admin_usage(
        tenant_id: str | None = None,
        from_time: str | None = Query(default=None, alias="from"),
        to_time: str | None = Query(default=None, alias="to"),
        cursor: str | None = None,
        limit: int = 100,
    ) -> Response:
        try:
            result = store.usage(
                tenant_id, from_time=from_time, to_time=to_time, cursor=cursor, limit=limit
            )
            return JSONResponse(content=result)
        except LedgerError as error:
            return ledger_error(error)

    @router.get("/admin/audit")
    async def admin_audit(tenant_id: str | None = None) -> Response:
        try:
            return JSONResponse(content={"items": store.audit_events(tenant_id)})
        except LedgerError as error:
            return ledger_error(error)

    @router.get("/admin/pricing")
    async def admin_get_pricing() -> Response:
        try:
            return JSONResponse(content=store.get_pricing(model))
        except LedgerError as error:
            return ledger_error(error)

    @router.put("/admin/pricing")
    async def admin_set_pricing(request: Request) -> Response:
        try:
            payload = await _json_object(request)
            return JSONResponse(
                content=store.set_pricing(
                    payload.get("model", model),
                    input_price=payload.get("input_micro_usd_per_token"),
                    output_price=payload.get("output_micro_usd_per_token"),
                )
            )
        except LedgerError as error:
            return ledger_error(error)

    return router


async def _json_object(request: Request) -> dict[str, object]:
    try:
        payload = await request.json()
    except (ValueError, UnicodeDecodeError) as error:
        raise LedgerError(400, "invalid_json", "Request body must be valid JSON") from error
    if not isinstance(payload, dict):
        raise LedgerError(400, "invalid_request", "Request body must be a JSON object")
    return payload
