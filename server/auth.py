"""Bearer authentication for inference-facing routes."""

from fastapi import Request
from fastapi.responses import JSONResponse
import hmac

from control_plane.storage import LedgerError, Store


PUBLIC_PATHS = frozenset({"/healthz", "/console"})


def ledger_error(error: LedgerError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status,
        content={
            "error": {
                "code": error.code,
                "message": error.message,
            }
        },
    )


def install_authentication(app, store: Store, admin_api_key: str = "") -> None:
    """Protect every route except the explicit public surface."""

    @app.middleware("http")
    async def authentication(request: Request, call_next):
        if request.url.path not in PUBLIC_PATHS:
            header = request.headers.get("authorization", "")
            raw_key = header[7:] if header.startswith("Bearer ") else ""
            if request.url.path == "/admin" or request.url.path.startswith("/admin/"):
                if not admin_api_key:
                    return ledger_error(
                        LedgerError(503, "admin_api_disabled", "Set ADMIN_API_KEY to enable management APIs")
                    )
                if hmac.compare_digest(raw_key, admin_api_key):
                    request.state.admin = True
                else:
                    try:
                        store.authenticate(raw_key)
                    except LedgerError as error:
                        return ledger_error(error)
                    return ledger_error(LedgerError(403, "admin_required"))
            else:
                if (
                    "tenant_id" in request.query_params
                    or request.headers.get("X-Tenant-ID")
                    or request.headers.get("X-User-ID")
                ):
                    return ledger_error(LedgerError(403, "tenant_override_forbidden"))
                try:
                    request.state.auth = store.authenticate(raw_key)
                except LedgerError as error:
                    return ledger_error(error)

        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response
