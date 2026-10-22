"""Bearer authentication for inference-facing routes."""

from fastapi import Request
from fastapi.responses import JSONResponse

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


def install_authentication(app, store: Store) -> None:
    """Protect every route except the explicit public surface."""

    @app.middleware("http")
    async def authentication(request: Request, call_next):
        if request.url.path not in PUBLIC_PATHS:
            header = request.headers.get("authorization", "")
            raw_key = header[7:] if header.startswith("Bearer ") else ""
            try:
                request.state.auth = store.authenticate(raw_key)
            except LedgerError as error:
                return ledger_error(error)

        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response
