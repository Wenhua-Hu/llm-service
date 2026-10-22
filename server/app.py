"""FastAPI application for the LLM service gateway."""

from fastapi import FastAPI

from control_plane.storage import Store
from server.auth import install_authentication
from server.config import Settings


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
    )
    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
    )
    app.state.store = store
    install_authentication(app, store)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Report process liveness without exposing configuration or secrets."""

        return {"status": "ok"}

    @app.get("/info")
    async def info() -> dict[str, str | bool]:
        """Return authenticated service metadata used by contract checks."""

        return {
            "contract": "llm-lab/1.0",
            "service": settings.service_name,
            "version": settings.service_version,
            "model_id": settings.model_id,
            "model_revision": settings.model_revision,
            "engine": "sglang",
            "engine_configured": settings.engine_base_url is not None,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        """List the model exposed by this gateway."""

        return {
            "object": "list",
            "data": [{"id": settings.model_id, "object": "model"}],
        }

    return app


app = create_app()
