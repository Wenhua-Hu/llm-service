"""HTTP adapter helpers for the configured SGLang-compatible engine."""

import httpx


async def tokenize_request(engine_base_url: str, model: str, payload: dict[str, object]) -> int:
    """Ask the engine's real tokenizer for the prompt token count."""

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{engine_base_url}/tokenize",
            json={"model": model, "messages": payload.get("messages", [])},
        )
    response.raise_for_status()
    result = response.json()
    count = result.get("count") if isinstance(result, dict) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("tokenize endpoint returned invalid count")
    return count


async def abort_request(engine_base_url: str, request_id: str, metrics) -> None:
    """Ask the engine to cancel the exact request whose client stream ended early."""

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{engine_base_url}/abort_request", json={"rid": request_id}
            )
        if response.is_success:
            await metrics.increment("abort_acknowledged")
    except httpx.HTTPError:
        return
