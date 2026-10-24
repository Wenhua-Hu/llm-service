"""SSE helpers for safely relaying engine streams to API clients."""

import json


def sse_error(code: str, message: str) -> bytes:
    """Format an OpenAI gateway error as an SSE event."""

    data = json.dumps({"error": {"code": code, "message": message}}, separators=(",", ":"))
    return f"event: error\ndata: {data}\n\n".encode("utf-8")


class SSEUsageParser:
    """Parse complete SSE data events while accepting arbitrary network chunking."""

    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, chunk: bytes) -> list[dict[str, object] | str]:
        self._buffer += chunk
        self._buffer = self._buffer.replace(b"\r\n", b"\n")
        events: list[dict[str, object] | str] = []
        while b"\n\n" in self._buffer:
            block, self._buffer = self._buffer.split(b"\n\n", 1)
            data_lines = []
            for raw_line in block.split(b"\n"):
                line = raw_line.rstrip(b"\r")
                if line.startswith(b"data:"):
                    data_lines.append(line[5:].lstrip(b" ").decode("utf-8"))
            if not data_lines:
                continue
            data = "\n".join(data_lines)
            if data == "[DONE]":
                events.append(data)
                continue
            try:
                value = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
        return events


def usage_counts(usage: dict[str, object]) -> tuple[int, int, int] | None:
    """Read validated input, output, and cached token counts from engine usage."""

    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    details = usage.get("prompt_tokens_details", {})
    cached_tokens = usage.get(
        "cached_tokens", details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    )
    values = (input_tokens, output_tokens, cached_tokens)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        return None
    return input_tokens, output_tokens, cached_tokens
