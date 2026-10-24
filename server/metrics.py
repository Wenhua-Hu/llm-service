"""Small process-local service counters and request-gate snapshot."""

import asyncio


class ServiceMetrics:
    COUNTERS = ("completed", "rejected", "cancelled", "timed_out", "abort_acknowledged")

    def __init__(self) -> None:
        self._counters = {name: 0 for name in self.COUNTERS}
        self._lock = asyncio.Lock()

    async def increment(self, name: str) -> None:
        if name not in self._counters:
            raise KeyError(f"unknown metric counter: {name}")
        async with self._lock:
            self._counters[name] += 1

    async def snapshot(self, gate) -> dict[str, object]:
        capacity = await gate.snapshot()
        async with self._lock:
            return {**capacity, "counters": dict(self._counters)}
