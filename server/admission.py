"""A bounded async gate for inference requests."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class GateFull(Exception):
    """Raised when the bounded wait queue is full or times out."""

    def __init__(self, reason: str = "queue_full") -> None:
        super().__init__(reason)
        self.reason = reason


class RequestGate:
    def __init__(self, slots: int, queue: int, queue_timeout_s: float = 30.0) -> None:
        if not 1 <= slots <= 64 or not 0 <= queue <= 112 or slots + queue > 120:
            raise ValueError("invalid gateway capacity")
        self.slots = slots
        self.queue_limit = queue
        self.queue_timeout_s = queue_timeout_s
        self.active = 0
        self.queued = 0
        self._condition = asyncio.Condition()

    async def acquire(self) -> None:
        async with self._condition:
            if self.active < self.slots:
                self.active += 1
                return
            if self.queued >= self.queue_limit:
                raise GateFull
            self.queued += 1
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(lambda: self.active < self.slots),
                    timeout=self.queue_timeout_s,
                )
                self.active += 1
            except TimeoutError as error:
                raise GateFull("timeout") from error
            finally:
                self.queued -= 1

    async def release(self) -> None:
        async with self._condition:
            if self.active < 1:
                raise RuntimeError("request gate released without an active slot")
            self.active -= 1
            self._condition.notify_all()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        await self.acquire()
        try:
            yield
        finally:
            await self.release()

    async def snapshot(self) -> dict[str, int]:
        async with self._condition:
            return {"active": self.active, "queued": self.queued}
