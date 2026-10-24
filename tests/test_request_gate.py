import asyncio

import pytest

from server.admission import GateFull, RequestGate


def test_gate_rejects_when_slot_and_queue_are_full() -> None:
    async def run() -> None:
        gate = RequestGate(slots=1, queue=0)
        await gate.acquire()

        with pytest.raises(GateFull):
            await gate.acquire()

        assert await gate.snapshot() == {"active": 1, "queued": 0}
        await gate.release()
        assert await gate.snapshot() == {"active": 0, "queued": 0}

    asyncio.run(run())


def test_gate_queues_one_request_and_releases_capacity() -> None:
    async def run() -> None:
        gate = RequestGate(slots=1, queue=1, queue_timeout_s=1.0)
        await gate.acquire()
        waiter = asyncio.create_task(gate.acquire())
        await asyncio.sleep(0)

        assert await gate.snapshot() == {"active": 1, "queued": 1}
        await gate.release()
        await waiter

        assert await gate.snapshot() == {"active": 1, "queued": 0}
        await gate.release()

    asyncio.run(run())


def test_gate_times_out_waiter_and_restores_queue_count() -> None:
    async def run() -> None:
        gate = RequestGate(slots=1, queue=1, queue_timeout_s=0.01)
        await gate.acquire()

        with pytest.raises(GateFull):
            await gate.acquire()

        assert await gate.snapshot() == {"active": 1, "queued": 0}
        await gate.release()

    asyncio.run(run())
