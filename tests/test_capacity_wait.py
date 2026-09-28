import asyncio
from types import SimpleNamespace

import pytest

from hear.queue.topology import RabbitMQTopology
from hear.runtime.pod import PodRuntime
from hear.runtime.roles import WorkerRole


class Exchange:
    def __init__(self):
        self.messages = []

    async def publish(self, message, **kwargs):
        self.messages.append(message)


@pytest.mark.parametrize("expired", [False, True])
def test_capacity_wait_does_not_spend_failure_retries(expired):
    runtime = object.__new__(PodRuntime)
    runtime._topology = RabbitMQTopology()
    runtime._role = WorkerRole.RECONSTRUCTION
    runtime._rabbitmq_dead_exchange = Exchange()
    runtime._rabbitmq_retry_exchange = Exchange()
    runtime._rabbitmq_provider = SimpleNamespace(
        Message=lambda **kw: SimpleNamespace(**kw), DeliveryMode=SimpleNamespace(PERSISTENT=2)
    )
    acknowledgements = []

    async def ack():
        acknowledgements.append(True)

    message = SimpleNamespace(
        headers={"x-hear-retry-count": 0},
        body=b"{}",
        content_type="application/json",
        message_id="attempt",
        correlation_id="job",
        reply_to="reply",
        ack=ack,
    )

    async def run():
        for _ in range(runtime._topology.max_retries + 3):
            dead, retries = await runtime._retry_or_dead_letter(
                message, capacity_wait=True, expired=expired
            )
            assert dead is expired
            assert retries == 0

    asyncio.run(run())
    chosen = runtime._rabbitmq_dead_exchange if expired else runtime._rabbitmq_retry_exchange
    assert len(chosen.messages) == runtime._topology.max_retries + 3
    assert len(acknowledgements) == len(chosen.messages)
