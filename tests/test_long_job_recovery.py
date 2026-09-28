import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hear.runtime.attempt_stream import AttemptRejection
from hear.runtime.pod import PodRuntime
from scripts.simulation_backend import SimulationBackend
from tests.test_cleaning_profile_workflow import envelope


def test_broker_timeout_covers_long_model_jobs():
    lines = Path("deploy/runtime/rabbitmq.conf").read_text().splitlines()
    settings = dict(
        line.split("=", 1) for line in lines if "=" in line and not line.lstrip().startswith("#")
    )
    value = next(value for key, value in settings.items() if key.strip() == "consumer_timeout")
    assert int(value.strip()) >= 10800000


def test_simulator_does_not_count_expired_leases_as_active():
    assert SimulationBackend.lease_live({"status": "running", "last_heartbeat": time.time()})
    assert not SimulationBackend.lease_live(
        {"status": "running", "last_heartbeat": time.time() - 91}
    )
    assert not SimulationBackend.lease_live({"status": "completed", "last_heartbeat": time.time()})


@pytest.mark.parametrize(
    "decision,retry", [("lease_unavailable", True), ("already_completed", False)]
)
def test_lease_wait_redelivery_is_not_discarded(decision, retry):
    value = envelope("natural")
    runtime = PodRuntime.__new__(PodRuntime)
    runtime.prepare_attempt = AsyncMock(
        return_value=SimpleNamespace(
            result=AttemptRejection("claim_rejected", {"decision": decision})
        )
    )
    runtime._retry_or_dead_letter = AsyncMock(return_value=(False, 0))
    runtime._publish_local = AsyncMock()
    message = SimpleNamespace(
        body=value.model_dump_json().encode(), reply_to=None, ack=AsyncMock(), reject=AsyncMock()
    )
    asyncio.run(runtime._on_rabbitmq_message(message))
    assert runtime._retry_or_dead_letter.await_count == int(retry)
    assert message.ack.await_count == int(not retry)
    if retry:
        assert runtime._retry_or_dead_letter.call_args.kwargs["capacity_wait"] is True
