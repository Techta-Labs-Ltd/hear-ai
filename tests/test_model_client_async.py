from unittest.mock import Mock

import pytest

from hear.services.model_client import RayModelClient


@pytest.mark.anyio
async def test_sync_adapter_rejects_event_loop_blocking_and_cancels_response():
    response = Mock()
    with pytest.raises(RuntimeError, match="synchronous_model_call_in_async_context"):
        RayModelClient({})._resolve_sync(response)
    response.cancel.assert_called_once()
    response.result.assert_not_called()


def test_legacy_sync_adapter_resolves_on_worker_thread():
    response = Mock()
    response.result.return_value = {"scores": [0.9]}
    assert RayModelClient({})._resolve_sync(response) == {"scores": [0.9]}
    response.cancel.assert_not_called()
