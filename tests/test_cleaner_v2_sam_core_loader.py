import os
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_core_loader import SamCoreBuilder
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@pytest.fixture
def guard(tmp_path):
    return ResourceGuard(
        ResourceBudget(1000000, 1000000, 10000), tmp_path, time.monotonic() + 30, threading.Event()
    )


def test_missing_source_does_not_create_module_namespace(guard):
    before = set(sys.modules)
    with pytest.raises(CleanExecutionError) as error:
        SamCoreBuilder.build(guard.workspace, guard.workspace / "config.json", guard)
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE
    assert not (set(sys.modules) - before)


def test_mismatched_source_is_not_executed(guard):
    path = guard.workspace / "untrusted.py"
    path.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(CleanExecutionError):
        SamCoreBuilder._read(path, "0" * 64, guard)


@pytest.mark.skipif(
    not os.environ.get("HEAR_SAM_TEST_SOURCE"), reason="explicit pinned source required"
)
def test_real_meta_core_shapes_buffers_and_namespace_cleanup(guard):
    before = set(sys.modules)
    rng = torch.random.get_rng_state().clone()
    built = SamCoreBuilder.build(
        Path(os.environ["HEAR_SAM_TEST_SOURCE"]), Path(os.environ["HEAR_SAM_TEST_CONFIG"]), guard
    )
    try:
        state = built.core.state_dict()
        assert len(state) == 247
        assert all(
            value.device.type == "meta" and value.dtype == torch.float32 for value in state.values()
        )
        buffers = list(built.core.buffers())
        assert buffers and all(
            value.device.type == "cpu" and torch.isfinite(value).all() for value in buffers
        )
        assert torch.equal(rng, torch.random.get_rng_state())
        assert not any(
            name.startswith(("sam_audio", "core.audio_visual_encoder"))
            for name in set(sys.modules) - before
        )
        assert not built.core.training
    finally:
        built.close()
        built.close()
    assert built.core is None
    assert not any(name.startswith("_hear_sam_core_") for name in set(sys.modules) - before)
