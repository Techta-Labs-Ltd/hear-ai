import os
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest
import torch

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_codec_loader import SamCodecBuilder
from hear.services.magic_clean.contracts import CleanExecutionError


@pytest.fixture
def guard(tmp_path):
    return ResourceGuard(
        ResourceBudget(1000000, 1000000, 10000), tmp_path, time.monotonic() + 60, threading.Event()
    )


def test_foreign_namespace_preserved_and_lease_released(guard, monkeypatch):
    foreign = ModuleType("dacvae")
    monkeypatch.setitem(sys.modules, "dacvae", foreign)
    for _ in range(2):
        with pytest.raises(CleanExecutionError, match="foreign DACVAE"):
            SamCodecBuilder.build(guard.workspace, guard.workspace / "config", guard)
        assert sys.modules["dacvae"] is foreign


@pytest.mark.skipif(
    not os.environ.get("HEAR_DACVAE_TEST_SOURCE"), reason="explicit pinned codec source required"
)
def test_real_meta_codec_is_bounded_and_owns_namespace(guard):
    original_rng = torch.random.get_rng_state().clone()
    for _ in range(2):
        built = SamCodecBuilder.build(
            Path(os.environ["HEAR_DACVAE_TEST_SOURCE"]),
            Path(os.environ["HEAR_SAM_TEST_CONFIG"]),
            guard,
        )
        try:
            state = built.codec.state_dict()
            assert len(state) == 317
            assert all(
                value.device.type == "meta" and value.dtype == torch.float32
                for value in state.values()
            )
            assert built.codec.decoder.alpha == 0.25
            assert built.codec.decoder.wm_model.msg_processor.nbits == 16
            # Resource preflight's lower bound is tied to this pinned final stage.
            final_stage = built.codec.decoder.model[-1]
            upsample = next(
                layer
                for layer in final_stage.modules()
                if isinstance(layer, torch.nn.ConvTranspose1d)
            )
            assert upsample.out_channels == 96 and upsample.stride == (2,)
            first_activation = final_stage.block[4].block[0]
            with torch.device("meta"):
                assert first_activation(torch.empty(2, 96, 5760)).shape == (2, 96, 5760)
            assert torch.equal(original_rng, torch.random.get_rng_state())
            with pytest.raises(CleanExecutionError, match="namespace occupied"):
                SamCodecBuilder.build(guard.workspace, guard.workspace / "absent", guard)
        finally:
            built.close()
            built.close()
        assert built.codec is None
        assert not any(name == "dacvae" or name.startswith("dacvae.") for name in sys.modules)
