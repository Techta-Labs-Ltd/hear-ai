import hashlib
import json
import pickle
import threading
import time
from pathlib import Path

import pytest
import torch

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_checkpoint import SamCheckpointLoader
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


def test_small_optional_manifest_is_pinned_and_exact():
    path = Path(__file__).resolve().parents[1] / "deploy/cleaner/sam-small-optional-keys.json"
    payload = path.read_bytes()
    assert (
        hashlib.sha256(payload).hexdigest()
        == "df43a1d8d8306fab6d9e9d70c991eb364efbdacdddac51bdc2e981a914bbed7e"
    )
    manifest = json.loads(payload)
    assert manifest["schema"] == "sam-optional-keys-v1"
    assert (
        manifest["checkpoint_sha256"]
        == "8c44fda9821fd9f2ec8977304e3c0f55290d9eacb6bbf25b4b8fb1f69c2a8c06"
    )
    keys = manifest["excluded_keys"]
    assert keys == sorted(set(keys)) and len(keys) == 601
    assert all(key.startswith("vision_encoder.") for key in keys)


@pytest.mark.parametrize(
    "fault", [None, "hash", "missing", "extra", "optional_extra", "shape", "dtype"]
)
def test_checkpoint_admission_before_assignment(tmp_path, fault):
    core, codec = torch.nn.Linear(2, 3), torch.nn.Linear(3, 2)
    before = core.weight.detach().clone()
    state = {key: torch.ones_like(value) for key, value in core.state_dict().items()}
    state.update(
        {"audio_codec." + key: torch.ones_like(value) for key, value in codec.state_dict().items()}
    )
    state["vision_encoder.weight"] = torch.ones(2)
    if fault == "missing":
        del state["bias"]
    elif fault == "extra":
        state["unknown"] = torch.ones(1)
    elif fault == "optional_extra":
        state["vision_encoder.unapproved"] = torch.ones(1)
    elif fault == "shape":
        state["audio_codec.weight"] = torch.ones(1)
    elif fault == "dtype":
        state["audio_codec.weight"] = state["audio_codec.weight"].double()
    path = tmp_path / "checkpoint.pt"
    torch.save(state, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    guard = ResourceGuard(
        ResourceBudget(1000000, 1000000, 10000), tmp_path, time.monotonic() + 30, threading.Event()
    )
    kwargs = dict(
        sha256="0" * 64 if fault == "hash" else digest,
        core=core,
        codec=codec,
        optional_keys=frozenset({"vision_encoder.weight"}),
        guard=guard,
    )
    if fault:
        with pytest.raises(CleanExecutionError) as error:
            SamCheckpointLoader.load(path, **kwargs)
        assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE
        assert torch.equal(core.weight, before)
    else:
        assert SamCheckpointLoader.load(path, **kwargs) == (2, 2)
        assert torch.equal(core.weight, torch.ones_like(core.weight))
        assert torch.equal(codec.weight, torch.ones_like(codec.weight))
        assert not core.training and not codec.training
        assert core.weight.device.type == codec.weight.device.type == "cpu"


def test_optional_manifest_cannot_exclude_required_audio_keys(tmp_path):
    guard = ResourceGuard(
        ResourceBudget(1000000, 1000000, 10000), tmp_path, time.monotonic() + 30, threading.Event()
    )
    with pytest.raises(ValueError):
        SamCheckpointLoader.load(
            tmp_path / "absent",
            sha256="a" * 64,
            core=None,
            codec=None,
            optional_keys=frozenset({"audio_codec.weight"}),
            guard=guard,
        )


def test_restricted_unpickling_failure_is_typed(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"fixture")
    guard = ResourceGuard(
        ResourceBudget(1000000, 1000000, 10000), tmp_path, time.monotonic() + 30, threading.Event()
    )

    def fail(*args, **kwargs):
        assert kwargs["weights_only"] and kwargs["map_location"] == "cpu"
        raise pickle.UnpicklingError("fixture untrusted object")

    monkeypatch.setattr(torch, "load", fail)
    with pytest.raises(CleanExecutionError) as error:
        SamCheckpointLoader.load(
            path,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            core=torch.nn.Linear(1, 1),
            codec=torch.nn.Linear(1, 1),
            optional_keys=frozenset({"vision_encoder.weight"}),
            guard=guard,
        )
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE
    assert error.value.__suppress_context__
