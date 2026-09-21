import configparser
import gc
import hashlib
import importlib.metadata
import os
import pickle
import threading
import time
import weakref
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hear.runtime.cleaner.deepfilter_loader import (
    LoadedDeepFilter,
    PinnedDeepFilterAssets,
    PinnedDeepFilterFactory,
)
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@pytest.fixture(autouse=True)
def isolate_fault_state(monkeypatch):
    # Production intentionally has no reset operation; tests isolate their processes' state.
    monkeypatch.setattr(PinnedDeepFilterFactory, "_worker_faulted", threading.Event())


@pytest.fixture
def assets(tmp_path):
    config = tmp_path / "config.ini"
    checkpoint = tmp_path / "model.ckpt"
    config.write_bytes(b"test config")
    checkpoint.write_bytes(b"test checkpoint")
    versions = tuple(
        (name, "test-pinned-version")
        for name in ("deepfilternet", "deepfilterlib", "torch", "torchaudio", "numpy")
    )
    return PinnedDeepFilterAssets(
        config,
        hashlib.sha256(config.read_bytes()).hexdigest(),
        checkpoint,
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        versions,
        "cpu",
    )


@pytest.fixture
def guard(tmp_path):
    return ResourceGuard(
        ResourceBudget(100000, 100000, 48000), tmp_path, time.monotonic() + 20, threading.Event()
    )


def test_assets_verified_without_importing_or_allocating_models(assets, guard, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test-pinned-version")
    assets.verify(guard)


def test_wrong_asset_rejected_before_package_lookup(assets, guard, monkeypatch):
    assets.checkpoint_path.write_bytes(b"changed")
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: pytest.fail("must reject first")
    )
    with pytest.raises(CleanExecutionError) as error:
        assets.verify(guard)
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE


def test_missing_package_is_typed_unavailable(assets, guard, monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(CleanExecutionError) as error:
        assets.verify(guard)
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE


def test_package_drift_rejected(assets, guard, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "another-version")
    with pytest.raises(CleanExecutionError):
        assets.verify(guard)


def test_failed_load_releases_global_config_lease(assets, guard, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test-pinned-version")
    factory = PinnedDeepFilterFactory(assets)

    def broken(_guard):
        raise RuntimeError("incompatible native module")

    monkeypatch.setattr(factory, "_load", broken)
    for _ in range(2):
        with pytest.raises(CleanExecutionError) as error:
            factory.open(guard)
        assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE
    assert not factory._runtime_lease.locked()


def test_missing_asset_never_downloads(assets, guard):
    assets.checkpoint_path.unlink()
    with pytest.raises(CleanExecutionError):
        PinnedDeepFilterFactory(assets).open(guard)
    assert not assets.checkpoint_path.exists()


@pytest.mark.parametrize("key", ["DEVICE", "MASK_PF", "SR", "EMB_GRU_SKIP_ENC"])
def test_environment_cannot_override_pinned_config(key, monkeypatch):
    parser = configparser.ConfigParser()
    parser.read("deploy/cleaner/deepfilter3.ini")
    monkeypatch.setenv(key, "unexpected")
    with pytest.raises(CleanExecutionError) as error:
        PinnedDeepFilterFactory.verify_environment(SimpleNamespace(parser=parser))
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE


def test_unrelated_environment_is_not_modified(monkeypatch):
    parser = configparser.ConfigParser()
    parser.read_string("[df]\nsr = 48000\n")
    monkeypatch.delenv("SR", raising=False)
    monkeypatch.setenv("HEAR_SERVICE_KEY", "private-value")
    PinnedDeepFilterFactory.verify_environment(SimpleNamespace(parser=parser))
    assert os.environ["HEAR_SERVICE_KEY"] == "private-value"


@pytest.mark.parametrize(
    "field", ["runtime_sha256", "checkpoint_sha256", "precision_policy_sha256"]
)
def test_factory_rejects_misreported_runtime_provenance(assets, field):
    factory = PinnedDeepFilterFactory(assets)
    identity = factory.identity("a" * 64)
    factory.validate_identity(identity)
    with pytest.raises(CleanExecutionError) as error:
        factory.validate_identity(identity.model_copy(update={field: "b" * 64}))
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE


def test_asset_descriptor_is_order_independent_and_changes_with_configuration(assets):
    reordered = replace(assets, package_versions=tuple(reversed(assets.package_versions)))
    assert reordered.runtime_sha256 == assets.runtime_sha256
    for changed in (
        replace(assets, config_sha256="a" * 64),
        replace(assets, checkpoint_sha256="b" * 64),
        replace(assets, device="cuda:0"),
        replace(assets, package_versions=assets.package_versions + (("appdirs", "1.4.4"),)),
    ):
        assert changed.runtime_sha256 != assets.runtime_sha256


@pytest.mark.parametrize("fault", [MemoryError, torch.cuda.OutOfMemoryError])
def test_load_oom_is_typed_and_blocks_reuse_without_retaining_exception(
    assets, guard, monkeypatch, fault
):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test-pinned-version")
    factory = PinnedDeepFilterFactory(assets)

    def broken(_guard):
        raise fault("private-allocation-diagnostics")

    monkeypatch.setattr(factory, "_load", broken)
    with pytest.raises(CleanExecutionError) as error:
        factory.open(guard)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    assert error.value.worker_restart_required
    transported = pickle.loads(pickle.dumps(error.value))
    assert transported.code == ErrorCode.RESOURCE_EXHAUSTED
    assert transported.worker_restart_required
    assert error.value.__context__ is None
    assert error.value.__cause__ is None
    assert "private" not in str(error.value)
    assert not factory._runtime_lease.locked()
    with pytest.raises(CleanExecutionError) as next_error:
        PinnedDeepFilterFactory(assets).open(guard)
    assert next_error.value.code == ErrorCode.ENGINE_UNAVAILABLE
    assert next_error.value.worker_restart_required


@pytest.mark.parametrize(
    "device,fault,code,restart",
    [
        ("cpu", MemoryError, ErrorCode.RESOURCE_EXHAUSTED, True),
        ("cuda", torch.cuda.OutOfMemoryError, ErrorCode.RESOURCE_EXHAUSTED, True),
        ("cuda", RuntimeError, ErrorCode.PROCESS_FAILED, True),
        ("cpu", RuntimeError, ErrorCode.PROCESS_FAILED, False),
        (
            "cpu",
            lambda _: CleanExecutionError(ErrorCode.CANCELLED, "cancelled"),
            ErrorCode.CANCELLED,
            False,
        ),
    ],
)
def test_inference_fault_closes_model_and_does_not_retain_native_traceback(
    guard, device, fault, code, restart
):
    class Model:
        pass

    def broken(model, state, samples, **kwargs):
        raise fault("private-native-path-and-buffer-details")

    lease = threading.Lock()
    lease.acquire()
    model = Model()
    reference = weakref.ref(model)
    backend = LoadedDeepFilter(
        model,
        lambda **kw: object(),
        {},
        broken,
        lease,
        guard,
        SimpleNamespace(parser=configparser.ConfigParser()),
        device,
    )
    del model
    with pytest.raises(CleanExecutionError) as error:
        backend.enhance(np.zeros((1, 512), dtype=np.float32), 18)
    assert error.value.code == code
    assert error.value.worker_restart_required == restart
    assert error.value.__context__ is None
    assert error.value.__cause__ is None
    assert backend.closed and backend.model is None
    assert not lease.locked()
    gc.collect()
    assert reference() is None
    backend.close()  # Idempotent after the fault path already closed the backend.


def test_cancelled_load_does_not_retain_partial_model_or_poison_worker(assets, guard, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "test-pinned-version")
    factory = PinnedDeepFilterFactory(assets)
    references = []

    class Model:
        pass

    def interrupted(_guard):
        model = Model()
        references.append(weakref.ref(model))
        raise CleanExecutionError(ErrorCode.CANCELLED, "cancelled during load")

    monkeypatch.setattr(factory, "_load", interrupted)
    with pytest.raises(CleanExecutionError) as error:
        factory.open(guard)
    assert error.value.code == ErrorCode.CANCELLED
    assert not error.value.worker_restart_required
    assert error.value.__context__ is None
    assert not factory._runtime_lease.locked()
    factory.assert_healthy()
    gc.collect()
    assert references[0]() is None
