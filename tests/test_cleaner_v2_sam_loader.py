from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import hear.runtime.cleaner.sam_loader as loader
from hear.runtime.cleaner.longform_sam import SolverPolicy
from hear.runtime.cleaner.sam_loader import PinnedSamAssets, PinnedSamFactory
from hear.services.magic_clean.contracts import CleanExecutionError
from tests.test_cleaner_v2_sam_plan import binding as binding_fixture

binding = binding_fixture


@pytest.fixture
def factory(binding):
    _, prompt, cache, guard = binding
    root = Path(__file__).resolve().parents[1]
    assets = PinnedSamAssets(
        guard.workspace,
        guard.workspace,
        guard.workspace / "config",
        guard.workspace / "weights",
        root / "deploy/cleaner/sam-small-optional-keys.json",
    )
    return PinnedSamFactory(assets, prompt, cache), guard


def test_policy_changes_change_identity(factory):
    value, _ = factory
    different_tile = PinnedSamFactory(
        value.assets, value.prompt, value.cache, codec_tile_frames=257
    )
    different_solver = PinnedSamFactory(
        value.assets, value.prompt, value.cache, solver=SolverPolicy(200, 40, 16)
    )
    assert value.identity != different_tile.identity != different_solver.identity
    with pytest.raises(CleanExecutionError):
        value.validate_identity(different_tile.identity)


@pytest.mark.parametrize("failure", [None, "codec", "checkpoint", "postload"])
def test_factory_owns_partial_loading_and_borrows_cache(factory, monkeypatch, failure):
    value, guard = factory
    closed = []
    core = SimpleNamespace(core=torch.nn.Linear(1, 1), close=lambda: closed.append("core"))
    codec = SimpleNamespace(codec=torch.nn.Linear(1, 1), close=lambda: closed.append("codec"))
    monkeypatch.setattr(loader.SamCoreBuilder, "build", lambda *args: core)

    def build(*args):
        if failure == "codec":
            raise RuntimeError("fixture codec construction")
        return codec

    def load(*args, **kwargs):
        assert kwargs["sha256"] == value.CHECKPOINT
        assert len(kwargs["optional_keys"]) == 601
        if failure == "checkpoint":
            raise RuntimeError("fixture checkpoint failure")
        if failure == "postload":
            guard.cancelled.set()
        return (247, 317)

    monkeypatch.setattr(loader.SamCodecBuilder, "build", build)
    monkeypatch.setattr(loader.SamCheckpointLoader, "load", load)
    if failure:
        with pytest.raises((RuntimeError, CleanExecutionError)):
            value.open(guard)
        assert closed == (["core"] if failure == "codec" else ["codec", "core"])
    else:
        backend = value.open(guard)
        assert backend.pipeline.core is core.core and backend.pipeline.codec is codec.codec
        backend.close()
        backend.close()
        assert closed == ["codec", "core"] and backend.pipeline is None
    assert value.cache.get(value.prompt)[0].shape == (1, 2, 768)


def test_missing_prompt_fails_before_construction(factory, monkeypatch):
    value, guard = factory
    value.cache.close()

    def forbidden(*args):
        pytest.fail("constructed before cache admission")

    monkeypatch.setattr(loader.SamCoreBuilder, "build", forbidden)
    with pytest.raises(CleanExecutionError):
        value.open(guard)


def test_cpu_autocast_rejected_before_construction_without_mutating_context(factory, monkeypatch):
    value, guard = factory

    def forbidden(*args):
        pytest.fail("constructed under unsupported autocast")

    monkeypatch.setattr(loader.SamCoreBuilder, "build", forbidden)
    assert not torch.is_autocast_enabled("cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(CleanExecutionError, match="precision policy mismatch"):
            value.open(guard)
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16
    assert not torch.is_autocast_enabled("cpu")
    value.validate_identity(value.identity)


@pytest.mark.parametrize(
    "setting", ["mkldnn", "mkldnn_deterministic", "deterministic", "matmul", "dtype", "device"]
)
def test_cpu_policy_mismatch_rejected_without_changing_settings(factory, monkeypatch, setting):
    value, _ = factory
    if setting == "mkldnn":
        monkeypatch.setattr(torch.backends.mkldnn, "enabled", False)
    elif setting == "mkldnn_deterministic":
        monkeypatch.setattr(torch.backends.mkldnn, "deterministic", True)
    elif setting == "deterministic":
        monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    elif setting == "matmul":
        monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "medium")
    elif setting == "dtype":
        monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.float64)
    else:
        monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("meta"))
    before = (
        torch.backends.mkldnn.enabled,
        torch.backends.mkldnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
        torch.get_float32_matmul_precision(),
        torch.get_default_dtype(),
        torch.get_default_device(),
    )
    with pytest.raises(CleanExecutionError, match="precision policy mismatch"):
        value.validate_identity(value.identity)
    assert before == (
        torch.backends.mkldnn.enabled,
        torch.backends.mkldnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
        torch.get_float32_matmul_precision(),
        torch.get_default_dtype(),
        torch.get_default_device(),
    )


def test_cuda_identity_is_distinct_and_device_is_explicit(factory):
    value, _ = factory
    cuda = PinnedSamFactory(value.assets, value.prompt, value.cache, device="cuda:0")
    assert cuda.identity != value.identity
    assert cuda.identity.precision_policy_sha256 != value.identity.precision_policy_sha256
    assert cuda.identity.longform_policy_sha256 == value.identity.longform_policy_sha256
    for device in ("cuda", "cuda:1", "mps", "meta"):
        with pytest.raises(ValueError, match="unsupported SAM device"):
            PinnedSamFactory(value.assets, value.prompt, value.cache, device=device)


@pytest.mark.parametrize("flag", ["tf32", "benchmark", "deterministic", "disabled"])
def test_cuda_backend_policy_rejected_without_initialization(factory, monkeypatch, flag):
    value, _ = factory
    cuda = PinnedSamFactory(value.assets, value.prompt, value.cache, device="cuda:0")
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("premature CUDA probe"))
    cuda.validate_identity(cuda.identity)
    attribute = {
        "tf32": "allow_tf32",
        "benchmark": "benchmark",
        "deterministic": "deterministic",
        "disabled": "enabled",
    }[flag]
    changed = flag != "disabled"
    monkeypatch.setattr(torch.backends.cudnn, attribute, changed)
    with pytest.raises(CleanExecutionError, match="CUDA precision policy mismatch"):
        cuda.validate_identity(cuda.identity)
    assert getattr(torch.backends.cudnn, attribute) == changed


@pytest.mark.parametrize("failure", [None, "unavailable", "multiple", "budget", "transfer"])
def test_cuda_cap_precedes_required_module_transfer(factory, monkeypatch, failure):
    value, guard = factory
    cuda = PinnedSamFactory(value.assets, value.prompt, value.cache, device="cuda:0")
    events = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: failure != "unavailable")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2 if failure == "multiple" else 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=1 if failure == "budget" else 18_000_000_000),
    )

    def cap(fraction, device):
        assert fraction == 0.5 and device == 0
        events.append("cap")

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", cap)
    modules = []
    for name in ("core", "codec"):
        tensor = SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.float32,
            numel=lambda: 4,
            element_size=lambda: 4,
        )

        def transfer(*, device, dtype, tensor=tensor, name=name):
            assert events[0] == "cap"
            assert dtype == torch.float32 and device == "cuda:0"
            events.append(name)
            if failure != "transfer":
                tensor.device = torch.device(device)

        modules.append(
            SimpleNamespace(
                parameters=lambda tensor=tensor: iter((tensor,)),
                buffers=lambda: iter(()),
                to=transfer,
            )
        )
    if failure:
        with pytest.raises(CleanExecutionError):
            cuda._place_modules(*modules, guard)
        assert events == (["cap", "core", "codec"] if failure == "transfer" else [])
    else:
        cuda._place_modules(*modules, guard)
        assert events == ["cap", "core", "codec"]
