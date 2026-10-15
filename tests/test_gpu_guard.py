"""Workers refuse hardware the images were not built for, loudly."""

from hear.bootstrap import RuntimeBootstrap
from hear.config import RuntimeSettings

MIG = "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb"


def test_mig_partitions_are_refused_and_logged(caplog):
    assert not RuntimeBootstrap._gpu_supported("", True, name=MIG)
    assert "gpu_unsupported reason=mig_partition" in caplog.text
    assert RuntimeBootstrap._gpu_supported("", True, name="NVIDIA RTX A5000")
    assert RuntimeBootstrap._gpu_supported("", False, name=MIG)


def test_pattern_narrows_accepted_cards():
    assert RuntimeBootstrap._gpu_supported(r"A5000|A40\b", True, name="NVIDIA A40")
    assert not RuntimeBootstrap._gpu_supported(r"A5000|A40\b", True, name="NVIDIA L4")


def test_no_cuda_device_is_not_this_checks_failure(monkeypatch):
    monkeypatch.setattr(RuntimeBootstrap, "_gpu_device_name", staticmethod(lambda: None))
    assert RuntimeBootstrap._gpu_supported("", True)


def test_guard_is_on_by_default_and_configurable():
    settings = RuntimeSettings.from_environment(
        {"HEAR_IMAGE_REVISION": "x", "HEAR_WORKER_ROLE": "pipeline"}
    )
    assert settings.reject_mig_gpus is True and settings.allowed_gpu_name_pattern == ""
    settings = RuntimeSettings.from_environment(
        {
            "HEAR_IMAGE_REVISION": "x",
            "HEAR_WORKER_ROLE": "pipeline",
            "HEAR_REJECT_MIG_GPUS": "false",
            "HEAR_ALLOWED_GPU_NAME_PATTERN": "A40",
        }
    )
    assert settings.reject_mig_gpus is False and settings.allowed_gpu_name_pattern == "A40"
