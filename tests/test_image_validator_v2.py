from pathlib import Path

from hear.runtime.roles import WorkerRole
from hear.tools.validate_image import ImageValidator


class PatchManager:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, check=False):
        self.calls += 1


class Manifest:
    def validate_local(self, model_root, role, enabled_features=frozenset()):
        return ()


def validator(tmp_path: Path):
    instance = ImageValidator(Path(__file__).resolve().parents[1])
    instance._patch_manager = PatchManager()
    instance._manifest = Manifest()
    return instance


def test_transcription_requires_patch(tmp_path):
    instance = validator(tmp_path)
    instance.validate(
        WorkerRole.TRANSCRIPTION,
        tmp_path,
        enabled_features=frozenset(),
        require_models=True,
    )
    assert instance._patch_manager.calls == 1


def test_reconstruction_does_not_require_patch(tmp_path):
    instance = validator(tmp_path)
    instance.validate(
        WorkerRole.RECONSTRUCTION,
        tmp_path,
        enabled_features=frozenset(),
        require_models=True,
    )
    assert instance._patch_manager.calls == 0
