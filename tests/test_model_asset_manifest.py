from pathlib import Path

from hear.contracts.jobs import WorkerRole
from hear.model_assets.manifest import ModelManifestLoader
from hear.model_assets.provisioner import ModelProvisioner


def test_manifest_filters_by_worker_role(tmp_path: Path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        """
version: 1
assets:
  - name: asr
    source: example/asr
    revision: abc123
    local_path: /models/asr
    roles:
      - pipeline
      - transcription
  - name: tts
    source: example/tts
    revision: def456
    local_path: /models/tts
    roles:
      - reconstruction
"""
    )
    manifest = ModelManifestLoader(manifest_path).load()
    provisioner = ModelProvisioner(manifest, tmp_path / "cache")
    assert [asset.name for asset in provisioner.assets_for_role(WorkerRole.PIPELINE)] == ["asr"]
    assert [asset.name for asset in provisioner.assets_for_role(WorkerRole.RECONSTRUCTION)] == [
        "tts"
    ]


def test_manifest_requires_revision(tmp_path: Path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        """
version: 1
assets:
  - name: asr
    source: example/asr
    local_path: /models/asr
    roles:
      - transcription
"""
    )
    try:
        ModelManifestLoader(manifest_path).load()
    except Exception:
        return
    raise AssertionError("manifest accepted asset without revision")
