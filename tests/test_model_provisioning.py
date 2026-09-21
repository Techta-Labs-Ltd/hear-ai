import io
from pathlib import Path

from hear.tools.model_provisioning import (
    DEMUCS_CHECKPOINT_URL,
    DEMUCS_MANIFEST,
    MODEL_MANIFEST,
    ModelProvisioner,
)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_demucs_provisioning_writes_the_supported_local_repository(monkeypatch, tmp_path):
    requested = []

    def fake_urlopen(url, timeout):
        requested.append((url, timeout))
        return _Response(b"demucs-checkpoint")

    monkeypatch.setattr("hear.tools.model_provisioning.urlopen", fake_urlopen)
    destination = tmp_path / "demucs"

    assert ModelProvisioner._provision_demucs(destination) == destination
    assert (destination / "htdemucs.yaml").read_text() == DEMUCS_MANIFEST
    assert (destination / "955717e8-8726e21a.th").read_bytes() == b"demucs-checkpoint"
    assert requested == [(DEMUCS_CHECKPOINT_URL, 300)]

    ModelProvisioner._provision_demucs(destination)
    assert requested == [(DEMUCS_CHECKPOINT_URL, 300)]


def test_provisioning_places_every_hugging_face_snapshot_under_the_model_root(
    monkeypatch, tmp_path
):
    calls = []

    def fake_snapshot_download(*, repo_id, local_dir, cache_dir, ignore_patterns):
        calls.append((repo_id, local_dir, cache_dir, ignore_patterns))
        return str(local_dir)

    monkeypatch.setattr(
        "hear.tools.model_provisioning.snapshot_download", fake_snapshot_download
    )
    monkeypatch.setattr(
        ModelProvisioner,
        "_provision_demucs",
        staticmethod(lambda destination: destination),
    )

    results = ModelProvisioner.provision(str(tmp_path / "models"))

    root = (tmp_path / "models").resolve()
    assert set(results) == {*MODEL_MANIFEST, "demucs", "dnsmos"}
    assert {repo_id for repo_id, *_args in calls} == set(MODEL_MANIFEST.values())
    assert all(Path(local_dir).is_relative_to(root) for _repo, local_dir, *_args in calls)
    assert all(Path(cache_dir).is_relative_to(root) for *_before, cache_dir, _patterns in calls)
