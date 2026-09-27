import os

from hear.audio.workspace import AudioWorkspace


def test_attempt_workspaces_are_isolated_and_cleanup_is_scoped(tmp_path):
    first = AudioWorkspace(tmp_path, "job:one", "attempt/one")
    second = AudioWorkspace(tmp_path, "job:one", "attempt/two")
    first_file = first.file("source.wav")
    second_file = second.file("source.wav")
    first_directory = first_file.parent
    first_file.write_bytes(b"first")
    second_file.write_bytes(b"second")

    first.cleanup()

    assert not first_directory.exists()
    assert second_file.read_bytes() == b"second"


def test_sweeper_removes_only_expired_attempts_and_reports_bytes(tmp_path):
    old = AudioWorkspace(tmp_path, "job", "old")
    fresh = AudioWorkspace(tmp_path, "job", "fresh")
    old_file = old.file("audio.wav")
    fresh_file = fresh.file("audio.wav")
    old_file.write_bytes(b"old-bytes")
    fresh_file.write_bytes(b"fresh-bytes")
    os.utime(old_file.parent, (1, 1))

    result = AudioWorkspace.sweep(tmp_path, 60)

    assert result == {"removed": 1, "bytes_freed": len(b"old-bytes")}
    assert not old_file.parent.exists()
    assert fresh_file.read_bytes() == b"fresh-bytes"


def test_sweeper_does_not_follow_jobs_symlink(tmp_path):
    root = tmp_path / "scratch"
    external = tmp_path / "external"
    external.mkdir()
    protected = external / "keep.wav"
    protected.write_bytes(b"keep")
    root.mkdir()
    (root / "jobs").symlink_to(external, target_is_directory=True)

    AudioWorkspace.sweep(root, 1)

    assert protected.read_bytes() == b"keep"
