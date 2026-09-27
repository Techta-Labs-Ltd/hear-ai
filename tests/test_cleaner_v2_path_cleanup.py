import pytest

from hear.runtime.cleaner.path_cleanup import AttemptPathCleanup
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


def test_removes_only_the_recorded_inode(tmp_path):
    path = tmp_path / "output.wav"
    path.write_bytes(b"owned")
    identity = AttemptPathCleanup.identity(path)
    replacement = tmp_path / "replacement.wav"
    replacement.write_bytes(b"replacement")
    replacement.replace(path)

    AttemptPathCleanup.remove_if_owned(path, identity, None)

    assert path.read_bytes() == b"replacement"


def test_private_cleanup_preserves_primary_error_code(tmp_path, monkeypatch):
    path = tmp_path / "output.wav"
    path.write_bytes(b"owned")
    primary = CleanExecutionError(ErrorCode.CANCELLED, "cancelled")

    def fail_unlink(self, *args, **kwargs):
        raise PermissionError("unavailable")

    monkeypatch.setattr(type(path), "unlink", fail_unlink)

    with pytest.raises(CleanExecutionError) as error:
        AttemptPathCleanup.remove_private(path, primary)

    assert error.value.code == ErrorCode.CANCELLED
    assert error.value.worker_restart_required
    assert str(error.value) == "attempt file cleanup failed; worker requires process restart"


def test_private_cleanup_rejects_symlinks(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"keep")
    path = tmp_path / "output.wav"
    path.symlink_to(target)

    with pytest.raises(CleanExecutionError) as error:
        AttemptPathCleanup.remove_private(path, None)

    assert error.value.worker_restart_required
    assert target.read_bytes() == b"keep"
