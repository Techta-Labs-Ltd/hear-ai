from dataclasses import replace
from unittest.mock import patch

import pytest

from hear.tools.dependency_patches import DependencyPatch, DependencyPatchManager


class TestDependencyPatches:
    def fixture(self, tmp_path):
        before = b"value = 1\n"
        after = b"value = 2\n"
        diff = b"--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
        target = tmp_path / "example.py"
        patch_file = tmp_path / "example.patch"
        target.write_bytes(before)
        patch_file.write_bytes(diff)
        digest = DependencyPatchManager.digest
        spec = DependencyPatch(
            "example",
            "example.py",
            "example.patch",
            "revision",
            digest(before),
            digest(after),
            digest(diff),
        )
        return target, patch_file, spec

    def test_apply_is_idempotent_and_verifiable(self, tmp_path):
        target, patch_file, spec = self.fixture(tmp_path)
        assert DependencyPatchManager.apply_file(target, patch_file, spec, False) == "applied"
        assert target.read_bytes() == b"value = 2\n"
        assert DependencyPatchManager.apply_file(target, patch_file, spec, False) == "verified"
        assert DependencyPatchManager.apply_file(target, patch_file, spec, True) == "verified"

    def test_check_does_not_write(self, tmp_path):
        target, patch_file, spec = self.fixture(tmp_path)
        with pytest.raises(RuntimeError, match="dependency_patch_required"):
            DependencyPatchManager.apply_file(target, patch_file, spec, True)
        assert target.read_bytes() == b"value = 1\n"

    @pytest.mark.parametrize("field", ["before_sha256", "patch_sha256", "after_sha256"])
    def test_unknown_content_is_not_modified(self, tmp_path, field):
        target, patch_file, spec = self.fixture(tmp_path)
        spec = replace(spec, **{field: "invalid"})
        with pytest.raises(RuntimeError):
            DependencyPatchManager.apply_file(target, patch_file, spec, False)
        assert target.read_bytes() == b"value = 1\n"

    def test_context_mismatch_is_rejected(self):
        with pytest.raises(RuntimeError, match="patch_context_mismatch"):
            DependencyPatchManager.patched_content(
                b"other = 1\n", b"@@ -1 +1 @@\n-value = 1\n+value = 2\n"
            )

    def test_revision_is_verified_before_file_access(self):
        with patch("hear.tools.dependency_patches.distribution") as distribution:
            distribution.return_value.read_text.return_value = "{}"
            with pytest.raises(RuntimeError, match="unsupported_dependency_revision"):
                DependencyPatchManager().run()
            distribution.return_value.locate_file.assert_not_called()
