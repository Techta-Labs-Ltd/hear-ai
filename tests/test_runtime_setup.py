import subprocess
from unittest.mock import Mock

from scripts.setup_runtime import RuntimeSetup


class TestRuntimeSetup:
    def test_install_automatically_patches_locked_environment(self, monkeypatch):
        runner = Mock()
        monkeypatch.setattr("sys.argv", ["setup_runtime.py", "--no-dev"])
        monkeypatch.setattr("scripts.setup_runtime.shutil.which", lambda name: "/usr/bin/uv")
        monkeypatch.setattr("scripts.setup_runtime.subprocess.run", runner)
        assert RuntimeSetup.main() == 0
        assert runner.call_args_list[0].args[0] == ["/usr/bin/uv", "sync", "--locked", "--no-dev"]
        assert runner.call_args_list[1].args[0] == [
            "/usr/bin/uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "hear.tools.dependency_patches",
        ]

    def test_check_never_installs_or_applies(self, monkeypatch):
        runner = Mock()
        monkeypatch.setattr("sys.argv", ["setup_runtime.py", "--check"])
        monkeypatch.setattr("scripts.setup_runtime.shutil.which", lambda name: "/usr/bin/uv")
        monkeypatch.setattr("scripts.setup_runtime.subprocess.run", runner)
        assert RuntimeSetup.main() == 0
        runner.assert_called_once()
        assert runner.call_args.args[0][-1] == "--check"
        assert "sync" not in runner.call_args.args[0]

    def test_failed_install_does_not_patch(self, monkeypatch):
        runner = Mock(side_effect=subprocess.CalledProcessError(7, "uv"))
        monkeypatch.setattr("sys.argv", ["setup_runtime.py"])
        monkeypatch.setattr("scripts.setup_runtime.shutil.which", lambda name: "/usr/bin/uv")
        monkeypatch.setattr("scripts.setup_runtime.subprocess.run", runner)
        assert RuntimeSetup.main() == 7
        runner.assert_called_once()
