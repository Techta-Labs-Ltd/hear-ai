import subprocess
import sys
from types import SimpleNamespace

import pytest

from hear.bootstrap import RuntimeBootstrap
from hear.runtime.roles import WorkerRole


class TestRuntimeBootstrap:
    def test_worker_identity_uses_explicit_revision(self):
        bootstrap = RuntimeBootstrap(
            {
                "HEAR_WORKER_ID": "worker-1",
                "HEAR_WORKER_GENERATION": "generation-1",
                "HEAR_IMAGE_REVISION": "image-1",
                "HEAR_ENGINE_REVISION": "engine-1",
            }
        )
        identity = bootstrap.worker_identity()
        assert identity.worker_id == "worker-1"
        assert identity.image_revision == "image-1"

    def test_worker_identity_rejects_missing_revision(self):
        bootstrap = RuntimeBootstrap({"HEAR_WORKER_ID": "worker-1"})
        with pytest.raises(RuntimeError):
            bootstrap.worker_identity()


    def test_bootstrap_import_does_not_load_qwen_module(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import hear.bootstrap; "
                "raise SystemExit(1 if 'hear.inference.qwen_asr' in sys.modules else 0)",
            ],
            check=False,
        )
        assert result.returncode == 0

    def test_readiness_checks_required_scratch_capacity(self, monkeypatch, tmp_path):
        bootstrap = RuntimeBootstrap(
            {
                "HEAR_TEMP_DIR": str(tmp_path / "scratch"),
                "HEAR_MIN_FREE_SCRATCH_BYTES": "1024",
            }
        )
        monkeypatch.setattr(
            "hear.bootstrap.shutil.disk_usage",
            lambda path: SimpleNamespace(total=4096, used=3073, free=1023),
        )

        readiness = bootstrap.readiness(WorkerRole.TRANSCRIPTION)
        snapshot = readiness.snapshot()

        assert snapshot["checks"]["scratch"] is False
        assert (tmp_path / "scratch").is_dir()
