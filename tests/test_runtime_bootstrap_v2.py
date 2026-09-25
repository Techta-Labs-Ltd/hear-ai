import pytest

from hear.bootstrap import RuntimeBootstrap


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
        import sys

        assert "hear.inference.qwen_asr" not in sys.modules