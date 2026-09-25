from pathlib import Path


class TestDockerRuntime:
    def test_transcription_images_apply_and_verify_patch(self):
        text = Path("Dockerfile").read_text()
        assert "python -m hear.tools.dependency_patches" in text
        assert "python -m hear.tools.dependency_patches --check" in text
        assert "transcription-pod" in text
        assert "transcription-serverless" in text

    def test_runtime_lock_is_isolated_from_legacy_root(self):
        runtime = Path("deploy/runtime/pyproject.toml").read_text()
        root = Path("pyproject.toml").read_text()
        assert "runpod>=1.12,<2" in runtime
        assert "aio-pika>=9.5,<10" in runtime
        assert "runpod>=1.12,<2" not in root
        assert "aio-pika>=9.5,<10" not in root

    def test_production_image_excludes_tests_and_evidence(self):
        text = Path(".dockerignore").read_text().splitlines()
        assert "tests" in text
        assert "deploy/cleaner/evidence" in text
        assert "models" in text