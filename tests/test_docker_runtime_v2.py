import re
from pathlib import Path


class TestDockerRuntime:
    def test_transcription_images_apply_and_verify_patch(self):
        text = Path("Dockerfile").read_text()
        assert "python -m hear.tools.dependency_patches" in text
        assert "python -m hear.tools.dependency_patches --check" in text
        assert "transcription-pod" in text
        assert "transcription-serverless" in text

    def test_runtime_dependencies_have_one_canonical_project(self):
        runtime = Path("deploy/runtime/pyproject.toml").read_text()
        root = Path("pyproject.toml").read_text()
        dockerfile = Path("Dockerfile").read_text()
        assert "runpod>=1.12,<2" in runtime
        assert "uvicorn>=0.34,<1" in runtime
        assert '"aio-pika>=9.5,<10"' in runtime
        assert "HEAR_RABBITMQ_URL" in Path("hear/config.py").read_text()
        assert "AS runtime-pod-base" in dockerfile
        assert "rabbitmq-server" in dockerfile
        assert 'CMD ["/usr/local/bin/run_pod.sh"]' in dockerfile
        assert "[project]" not in root
        assert "dependencies =" not in root
        assert "https://astral.sh/uv/0.10.9/install.sh" in dockerfile

    def test_production_image_excludes_tests_and_evidence(self):
        text = Path(".dockerignore").read_text().splitlines()
        assert "tests" in text
        assert "docs" in text
        assert "outputs" in text
        assert "models" in text

    def test_ci_matrix_covers_every_pod_and_serverless_target(self):
        dockerfile = Path("Dockerfile").read_text()
        workflow = Path(".github/workflows/ci.yml").read_text()
        targets = {
            "pipeline-pod",
            "pipeline-serverless",
            "pipeline-llm-pod",
            "pipeline-llm-serverless",
            "transcription-pod",
            "transcription-serverless",
            "reconstruction-pod",
            "reconstruction-serverless",
            "magic-clean-natural-pod",
            "magic-clean-natural-serverless",
        }
        docker_targets = set(re.findall(r"^FROM .* AS ([a-z][a-z0-9-]+)$", dockerfile, re.M))
        workflow_targets = set(
            re.findall(
                r"^\s+- (hear-[a-z0-9-]+|pipeline-[a-z0-9-]+|transcription-[a-z0-9-]+|reconstruction-[a-z0-9-]+|magic-clean-[a-z0-9-]+)$",
                workflow,
                re.M,
            )
        )

        assert not any("sam-audio" in target for target in docker_targets)
        assert not any("sam-audio" in target for target in workflow_targets)
        assert targets <= docker_targets
        assert targets <= workflow_targets
