import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from hear.runtime.cleaner.asset_probe import PinnedAssetProbe, PinnedAssetSet
from hear.runtime.cleaner.deepfilter_available import DeepFilterNetCleaner
from scripts.deduplicate_image_dependencies import ImageDependencyDeduplication
from scripts.provision_release_sound_assets import ReleaseSoundAssetProvisioner


def test_deduplication_copies_packages_across_overlay_devices(tmp_path, monkeypatch):
    for role in ("pipeline", "reconstruction"):
        package = tmp_path / "venvs" / role / "lib/python3.12/site-packages/torch"
        package.mkdir(parents=True)
        (package / "native.so").write_bytes(b"identical-native-library")
        (package / "alias.so").symlink_to("native.so")

    def cross_device_rename(*args, **kwargs):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(Path, "rename", cross_device_rename)
    assert ImageDependencyDeduplication.deduplicate(tmp_path) == ["torch"]
    assert (tmp_path / "shared/torch/native.so").read_bytes() == b"identical-native-library"
    assert os.readlink(tmp_path / "shared/torch/alias.so") == "native.so"
    for role in ("pipeline", "reconstruction"):
        assert (
            os.readlink(tmp_path / "venvs" / role / "lib/python3.12/site-packages/torch")
            == "/opt/hear-ai-v11/shared/torch"
        )


def test_cleaner_readiness_reuses_hashes_and_rejects_changed_assets(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint"
    path.write_bytes(b"verified-checkpoint")
    cleaner = DeepFilterNetCleaner.__new__(DeepFilterNetCleaner)
    cleaner._pinned_assets = PinnedAssetSet(
        ((path, hashlib.sha256(path.read_bytes()).hexdigest(), None),)
    )
    cleaner._identity = object()
    cleaner._factory = SimpleNamespace(validate_identity=lambda identity: None)
    cleaner._dsp = SimpleNamespace(is_ready=lambda: True)
    original = PinnedAssetProbe.sha256
    checks = []

    def verify(*args, **kwargs):
        checks.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(PinnedAssetProbe, "sha256", verify)
    assert cleaner.is_ready()
    assert cleaner.is_ready()
    assert checks == [path]
    path.write_bytes(b"corrupted-checkpoint")
    assert not cleaner.is_ready()
    assert checks == [path, path]


def test_safe_dotenv_loader_can_load_configuration_and_build_metadata(tmp_path):
    configuration = tmp_path / "production.env"
    metadata = tmp_path / "release.env"
    configuration.write_text("HEAR_TEST_CONFIG='literal $(exit 73)'\n")
    metadata.write_text("HEAR_TEST_DIGEST=abc123\n")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source scripts/load-env.sh "$1"; source scripts/load-env.sh "$2"; '
            'printf "%s|%s" "$HEAR_TEST_CONFIG" "$HEAR_TEST_DIGEST"',
            "dotenv-test",
            str(configuration),
            str(metadata),
        ],
        env={**os.environ, "HEAR_ENV_PARSER_PYTHON": sys.executable},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "literal $(exit 73)|abc123"


def test_sound_asset_release_records_exported_hashes_only_after_success(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    calls = []

    def provision(root, work, relative):
        calls.append(relative)
        destination = root / relative
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_text(json.dumps({"export": relative}))

    monkeypatch.setattr(
        ReleaseSoundAssetProvisioner,
        "provision_analysis",
        lambda root, work: provision(root, work, "sound-cleanup-v1-runtime"),
    )
    monkeypatch.setattr(
        ReleaseSoundAssetProvisioner,
        "provision_separator",
        lambda root, work: provision(root, work, "sound-cleanup-specialist/runtime"),
    )
    monkeypatch.setattr(sys, "argv", ["provision", "--model-root", str(model_root)])
    ReleaseSoundAssetProvisioner.main()
    assert calls == ["sound-cleanup-v1-runtime", "sound-cleanup-specialist/runtime"]
    metadata = (model_root / "sound-cleanup-release.env").read_text()
    for relative in calls:
        assert (
            hashlib.sha256((model_root / relative / "manifest.json").read_bytes()).hexdigest()
            in metadata
        )


def test_sound_asset_provisioning_failure_invalidates_old_metadata(tmp_path, monkeypatch):
    import pytest

    model_root = tmp_path / "models"
    model_root.mkdir()
    metadata = model_root / "sound-cleanup-release.env"
    metadata.write_text("stale-release")

    def fail(root, work):
        raise RuntimeError("unverified-export")

    monkeypatch.setattr(ReleaseSoundAssetProvisioner, "provision_analysis", fail)
    monkeypatch.setattr(sys, "argv", ["provision", "--model-root", str(model_root)])
    with pytest.raises(RuntimeError, match="unverified-export"):
        ReleaseSoundAssetProvisioner.main()
    assert not metadata.exists()


def test_container_launcher_uses_gpu_and_external_config_without_host_models(tmp_path):
    docker = tmp_path / "docker"
    arguments = tmp_path / "arguments.json"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        f"output = Path({str(arguments)!r})\n"
        "if sys.argv[1] == 'run':\n"
        "    output.write_text(json.dumps(sys.argv[1:]))\n"
    )
    docker.chmod(0o755)
    result = subprocess.run(
        [
            "bash", "scripts/run_production_container.sh",
            "--image", "hear-ai@sha256:release",
            "--env-file", "/srv/hear config/production.env",
        ],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(arguments.read_text())
    assert argv[argv.index("--gpus") + 1] == "all"
    assert argv[argv.index("--mount") + 1] == (
        "type=bind,src=/srv/hear config/production.env,"
        "dst=/root/hear-ai-config/production.env,readonly"
    )
    assert argv.count("--mount") == 1
    assert "hear-ai@sha256:release" in argv
    assert "--privileged" not in argv
    assert "HEAR_RUNTIME_MODE=production" in argv


def test_container_launcher_rejects_simulation_env(tmp_path):
    docker = tmp_path / "docker"
    bootstrap = tmp_path / "bootstrap.sh"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"output = Path({str(bootstrap)!r})\n"
        "if sys.argv[1] == 'run':\n"
        "    output.write_text(sys.argv[-1])\n"
    )
    docker.chmod(0o755)
    subprocess.run(
        ["bash", "scripts/run_production_container.sh"],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        check=True,
    )
    # Exercise the actual container bootstrap with a local path to the loader.
    loader = str(Path("scripts/load-env.sh").resolve())
    command = bootstrap.read_text().replace("/app/scripts/load-env.sh", loader)
    configuration = tmp_path / "production.env"
    configuration.write_text("HEAR_RUNTIME_MODE=simulation\n")
    result = subprocess.run(
        ["bash", "-c", command],
        env={
            **os.environ,
            "HEAR_ENV_FILE": str(configuration),
            "HEAR_ENV_PARSER_PYTHON": sys.executable,
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "requires HEAR_RUNTIME_MODE=production" in result.stderr


def test_container_launcher_can_run_cpu_diagnostics_without_a_gpu_driver():
    result = subprocess.run(
        ["bash", "scripts/run_production_container.sh", "--gpus", "none", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--gpus" not in result.stdout
    assert "HEAR_RUNTIME_MODE=production" in result.stdout
    assert "readonly" in result.stdout


def test_image_builder_stops_before_build_when_docker_daemon_is_unavailable(tmp_path):
    docker = tmp_path / "docker"
    arguments = tmp_path / "arguments.jsonl"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        f"output = Path({str(arguments)!r})\n"
        "with output.open('a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'info':\n"
        "    raise SystemExit(23)\n"
    )
    docker.chmod(0o755)
    runfiles = tmp_path / "runfiles/_main"
    runfiles.mkdir(parents=True)
    (runfiles / "hear-runtime-context.tar").write_bytes(b"context")
    result = subprocess.run(
        ["bash", "scripts/build_runpod_image.sh"],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "RUNFILES_DIR": str(runfiles.parent),
            "HEAR_IMAGE_TAR": str(tmp_path / "image.tar"),
            "HEAR_IMAGE_BUILDER": "kaniko",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 23
    assert [json.loads(line) for line in arguments.read_text().splitlines()] == [
        ["buildx", "version"], ["info"],
    ]
    assert not (tmp_path / "image.tar").exists()
