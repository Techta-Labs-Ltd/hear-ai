import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "runpod-workspace-env.sh"


def test_runpod_workspace_script_prepares_only_local_directories(tmp_path):
    command = (
        f'HEAR_WORKSPACE_ROOT="{tmp_path}"; '
        f'source "{SCRIPT}"; '
        "printf '%s\\n' \"$MODEL_CACHE_DIR\" \"$HF_HUB_CACHE\" \"$UV_CACHE_DIR\""
    )

    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        str(tmp_path / "models"),
        str(tmp_path / ".cache/huggingface/hub"),
        str(tmp_path / ".cache/uv"),
    ]
    assert (tmp_path / "fish-speech").is_dir()
    assert (tmp_path / "checkpoints").is_dir()


def test_runpod_workspace_script_contains_no_download_command():
    script = SCRIPT.read_text()

    forbidden = ("snapshot_download", "huggingface-cli", "hf download", "wget", "curl")
    assert not any(command in script for command in forbidden)
