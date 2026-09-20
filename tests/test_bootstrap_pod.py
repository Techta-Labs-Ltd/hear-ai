from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_installs_and_prepares_the_full_root_runtime():
    script = (PROJECT_ROOT / "scripts" / "bootstrap-pod.sh").read_text()

    assert "postgresql" in script
    assert "supervisor" in script
    assert "command -v uv" in script
    assert "uv sync --frozen --group dev --inexact" in script
    assert "hf-transfer" in (PROJECT_ROOT / "pyproject.toml").read_text()
    assert "git clone --depth 1 https://github.com/fishaudio/fish-speech.git" in script
    assert "--no-deps -e \"$fish_speech_root\"" in script
    assert "Fish Speech inference imports are ready" in script
    for directory in ("/models", "/cache", "/audio", "/postgres"):
        assert directory in script
    assert "supervisord -c" in script
