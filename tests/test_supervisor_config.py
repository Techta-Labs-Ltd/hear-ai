from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_supervisor_starts_ray_before_the_hear_server():
    config = (PROJECT_ROOT / "deploy" / "supervisord.conf").read_text()
    launcher = (PROJECT_ROOT / "scripts" / "start-hear-ray-server.sh").read_text()

    assert "[program:ray-head]" in config
    assert "ray start --head --block" in config
    assert "--dashboard-port=8282" in config
    assert "[program:hear-ray-server]" in config
    assert 'RAY_ADDRESS="auto"' in config
    assert "priority=10" in config
    assert "priority=20" in config
    assert "ray status" in launcher
    assert "exec uv run --no-project python main.py" in launcher
