import asyncio
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from hear.core.downloader import download_audio
from hear.core.hear_temp import (
    cleanup_job_temp,
    hear_temp_job_dir,
    purge_all_temp,
    sweep_tracked_temp_files,
)


def test_job_temp_is_scoped_and_removed(monkeypatch, tmp_path):
    monkeypatch.setattr("hear.core.hear_temp.settings.HEAR_TEMP_DIR", str(tmp_path / "hear-ai"))
    path = hear_temp_job_dir("job/../../escape", "run/value")
    audio_path = os.path.join(path, "source.wav")
    with open(audio_path, "wb") as audio:
        audio.write(b"audio")

    assert os.path.commonpath([path, str(tmp_path / "hear-ai")]) == str(tmp_path / "hear-ai")

    cleanup_job_temp(None, "job/../../escape", "run/value")

    assert not os.path.exists(path)


def test_download_audio_streams_into_job_scope(monkeypatch, tmp_path):
    monkeypatch.setattr("hear.core.hear_temp.settings.HEAR_TEMP_DIR", str(tmp_path / "hear-ai"))
    payload = b"audio-data" * 200_000

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        path = asyncio.run(
            download_audio(
                f"http://127.0.0.1:{server.server_port}/audio",
                job_id="job",
                run_id="run",
                purpose="source",
            )
        )
        with open(path, "rb") as downloaded:
            assert downloaded.read() == payload
        assert not os.path.exists(f"{path}.part")
    finally:
        server.shutdown()
        server.server_close()


def test_download_audio_can_decode_source_to_wav(monkeypatch, tmp_path):
    payload = b"encoded source audio"

    class Response:
        def raise_for_status(self):
            return None

        async def aiter_bytes(self, chunk_size):
            yield payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def stream(self, *args, **kwargs):
            return Response()

    def fake_convert(source_path, wav_path):
        with open(source_path, "rb") as source, open(wav_path, "wb") as output:
            assert source.read() == payload
            output.write(b"RIFF decoded wav")

    monkeypatch.setattr("hear.core.downloader.httpx.AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr("hear.core.downloader._convert_to_wav", fake_convert)
    monkeypatch.setattr("hear.core.downloader.hear_temp_job_dir", lambda *args: str(tmp_path))

    path = asyncio.run(download_audio(
        "https://example.test/source.mp3",
        job_id="job",
        run_id="run",
        purpose="magic_clean",
        convert_to_wav=True,
    ))

    assert path.endswith("magic_clean.wav")
    with open(path, "rb") as decoded:
        assert decoded.read() == b"RIFF decoded wav"
    assert not os.path.exists(f"{path}.source")


def test_sweep_removes_old_orphan_audio_from_temp_root(monkeypatch, tmp_path):
    temp_root = tmp_path / "hear-ai"
    monkeypatch.setattr("hear.core.hear_temp.settings.HEAR_TEMP_DIR", str(temp_root))
    temp_root.mkdir(parents=True)
    orphan = temp_root / "tmp-crashed.wav"
    orphan.write_bytes(b"orphan audio")
    old = __import__("time").time() - (25 * 60 * 60)
    os.utime(orphan, (old, old))

    result = sweep_tracked_temp_files()

    assert not orphan.exists()
    assert result["orphan_fs"] == 1
    assert result["bytes_freed"] == len(b"orphan audio")




def test_purge_never_removes_unmanaged_legacy_files(monkeypatch, tmp_path):
    monkeypatch.setattr("hear.core.hear_temp.settings.HEAR_TEMP_DIR", str(tmp_path / "hear-ai"))
    legacy = tmp_path / "hear-ai" / "jobs" / "legacy" / "source.wav"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"keep")
    managed = hear_temp_job_dir("managed-job", "run")
    with open(os.path.join(managed, "source.wav"), "wb") as audio:
        audio.write(b"remove")

    purge_all_temp()

    assert legacy.read_bytes() == b"keep"
    assert not os.path.exists(managed)


def test_default_audio_directory_is_inside_project_workspace():
    from hear.config import PROJECT_ROOT, Settings

    runtime = Settings(_env_file=None)

    assert runtime.HEAR_TEMP_DIR == str(PROJECT_ROOT / "audio")
    assert not runtime.HEAR_TEMP_DIR.startswith("/tmp")


def test_sweep_uses_configured_max_age(monkeypatch, tmp_path):
    temp_root = tmp_path / "audio"
    monkeypatch.setattr("hear.core.hear_temp.settings.HEAR_TEMP_DIR", str(temp_root))
    monkeypatch.setattr("hear.core.hear_temp.settings.AUDIO_MAX_AGE_SECONDS", 10)
    temp_root.mkdir()
    stale = temp_root / "stale.wav"
    fresh = temp_root / "fresh.wav"
    stale.write_bytes(b"stale")
    fresh.write_bytes(b"fresh")
    now = __import__("time").time()
    os.utime(stale, (now - 11, now - 11))
    os.utime(fresh, (now - 9, now - 9))

    sweep_tracked_temp_files()

    assert not stale.exists()
    assert fresh.exists()
