import ast
import asyncio
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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


async def _exercise_cancelled_download_removes_partial(monkeypatch, tmp_path):
    started = asyncio.Event()

    class Response:
        headers = None

        @staticmethod
        def raise_for_status():
            return None

        async def aiter_bytes(self, chunk_size):
            del chunk_size
            yield b"partial-audio"
            started.set()
            await asyncio.Event().wait()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        @staticmethod
        def stream(*_args, **_kwargs):
            return Response()

    monkeypatch.setattr(
        "hear.core.downloader.httpx.AsyncClient",
        lambda **_kwargs: Client(),
    )
    monkeypatch.setattr(
        "hear.core.downloader.hear_temp_job_dir",
        lambda *_args: str(tmp_path),
    )
    task = asyncio.create_task(
        download_audio(
            "https://example.test/source.wav",
            job_id="job",
            run_id="run",
            purpose="source",
        )
    )
    await started.wait()

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("download cancellation must propagate")

    assert not (tmp_path / "source.wav.part").exists()
    assert not (tmp_path / "source.wav").exists()


def test_cancelled_download_does_not_leave_partial_audio(monkeypatch, tmp_path):
    asyncio.run(_exercise_cancelled_download_removes_partial(monkeypatch, tmp_path))


async def _exercise_cancelled_conversion_waits_before_cleanup(monkeypatch, tmp_path):
    payload = b"encoded source audio"
    conversion_started = threading.Event()
    release_conversion = threading.Event()

    class Response:
        headers = {"content-length": str(len(payload))}

        @staticmethod
        def raise_for_status():
            return None

        async def aiter_bytes(self, chunk_size):
            del chunk_size
            yield payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        @staticmethod
        def stream(*_args, **_kwargs):
            return Response()

    def delayed_convert(_source_path, wav_path):
        conversion_started.set()
        assert release_conversion.wait(timeout=5)
        with open(wav_path, "wb") as output:
            output.write(b"late decoded output")

    monkeypatch.setattr(
        "hear.core.downloader.httpx.AsyncClient",
        lambda **_kwargs: Client(),
    )
    monkeypatch.setattr(
        "hear.core.downloader.hear_temp_job_dir",
        lambda *_args: str(tmp_path),
    )
    monkeypatch.setattr("hear.core.downloader._convert_to_wav", delayed_convert)
    task = asyncio.create_task(
        download_audio(
            "https://example.test/source.mp3",
            job_id="job",
            run_id="run",
            purpose="source",
            convert_to_wav=True,
        )
    )
    assert await asyncio.to_thread(conversion_started.wait, 5)

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release_conversion.set()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("conversion cancellation must propagate")

    assert not (tmp_path / "source.wav").exists()
    assert not (tmp_path / "source.wav.source").exists()
    assert not (tmp_path / "source.wav.source.part").exists()


def test_cancelled_conversion_cannot_recreate_cleaned_audio(monkeypatch, tmp_path):
    asyncio.run(
        _exercise_cancelled_conversion_waits_before_cleanup(monkeypatch, tmp_path)
    )


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


def test_runtime_audio_tempfiles_always_set_workspace_directory():
    project_root = Path(__file__).resolve().parents[1]
    runtime_files = [
        *project_root.joinpath("hear").rglob("*.py"),
        *project_root.joinpath("scripts").glob("*.py"),
    ]
    audio_suffixes = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    for source_path in runtime_files:
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = getattr(node.func, "attr", "")
            if function_name not in {"mkstemp", "NamedTemporaryFile"}:
                continue
            suffix = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "suffix"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ),
                "",
            )
            if suffix not in audio_suffixes:
                continue
            assert any(keyword.arg == "dir" for keyword in node.keywords), (
                f"audio tempfile must set the workspace directory: {source_path}"
            )


def test_live_regeneration_harness_has_no_system_tmp_audio_paths():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "live_regeneration_local_test.py"
    )
    source = script_path.read_text()

    assert 'AUDIO_ROOT = Path(settings.HEAR_TEMP_DIR)' in source
    assert 'Path("/tmp/' not in source


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
