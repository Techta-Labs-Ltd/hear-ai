import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from hear.orchestrator import Orchestrator, audio_tag_result


def test_audio_tag_result_contains_only_transcription_output():
    job = SimpleNamespace(
        id="job-1", run_id="run-1", backend_id="backend-a", job_type="audio_tag", track_id="track-1",
        input_url="https://audio.test/a.mp3", job_options={}, existing_transcript=None,
    )
    track = SimpleNamespace(track_id="track-1")
    result = audio_tag_result(
        job,
        track,
        "Tag this as construction news.",
        ["#construction", "#news", "#ignored"],
    )

    assert result == {
        "job_id": "job-1",
        "run_id": "run-1",
        "backend_id": "backend-a",
        "job_type": "audio_tag",
        "track_id": "track-1",
        "transcription": "Tag this as construction news.",
        "suggestions": ["#construction", "#news"],
    }
    assert "moderation" not in result
    assert "categorization" not in result
    assert "discovery" not in result


def test_audio_tag_pipeline_returns_text_and_two_suggestions_without_moderation(
    monkeypatch, tmp_path
):
    asyncio.run(_assert_audio_tag_pipeline_stops_after_transcription(monkeypatch, tmp_path))


async def _assert_audio_tag_pipeline_stops_after_transcription(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio fixture")
    transcription = {
        "transcript": "A short audio clip.",
        "segments": [],
        "language": "en",
    }
    job = SimpleNamespace(
        id="job-1", run_id="run-1", backend_id="backend-a", job_type="audio_tag", track_id="track-1",
        input_url="https://audio.test/a.mp3", job_options={}, existing_transcript=None,
    )
    track = SimpleNamespace(track_id="track-1", audio_url="https://audio.test/a.mp3")
    track_job = SimpleNamespace(updated_at=None)

    orchestrator_class = Orchestrator.func_or_class
    orchestrator = orchestrator_class.__new__(orchestrator_class)
    orchestrator._set_stage = AsyncMock(return_value=True)
    orchestrator._transcriber = SimpleNamespace(
        transcribe=AsyncMock(return_value=transcription)
    )
    orchestrator._moderator = SimpleNamespace(moderate=AsyncMock())
    orchestrator._categorizer = SimpleNamespace(
        categorize=AsyncMock(
            return_value={"tags": ["#construction", "#news", "#ignored"]}
        )
    )
    completed = []
    orchestrator_platform = SimpleNamespace(auto_tag_keywords=[])

    async def complete(db, completed_job, completed_track_job, result):
        completed.append(result)
        return True

    orchestrator._complete = complete
    monkeypatch.setattr(
        "hear.orchestrator.fetch_platform_settings",
        AsyncMock(return_value=orchestrator_platform),
    )
    monkeypatch.setattr(
        "hear.orchestrator.download_audio", AsyncMock(return_value=str(audio_path))
    )
    monkeypatch.setattr("hear.orchestrator.commit_with_retry", AsyncMock())

    await orchestrator._process_pipeline(job, track_job, object())

    assert completed == [
        {
            "job_id": "job-1",
            "run_id": "run-1",
            "backend_id": "backend-a",
            "job_type": "audio_tag",
            "track_id": "track-1",
            "transcription": "A short audio clip.",
            "suggestions": ["#construction", "#news"],
        }
    ]
    orchestrator._transcriber.transcribe.assert_awaited_once_with(
        b"audio fixture",
        job_id="job-1",
        run_id="run-1",
        track_id="track-1",
        short_utterance=True,
    )
    orchestrator._moderator.moderate.assert_not_awaited()
    orchestrator._categorizer.categorize.assert_awaited_once_with(
        transcript="A short audio clip.",
        segments=[],
        custom_tags=orchestrator_platform.auto_tag_keywords,
        max_tags=2,
        per_track_transcripts={"track-1": "A short audio clip."},
    )
