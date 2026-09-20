from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from hear.orchestrator import Orchestrator
from hear.services.reconstruction.service import RegenerationService


@pytest.mark.anyio
@pytest.mark.parametrize("same_speaker", [False, True])
async def test_edit_transcript_forwards_stored_speaker_control(monkeypatch, tmp_path, same_speaker):
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    changes = [{"segment_start": 0, "segment_end": 1, "new_text": "new"}]
    job = SimpleNamespace(
        id="job",
        run_id="run",
        backend_id="backend",
        job_type="edit_transcript",
        input_url="https://audio.test/source",
        edited_transcript="new",
        custom_tags={"same_speaker": same_speaker},
    )
    runner = object.__new__(Orchestrator.func_or_class)
    runner._track_from_job = Mock(
        return_value=SimpleNamespace(track_id="track", audio_url=job.input_url)
    )
    runner._transcriber = SimpleNamespace(
        transcribe_file=AsyncMock(
            return_value={
                "transcript": "old",
                "segments": [{"text": "old", "start": 0, "end": 1}],
            }
        )
    )
    runner._download_reconstruction_root = AsyncMock(return_value=None)
    runner._set_stage = AsyncMock(return_value=True)
    runner._complete = AsyncMock(return_value=True)
    synthesize = AsyncMock(
        return_value=SimpleNamespace(
            audio_url="https://audio.test/result",
            b2_key="key",
            duration=1,
            bucket_name="bucket",
            segments=[],
        )
    )
    runner._synthesizer = SimpleNamespace(reconstruct_segments=synthesize)
    monkeypatch.setattr("hear.orchestrator.download_audio", AsyncMock(return_value=str(source)))
    monkeypatch.setattr("hear.orchestrator.compute_edit_segments", lambda *_args: ["edit"])
    monkeypatch.setattr("hear.orchestrator.edit_segments_to_changes", lambda *_args: changes)
    monkeypatch.setattr("hear.orchestrator.storage_for_job", lambda _job: object())

    await runner._process_edit_transcript(job, object(), object())

    assert synthesize.await_args.kwargs["same_speaker"] is same_speaker
    assert synthesize.await_args.kwargs["voice_reference_audio_path"] == (
        str(source) if same_speaker else None
    )


@pytest.mark.anyio
async def test_preview_quality_exception_is_not_passed(monkeypatch):
    preview = SimpleNamespace(
        audio_url="https://audio.test/preview",
        b2_key="key",
        duration=1,
        segments=[],
    )
    synthesizer = SimpleNamespace(
        TARGET_SR=100,
        generate_preview=AsyncMock(return_value=preview),
        _compute_seed=Mock(return_value=1),
    )
    assessor = SimpleNamespace(assess=Mock(side_effect=RuntimeError("private model detail")))
    service = RegenerationService(synthesizer, assessor)
    service._broadcast_event = AsyncMock()
    service._download_to_temp = Mock(return_value="preview.wav")
    service._commit = AsyncMock()
    monkeypatch.setattr(
        "hear.services.reconstruction.service.download_audio", AsyncMock(return_value="source.wav")
    )
    monkeypatch.setattr("hear.services.reconstruction.service.drop_temp_standalone", Mock())
    monkeypatch.setattr(
        "hear.services.reconstruction.service.torchaudio.load",
        lambda _path: (torch.ones(1, 100), 100),
    )
    monkeypatch.setattr(
        "hear.services.reconstruction.service.SessionLocal", Mock(return_value=Mock())
    )
    monkeypatch.setattr(
        "hear.services.reconstruction.service.encrypt_storage_context", lambda _context: "encrypted"
    )

    result = await service.create_preview(
        "track",
        "https://audio.test/source",
        [{"segment_start": 0, "segment_end": 1, "new_text": "new"}],
        backend_id="backend",
        storage=SimpleNamespace(context=object()),
    )

    assert result.quality_metrics == {
        "passed": False,
        "status": "unavailable",
        "error": "quality_assessment_failed",
    }
