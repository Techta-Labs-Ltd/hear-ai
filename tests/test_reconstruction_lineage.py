import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hear.orchestrator import OrchestrationResults, Orchestrator

RETRY_CHANGES = [{"segment_start": 0.25, "segment_end": 1.25}]


def _job(
    job_id: str,
    input_url: str,
    output_url: str,
    *,
    backend_id: str = "backend-a",
    track_id: str = "track-a",
    segment_url: str | None = None,
    root_hint: str | None = None,
    changes: list[dict] | None = None,
):
    segments = [{"audio_url": segment_url}] if segment_url else []
    options = {"voice_reference_audio_url": root_hint} if root_hint else {}
    return SimpleNamespace(
        id=job_id,
        backend_id=backend_id,
        track_id=track_id,
        status="completed",
        job_type="reconstruct",
        input_url=input_url,
        custom_tags={"changes": RETRY_CHANGES if changes is None else changes},
        result_json={"rebuilt_audio": {"audio_url": output_url}, "segments": segments},
        job_options=options,
    )


def test_resolves_multi_generation_chain_to_original_input():
    first = _job("job-1", "https://audio/root.wav", "https://audio/one.mp3")
    second = _job("job-2", "https://audio/one.mp3", "https://audio/two.mp3")
    resolved, hops = OrchestrationResults.resolve_reconstruction_reference_url(
        "https://audio/two.mp3",
        backend_id="backend-a",
        track_id="track-a",
        changes=RETRY_CHANGES,
        jobs=[second, first],
    )
    assert resolved == "https://audio/root.wav"
    assert hops == 2


def test_persisted_root_hint_collapses_a_retry_chain():
    parent = _job(
        "job-2",
        "https://audio/one.mp3",
        "https://audio/two.mp3",
        root_hint="https://audio/root.wav",
    )
    resolved, hops = OrchestrationResults.resolve_reconstruction_reference_url(
        "https://audio/two.mp3",
        backend_id="backend-a",
        track_id="track-a",
        changes=RETRY_CHANGES,
        jobs=[parent],
    )
    assert resolved == "https://audio/root.wav"
    assert hops == 1


def test_changed_retry_intervals_fail_closed():
    parent = _job("job-1", "https://audio/root.wav", "https://audio/rebuilt.mp3")
    with pytest.raises(ValueError, match="same original intervals"):
        OrchestrationResults.resolve_reconstruction_reference_url(
            "https://audio/rebuilt.mp3",
            backend_id="backend-a",
            track_id="track-a",
            changes=[{"segment_start": 2.0, "segment_end": 3.0}],
            jobs=[parent],
        )


def test_isolated_segment_timeline_fails_closed():
    parent = _job(
        "job-1",
        "https://audio/root.wav",
        "https://audio/rebuilt.mp3",
        segment_url="https://audio/segment.mp3",
    )
    with pytest.raises(ValueError, match="isolated reconstruction segments"):
        OrchestrationResults.resolve_reconstruction_reference_url(
            "https://audio/segment.mp3",
            backend_id="backend-a",
            track_id="track-a",
            changes=RETRY_CHANGES,
            jobs=[parent],
        )


def test_lineage_never_crosses_backend_or_track_boundaries():
    wrong_backend = _job(
        "job-1", "https://audio/root-a.wav", "https://audio/rebuilt.mp3", backend_id="backend-b"
    )
    wrong_track = _job(
        "job-2", "https://audio/root-b.wav", "https://audio/rebuilt.mp3", track_id="track-b"
    )
    resolved, hops = OrchestrationResults.resolve_reconstruction_reference_url(
        "https://audio/rebuilt.mp3",
        backend_id="backend-a",
        track_id="track-a",
        changes=RETRY_CHANGES,
        jobs=[wrong_backend, wrong_track],
    )
    assert resolved == "https://audio/rebuilt.mp3"
    assert hops == 0


def test_ambiguous_lineage_fails_closed():
    first = _job("job-1", "https://audio/root-a.wav", "https://audio/same.mp3")
    second = _job("job-2", "https://audio/root-b.wav", "https://audio/same.mp3")
    with pytest.raises(ValueError, match="ambiguous"):
        OrchestrationResults.resolve_reconstruction_reference_url(
            "https://audio/same.mp3",
            backend_id="backend-a",
            track_id="track-a",
            changes=RETRY_CHANGES,
            jobs=[first, second],
        )


def test_cyclic_lineage_fails_closed():
    first = _job("job-1", "https://audio/two.mp3", "https://audio/one.mp3")
    second = _job("job-2", "https://audio/one.mp3", "https://audio/two.mp3")
    with pytest.raises(ValueError, match="cycle"):
        OrchestrationResults.resolve_reconstruction_reference_url(
            "https://audio/one.mp3",
            backend_id="backend-a",
            track_id="track-a",
            changes=RETRY_CHANGES,
            jobs=[first, second],
        )


def test_malformed_results_are_ignored():
    malformed = SimpleNamespace(
        id="job-1",
        backend_id="backend-a",
        track_id="track-a",
        status="completed",
        job_type="reconstruct",
        input_url="https://audio/root.wav",
        result_json={"rebuilt_audio": None, "segments": "invalid"},
        job_options={},
    )
    resolved, hops = OrchestrationResults.resolve_reconstruction_reference_url(
        "https://audio/current.mp3",
        backend_id="backend-a",
        track_id="track-a",
        changes=RETRY_CHANGES,
        jobs=[malformed],
    )
    assert resolved == "https://audio/current.mp3"
    assert hops == 0


@pytest.mark.parametrize(
    ("same_speaker", "expected_voice_reference"), [(True, "/tmp/immutable-root.wav"), (False, None)]
)
def test_reconstruct_retry_splices_from_immutable_root(
    monkeypatch, same_speaker, expected_voice_reference
):
    changes = [
        {
            "segment_start": 0.25,
            "segment_end": 1.25,
            "new_text": "replacement speech",
            "original_text": "original speech",
        }
    ]
    job = SimpleNamespace(
        id="job-retry",
        run_id="run-retry",
        backend_id="backend-a",
        job_type="reconstruct",
        track_id="track-a",
        input_url="https://audio/rebuilt.mp3",
        custom_tags={"changes": changes, "same_speaker": same_speaker},
        job_options={},
    )
    rebuilt = SimpleNamespace(
        audio_url="https://audio/new-rebuilt.mp3",
        b2_key="rebuilt/key.mp3",
        duration=2.0,
        bucket_name="audio",
        segments=[],
    )
    orchestrator_class = Orchestrator.func_or_class
    orchestrator = orchestrator_class.__new__(orchestrator_class)
    orchestrator._download_reconstruction_root = AsyncMock(return_value="/tmp/immutable-root.wav")
    orchestrator._set_stage = AsyncMock(return_value=True)
    orchestrator._synthesizer = SimpleNamespace(
        reconstruct_segments=AsyncMock(return_value=rebuilt)
    )
    orchestrator._complete = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "hear.orchestrator.AudioDownloader.download_audio",
        AsyncMock(return_value="/tmp/submitted-rebuilt.wav"),
    )
    monkeypatch.setattr("hear.orchestrator.StorageContexts.storage_for_job", lambda _: object())
    asyncio.run(orchestrator._process_reconstruct(job, SimpleNamespace(), object()))
    call = orchestrator._synthesizer.reconstruct_segments.await_args
    assert call.kwargs["original_audio_path"] == "/tmp/immutable-root.wav"
    assert call.kwargs["voice_reference_audio_path"] == expected_voice_reference
    assert call.kwargs["same_speaker"] is same_speaker
