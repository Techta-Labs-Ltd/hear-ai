from types import SimpleNamespace

from hear.orchestrator import Orchestrator


def test_no_content_report_is_structured_and_includes_transcription():
    transcript = {"transcript": "", "language": "en", "segments": []}

    report = Orchestrator.func_or_class._no_content_report(transcript)

    assert report == {
        "flagged": True,
        "code": "content_not_detected",
        "reason": "No usable spoken content was detected in the transcription",
        "transcription": transcript,
    }


def test_stage_result_is_emitted_as_incremental_grpc_payload():
    orchestrator_class = Orchestrator.func_or_class
    orchestrator = orchestrator_class.__new__(orchestrator_class)
    events = []
    orchestrator._push_event = lambda job_id, event: events.append(event)
    job = SimpleNamespace(
        id="job-1", run_id="run-1", job_type="pipeline"
    )
    track_job = SimpleNamespace(track_id="track-1")

    orchestrator._push_stage_result(
        job, track_job, "transcribing", {"transcript": "Hello"}
    )

    assert events[0]["event"] == "stage_result"
    assert events[0]["progress_pct"] == 25
    assert events[0]["result"] == {
        "stage": "transcribing",
        "data": {"transcript": "Hello"},
    }


def test_track_context_is_built_from_audio_request_without_backend_fetch():
    job = SimpleNamespace(
        track_id="track-1",
        input_url="https://audio.test/source.mp3",
        job_options={"source": "upload"},
    )

    track = Orchestrator.func_or_class._track_from_job(job)

    assert track.track_id == "track-1"
    assert track.audio_url == "https://audio.test/source.mp3"
    assert track.transcription is None
    assert track.name == ""
    assert track.duration == 0
