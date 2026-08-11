from hear.services.transcription.service import TranscriptionService


def _result(text="Thank you.", *, logprob=-0.2, audio_duration=30.0):
    return {
        "audio_duration": audio_duration,
        "language": "en",
        "segments": [{
            "text": text,
            "start": 12.0,
            "end": 13.2,
            "avg_logprob": logprob,
        }],
    }


def test_noise_only_thank_you_is_treated_as_silent():
    result = TranscriptionService()._process_result(_result())
    assert result["silent"] is True
    assert result["transcript"] == ""
    assert result["segments"] == []


def test_low_confidence_segment_is_treated_as_silent():
    result = TranscriptionService()._process_result(
        _result("invented words", logprob=-1.2)
    )
    assert result["silent"] is True


def test_short_utterance_can_legitimately_say_thank_you():
    result = TranscriptionService()._process_result(
        _result(audio_duration=2.0), short_utterance=True
    )
    assert result["silent"] is False
    assert result["transcript"] == "Thank you."


def test_normal_credible_speech_is_preserved():
    result = TranscriptionService()._process_result(
        _result("This is a complete spoken recording", audio_duration=30.0)
    )
    assert result["silent"] is False
    assert result["transcript"] == "This is a complete spoken recording"
