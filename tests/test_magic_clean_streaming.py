from __future__ import annotations

import math
import threading

import numpy as np
import pytest
import soundfile as sf
import torch

from hear.services.magic_clean.models import ContentMode
from hear.services.magic_clean.pipeline import MagicCleanPipeline
from hear.services.magic_clean.processing.audio_io import AudioIO
from hear.services.magic_clean.processing.quality import QualityMetrics
from hear.services.magic_clean.processing.validation import AudioValidator
from hear.services.magic_clean.streaming import MagicCleanProcessingCancelled, StreamingAudioCleaner


class IdentityPipeline:
    def __init__(self) -> None:
        self.process_calls = 0
        self.finalise_calls = 0

    _chunk_sizes = staticmethod(MagicCleanPipeline._chunk_sizes)

    def process(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        mode: ContentMode,
        levels,
        cut_silence: bool,
        finalise: bool,
    ) -> torch.Tensor:
        del sample_rate, mode, levels
        assert cut_silence is False
        assert finalise is False
        self.process_calls += 1
        return waveform.clone()

    def finalise(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        mode: ContentMode,
        *,
        cut_silence: bool,
        master: bool,
    ) -> torch.Tensor:
        del sample_rate, mode
        assert cut_silence is True
        assert master is False
        self.finalise_calls += 1
        return waveform[:, : waveform.shape[1] * 3 // 4]


class CancelAfterFirstWindowPipeline(IdentityPipeline):
    def __init__(self, cancel_event: threading.Event) -> None:
        super().__init__()
        self._cancel_event = cancel_event

    def process(self, *args, **kwargs) -> torch.Tensor:
        cleaned = super().process(*args, **kwargs)
        self._cancel_event.set()
        return cleaned


def _write_stereo_fixture(path, *, sample_rate: int = 48000, seconds: float = 1.25):
    sample_count = round(sample_rate * seconds)
    timeline = np.arange(sample_count, dtype=np.float64) / sample_rate
    left = 0.08 * np.sin(2 * math.pi * 311.0 * timeline)
    right = 0.04 * np.sin(2 * math.pi * 733.0 * timeline + 0.25)
    waveform = np.stack((left, right), axis=1).astype(np.float32)
    sf.write(path, waveform, sample_rate, subtype="FLOAT")
    return sample_count


@pytest.mark.parametrize("chunk_seconds", [0.3, 60.0])
def test_disk_backed_engine_preserves_stereo_timeline_and_validates_mp3(tmp_path, chunk_seconds):
    source_path = tmp_path / "source.wav"
    output_path = tmp_path / "cleaned.mp3"
    reference_path = tmp_path / "retained-reference.wav"
    sample_count = _write_stereo_fixture(source_path)
    pipeline = IdentityPipeline()
    result = StreamingAudioCleaner.clean_file_streaming(
        pipeline,
        str(source_path),
        str(output_path),
        device=torch.device("cpu"),
        mode=ContentMode.SPEECH,
        levels=None,
        chunk_seconds=chunk_seconds,
        overlap_seconds=0.05 if chunk_seconds < 1 else 2.0,
        bitrate_kbps=96,
        validation_reference_path=str(reference_path),
    )
    assert output_path.is_file()
    assert reference_path.is_file()
    assert sf.info(reference_path).format == "RF64"
    assert result.sample_rate == 48000
    assert result.channels == 2
    assert result.input_samples == sample_count
    assert result.output_samples == sample_count
    assert pipeline.process_calls == result.chunks_processed
    assert pipeline.finalise_calls == 0
    assert set(result.stage_times) == {
        "decode",
        "enhance_and_stitch",
        "measure",
        "master_and_encode",
    }
    delivered = AudioValidator.validate_delivered_audio(
        str(source_path),
        str(output_path),
        cut_silence=False,
        expect_audible=True,
        metrics=QualityMetrics(),
        retained_reference_path=str(reference_path),
    )
    assert delivered.channels == 2
    assert delivered.peak_db <= -1.0
    assert delivered.clipped_sample_count == 0


def test_lossless_decode_enables_automatic_rf64_upgrade(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(
        "hear.services.magic_clean.streaming.StreamingAudioCleaner._run",
        lambda command: commands.append(command),
    )
    StreamingAudioCleaner._decode_lossless("source.audio", "decoded.wav")
    assert len(commands) == 1
    rf64_option = commands[0].index("-rf64")
    assert commands[0][rf64_option + 1] == "auto"


def test_silence_edit_runs_once_after_all_context_cores_are_stitched(tmp_path):
    source_path = tmp_path / "source.wav"
    output_path = tmp_path / "shortened.mp3"
    sample_count = _write_stereo_fixture(source_path, seconds=1.8)
    source, sample_rate = sf.read(source_path, dtype="float32", always_2d=True)
    source[sample_count * 3 // 4 :] = 0.0
    sf.write(source_path, source, sample_rate, subtype="FLOAT")
    pipeline = IdentityPipeline()
    result = StreamingAudioCleaner.clean_file_streaming(
        pipeline,
        str(source_path),
        str(output_path),
        device=torch.device("cpu"),
        cut_silence=True,
        chunk_seconds=0.4,
        overlap_seconds=0.05,
    )
    assert pipeline.process_calls > 1
    assert pipeline.finalise_calls == 1
    assert result.output_samples == sample_count * 3 // 4
    assert "silence_edit" in result.stage_times
    delivered = AudioValidator.validate_delivered_audio(
        str(source_path),
        str(output_path),
        cut_silence=True,
        expect_audible=True,
        metrics=QualityMetrics(),
    )
    assert delivered.duration_seconds < result.input_duration_seconds


def test_cooperative_cancellation_stops_before_the_next_window(tmp_path):
    source_path = tmp_path / "source.wav"
    output_path = tmp_path / "cancelled.mp3"
    _write_stereo_fixture(source_path, seconds=1.2)
    cancel_event = threading.Event()
    pipeline = CancelAfterFirstWindowPipeline(cancel_event)
    with pytest.raises(MagicCleanProcessingCancelled, match="was cancelled"):
        StreamingAudioCleaner.clean_file_streaming(
            pipeline,
            str(source_path),
            str(output_path),
            device=torch.device("cpu"),
            chunk_seconds=0.3,
            overlap_seconds=0.05,
            cancel_event=cancel_event,
        )
    assert pipeline.process_calls == 1
    assert not output_path.exists()
    assert not (tmp_path / "cancelled.mp3.partial.mp3").exists()


def test_high_sample_rate_source_uses_supported_mp3_delivery_rate(tmp_path):
    source_path = tmp_path / "source-96k.wav"
    output_path = tmp_path / "cleaned.mp3"
    _write_stereo_fixture(source_path, sample_rate=96000, seconds=0.6)
    result = StreamingAudioCleaner.clean_file_streaming(
        IdentityPipeline(),
        str(source_path),
        str(output_path),
        device=torch.device("cpu"),
        chunk_seconds=60,
        overlap_seconds=2,
    )
    decoded, decoded_rate = AudioIO.load(str(output_path))
    assert result.sample_rate == 96000
    assert result.delivery_sample_rate == 48000
    assert decoded_rate == 48000
    assert decoded.shape[0] == 2


def test_limiter_compensates_lookahead_without_shifting_or_losing_tail(tmp_path):
    sample_rate = 48000
    source_path = tmp_path / "impulses.wav"
    output_path = tmp_path / "impulses.mp3"
    waveform = np.zeros((sample_rate, 1), dtype=np.float32)
    waveform[100, 0] = 0.7
    waveform[-101, 0] = -0.7
    sf.write(source_path, waveform, sample_rate, subtype="FLOAT")
    StreamingAudioCleaner.clean_file_streaming(
        IdentityPipeline(),
        str(source_path),
        str(output_path),
        device=torch.device("cpu"),
        chunk_seconds=60,
        overlap_seconds=2,
    )
    decoded, decoded_rate = AudioIO.load(str(output_path))
    absolute = decoded[0].abs()
    assert decoded_rate == sample_rate
    assert int(absolute[:1000].argmax()) == pytest.approx(100, abs=2)
    tail_peak = int(absolute[-1000:].argmax()) + decoded.shape[1] - 1000
    assert tail_peak == pytest.approx(decoded.shape[1] - 101, abs=2)


def test_codec_true_peak_overshoot_is_corrected_before_delivery(tmp_path):
    sample_rate = 48000
    source_path = tmp_path / "bursty-square.wav"
    output_path = tmp_path / "bursty-square.mp3"
    waveform = np.zeros((4 * sample_rate, 1), dtype=np.float32)
    burst_samples = round(0.01 * sample_rate)
    phase = np.arange(burst_samples)
    burst = 0.794 * np.where(phase % 48 < 24, 1.0, -1.0)
    for second in range(4):
        start = second * sample_rate
        waveform[start : start + burst_samples, 0] = burst
    sf.write(source_path, waveform, sample_rate, subtype="FLOAT")
    result = StreamingAudioCleaner.clean_file_streaming(
        IdentityPipeline(),
        str(source_path),
        str(output_path),
        device=torch.device("cpu"),
        chunk_seconds=60,
        overlap_seconds=2,
    )
    delivered = AudioValidator.validate_delivered_audio(
        str(source_path),
        str(output_path),
        cut_silence=False,
        expect_audible=True,
        metrics=QualityMetrics(),
    )
    assert result.peak_db <= -1.0
    assert delivered.peak_db <= -1.0


def test_silent_file_has_no_codec_true_peak_to_correct(tmp_path):
    sample_rate = 48000
    source_path = tmp_path / "silence.wav"
    output_path = tmp_path / "silence.mp3"
    sf.write(
        source_path, np.zeros((sample_rate, 1), dtype=np.float32), sample_rate, subtype="FLOAT"
    )
    result = StreamingAudioCleaner.clean_file_streaming(
        IdentityPipeline(),
        str(source_path),
        str(output_path),
        device=torch.device("cpu"),
        chunk_seconds=60,
        overlap_seconds=2,
    )
    delivered = AudioValidator.validate_delivered_audio(
        str(source_path),
        str(output_path),
        cut_silence=False,
        expect_audible=True,
        metrics=QualityMetrics(),
    )
    assert result.peak_db == -99.0
    assert delivered.peak_db == -99.0
    assert delivered.lufs == -99.0
