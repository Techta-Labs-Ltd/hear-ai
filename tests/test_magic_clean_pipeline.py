import numpy as np
import torch

from hear.services.magic_clean.models import ContentMode, StemLevels
from hear.services.magic_clean.pipeline import MagicCleanPipeline, MagicCleanProfile
from hear.services.magic_clean.processing.dynamics import DynamicsProcessor


class RecordingProcessor:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self._loaded = False

    def load(self) -> None:
        self.calls.append("mossformer.load")

    def enhance(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        self.calls.append("mossformer")
        return waveform

    def spectral_suppress(
        self, waveform: torch.Tensor, sr: int, strength: float
    ) -> torch.Tensor:
        self.calls.append(f"spectral:{strength}")
        return waveform

    def apply_eq_speech(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        self.calls.append("speech_eq")
        return waveform

    def apply_deesser(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        self.calls.append("deesser")
        return waveform

    def apply_eq_music(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        self.calls.append("music_eq")
        return waveform

    def compress(
        self, waveform: torch.Tensor, sr: int, mode: ContentMode
    ) -> torch.Tensor:
        self.calls.append("compress")
        return waveform

    def normalise_lufs(self, waveform: torch.Tensor) -> torch.Tensor:
        self.calls.append("normalise")
        return waveform

    def lookahead_limit(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        self.calls.append("limit")
        return waveform

    def detect_and_strip_silence(
        self, waveform: torch.Tensor, sr: int
    ) -> torch.Tensor:
        self.calls.append("cut_silence")
        return waveform[:, : waveform.shape[1] // 2]

    def load_stem(self, model: str) -> None:
        self.calls.append("stem.load")

    def load(self, model: str | None = None) -> None:
        self._loaded = True
        self.calls.append("stem.load" if model else "mossformer.load")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def separate(self, waveform: torch.Tensor, sr: int) -> dict[str, torch.Tensor]:
        self.calls.append("stem")
        return {"vocals": waveform, "other": torch.zeros_like(waveform)}


def make_pipeline(calls: list[str], profile: MagicCleanProfile | None = None):
    mossformer = RecordingProcessor(calls)
    shared = RecordingProcessor(calls)
    stem = RecordingProcessor(calls)
    return MagicCleanPipeline(
        mossformer=mossformer,
        noise=shared,
        speech=shared,
        dynamics=shared,
        stem=stem,
        silence=shared,
        profile=profile,
    )


def test_default_speech_pipeline_preserves_timeline_and_skips_stem_separation():
    calls: list[str] = []
    pipeline = make_pipeline(calls)
    waveform = torch.randn(1, 44_100)

    result = pipeline.process(waveform, 44_100, ContentMode.SPEECH)

    assert result.shape == waveform.shape
    assert calls == [
        "mossformer",
        "spectral:0.65",
        "speech_eq",
        "deesser",
        "compress",
        "normalise",
        "limit",
    ]
    assert "stem" not in calls


def test_default_profile_does_not_load_demucs():
    calls: list[str] = []
    pipeline = make_pipeline(calls)

    pipeline.load("htdemucs")

    assert calls == ["mossformer.load"]


def test_pipeline_disables_autograd_during_inference():
    class GradRecordingProcessor(RecordingProcessor):
        def enhance(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
            assert not torch.is_grad_enabled()
            assert torch.is_inference_mode_enabled()
            return super().enhance(waveform, sr)

    calls: list[str] = []
    shared = RecordingProcessor(calls)
    pipeline = MagicCleanPipeline(
        mossformer=GradRecordingProcessor(calls),
        noise=shared,
        speech=shared,
        dynamics=shared,
        stem=RecordingProcessor(calls),
        silence=shared,
    )

    pipeline.process(torch.ones(1, 100, requires_grad=True), 44_100, ContentMode.SPEECH)


def test_silence_cutting_is_opt_in_and_runs_before_final_loudness():
    calls: list[str] = []
    pipeline = make_pipeline(calls)

    result = pipeline.process(
        torch.ones(1, 100),
        44_100,
        ContentMode.SPEECH,
        cut_silence=True,
    )

    assert result.shape == (1, 50)
    assert calls[-4:] == ["compress", "cut_silence", "normalise", "limit"]


def test_user_stem_percentages_are_applied_to_the_mix():
    calls: list[str] = []
    pipeline = make_pipeline(calls)
    pipeline.load("htdemucs")
    waveform = torch.ones(1, 100)

    result = pipeline.process(
        waveform,
        44_100,
        ContentMode.SPEECH,
        StemLevels(speech=50, music=10, background=10),
    )

    torch.testing.assert_close(result, waveform * 0.5)
    assert "stem.load" in calls
    assert "stem" in calls
    assert "spectral:0.9" in calls


def test_background_percentage_inversely_controls_bounded_noise_suppression():
    assert MagicCleanPipeline._suppression_strength_for_background(100) == 0.0
    assert MagicCleanPipeline._suppression_strength_for_background(50) == 0.5
    assert MagicCleanPipeline._suppression_strength_for_background(10) == 0.9
    assert MagicCleanPipeline._suppression_strength_for_background(0) == 0.9


def _reference_lookahead_limit(
    processor: DynamicsProcessor, waveform: torch.Tensor, sr: int
) -> torch.Tensor:
    ceiling = 10 ** (processor.TRUE_PEAK_DBTP / 20)
    lookahead = int(sr * processor.LIMITER_LOOKAHEAD_MS / 1000)
    release = np.exp(-1.0 / (sr * processor.LIMITER_RELEASE_MS / 1000))
    signal = waveform.squeeze(0).numpy().astype(np.float64)
    absolute = np.abs(signal)
    peaks = np.array(
        [absolute[i : min(i + lookahead + 1, len(signal))].max() for i in range(len(signal))]
    )
    gain = np.ones(len(signal), dtype=np.float64)
    previous = 1.0
    for index, peak in enumerate(peaks):
        required = min(ceiling / (peak + 1e-10), 1.0)
        previous = required if required < previous else release * previous + (1 - release) * required
        gain[index] = previous
    delayed = np.ones(len(signal), dtype=np.float64)
    delayed[lookahead:] = gain[: len(signal) - lookahead]
    delayed[:lookahead] = gain[0]
    return waveform * torch.from_numpy(delayed.astype(np.float32)).unsqueeze(0)


def test_rolling_limiter_matches_previous_algorithm():
    generator = torch.Generator().manual_seed(7)
    waveform = torch.randn(1, 8_000, generator=generator) * 0.8
    processor = DynamicsProcessor(torch.device("cpu"))

    expected = _reference_lookahead_limit(processor, waveform, 8_000)
    actual = processor.lookahead_limit(waveform, 8_000)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_limiter_handles_audio_shorter_than_lookahead():
    waveform = torch.tensor([[0.1, 1.0, -0.2]], dtype=torch.float32)
    processor = DynamicsProcessor(torch.device("cpu"))

    result = processor.lookahead_limit(waveform, 44_100)

    assert result.shape == waveform.shape
    assert torch.isfinite(result).all()
    assert result.abs().max() <= 10 ** (processor.TRUE_PEAK_DBTP / 20)


def test_chunked_pipeline_bounds_each_processing_window_and_preserves_length():
    calls: list[str] = []
    pipeline = make_pipeline(calls)
    waveform = torch.ones(1, 25)

    result = pipeline.process_chunked(
        waveform,
        10,
        ContentMode.SPEECH,
        chunk_seconds=1.0,
        overlap_seconds=0.2,
    )

    assert result.shape == waveform.shape
    torch.testing.assert_close(result, waveform)
    assert calls.count("mossformer") == 4


def test_chunked_pipeline_rejects_overlap_as_large_as_chunk():
    calls: list[str] = []
    pipeline = make_pipeline(calls)

    try:
        pipeline.process_chunked(
            torch.ones(1, 20),
            10,
            ContentMode.SPEECH,
            chunk_seconds=1.0,
            overlap_seconds=1.0,
        )
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("expected invalid overlap to be rejected")

