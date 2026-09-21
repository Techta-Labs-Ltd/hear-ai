from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .models import ContentMode, StemLevels
from .processing.validation import SILENCE_PEAK_THRESHOLD, SILENCE_RMS_THRESHOLD


@dataclass(frozen=True, slots=True)
class MagicCleanProfile:
    use_stem_separation: bool = False


    residual_suppression_strength: float = 0.0
    enable_spectral_suppression: bool = False
    apply_fixed_tone_shaping: bool = False
    speech_protection_dry_mix: float = 1.0


class MagicCleanPipeline:
    MAX_USER_SUPPRESSION_STRENGTH = 0.90
    MAX_ERASED_ACTIVE_SAMPLE_FRACTION = 0.80
    SEVERE_SAMPLE_ATTENUATION_RATIO = 0.05

    def __init__(
        self,
        *,
        mossformer: Any,
        noise: Any,
        speech: Any,
        dynamics: Any,
        stem: Any,
        silence: Any | None = None,
        profile: MagicCleanProfile | None = None,
    ) -> None:
        self._mossformer = mossformer
        self._noise = noise
        self._speech = speech
        self._dynamics = dynamics
        self._stem = stem
        self._silence = silence
        self.profile = profile or MagicCleanProfile()
        self._demucs_model: str | None = None
        self._demucs_model_path: str | None = None

    def load(
        self,
        demucs_model: str,
        mossformer_model_path: str | None = None,
        demucs_model_path: str | None = None,
    ) -> None:
        self._mossformer.load(mossformer_model_path)
        self._demucs_model = demucs_model
        self._demucs_model_path = demucs_model_path


        if demucs_model_path is not None:
            self._ensure_stem_loaded()

    @torch.inference_mode()
    def process(
        self,
        waveform: torch.Tensor,
        sr: int,
        mode: ContentMode,
        levels: StemLevels | None = None,
        cut_silence: bool = False,
        finalise: bool = True,
    ) -> torch.Tensor:
        self._validate_waveform(waveform, sr)
        if levels is not None:
            enhanced = self._process_stem_mix(waveform, sr, levels)
            final_mode = ContentMode.SPEECH
        elif mode == ContentMode.MUSIC:
            enhanced = (
                self._speech.apply_eq_music(waveform, sr)
                if self.profile.apply_fixed_tone_shaping
                else waveform
            )
            final_mode = mode
        else:
            enhanced = waveform
            if self.profile.use_stem_separation:
                enhanced = self._stem.separate(enhanced, sr)["vocals"]
            enhanced = self._mossformer.enhance(enhanced, sr)
            enhanced = self._apply_residual_suppression(
                enhanced,
                sr,
                self.profile.residual_suppression_strength,
            )
            enhanced = self._apply_speech_tone_shaping(enhanced, sr)
            enhanced = self._protect_source_activity(waveform, enhanced, sr)
            final_mode = mode

        self._validate_processed(enhanced, waveform.shape)
        if not finalise:
            return enhanced
        return self.finalise(
            enhanced,
            sr,
            final_mode,
            cut_silence=cut_silence,
        )

    @torch.inference_mode()
    def process_chunked(
        self,
        waveform: torch.Tensor,
        sr: int,
        mode: ContentMode,
        levels: StemLevels | None = None,
        cut_silence: bool = False,
        *,
        chunk_seconds: float = 60.0,
        overlap_seconds: float = 2.0,
    ) -> torch.Tensor:
        """Process context windows, retain their cores, then master exactly once."""
        self._validate_waveform(waveform, sr)
        chunk_samples, overlap_samples = self._chunk_sizes(sr, chunk_seconds, overlap_seconds)
        if waveform.shape[1] <= chunk_samples:
            return self.process(waveform, sr, mode, levels, cut_silence)

        core_samples = chunk_samples - (2 * overlap_samples)
        if core_samples <= 0:
            raise ValueError("chunk context must leave a non-empty core")
        pieces: list[torch.Tensor] = []
        total_samples = waveform.shape[1]
        for core_start in range(0, total_samples, core_samples):
            core_end = min(core_start + core_samples, total_samples)
            window_start = max(0, core_start - overlap_samples)
            window_end = min(total_samples, core_end + overlap_samples)
            window = waveform[:, window_start:window_end]
            processed = self.process(
                window,
                sr,
                mode,
                levels,
                False,
                False,
            )
            self._validate_processed(processed, window.shape)
            core_offset = core_start - window_start
            pieces.append(processed[:, core_offset : core_offset + (core_end - core_start)])

        joined = torch.cat(pieces, dim=1)
        self._validate_processed(joined, waveform.shape)
        return self.finalise(
            joined,
            sr,
            ContentMode.SPEECH if levels is not None else mode,
            cut_silence=cut_silence,
        )

    @staticmethod
    def _chunk_sizes(sr: int, chunk_seconds: float, overlap_seconds: float) -> tuple[int, int]:
        if sr <= 0 or chunk_seconds <= 0 or overlap_seconds < 0:
            raise ValueError("sample rate and chunk duration must be positive")
        chunk_samples = max(1, round(sr * chunk_seconds))
        overlap_samples = round(sr * overlap_seconds)
        if overlap_samples * 2 >= chunk_samples:
            raise ValueError("two overlap/context margins must be shorter than the chunk")
        return chunk_samples, overlap_samples

    def _suppression_strength_for_background(self, background: int) -> float:
        """Convert retained-background percentage into bounded noise suppression."""
        if not self.profile.enable_spectral_suppression:
            return 0.0
        requested = 1.0 - (background / 100.0)
        return max(0.0, min(requested, self.MAX_USER_SUPPRESSION_STRENGTH))

    def _process_stem_mix(
        self,
        waveform: torch.Tensor,
        sr: int,
        levels: StemLevels,
    ) -> torch.Tensor:
        self._ensure_stem_loaded()
        stems = self._stem.separate(waveform, sr)
        speech = stems["vocals"]
        music_stems = [stem for name, stem in stems.items() if name != "vocals"]
        music = torch.stack(music_stems).sum(dim=0) if music_stems else torch.zeros_like(waveform)
        self._validate_processed(speech, waveform.shape)
        self._validate_processed(music, waveform.shape)
        speech = self._mossformer.enhance(speech, sr)
        speech = self._apply_residual_suppression(
            speech,
            sr,
            self.profile.residual_suppression_strength,
        )
        speech = self._apply_speech_tone_shaping(speech, sr)
        self._validate_processed(speech, waveform.shape)
        background = waveform - speech - music

        mixed = (
            speech * (levels.speech / 100.0)
            + music * (levels.music / 100.0)
            + background * (levels.background / 100.0)
        )
        return mixed

    def _ensure_stem_loaded(self) -> None:
        if self._stem.is_loaded:
            return
        if not self._demucs_model or not self._demucs_model_path:
            raise RuntimeError("Magic Clean must be loaded before stem mixing")
        self._stem.load(self._demucs_model, self._demucs_model_path)

    def finalise(
        self,
        waveform: torch.Tensor,
        sr: int,
        mode: ContentMode,
        *,
        cut_silence: bool = False,
        master: bool = True,
    ) -> torch.Tensor:
        """Apply one global silence edit and, optionally, one mastering pass."""
        self._validate_waveform(waveform, sr)
        if cut_silence:
            if self._silence is None:
                raise RuntimeError("Silence cutting was requested but is unavailable")
            waveform = self._silence.detect_and_strip_silence(waveform, sr)
            self._validate_waveform(waveform, sr)
        if not master:
            return waveform
        waveform = self._dynamics.compress(waveform, sr, mode)
        waveform = self._dynamics.normalise_lufs(waveform)
        waveform = self._dynamics.lookahead_limit(waveform, sr)
        self._validate_waveform(waveform, sr)
        return waveform

    def strip_silence_file(
        self,
        source_path: str,
        enhanced_path: str,
        output_path: str,
    ) -> int:
        """Run one disk-backed source/output-union silence decision pass."""
        if self._silence is None:
            raise RuntimeError("Silence cutting was requested but is unavailable")
        return self._silence.detect_and_strip_silence_file(
            source_path,
            enhanced_path,
            output_path,
        )

    def _apply_residual_suppression(
        self,
        waveform: torch.Tensor,
        sr: int,
        strength: float,
    ) -> torch.Tensor:
        safe_strength = strength if self.profile.enable_spectral_suppression else 0.0
        return self._noise.spectral_suppress(waveform, sr, strength=safe_strength)

    def _apply_speech_tone_shaping(
        self,
        waveform: torch.Tensor,
        sr: int,
    ) -> torch.Tensor:
        if not self.profile.apply_fixed_tone_shaping:
            return waveform
        shaped = self._speech.apply_eq_speech(waveform, sr)
        return self._speech.apply_deesser(shaped, sr)

    def _protect_source_activity(
        self,
        source: torch.Tensor,
        speech: torch.Tensor,
        sr: int,
    ) -> torch.Tensor:
        """Conservatively restore source in frames where separation collapses."""
        self._validate_processed(speech, source.shape)
        dry_mix = max(0.0, min(self.profile.speech_protection_dry_mix, 1.0))
        if dry_mix == 0.0:
            return speech

        frame_samples = max(1, round(sr * 0.02))
        sample_count = source.shape[1]
        padding = (-sample_count) % frame_samples
        source_frames = F.pad(source, (0, padding)).reshape(
            source.shape[0],
            -1,
            frame_samples,
        )
        speech_frames = F.pad(speech, (0, padding)).reshape_as(source_frames)
        source_rms = source_frames.square().mean(dim=2).sqrt()
        speech_rms = speech_frames.square().mean(dim=2).sqrt()
        source_peak = source_frames.abs().amax(dim=2)
        source_active = (source_peak >= SILENCE_PEAK_THRESHOLD) | (
            source_rms >= SILENCE_RMS_THRESHOLD
        )
        active_samples = source_frames.abs() >= SILENCE_RMS_THRESHOLD
        severely_attenuated_samples = active_samples & (
            speech_frames.abs() < source_frames.abs() * self.SEVERE_SAMPLE_ATTENUATION_RATIO
        )
        active_sample_count = active_samples.sum(dim=2)
        severely_attenuated_fraction = severely_attenuated_samples.sum(
            dim=2
        ) / active_sample_count.clamp(min=1)
        fragmented = (active_sample_count > 0) & (
            severely_attenuated_fraction >= self.MAX_ERASED_ACTIVE_SAMPLE_FRACTION
        )
        collapsed = source_active & ((speech_rms < source_rms * 0.25) | fragmented)
        restored = speech_frames * (1.0 - dry_mix) + source_frames * dry_mix
        protected = torch.where(
            collapsed[:, :, None],
            restored,
            speech_frames,
        )
        return protected.reshape(source.shape[0], -1)[:, :sample_count]

    @staticmethod
    def _validate_waveform(waveform: torch.Tensor, sr: int) -> None:
        if sr <= 0:
            raise ValueError("sample rate must be positive")
        if waveform.ndim != 2 or waveform.shape[0] < 1 or waveform.shape[1] < 1:
            raise ValueError("waveform must be non-empty [channels, samples] audio")
        if not torch.isfinite(waveform).all():
            raise ValueError("waveform contains non-finite samples")

    @staticmethod
    def _validate_processed(
        waveform: torch.Tensor,
        expected_shape: torch.Size | tuple[int, ...],
    ) -> None:
        if tuple(waveform.shape) != tuple(expected_shape):
            raise RuntimeError(
                "Magic Clean processor changed channel count or sample length: "
                f"expected={tuple(expected_shape)}, actual={tuple(waveform.shape)}"
            )
        if not torch.isfinite(waveform).all():
            raise RuntimeError("Magic Clean processor returned non-finite samples")
