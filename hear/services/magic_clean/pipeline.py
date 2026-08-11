from dataclasses import dataclass
from typing import Any
import torch
from .models import ContentMode, StemLevels

@dataclass(frozen=True, slots=True)
class MagicCleanProfile:
    use_stem_separation: bool = False
    residual_suppression_strength: float = 0.65


class MagicCleanPipeline:
    MAX_USER_SUPPRESSION_STRENGTH = 0.90

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

    def load(self, demucs_model: str) -> None:
        self._mossformer.load()
        self._demucs_model = demucs_model
        if self.profile.use_stem_separation:
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
        if levels is not None:
            return self._process_stem_mix(waveform, sr, levels, cut_silence, finalise)

        if mode == ContentMode.MUSIC:
            shaped = self._speech.apply_eq_music(waveform, sr)
            return self._finish(shaped, sr, mode, cut_silence, finalise)

        enhanced = waveform
        if self.profile.use_stem_separation:
            enhanced = self._stem.separate(enhanced, sr)["vocals"]

        enhanced = self._mossformer.enhance(enhanced, sr)
        enhanced = self._noise.spectral_suppress(
            enhanced,
            sr,
            strength=self.profile.residual_suppression_strength,
        )
        enhanced = self._speech.apply_eq_speech(enhanced, sr)
        enhanced = self._speech.apply_deesser(enhanced, sr)
        return self._finish(enhanced, sr, mode, cut_silence, finalise)

    @torch.inference_mode()
    def process_chunked(
        self, waveform: torch.Tensor, sr: int, mode: ContentMode,
        levels: StemLevels | None = None, cut_silence: bool = False, *,
        chunk_seconds: float = 60.0, overlap_seconds: float = 2.0,
    ) -> torch.Tensor:
        """Process bounded windows and join them with equal-power crossfades."""
        chunk_samples, overlap_samples = self._chunk_sizes(
            sr, chunk_seconds, overlap_seconds
        )
        if waveform.shape[1] <= chunk_samples:
            return self.process(waveform, sr, mode, levels, cut_silence)
        if cut_silence:
            overlap_samples = 0
        pieces: list[torch.Tensor] = []
        pending: torch.Tensor | None = None
        step = chunk_samples - overlap_samples
        for start in range(0, waveform.shape[1], step):
            current = self.process(
                waveform[:, start:start + chunk_samples], sr, mode, levels, cut_silence
            )
            if pending is None:
                pending = current
                continue
            crossfade = min(overlap_samples, pending.shape[1], current.shape[1])
            if crossfade:
                fade = torch.linspace(
                    0.0, 1.0, crossfade, device=current.device, dtype=current.dtype
                )
                pieces.append(pending[:, :-crossfade])
                pieces.append(
                    pending[:, -crossfade:] * torch.cos(fade * torch.pi / 2).square()
                    + current[:, :crossfade] * torch.sin(fade * torch.pi / 2).square()
                )
                pending = current[:, crossfade:]
            else:
                pieces.append(pending)
                pending = current
        if pending is not None:
            pieces.append(pending)
        return torch.cat(pieces, dim=1)

    @staticmethod
    def _chunk_sizes(
        sr: int, chunk_seconds: float, overlap_seconds: float
    ) -> tuple[int, int]:
        if sr <= 0 or chunk_seconds <= 0 or overlap_seconds < 0:
            raise ValueError("sample rate and chunk duration must be positive")
        chunk_samples = max(1, round(sr * chunk_seconds))
        overlap_samples = round(sr * overlap_seconds)
        if overlap_samples >= chunk_samples:
            raise ValueError("chunk overlap must be shorter than the chunk")
        return chunk_samples, overlap_samples

    @classmethod
    def _suppression_strength_for_background(cls, background: int) -> float:
        """Convert retained-background percentage into bounded noise suppression."""
        requested = 1.0 - (background / 100.0)
        return max(0.0, min(requested, cls.MAX_USER_SUPPRESSION_STRENGTH))

    def _process_stem_mix(
        self,
        waveform: torch.Tensor,
        sr: int,
        levels: StemLevels,
        cut_silence: bool = False,
        finalise: bool = True,
    ) -> torch.Tensor:
        self._ensure_stem_loaded()
        stems = self._stem.separate(waveform, sr)
        speech = stems["vocals"]
        music_stems = [stem for name, stem in stems.items() if name != "vocals"]
        music = torch.stack(music_stems).sum(dim=0)
        background = waveform - speech - music

        speech = self._mossformer.enhance(speech, sr)
        speech = self._noise.spectral_suppress(
            speech,
            sr,
            strength=self._suppression_strength_for_background(levels.background),
        )
        speech = self._speech.apply_eq_speech(speech, sr)
        speech = self._speech.apply_deesser(speech, sr)

        mixed = (
            speech * (levels.speech / 100.0)
            + music * (levels.music / 100.0)
            + background * (levels.background / 100.0)
        )
        return self._finish(mixed, sr, ContentMode.MUSIC, cut_silence, finalise)

    def _ensure_stem_loaded(self) -> None:
        if self._stem.is_loaded:
            return
        if not self._demucs_model:
            raise RuntimeError("Magic Clean must be loaded before stem mixing")
        self._stem.load(self._demucs_model)

    def _finish(
        self,
        waveform: torch.Tensor,
        sr: int,
        mode: ContentMode,
        cut_silence: bool = False,
        finalise: bool = True,
    ) -> torch.Tensor:
        waveform = self._dynamics.compress(waveform, sr, mode)
        if cut_silence:
            if self._silence is None:
                raise RuntimeError("Silence cutting was requested but is unavailable")
            waveform = self._silence.detect_and_strip_silence(waveform, sr)
        if not finalise:
            return waveform
        waveform = self._dynamics.normalise_lufs(waveform)
        return self._dynamics.lookahead_limit(waveform, sr)
