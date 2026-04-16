import asyncio
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from demucs.apply import apply_model
from demucs.pretrained import get_model

from app.config import settings
from app.core.storage import storage


@dataclass
class EnhancementResult:
    b2_key: str
    enhanced_url: str
    local_path: str
    quality_score: float
    snr_db: float
    peak_db: float
    lufs: float
    clipping_detected: bool


class AudioEnhancer:
    TARGET_SR = 44100
    TARGET_LUFS = -16.0

    def __init__(self):
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        self._model = get_model(settings.DEMUCS_MODEL)
        self._model.to(self._device)
        self._model.eval()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_audio(self, path: str) -> tuple[torch.Tensor, int]:
        file_size = os.path.getsize(path)
        if file_size == 0:
            raise ValueError(f"Audio file is empty: {path}")

        for backend in ("soundfile", "ffmpeg", None):
            try:
                kwargs = {"backend": backend} if backend else {}
                waveform, sr = torchaudio.load(path, **kwargs)
                return waveform, sr
            except Exception:
                continue

        raise RuntimeError(
            f"Could not load audio from {path} ({file_size} bytes) — "
            "no compatible torchaudio backend. Install soundfile or ffmpeg."
        )

    def _to_mono(self, w: torch.Tensor) -> torch.Tensor:
        return w.mean(dim=0, keepdim=True) if w.shape[0] > 1 else w

    def _resample(self, w: torch.Tensor, orig: int, target: int) -> torch.Tensor:
        return F.resample(w, orig, target) if orig != target else w

    def _detect_clipping(self, w: torch.Tensor) -> bool:
        return (w.abs() > 0.99).float().mean().item() > 0.001

    def _denoise_demucs(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:

        if waveform.dim() == 2 and waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        waveform_resampled = self._resample(waveform, sr, self._model.samplerate)

        with torch.no_grad():
            sources = apply_model(
                self._model,
                waveform_resampled[None].to(self._device),
                progress=False,
            )[0]

        vocals_idx = self._model.sources.index("vocals")
        vocals = sources[vocals_idx].cpu()
        vocals = self._resample(vocals, self._model.samplerate, self.TARGET_SR)
        return self._to_mono(vocals)

    def _apply_eq(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            effects = [
                ["highpass", "f=80"],
                ["lowpass", "f=14000"],
                ["equalizer", "f=200", "width_type=o", "width=2", "g=-2"],
                ["equalizer", "f=3000", "width_type=o", "width=1", "g=2"],
                ["compand", "0.1,0.3", "6:-70,-60,-20", "-5", "-90", "0.1"],
                ["rate", str(sr)],
            ]
            result, _ = torchaudio.sox_effects.apply_effects_tensor(w, sr, effects, channels_first=True)
            return result
        except (OSError, RuntimeError):
            print("[ENHANCER] libsox not available — skipping EQ")
            return w

    def _de_ess(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            effects = [
                ["equalizer", "f=7000", "width_type=o", "width=1.5", "g=-3"],
                ["equalizer", "f=9000", "width_type=o", "width=1", "g=-2"],
            ]
            result, _ = torchaudio.sox_effects.apply_effects_tensor(w, sr, effects, channels_first=True)
            return result
        except (OSError, RuntimeError):
            print("[ENHANCER] libsox not available — skipping de-essing")
            return w

    def _strip_silence(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            effects = [
                ["silence", "1", "0.1", "0.1%"],
                ["reverse"],
                ["silence", "1", "0.1", "0.1%"],
                ["reverse"],
            ]
            trimmed, _ = torchaudio.sox_effects.apply_effects_tensor(w, sr, effects, channels_first=True)
            if trimmed.shape[1] < sr * 0.1:
                return w
            return trimmed
        except (OSError, RuntimeError):
            print("[ENHANCER] libsox not available — skipping silence strip")
            return w

    def _remove_internal_silence(self, w: torch.Tensor, sr: int, max_gap_ms: int = 500) -> torch.Tensor:
        frame_size = int(sr * 0.02)
        threshold = 0.005
        max_gap_frames = int(max_gap_ms / 20)

        num_frames = w.shape[1] // frame_size
        energy = torch.zeros(num_frames)
        for i in range(num_frames):
            chunk = w[:, i * frame_size:(i + 1) * frame_size]
            energy[i] = chunk.pow(2).mean().sqrt().item()

        voiced = energy > threshold
        output_frames = []
        silence_count = 0

        for i in range(num_frames):
            if voiced[i]:
                if silence_count > 0 and silence_count <= max_gap_frames:
                    start = (i - silence_count) * frame_size
                    end = i * frame_size
                    output_frames.append(w[:, start:end])
                output_frames.append(w[:, i * frame_size:(i + 1) * frame_size])
                silence_count = 0
            else:
                silence_count += 1

        if not output_frames:
            return w
        return torch.cat(output_frames, dim=1)

    def _normalise_loudness(self, w: torch.Tensor, target_lufs: float = -16.0) -> torch.Tensor:
        rms = w.pow(2).mean().sqrt().item()
        if rms < 1e-8:
            return w
        target_rms = 10 ** (target_lufs / 20)
        w = w * (target_rms / rms)
        peak = w.abs().max().item()
        if peak > 0.99:
            w = w * (0.99 / peak)
        return w

    def _compute_snr(self, raw: torch.Tensor, enhanced: torch.Tensor) -> float:
        min_len = min(raw.shape[1], enhanced.shape[1])
        noise = raw[:, :min_len] - enhanced[:, :min_len]
        signal_power = enhanced[:, :min_len].pow(2).mean().item()
        noise_power = noise.pow(2).mean().item() + 1e-10
        return 10 * np.log10(signal_power / noise_power)

    def _compute_lufs(self, w: torch.Tensor) -> float:
        rms = w.pow(2).mean().sqrt().item()
        return 20 * np.log10(rms + 1e-8)

    def _compute_quality_score(self, snr_db: float, clipping: bool, lufs: float) -> float:
        snr_score = min(1.0, max(0.0, (snr_db + 5) / 40))
        lufs_score = 1.0 - min(1.0, abs(lufs - self.TARGET_LUFS) / 20)
        clip_penalty = 0.3 if clipping else 0.0
        return round(max(0.0, (snr_score * 0.6 + lufs_score * 0.4) - clip_penalty), 3)

    async def enhance(self, input_path: str, recording_id: str, job_id: str) -> EnhancementResult:
        loop = asyncio.get_event_loop()

        waveform, sr = self._load_audio(input_path)
        raw_clone = waveform.clone()
        clipping_input = self._detect_clipping(waveform)

        enhanced = await loop.run_in_executor(None, self._denoise_demucs, waveform, sr)
        enhanced = self._apply_eq(enhanced, self.TARGET_SR)
        enhanced = self._de_ess(enhanced, self.TARGET_SR)
        enhanced = self._strip_silence(enhanced, self.TARGET_SR)
        enhanced = self._remove_internal_silence(enhanced, self.TARGET_SR)
        enhanced = self._normalise_loudness(enhanced, self.TARGET_LUFS)

        raw_mono = self._to_mono(raw_clone)
        raw_at_target = self._resample(raw_mono, sr, self.TARGET_SR)
        snr = self._compute_snr(raw_at_target, enhanced)
        lufs = self._compute_lufs(enhanced)
        peak_db = 20 * np.log10(enhanced.abs().max().item() + 1e-8)
        quality_score = self._compute_quality_score(snr, clipping_input, lufs)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        torchaudio.save(out_path, enhanced, self.TARGET_SR)

        b2_key = f"{settings.B2_ENHANCED_PREFIX}{recording_id}/{job_id}.wav"
        enhanced_url = await loop.run_in_executor(None, storage.upload_file, out_path, b2_key)

        return EnhancementResult(
            b2_key=b2_key,
            enhanced_url=enhanced_url,
            local_path=out_path,
            quality_score=quality_score,
            snr_db=round(snr, 2),
            peak_db=round(peak_db, 2),
            lufs=round(lufs, 2),
            clipping_detected=clipping_input,
        )
