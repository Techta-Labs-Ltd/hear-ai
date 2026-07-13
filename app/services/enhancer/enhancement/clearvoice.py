import io
import logging
import threading

import numpy as np
import torch
import torchaudio
import soundfile as sf

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext
from app.services.triton_client import get_triton_client

logger = logging.getLogger(__name__)


class ClearVoiceEnhancer(ProcessingStage):
    name = "clearvoice"

    SR = 48_000

    def __init__(self, config):
        self._c = config
        self._lock = threading.Lock()
        self._ready = True

    def load(self):
        pass

    @staticmethod
    def _cosine_fade(n: int, device: torch.device) -> torch.Tensor:
        t = torch.linspace(0, 1, n, device=device, dtype=torch.float32)
        return 0.5 * (1.0 - torch.cos(t * torch.pi))

    def _validate_output(self, enhanced: np.ndarray, original: np.ndarray, label: str) -> np.ndarray:
        if not np.all(np.isfinite(enhanced)):
            logger.warning("%s: NaN/Inf in output, using original", label)
            return original.astype(np.float32)
        if np.max(np.abs(enhanced)) < 1e-6:
            logger.warning("%s: near-silent output, using original", label)
            return original.astype(np.float32)
        orig_rms = np.sqrt(np.mean(original ** 2))
        proc_rms = np.sqrt(np.mean(enhanced ** 2))
        if orig_rms > 1e-6 and proc_rms > orig_rms * 10:
            logger.warning("%s: excessive gain (%.1fx), scaling down", label, proc_rms / orig_rms)
            enhanced = enhanced * (orig_rms / proc_rms)
        return enhanced

    def _match_level(self, processed: np.ndarray, original: np.ndarray) -> np.ndarray:
        orig_rms = np.sqrt(np.mean(original ** 2))
        proc_rms = np.sqrt(np.mean(processed ** 2))
        if orig_rms > 1e-6 and proc_rms > 1e-6:
            scale = orig_rms / proc_rms
            scale = max(0.5, min(scale, 2.0))
            processed = processed * scale
        return processed

    def _run_chunk(self, audio_np: np.ndarray) -> np.ndarray:
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)

        buf = io.BytesIO()
        sf.write(buf, audio_np.squeeze(0).astype(np.float32), self.SR, format="WAV")
        wav_bytes = buf.getvalue()

        try:
            with self._lock:
                result_bytes = get_triton_client().enhance_audio_sync(
                    "mossformer2", wav_bytes, sample_rate=self.SR
                )
            if not result_bytes or len(result_bytes) < 44:
                return audio_np.squeeze(0).astype(np.float32)
            output_np, _ = sf.read(io.BytesIO(result_bytes))
        except Exception as e:
            logger.warning("ClearVoice Triton call failed: %s", e)
            return audio_np.squeeze(0).astype(np.float32)

        return output_np.astype(np.float32)

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        if not self._ready:
            return ctx

        wet_dry = getattr(self._c, "clearvoice_wet_dry", 0.30)
        chunk_seconds = getattr(self._c, "clearvoice_chunk_seconds", 30)
        overlap_seconds = getattr(self._c, "clearvoice_overlap_seconds", 3.0)
        wet_dry = max(0.0, min(wet_dry, 1.0))

        try:
            w = ctx.audio.data
            sr = ctx.audio.sample_rate
            original_len = w.shape[1]
            original_device = w.device

            w_48k = torchaudio.functional.resample(w, sr, self.SR) if sr != self.SR else w
            audio_np = w_48k.squeeze(0).cpu().numpy().astype(np.float32)
            peak_val = np.max(np.abs(audio_np))
            if peak_val > 0.99:
                audio_np = audio_np * (0.99 / peak_val)

            chunk_samples = chunk_seconds * self.SR
            overlap_samples = int(self.SR * overlap_seconds)

            with self._lock:
                if len(audio_np) <= chunk_samples:
                    audio_in = audio_np.reshape(1, -1)
                    enhanced = self._run_chunk(audio_in)
                    enhanced = self._validate_output(enhanced, audio_np, "clearvoice_short")
                    enhanced = self._match_level(enhanced, audio_np)
                else:
                    chunks = []
                    pos = 0
                    while pos < len(audio_np):
                        end = min(pos + chunk_samples, len(audio_np))
                        chunk = audio_np[pos:end]
                        chunk_in = chunk.reshape(1, -1)
                        try:
                            chunk_out = self._run_chunk(chunk_in)
                            chunk_out = self._validate_output(chunk_out, chunk, "clearvoice_chunk")
                            chunk_out = self._match_level(chunk_out, chunk)
                        except Exception:
                            chunk_out = chunk.astype(np.float32)
                        if chunks and overlap_samples > 0 and len(chunk_out) > overlap_samples:
                            fade_len = min(overlap_samples, len(chunks[-1]), len(chunk_out))
                            fade = np.linspace(0, 1, fade_len).astype(np.float32)
                            cos_fade = 0.5 * (1.0 - np.cos(fade * np.pi))
                            prev_tail = chunks[-1][-fade_len:]
                            cur_head = chunk_out[:fade_len]
                            blended = prev_tail * (1.0 - cos_fade) + cur_head * cos_fade
                            chunks[-1] = chunks[-1][:-fade_len]
                            chunks.append(blended)
                            chunks.append(chunk_out[fade_len:])
                        else:
                            chunks.append(chunk_out)
                        pos += chunk_samples - overlap_samples
                    enhanced = np.concatenate(chunks)

            result = torch.from_numpy(enhanced.astype(np.float32))
            if sr != self.SR:
                result = torchaudio.functional.resample(result, self.SR, sr)
            result = result[:original_len]
            if len(result) < original_len:
                pad = torch.zeros(original_len - len(result), dtype=torch.float32)
                result = torch.cat([result, pad])
            result = result.unsqueeze(0)

            original_for_blend = w[:, :result.shape[1]]
            if original_for_blend.shape[1] < result.shape[1]:
                pad = torch.zeros(
                    1, result.shape[1] - original_for_blend.shape[1],
                    dtype=result.dtype, device=w.device,
                )
                original_for_blend = torch.cat([original_for_blend, pad], dim=1)
            elif result.shape[1] < original_for_blend.shape[1]:
                pad = torch.zeros(
                    1, original_for_blend.shape[1] - result.shape[1],
                    dtype=result.dtype, device=w.device,
                )
                result = torch.cat([result, pad], dim=1)

            blended = wet_dry * result.to(original_device) + (1.0 - wet_dry) * original_for_blend.to(original_device)
            ctx.audio.data = blended[:, :original_len]
        except Exception as e:
            logger.warning("ClearVoice process failed: %s", e)
        return ctx
