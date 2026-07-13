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


class DeepFilterNetEnhancer(ProcessingStage):
    name = "deepfilternet"

    SR = 48000

    def __init__(self, config):
        self._c = config
        self._lock = threading.Lock()
        self._ready = True

    def load(self):
        pass

    def _validate_output(self, result: torch.Tensor, original: torch.Tensor, label: str) -> torch.Tensor:
        if result.dim() == 1:
            result = result.unsqueeze(0)
        if not torch.isfinite(result).all():
            logger.warning("%s: NaN/Inf in output, using original", label)
            return original
        if result.abs().max().item() < 1e-6:
            logger.warning("%s: near-silent output, using original", label)
            return original
        result_rms = result.pow(2).mean().sqrt().item()
        original_rms = original.pow(2).mean().sqrt().item()
        if original_rms > 1e-6 and result_rms > original_rms * 10:
            logger.warning("%s: excessive gain (%.1fx), scaling down", label, result_rms / original_rms)
            result = result * (original_rms / result_rms)
        return result

    def _match_level(self, processed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        orig_rms = original.pow(2).mean().sqrt().item()
        proc_rms = processed.pow(2).mean().sqrt().item()
        if orig_rms > 1e-6 and proc_rms > 1e-6:
            scale = orig_rms / proc_rms
            scale = max(0.5, min(scale, 2.0))
            processed = processed * scale
        return processed

    def _process_chunk(self, w_chunk: torch.Tensor, sr: int, original_len: int) -> torch.Tensor:
        w_48k = torchaudio.functional.resample(w_chunk, sr, self.SR) if sr != self.SR else w_chunk
        w_cpu = w_48k.cpu()
        atten_lim = getattr(self._c, "deepfilter_atten_lim_db", 12.0)

        buf = io.BytesIO()
        sf.write(buf, w_cpu.squeeze(0).numpy().astype(np.float32), self.SR, format="WAV")
        wav_bytes = buf.getvalue()

        try:
            with self._lock:
                result_bytes = get_triton_client().enhance_audio_sync(
                    "deepfilternet", wav_bytes, sample_rate=self.SR, atten_lim_db=atten_lim
                )
            if not result_bytes or len(result_bytes) < 44:
                return w_chunk[:, :original_len]
            result_np, _ = sf.read(io.BytesIO(result_bytes))
            result = torch.from_numpy(result_np.astype(np.float32))
        except Exception as e:
            logger.warning("DeepFilterNet Triton call failed: %s", e)
            return w_chunk[:, :original_len]

        if result.dim() == 1:
            result = result.unsqueeze(0)
        if sr != self.SR:
            result = torchaudio.functional.resample(result, self.SR, sr)
        result = result[:, :original_len]
        if result.shape[1] < original_len:
            pad = torch.zeros(1, original_len - result.shape[1])
            result = torch.cat([result, pad], dim=1)
        return result

    @staticmethod
    def _cosine_fade(n: int, device: torch.device) -> torch.Tensor:
        t = torch.linspace(0, 1, n, device=device, dtype=torch.float32)
        return 0.5 * (1.0 - torch.cos(t * torch.pi))

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        if not self._ready:
            return ctx

        wet_dry = getattr(self._c, "deepfilter_wet_dry", 0.70)
        chunk_seconds = getattr(self._c, "deepfilter_chunk_seconds", 30)
        overlap_seconds = getattr(self._c, "deepfilter_overlap_seconds", 3.0)

        try:
            w = ctx.audio.data
            sr = ctx.audio.sample_rate
            n = w.shape[1]
            original_device = w.device
            chunk_samples = int(chunk_seconds * sr)
            overlap_samples = int(overlap_seconds * sr)

            wet_dry = max(0.0, min(wet_dry, 1.0))

            if n <= chunk_samples:
                result = self._process_chunk(w, sr, n)
                result = self._validate_output(result, w, "deepfilternet_short")
                result = self._match_level(result, w)
                blended = wet_dry * result.to(original_device) + (1.0 - wet_dry) * w
                ctx.audio.data = blended
                return ctx

            chunks = []
            pos = 0
            while pos < n:
                end = min(pos + chunk_samples, n)
                chunk = w[:, pos:end]
                chunk_len = chunk.shape[1]
                try:
                    processed = self._process_chunk(chunk, sr, chunk_len)
                    processed = self._validate_output(processed, chunk, "deepfilternet_chunk")
                    processed = self._match_level(processed, chunk)
                except Exception:
                    processed = chunk
                chunks.append(processed.to(original_device))
                pos += chunk_samples - overlap_samples

            if len(chunks) == 1:
                blended_data = wet_dry * chunks[0] + (1.0 - wet_dry) * w[:, :chunks[0].shape[1]]
                ctx.audio.data = blended_data
                return ctx

            fade = self._cosine_fade(overlap_samples, original_device).view(1, -1)
            blended_parts = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    blended_parts.append(chunk)
                else:
                    prev = blended_parts[-1]
                    if prev.shape[1] > overlap_samples and chunk.shape[1] > overlap_samples:
                        overlap_prev = prev[:, -overlap_samples:]
                        overlap_cur = chunk[:, :overlap_samples]
                        crossfade = overlap_prev * (1.0 - fade) + overlap_cur * fade
                        blended_parts[-1] = prev[:, :-overlap_samples]
                        blended_parts.append(crossfade)
                        blended_parts.append(chunk[:, overlap_samples:])
                    else:
                        blended_parts.append(chunk)

            if blended_parts:
                blended_data = torch.cat(blended_parts, dim=1)[:, :n]
            else:
                blended_data = w

            blend_with_original = wet_dry * blended_data + (1.0 - wet_dry) * w[:, :blended_data.shape[1]]
            if blend_with_original.shape[1] < n:
                pad = torch.zeros(1, n - blend_with_original.shape[1], device=original_device, dtype=w.dtype)
                blend_with_original = torch.cat([blend_with_original, pad], dim=1)
            ctx.audio.data = blend_with_original[:, :n]

        except Exception as e:
            logger.warning("DeepFilterNet process failed: %s", e)
        return ctx
