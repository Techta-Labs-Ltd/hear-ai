import logging
import os
import threading

import numpy as np
import torch
import torchaudio
import librosa

from clearvoice import ClearVoice

logger = logging.getLogger(__name__)

MAX_CHUNK_S = 4

class MossFormer2Enhancer:
    SR = 48_000

    def __init__(self):
        self._cv = None
        self._lock = threading.Lock()

    def load(self):
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        prev_cwd = os.getcwd()
        os.chdir(project_root)
        try:
            self._cv = ClearVoice(
                task="speech_enhancement",
                model_names=["MossFormer2_SE_48K"],
            )
        finally:
            os.chdir(prev_cwd)

    def _enhance_chunk(self, audio_np: np.ndarray) -> np.ndarray:
        """Run ClearVoice on a single chunk of audio (shape: [1, samples])."""
        output_np = self._cv(audio_np, False)
        if isinstance(output_np, np.ndarray):
            return output_np[0, :].astype(np.float32)
        elif isinstance(output_np, dict):

            first_key = list(output_np.keys())[0]
            return output_np[first_key][0, :].astype(np.float32)
        raise RuntimeError(f"Unexpected output type: {type(output_np)}")

    @torch.no_grad()
    def enhance(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        original_len = w.shape[1]
        original_sr = sr
        try:
            if original_sr != self.SR:
                w = torchaudio.functional.resample(w, original_sr, self.SR)

            audio_np = w.squeeze(0).cpu().numpy().astype(np.float32)


            peak_val = np.max(np.abs(audio_np))
            if peak_val > 1e-6:
                norm_factor = 0.9 / peak_val
                audio_np = audio_np * norm_factor
            else:
                norm_factor = 1.0

            resampled_len = len(audio_np)

            with self._lock:
                chunk_samples = MAX_CHUNK_S * self.SR

                if resampled_len <= chunk_samples:
                    audio_in = audio_np.reshape(1, -1)
                    enhanced = self._enhance_chunk(audio_in)
                else:
                    overlap_samples = int(self.SR * 0.5)
                    chunks = []
                    pos = 0
                    while pos < resampled_len:
                        end = min(pos + chunk_samples, resampled_len)
                        chunk = audio_np[pos:end]
                        chunk_in = chunk.reshape(1, -1)

                        try:
                            chunk_out = self._enhance_chunk(chunk_in)
                        except Exception as e:
                            logger.warning("Chunk failed at pos %d: %s, using original", pos, e)
                            chunk_out = chunk.astype(np.float32)

                        if chunks and overlap_samples > 0 and len(chunk_out) > overlap_samples:
                            fade_in = np.linspace(0, 1, overlap_samples, dtype=np.float32)
                            prev_tail = chunks[-1][-overlap_samples:]
                            cur_head = chunk_out[:overlap_samples]
                            blended = prev_tail * (1 - fade_in) + cur_head * fade_in
                            chunks[-1] = chunks[-1][:-overlap_samples]
                            chunks.append(blended)
                            chunks.append(chunk_out[overlap_samples:])
                        else:
                            chunks.append(chunk_out)

                        pos += chunk_samples - overlap_samples

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    enhanced = np.concatenate(chunks)

            enhanced = torch.from_numpy(enhanced)
            if original_sr != self.SR:
                enhanced = torchaudio.functional.resample(enhanced, self.SR, original_sr)

            enhanced = enhanced[:original_len]
            if len(enhanced) < original_len:
                pad = torch.zeros(original_len - len(enhanced), dtype=torch.float32)
                enhanced = torch.cat([enhanced, pad])

            enhanced = enhanced / norm_factor

            result = enhanced.unsqueeze(0)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result.to(w.device)
        except Exception as e:
            logger.error("MossFormer2 enhancement failed: %s", e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return w
