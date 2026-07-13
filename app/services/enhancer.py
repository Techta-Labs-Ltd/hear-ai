import asyncio
import json
import logging
import os
import time
import warnings

import numpy as np
import torch

from app.config import settings
from app.core.audio_utils import save_as_mp3
from app.core.hear_temp import drop_temp_standalone
from app.core.storage import get_storage

from app.services.enhancer_utils_old import (
    ContentMode,
    EnhancementResult,
    AudioIO,
    QualityMetrics,
    NoiseReducer,
    SpeechProcessor,
    SilenceProcessor,
    DynamicsProcessor,
    MossFormer2Enhancer,
    StemSeparator,
)

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

LOG_PATH = "/workspace/.cursor/debug-17b4a5.log"

def _dbg(step: str, w: torch.Tensor, sr: int, hypothesis: str):
    try:
        sig = w.squeeze(0).cpu().numpy().astype(np.float64)
        rms = float(np.sqrt(np.mean(sig ** 2)))
        peak = float(np.max(np.abs(sig)))
        dur = len(sig) / sr
        rms_db = 20 * np.log10(rms + 1e-10)
        peak_db = 20 * np.log10(peak + 1e-10)
        entry = json.dumps({
            "sessionId": "17b4a5",
            "timestamp": int(time.time() * 1000),
            "location": f"enhancer.py:{step}",
            "message": step,
            "data": {"duration_s": round(dur, 2), "samples": len(sig), "rms": round(rms, 6),
                     "rms_db": round(rms_db, 1), "peak": round(peak, 6), "peak_db": round(peak_db, 1), "sr": sr},
            "hypothesisId": hypothesis
        })
        with open(LOG_PATH, "a") as f:
            f.write(entry + "\n")
    except Exception:
        pass


class AudioEnhancer:
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mossformer = MossFormer2Enhancer()
        self._noise = NoiseReducer()
        self._speech = SpeechProcessor()
        self._silence = SilenceProcessor()
        self._dynamics = DynamicsProcessor(self._device)
        self._metrics = QualityMetrics()
        self._stem = StemSeparator(self._device)
        self._gpu_lock = asyncio.Lock()
        self._loaded = False

    def load(self):
        self._mossformer.load()
        self._stem.load(settings.DEMUCS_MODEL)
        self._silence.load()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def enhance(
        self,
        input_path: str,
        track_id: str,
        job_id: str,
        mode: ContentMode = ContentMode.AUTO,
        ai_job_id: str | None = None,
        ai_run_id: str | None = None,
        word_timestamps: list[dict] | None = None,
    ) -> EnhancementResult:
        async with self._gpu_lock:
            loop = asyncio.get_running_loop()
            out_path = None

            try:
                waveform, sr = AudioIO.load(input_path)
                raw_clone = waveform.clone()
                clipping_input = AudioIO.detect_clipping(waveform)

                mono = AudioIO.to_mono(waveform)
                enhanced = AudioIO.resample(mono, sr, AudioIO.TARGET_SR).to(self._device)

                if mode == ContentMode.AUTO:
                    mode = ContentMode.SPEECH

                _dbg("00_input_after_resample", enhanced, AudioIO.TARGET_SR, "H1")
                _dbg("00_input_after_resample", enhanced, AudioIO.TARGET_SR, "H2")

                raw_at_target = AudioIO.resample(AudioIO.to_mono(raw_clone), sr, AudioIO.TARGET_SR)

                if mode == ContentMode.SPEECH:

                    stems = await loop.run_in_executor(
                        None, self._stem.separate, enhanced, AudioIO.TARGET_SR
                    )
                    enhanced = stems['vocals']
                    _dbg("00_5_after_demucs", enhanced, AudioIO.TARGET_SR, "H1")


                enhanced = await loop.run_in_executor(
                    None, self._mossformer.enhance, enhanced, AudioIO.TARGET_SR
                )
                _dbg("01_after_mossformer2", enhanced, AudioIO.TARGET_SR, "H1")
                _dbg("01_after_mossformer2", enhanced, AudioIO.TARGET_SR, "H5")

                if mode == ContentMode.SPEECH:
                    if word_timestamps:
                        enhanced = await loop.run_in_executor(
                            None, self._silence.apply_transcript_gate, enhanced, word_timestamps, AudioIO.TARGET_SR
                        )
                        _dbg("02_after_transcript_gate", enhanced, AudioIO.TARGET_SR, "H2")

                    enhanced = await loop.run_in_executor(
                        None, self._noise.spectral_suppress, enhanced, AudioIO.TARGET_SR
                    )
                    _dbg("02_after_spectral_suppress", enhanced, AudioIO.TARGET_SR, "H3")


                    enhanced = await loop.run_in_executor(
                        None, self._silence.apply_vad_gate, enhanced, AudioIO.TARGET_SR
                    )
                    _dbg("02_5_after_vad_gate", enhanced, AudioIO.TARGET_SR, "H1")


                    enhanced = await loop.run_in_executor(
                        None, self._noise.noise_gate, enhanced, AudioIO.TARGET_SR
                    )
                    _dbg("03_after_noise_gate", enhanced, AudioIO.TARGET_SR, "H4")


                    enhanced = await loop.run_in_executor(
                        None, self._silence.detect_and_strip_silence, enhanced, AudioIO.TARGET_SR
                    )
                    _dbg("04_after_silence_strip", enhanced, AudioIO.TARGET_SR, "H1")


                    enhanced = await loop.run_in_executor(
                        None, self._speech.apply_eq_speech, enhanced, AudioIO.TARGET_SR
                    )
                    _dbg("05_after_eq", enhanced, AudioIO.TARGET_SR, "H1")

                    enhanced = await loop.run_in_executor(
                        None, self._speech.apply_deesser, enhanced, AudioIO.TARGET_SR
                    )
                    _dbg("06_after_deesser", enhanced, AudioIO.TARGET_SR, "H1")

                    enhanced = await loop.run_in_executor(
                        None, self._dynamics.compress, enhanced, AudioIO.TARGET_SR, mode
                    )
                    _dbg("07_after_compress_and_level", enhanced, AudioIO.TARGET_SR, "H1")
                else:
                    enhanced = await loop.run_in_executor(
                        None, self._speech.apply_eq_music, enhanced, AudioIO.TARGET_SR
                    )
                    enhanced = await loop.run_in_executor(
                        None, self._dynamics.compress, enhanced, AudioIO.TARGET_SR, mode
                    )

                snr = self._metrics.compute_snr(raw_at_target, enhanced)


                enhanced = await loop.run_in_executor(None, self._dynamics.normalise_lufs, enhanced)
                _dbg("08_after_lufs_normalise", enhanced, AudioIO.TARGET_SR, "H2")

                enhanced = await loop.run_in_executor(
                    None, self._dynamics.lookahead_limit, enhanced, AudioIO.TARGET_SR
                )
                _dbg("09_after_lookahead_limit", enhanced, AudioIO.TARGET_SR, "H1")


                enhanced = await loop.run_in_executor(
                    None, self._noise.noise_gate, enhanced, AudioIO.TARGET_SR,
                    NoiseReducer.POST_GATE_THRESHOLD_DB,
                )
                _dbg("10_after_final_noise_gate", enhanced, AudioIO.TARGET_SR, "H4")

                lufs = self._metrics.compute_lufs(enhanced)
                peak_db = 20 * np.log10(enhanced.abs().max().item() + 1e-8)
                quality_score = self._metrics.compute_quality_score(snr, clipping_input, lufs)

                out_path = save_as_mp3(
                    enhanced.cpu(),
                    AudioIO.TARGET_SR,
                    job_id=ai_job_id,
                    run_id=ai_run_id,
                    track_id=track_id,
                    purpose="enhance_output",
                )

                b2_key = f"{settings.B2_ENHANCED_PREFIX}{track_id}/{job_id}.mp3"
                enhanced_url = await loop.run_in_executor(None, get_storage().upload_file, out_path, b2_key)

                return EnhancementResult(
                    b2_key=b2_key,
                    enhanced_url=enhanced_url,
                    local_path=out_path,
                    quality_score=quality_score,
                    snr_db=round(snr, 2),
                    peak_db=round(peak_db, 2),
                    lufs=round(lufs, 2),
                    clipping_detected=clipping_input,
                    mode_used=mode.value,
                )

            except Exception:
                if out_path:
                    try:
                        drop_temp_standalone(out_path)
                    except Exception:
                        if os.path.exists(out_path):
                            try:
                                os.unlink(out_path)
                            except OSError:
                                pass
                raise
