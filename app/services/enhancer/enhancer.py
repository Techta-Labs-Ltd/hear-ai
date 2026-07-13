import asyncio
import logging
import math
import os
import time
import traceback
from typing import Callable

import numpy as np
import torch
import torchaudio

from app.config import settings
from app.core.audio_utils import save_as_mp3
from app.core.hear_temp import drop_temp_standalone
from app.core.storage import get_storage

from app.services.enhancer.config import EngineConfig, _apply_preset
from app.services.enhancer.models import ProcessingContext, AudioBuffer
from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.shaping.eq import DCOffsetRemover, SpeechEQ
from app.services.enhancer.shaping.deesser import DeEsser
from app.services.enhancer.dynamics.normalizer import PeakNormalizer, LoudnessNormalizer
from app.services.enhancer.dynamics.limiter import LookaheadLimiter
from app.services.enhancer.enhancement.clearvoice import ClearVoiceEnhancer
from app.services.enhancer.enhancement.deepfilternet import DeepFilterNetEnhancer
from app.services.enhancer.analysis.vad import VADGate
from app.services.enhancer.quality.dnsmos_scorer import DNSMOSScorer
from app.services.enhancer.quality.metrics import QualityMetrics
from app.services.enhancer.io.audio import AudioIO

logger = logging.getLogger(__name__)

DNSMOS_TOLERANCE = 0.05

HEARTBEAT_INTERVAL = 15.0


class OutputFade(ProcessingStage):
    name = "output_fade"

    def __init__(self, config):
        self._c = config

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        try:
            w = ctx.audio.data
            sr = ctx.audio.sample_rate
            fade_ms = getattr(self._c, "output_fade_ms", 5)
            fade_samples = max(1, int(sr * fade_ms / 1000))
            if w.shape[1] > fade_samples * 2:
                fade_in = 0.5 * (1.0 - torch.cos(torch.linspace(0, math.pi, fade_samples, device=w.device, dtype=w.dtype)))
                fade_out = fade_in.flip(0)
                w = w.clone()
                w[:, :fade_samples] *= fade_in.unsqueeze(0)
                w[:, -fade_samples:] *= fade_out.unsqueeze(0)
                ctx.audio.data = w
        except Exception:
            pass
        return ctx


class EnhancementResult:
    def __init__(self, audio: AudioBuffer, quality_score: float,
                 stage_times: dict, dnsmos_scores: list, events: list):
        self.audio = audio
        self.quality_score = quality_score
        self.stage_times = stage_times
        self.dnsmos_scores = dnsmos_scores
        self.events = events


class Enhancer:
    def __init__(self, config: EngineConfig | None = None):
        self._config = _apply_preset(config or EngineConfig())
        self._stages: list[ProcessingStage] = []
        self._dnsmos_stages: set[str] = set()
        self._dnsmos = DNSMOSScorer()
        self._gpu_lock = asyncio.Lock()
        self._loaded = False

    def load(self):
        self._stages = self._build_stages()
        for stage in self._stages:
            stage.load()
        self._dnsmos.load()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _build_stages(self) -> list[ProcessingStage]:
        c = self._config
        stages = [
            DCOffsetRemover(),
            PeakNormalizer(c),
            DeepFilterNetEnhancer(c),
            ClearVoiceEnhancer(c),
            SpeechEQ(c),
            DeEsser(c),
            VADGate(c),
            LoudnessNormalizer(c),
            LookaheadLimiter(c),
            OutputFade(c),
        ]
        self._dnsmos_stages = {"deepfilternet", "clearvoice"}
        return stages

    async def _score_audio(self, ctx: ProcessingContext) -> float:
        return await asyncio.get_running_loop().run_in_executor(
            None, self._dnsmos.score, ctx.audio
        )

    async def enhance(
        self,
        input_path: str,
        track_id: str = "track",
        job_id: str = "job",
        mode: str | None = None,
        ai_job_id: str | None = None,
        ai_run_id: str | None = None,
        on_stage: Callable[[str], None] | None = None,
        on_progress: Callable[[str, int, int, float], None] | None = None,
    ) -> dict:
        out_path = None
        try:
            wav, sr = torchaudio.load(input_path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            target_sr = self._config.target_sr
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr
            if torch.cuda.is_available():
                wav = wav.cuda()

            audio = AudioBuffer(data=wav, sample_rate=sr)
            ctx = ProcessingContext(audio=audio, raw=audio.clone())
            dnsmos_scores = []
            prev_audio = ctx.audio.data.clone()

            ready_stages = [s for s in self._stages if s._ready]
            total_stages = len(ready_stages)

            last_heartbeat = time.time()

            async def _heartbeat_loop(stop_event: asyncio.Event):
                nonlocal last_heartbeat
                while not stop_event.is_set():
                    await asyncio.sleep(HEARTBEAT_INTERVAL)
                    if stop_event.is_set():
                        break
                    now = time.time()
                    if on_progress and now - last_heartbeat >= HEARTBEAT_INTERVAL:
                        last_heartbeat = now
                        try:
                            on_progress("heartbeat", 0, total_stages, 0.0)
                        except Exception:
                            pass

            stop_heartbeat = asyncio.Event()
            hb_task = asyncio.create_task(_heartbeat_loop(stop_heartbeat))

            try:
                for stage_idx, stage in enumerate(ready_stages):
                    t0 = time.time()
                    before = None
                    gated = stage.name in self._dnsmos_stages

                    stage_pct = round(((stage_idx + 1) / total_stages) * 100.0, 1)

                    if on_progress:
                        try:
                            on_progress(stage.name, stage_idx, total_stages, stage_pct)
                        except Exception:
                            pass
                    if on_stage:
                        try:
                            on_stage(stage.name)
                        except Exception:
                            pass

                    last_heartbeat = time.time()

                    if gated:
                        before = await self._score_audio(ctx)

                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        async with self._gpu_lock:
                            ctx = await stage.process(ctx)
                    except Exception as e:
                        logger.error("Stage %s failed: %s\n%s", stage.name, e, traceback.format_exc())

                    elapsed = time.time() - t0
                    ctx.stage_times[stage.name] = elapsed
                    logger.info("Stage %s done in %.2fs", stage.name, elapsed)

                    if gated and before is not None:
                        after = await self._score_audio(ctx)
                        dnsmos_scores.append({
                            "stage": stage.name,
                            "before": round(before, 4),
                            "after": round(after, 4),
                        })
                        if after < before - DNSMOS_TOLERANCE:
                            logger.warning(
                                "DNSMOS regressed %.4f->%.4f for %s, reverting",
                                before, after, stage.name
                            )
                            ctx.audio.data = prev_audio.clone()
                            ctx.stage_times[stage.name] = time.time() - t0
                        else:
                            prev_audio = ctx.audio.data.clone()
                    else:
                        prev_audio = ctx.audio.data.clone()
            finally:
                stop_heartbeat.set()
                await hb_task

            final_score = await self._score_audio(ctx)
            dnsmos_scores.append({
                "stage": "final",
                "before": None,
                "after": round(final_score, 4),
            })

            wav = ctx.audio.data.cpu()
            sr = ctx.audio.sample_rate

            out_path = save_as_mp3(
                wav,
                sr,
                job_id=ai_job_id,
                run_id=ai_run_id,
                track_id=track_id,
                purpose="enhance_output",
            )

            b2_key = f"{settings.B2_ENHANCED_PREFIX}{track_id}/{job_id}.mp3"
            loop = asyncio.get_running_loop()
            enhanced_url = await loop.run_in_executor(None, get_storage().upload_file, out_path, b2_key)

            snr = QualityMetrics.compute_snr(ctx.raw, ctx.audio)
            lufs = QualityMetrics.compute_lufs(ctx.audio)
            clipping = AudioIO.detect_clipping(ctx.audio)
            quality_score = QualityMetrics.compute_quality_score(snr, clipping, lufs)
            peak_db = 20 * math.log10(ctx.audio.peak + 1e-8)

            return {
                "b2_key": b2_key,
                "enhanced_url": enhanced_url,
                "local_path": out_path,
                "quality_score": quality_score,
                "snr_db": round(snr, 2),
                "peak_db": round(peak_db, 2),
                "lufs": round(lufs, 2),
                "clipping_detected": clipping,
                "mode_used": "speech",
                "stage_times": dict(ctx.stage_times),
                "dnsmos_scores": dnsmos_scores,
                "events": [{"start_sample": e.start_sample, "end_sample": e.end_sample, "confidence": e.confidence} for e in ctx.events],
            }
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