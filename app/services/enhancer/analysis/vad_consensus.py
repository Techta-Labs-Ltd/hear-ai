import asyncio
import logging
import threading
import warnings
from enum import Enum

import numpy as np
import torch
import torchaudio

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext

logger = logging.getLogger(__name__)

VAD_MODE_FAST = "fast"
VAD_MODE_ACCURATE = "accurate"


class VotingMode(Enum):
    AND = "and"
    OR = "or"
    MAJORITY = "majority"


class VADConsensus(ProcessingStage):
    name = "vad_consensus"

    SILERO_SR = 16000
    WEBRTC_FRAME_MS = 30

    def __init__(self, config):
        self._c = config
        self._silero = None
        self._get_ts = None
        self._webrtc = None
        self._lock = threading.Lock()
        self._resamplers = {}

    def load(self):
        silero_ok = False
        webrtc_ok = False

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                silero_model, silero_utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    trust_repo=True,
                    verbose=False,
                )
            if torch.cuda.is_available():
                silero_model.cuda()
            self._silero = silero_model
            self._get_ts = silero_utils[0]
            silero_ok = True
        except Exception as e:
            logger.warning("Silero VAD load failed: %s", e)

        try:
            import webrtcvad
            self._webrtc = webrtcvad.Vad(2)
            webrtcvad  # suppress unused
            webrtc_ok = True
        except Exception as e:
            logger.warning("WebRTC VAD load failed: %s", e)

        self._ready = silero_ok or webrtc_ok

    def _silero_vad(self, w_16k: torch.Tensor) -> torch.Tensor:
        device = w_16k.device
        wav = w_16k.squeeze(0)
        if device != next(self._silero.parameters()).device:
            self._silero.to(device)
        ts = self._get_ts(
            wav,
            self._silero,
            sampling_rate=self.SILERO_SR,
            threshold=self._c.vad_threshold,
            min_speech_duration_ms=100,
            min_silence_duration_ms=250,
        )
        mask = torch.zeros(wav.shape[0], dtype=torch.float32, device=device)
        for seg in ts:
            s = int(seg["start"])
            e = int(seg["end"])
            mask[s:e] = 1.0
        return mask

    def _webrtc_vad_sync(self, wav: np.ndarray) -> torch.Tensor:
        frame_size = int(self.SILERO_SR * self.WEBRTC_FRAME_MS / 1000)
        n_frames = len(wav) // frame_size
        mask = torch.zeros(len(wav), dtype=torch.float32)
        for i in range(n_frames):
            start = i * frame_size
            frame = wav[start:start + frame_size].tobytes()
            try:
                if self._webrtc.is_speech(frame, self.SILERO_SR):
                    mask[start:start + frame_size] = 1.0
            except Exception:
                pass
        return mask

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        if not self._ready:
            return ctx

        w = ctx.audio.data
        sr = ctx.audio.sample_rate
        n = w.shape[1]
        device = w.device

        if sr != self.SILERO_SR:
            w_16k = torchaudio.functional.resample(w, sr, self.SILERO_SR)
        else:
            w_16k = w

        loop = asyncio.get_running_loop()
        masks = []

        duration_s = w_16k.shape[1] / self.SILERO_SR
        use_silero = self._silero is not None and duration_s < 60

        if use_silero:
            try:
                mask = self._silero_vad(w_16k)
                masks.append(mask)
            except Exception as e:
                logger.warning("Silero VAD failed: %s", e)

        if self._webrtc is not None:
            try:
                wav_np = w_16k.squeeze(0).cpu().numpy().astype(np.int16)
                mask = await loop.run_in_executor(None, self._webrtc_vad_sync, wav_np)
                masks.append(mask)
            except Exception as e:
                logger.warning("WebRTC VAD failed: %s", e)

        if not masks:
            return ctx

        consensus = masks[0]
        for m in masks[1:]:
            mode = VotingMode(self._c.vad_consensus_mode)
            if mode == VotingMode.AND:
                consensus = torch.min(consensus, m)
            elif mode == VotingMode.OR:
                consensus = torch.max(consensus, m)
            else:
                consensus = ((consensus + m) > 1.0).float()

        if sr != self.SILERO_SR:
            consensus = torchaudio.functional.resample(
                consensus.view(1, -1), self.SILERO_SR, sr
            ).squeeze(0)
        if consensus.shape[0] > n:
            consensus = consensus[:n]
        elif consensus.shape[0] < n:
            pad = torch.zeros(n - consensus.shape[0], device=consensus.device)
            consensus = torch.cat([consensus, pad])

        fade_ms = getattr(self._c, "vad_fade_ms", 50)
        fade_samples = int(sr * fade_ms / 1000)
        fade_in = (1.0 - torch.cos(torch.linspace(0, torch.pi / 2, fade_samples, device=device))) * 0.5 + 0.5
        fade_in = fade_in / fade_in[-1]
        fade_out = (1.0 - torch.cos(torch.linspace(torch.pi / 2, torch.pi, fade_samples, device=device))) * 0.5 + 0.5

        transitions = torch.where(torch.diff(consensus) != 0)[0]
        for t_idx in transitions:
            t = t_idx.item()
            if consensus[t + 1] > consensus[t]:
                e = min(t + fade_samples, n)
                consensus[t:e] = fade_in[:e - t]
            else:
                s = max(0, t - fade_samples + 1)
                consensus[s:t + 1] = fade_out[:t - s + 1].flip(0)

        speech_ratio = consensus.mean().item()
        if speech_ratio < 0.01:
            logger.info(
                "VAD detected only %.1f%% speech — audio returned unmodified",
                speech_ratio * 100,
            )
            return ctx

        speech_gain = getattr(self._c, "vad_speech_gain", 1.0)
        silence_gain = getattr(self._c, "vad_silence_gain", 0.0)
        gain = speech_gain * consensus + silence_gain * (1.0 - consensus)
        ctx.audio.data = w * gain.unsqueeze(0).to(device)

        return ctx
