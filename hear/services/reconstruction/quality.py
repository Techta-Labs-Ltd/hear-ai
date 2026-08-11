import logging
from dataclasses import dataclass, field

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio

from hear.services.reconstruction.audio_buffer import AudioBuffer
from hear.services.reconstruction.dnsmos import DNSMOSScorer

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    dnsmos_ovr: float = 0.0
    loudness_match_db: float = 0.0
    duration_delta_ms: float = 0.0
    clipping_detected: bool = False
    passed: bool = False
    details: dict = field(default_factory=dict)


class RegenerationQualityAssessor:
    MIN_DNSMOS = 3.0
    MAX_LOUDNESS_DIFF_DB = 6.0
    MAX_DURATION_DELTA_MS = 500
    CLIPPING_THRESHOLD = 0.99

    def __init__(self):
        self._scorer = None

    def _ensure_scorer(self) -> bool:
        if self._scorer is None:
            self._scorer = DNSMOSScorer()
        return self._scorer.load()

    def assess(
        self,
        tts_waveform: torch.Tensor,
        ref_waveform: torch.Tensor,
        sr: int,
    ) -> QualityReport:
        report = QualityReport()

        try:
            if self._ensure_scorer():
                audio_16k = (
                    torchaudio.functional.resample(tts_waveform, sr, 16000)
                    if sr != 16000 else tts_waveform
                )
                buf = AudioBuffer(data=audio_16k, sample_rate=16000)
                report.dnsmos_ovr = self._scorer.score(buf)
        except Exception as e:
            logger.warning("DNSMOS assessment failed: %s", e)

        try:
            if tts_waveform.shape[1] < int(sr * 0.2) or ref_waveform.shape[1] < int(sr * 0.2):
                report.loudness_match_db = 0.0
            else:
                meter = pyln.Meter(sr)
                tts_np = tts_waveform.squeeze(0).cpu().numpy().astype(np.float64)
                ref_np = ref_waveform.squeeze(0).cpu().numpy().astype(np.float64)
                tts_lufs = meter.integrated_loudness(tts_np)
                ref_lufs = meter.integrated_loudness(ref_np)
                if np.isfinite(tts_lufs) and np.isfinite(ref_lufs):
                    report.loudness_match_db = abs(ref_lufs - tts_lufs)
        except Exception as e:
            logger.warning("Loudness assessment failed: %s", e)

        tts_dur = tts_waveform.shape[1] / sr
        ref_dur = ref_waveform.shape[1] / sr
        report.duration_delta_ms = abs(tts_dur - ref_dur) * 1000.0

        peak = tts_waveform.abs().max().item()
        report.clipping_detected = peak > self.CLIPPING_THRESHOLD

        report.details = {
            "tts_duration_s": round(tts_dur, 3),
            "ref_duration_s": round(ref_dur, 3),
            "peak": round(peak, 4),
        }

        report.passed = (
            report.dnsmos_ovr >= self.MIN_DNSMOS
            and report.loudness_match_db <= self.MAX_LOUDNESS_DIFF_DB
            and report.duration_delta_ms <= self.MAX_DURATION_DELTA_MS
            and not report.clipping_detected
        )

        logger.info(
            "QualityReport: dnsmos=%.2f loudness=%.1fdB dur_delta=%.0fms clip=%s passed=%s",
            report.dnsmos_ovr, report.loudness_match_db,
            report.duration_delta_ms, report.clipping_detected, report.passed,
        )

        return report
