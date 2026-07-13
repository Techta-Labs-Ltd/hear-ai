import logging

import numpy as np
import torch
import torchaudio

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext

logger = logging.getLogger(__name__)


def _design_lr4(cross_hz: float, sr: int, lowpass: bool) -> tuple:
    from scipy.signal import butter
    order = 4
    nyq = sr / 2
    norm = cross_hz / nyq
    if norm >= 1.0:
        return None
    b, a = butter(order, norm, btype="low" if lowpass else "high", output="ba")
    return torch.tensor(b, dtype=torch.float64), torch.tensor(a, dtype=torch.float64)


def _apply_iir(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    sig = x.squeeze(0).cpu().double().numpy()
    from scipy.signal import filtfilt
    out = filtfilt(b.numpy(), a.numpy(), sig)
    return torch.from_numpy(out.astype(np.float32)).unsqueeze(0).to(x.device)


class MultibandCompressor(ProcessingStage):
    name = "multiband_compressor"

    def __init__(self, config):
        self._c = config
        self._bands = [
            {"name": "low",  "cross": 300.0},
            {"name": "mid",  "cross": 3000.0},
            {"name": "high", "cross": None},
        ]

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        try:
            w = ctx.audio.data
            sr = ctx.audio.sample_rate
            original_device = w.device
            sig = w.cpu().double()

            lp1 = _design_lr4(self._bands[0]["cross"], sr, lowpass=True)
            hp1 = _design_lr4(self._bands[0]["cross"], sr, lowpass=False)
            lp2 = _design_lr4(self._bands[1]["cross"], sr, lowpass=True)
            hp2 = _design_lr4(self._bands[1]["cross"], sr, lowpass=False)

            if lp1 is None or hp1 is None:
                return ctx

            band1 = _apply_iir(sig, lp1[0], lp1[1])
            band2 = _apply_iir(sig, *((hp1[0], hp1[1]) if hp1 else (None, None)))
            if band2 is not None and lp2 is not None:
                band2 = _apply_iir(band2, lp2[0], lp2[1])
            band3 = sig - band1 - (band2 if band2 is not None else sig * 0)

            if band2 is None:
                band2 = sig * 0
                band3 = sig * 0

            bands_processed = []
            for idx, band in enumerate([band1, band2, band3]):
                prefix = f"mb_band{idx + 1}_"
                threshold = getattr(self._c, f"{prefix}threshold", -20.0)
                ratio = getattr(self._c, f"{prefix}ratio", 3.0)
                attack_ms = getattr(self._c, f"{prefix}attack_ms", 5.0)
                release_ms = getattr(self._c, f"{prefix}release_ms", 100.0)
                makeup = getattr(self._c, f"{prefix}makeup_db", 2.0)

                abs_sig = band.abs().squeeze(0).numpy().astype(np.float64)
                attack_coef = np.exp(-1.0 / (sr * attack_ms / 1000.0))
                release_coef = np.exp(-1.0 / (sr * release_ms / 1000.0))

                env = np.zeros_like(abs_sig)
                prev = 0.0
                for i in range(len(abs_sig)):
                    c = attack_coef if abs_sig[i] > prev else release_coef
                    prev = c * prev + (1.0 - c) * abs_sig[i]
                    env[i] = prev

                env_db = 20 * np.log10(env + 1e-10)
                knee = 6.0
                over_db = env_db - threshold
                slope = 1.0 / ratio - 1.0
                gain_db = np.where(
                    over_db > knee,
                    over_db * slope,
                    np.where(
                        over_db > -knee,
                        ((over_db + knee) / (2 * knee)) ** 2 * knee * slope,
                        0.0,
                    ),
                )

                gain_linear = 10 ** ((gain_db + makeup) / 20.0)
                band_processed = band * gain_linear
                bands_processed.append(band_processed)

            combined = bands_processed[0] + bands_processed[1] + bands_processed[2]
            combined = torch.clamp(combined, -1.0, 1.0)

            ctx.audio.data = combined.float().to(original_device)
        except Exception as e:
            logger.warning("Multiband compressor failed: %s", e)

        return ctx
