import numpy as np
import torch

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext

GATE_FLOOR = 0.50
GATE_HYSTERESIS_DB = 3.0


class AdaptiveGate(ProcessingStage):
    name = "adaptive_gate"

    def __init__(self, config):
        self._c = config

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        try:
            w = ctx.audio.data
            sr = ctx.audio.sample_rate
            sig = w.squeeze(0).cpu().numpy().astype(np.float64)

            if ctx.noise_profile is not None:
                noise_db = 20 * np.log10(np.mean(ctx.noise_profile["noise_estimate"]) + 1e-10)
                threshold_db = max(noise_db + self._c.gate_threshold_offset_db, -60.0)
            else:
                frame_energy = []
                frame_size = int(sr * 0.030)
                for i in range(0, len(sig), frame_size):
                    frame = sig[i:i + frame_size]
                    if len(frame) > 0:
                        frame_energy.append(np.sqrt(np.mean(frame ** 2)))
                frame_energy = np.array(frame_energy)
                if len(frame_energy) > 10:
                    noise_floor = np.percentile(frame_energy, 10)
                else:
                    noise_floor = np.max(np.abs(sig)) * 0.01
                threshold_db = 20 * np.log10(noise_floor + 1e-10) + self._c.gate_threshold_offset_db

            threshold_lin = 10 ** (threshold_db / 20)
            hold_samples = int(sr * self._c.gate_hold_ms / 1000)

            # --- Signal envelope detector (fast — tracks transients for detection) ---
            # Uses a short 2ms attack / 80ms release regardless of gate_attack_ms so
            # the detector accurately follows speech onset/offset.
            env_attack = np.exp(-1.0 / (sr * 0.002))   # 2ms
            env_release = np.exp(-1.0 / (sr * 0.080))  # 80ms
            abs_sig = np.abs(sig)
            env = np.zeros_like(abs_sig)
            prev = 0.0
            for i in range(len(abs_sig)):
                c = env_attack if abs_sig[i] > prev else env_release
                prev = c * prev + (1.0 - c) * abs_sig[i]
                env[i] = prev

            hysteresis_lin = 10 ** (GATE_HYSTERESIS_DB / 20)
            open_threshold = threshold_lin
            close_threshold = threshold_lin / hysteresis_lin
            gate_open = np.zeros(len(env), dtype=bool)
            was_open = False
            for i in range(len(env)):
                if was_open:
                    gate_open[i] = env[i] > close_threshold
                else:
                    gate_open[i] = env[i] > open_threshold
                was_open = gate_open[i]

            held = np.zeros(len(gate_open), dtype=bool)
            counter = 0
            for i in range(len(gate_open)):
                if gate_open[i]:
                    counter = hold_samples
                    held[i] = True
                elif counter > 0:
                    held[i] = True
                    counter -= 1

            # --- Gate gain smoothing (slow — eliminates thumps on open/close) ---
            # gate_attack_ms controls how slowly the gain ramps from floor→1.0.
            # 20ms is below the audible click threshold (~30ms) while still opening
            # before the first phoneme has fully passed.
            gain_attack = np.exp(-1.0 / (sr * self._c.gate_attack_ms / 1000.0))
            gain_release = np.exp(-1.0 / (sr * self._c.gate_release_ms / 1000.0))

            # Target: GATE_FLOOR when gate is closed, 1.0 when open.
            # Never goes to zero — preserves natural room ambience.
            target = held.astype(np.float64) * (1.0 - GATE_FLOOR) + GATE_FLOOR
            gain_np = np.ones_like(target)
            prev_g = target[0]
            gain_np[0] = prev_g
            for i in range(1, len(target)):
                c = gain_attack if target[i] > prev_g else gain_release
                prev_g = c * prev_g + (1.0 - c) * target[i]
                gain_np[i] = prev_g

            gain = torch.from_numpy(gain_np.astype(np.float32)).unsqueeze(0).to(w.device)
            ctx.audio.data = w * gain
        except Exception:
            pass
        return ctx


