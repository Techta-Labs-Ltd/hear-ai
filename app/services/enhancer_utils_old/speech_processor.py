import torch
import numpy as np
import torchaudio.functional as F


class SpeechProcessor:
    DEESSER_FREQ_HZ      = 6000.0
    DEESSER_Q            = 2.0
    DEESSER_THRESHOLD_DB = -20.0
    DEESSER_ATTACK_MS    = 1
    DEESSER_RELEASE_MS   = 50
    DEESSER_REDUCTION_DB = -6.0

    def apply_eq_speech(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            w = F.highpass_biquad(w, sr, cutoff_freq=80.0)
            w = F.bass_biquad(w, sr, gain=-1.0, central_freq=250.0)
            w = F.treble_biquad(w, sr, gain=1.0, central_freq=6000.0)
            return w
        except Exception:
            return w

    def apply_eq_music(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            w = F.highpass_biquad(w, sr, cutoff_freq=30.0)
            w = F.equalizer_biquad(w, sr, center_freq=100.0,  gain=1.0,  Q=1.2)
            w = F.equalizer_biquad(w, sr, center_freq=3000.0, gain=-1.0, Q=2.0)
            w = F.equalizer_biquad(w, sr, center_freq=8000.0, gain=2.0,  Q=1.5)
            return w
        except Exception:
            return w

    def apply_deesser(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            s_freq = self.DEESSER_FREQ_HZ
            s_q = self.DEESSER_Q
            theta = 2.0 * np.pi * s_freq / sr
            alpha_d = np.sin(theta) / (2.0 * s_q)

            a0 = 1.0 + alpha_d
            b0 = 1.0 / a0
            b1 = -2.0 * np.cos(theta) / a0
            b2 = 1.0 / a0
            a1 = -2.0 * np.cos(theta) / a0
            a2 = (1.0 - alpha_d) / a0

            sig = w.squeeze(0).cpu().numpy().astype(np.float64)

            sos = np.zeros(len(sig))
            for i in range(2, len(sig)):
                sos[i] = (
                    b0 * sig[i] + b1 * sig[i - 1] + b2 * sig[i - 2]
                    - a1 * sos[i - 1] - a2 * sos[i - 2]
                )

            sibilance = np.abs(sig - sos)
            env = np.zeros_like(sibilance)
            attack_c = np.exp(-1.0 / (sr * self.DEESSER_ATTACK_MS / 1000.0))
            release_c = np.exp(-1.0 / (sr * self.DEESSER_RELEASE_MS / 1000.0))
            threshold_lin = 10.0 ** (self.DEESSER_THRESHOLD_DB / 20.0)
            reduction_lin = 10.0 ** (self.DEESSER_REDUCTION_DB / 20.0)

            prev_env = 0.0
            for i in range(len(sibilance)):
                c = attack_c if sibilance[i] > prev_env else release_c
                prev_env = c * prev_env + (1.0 - c) * sibilance[i]
                env[i] = prev_env

            gain = np.ones_like(env)
            over = env > threshold_lin
            if over.any():
                overshoot = (env[over] - threshold_lin) / (env[over] + 1e-10)
                gain[over] = 1.0 - overshoot * (1.0 - reduction_lin)
                gain = np.clip(gain, reduction_lin, 1.0)

            sos_filtered = sos + (sig - sos) * gain

            return torch.from_numpy(
                sos_filtered.astype(np.float32)
            ).unsqueeze(0).to(w.device)
        except Exception:
            return w
