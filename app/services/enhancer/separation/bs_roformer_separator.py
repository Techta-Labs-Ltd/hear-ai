import logging
import threading

import torch
import torchaudio

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext

logger = logging.getLogger(__name__)


class BSRoFormerSeparator(ProcessingStage):
    name = "bs_roformer"

    def __init__(self, config):
        self._c = config
        self._model = None
        self._lock = threading.Lock()

    def _try_torchscript(self) -> bool:
        candidates = [
            ("ZFTurbo/BS-RoFormer-V2", "model.pt"),
            ("pcunwa/BS-Roformer-Revive", "model.pt"),
        ]
        for repo_id, filename in candidates:
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(repo_id=repo_id, filename=filename)
                self._model = torch.jit.load(model_path, map_location="cpu")
                self._model.eval()
                logger.info("BS-RoFormer loaded via torch.jit from %s", repo_id)
                return True
            except Exception:
                continue
        return False

    def _try_bsroformer_package(self) -> bool:
        try:
            from bs_roformer import BSRoformer
            from huggingface_hub import hf_hub_download
            import yaml

            repo_id = "pcunwa/BS-Roformer-Revive"
            ckpt_name = "bs_roformer_revive.ckpt"

            config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)

            with open(config_path) as f:
                cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

            model_cfg = cfg["model"]
            self._model = BSRoformer(
                dim=model_cfg["dim"],
                depth=model_cfg["depth"],
                stereo=model_cfg.get("stereo", False),
                num_stems=model_cfg.get("num_stems", 1),
                time_transformer_depth=model_cfg.get("time_transformer_depth", 1),
                freq_transformer_depth=model_cfg.get("freq_transformer_depth", 1),
                dim_head=model_cfg.get("dim_head", 64),
                heads=model_cfg.get("heads", 8),
                attn_dropout=model_cfg.get("attn_dropout", 0.0),
                ff_dropout=model_cfg.get("ff_dropout", 0.0),
                flash_attn=model_cfg.get("flash_attn", True),
                dim_freqs_in=model_cfg.get("dim_freqs_in", 1025),
                stft_n_fft=model_cfg.get("stft_n_fft", 2048),
                stft_hop_length=model_cfg.get("stft_hop_length", 512),
                stft_win_length=model_cfg.get("stft_win_length", 2048),
                stft_normalized=model_cfg.get("stft_normalized", False),
                mask_estimator_depth=model_cfg.get("mask_estimator_depth", 2),
                multi_stft_resolution_loss_weight=model_cfg.get("multi_stft_resolution_loss_weight", 1.0),
                multi_stft_resolutions_window_sizes=model_cfg.get("multi_stft_resolutions_window_sizes", (4096, 2048, 1024, 512, 256)),
                multi_stft_hop_size=model_cfg.get("multi_stft_hop_size", 147),
                multi_stft_normalized=model_cfg.get("multi_stft_normalized", False),
            )
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state_dict, strict=False)
            self._model.eval()
            self._model = self._model.cpu()
            logger.info("BS-RoFormer loaded via bs_roformer package from %s", repo_id)
            return True
        except Exception as e:
            logger.warning("BS-RoFormer bs_roformer package method failed: %s", e)
            return False

    def load(self):
        if self._try_torchscript():
            self._ready = True
            return
        if self._try_bsroformer_package():
            self._ready = True
            return
        logger.warning("BS-RoFormer all load methods failed — separation stage disabled")

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        if not self._ready or self._model is None:
            return ctx

        w = ctx.audio.data
        sr = ctx.audio.sample_rate
        original_len = w.shape[1]
        original_device = w.device

        model_sr = getattr(self._model, "sample_rate", 44100)
        if sr != model_sr:
            w_resampled = torchaudio.functional.resample(w, sr, model_sr)
        else:
            w_resampled = w

        model_stereo = getattr(self._model, "stereo", False)
        is_stereo = w_resampled.shape[0] >= 2
        if model_stereo and not is_stereo:
            w_resampled = w_resampled.repeat(2, 1)
        elif not model_stereo and is_stereo:
            w_resampled = w_resampled.mean(dim=0, keepdim=True)

        try:
            with self._lock:
                with torch.no_grad():
                    sources = self._model(w_resampled.unsqueeze(0).cpu())
            if isinstance(sources, dict):
                vocals = sources.get("vocals", sources.get("vocals", list(sources.values())[0]))
            elif isinstance(sources, (list, tuple)):
                vocals = sources[0]
            elif isinstance(sources, torch.Tensor) and sources.dim() == 3:
                vocals = sources[:, 0:1]
            else:
                vocals = w_resampled.unsqueeze(0)

            if vocals.dim() == 3:
                if vocals.shape[0] > 1 and vocals.shape[1] == 1:
                    vocals = vocals[:1]
                elif vocals.shape[0] > 1:
                    vocals = vocals[:1]
                elif vocals.shape[0] == 1:
                    pass
            if vocals.dim() == 3:
                vocals = vocals.squeeze(0)
            if vocals.shape[0] > 1:
                vocals = vocals.mean(dim=0, keepdim=True)

            if model_sr != sr:
                vocals = torchaudio.functional.resample(vocals, model_sr, sr)
            if vocals.shape[1] > original_len:
                vocals = vocals[:, :original_len]
            elif vocals.shape[1] < original_len:
                pad = torch.zeros(1, original_len - vocals.shape[1], device=vocals.device)
                vocals = torch.cat([vocals, pad], dim=1)

            ctx.audio.data = vocals.to(original_device)
        except Exception as e:
            logger.warning("BS-RoFormer process failed: %s", e)

        return ctx
