from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

import ray
from huggingface_hub import snapshot_download

MODEL_MANIFEST = {
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-forced-aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "toxic-bert": "unitary/toxic-bert",
    "twitter-roberta-sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "nli-distilroberta": "cross-encoder/nli-distilroberta-base",
    "fish-speech/s2-pro": "fishaudio/s2-pro",
    "mossformer2-se-48k": "alibabasglab/MossFormer2_SE_48K",
}
UNUSED_MODEL_FORMATS = ("*.msgpack", "flax_model*", "tf_model*", "*.h5", "*.onnx")
DEMUCS_MANIFEST = "models: ['955717e8']\n"
DEMUCS_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"
)
DNSMOS_MODEL_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)


class ModelProvisioner:
    @staticmethod
    def provision(model_root: str, hub_cache: str | None = None) -> dict[str, str]:
        """Download missing model artifacts into one durable model root."""
        root = Path(model_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(hub_cache).expanduser().resolve() if hub_cache else root / ".hub-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

        results = {
            relative_path: snapshot_download(
                repo_id=repo_id,
                local_dir=root / relative_path,
                cache_dir=cache_dir,
                ignore_patterns=UNUSED_MODEL_FORMATS,
            )
            for relative_path, repo_id in MODEL_MANIFEST.items()
        }
        results["demucs"] = str(ModelProvisioner._provision_demucs(root / "demucs"))
        results["dnsmos"] = str(
            ModelProvisioner._provision_file(
                DNSMOS_MODEL_URL, root / "dnsmos" / "sig_bak_ovr.onnx"
            )
        )
        return results

    @staticmethod
    def _provision_demucs(destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        manifest = destination / "htdemucs.yaml"
        if not manifest.is_file() or manifest.read_text() != DEMUCS_MANIFEST:
            manifest.write_text(DEMUCS_MANIFEST)
        ModelProvisioner._provision_file(
            DEMUCS_CHECKPOINT_URL, destination / "955717e8-8726e21a.th"
        )
        return destination

    @staticmethod
    def _provision_file(url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size > 0:
            return destination
        with urlopen(url, timeout=300) as response:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=f".{destination.name}-", suffix=".part", delete=False
            ) as stream:
                partial = Path(stream.name)
                try:
                    shutil.copyfileobj(response, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                    if stream.tell() == 0:
                        raise RuntimeError(f"empty_model_file:{destination.name}")
                    partial.replace(destination)
                finally:
                    partial.unlink(missing_ok=True)
        return destination


class ModelProvisioningTask:
    @staticmethod
    @ray.remote(num_cpus=0.1)
    def provision_models_on_ray(
        model_root: str, hub_cache: str | None = None
    ) -> dict[str, str]:
        return ModelProvisioner.provision(model_root, hub_cache)


provision_models_on_ray = ModelProvisioningTask.provision_models_on_ray
