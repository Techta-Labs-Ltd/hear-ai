from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import ray
from huggingface_hub import snapshot_download

MODEL_MANIFEST = {
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-forced-aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "toxic-bert": "unitary/toxic-bert",
    "twitter-roberta-sentiment": (
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ),
    "nli-distilroberta": "cross-encoder/nli-distilroberta-base",
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "fish-speech/s2-pro": "fishaudio/s2-pro",
    "mossformer2-se-48k": "alibabasglab/MossFormer2_SE_48K",
}

UNUSED_MODEL_FORMATS = (
    "*.msgpack",
    "flax_model*",
    "tf_model*",
    "*.h5",
    "*.onnx",
)

@ray.remote(num_cpus=0.1)
def download_model(repo_id: str, destination: str) -> dict[str, str]:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=destination,
        ignore_patterns=UNUSED_MODEL_FORMATS,
    )
    return {"repo_id": repo_id, "path": path}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/models")
    parser.add_argument("--ray-address", default="local")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    address = None if args.ray_address == "local" else args.ray_address
    ray.init(address=address, ignore_reinit_error=False)
    try:
        refs = [
            download_model.remote(repo_id, str(root / relative_path))
            for relative_path, repo_id in MODEL_MANIFEST.items()
        ]
        results = ray.get(refs)
    finally:
        ray.shutdown()

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
