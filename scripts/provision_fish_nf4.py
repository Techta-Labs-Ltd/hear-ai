"""Download and verify the pinned low-VRAM Fish model outside job execution."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

from hear.inference.fish_nf4_assets import FishNF4Assets
from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole


class ProvisionFishNF4:
    @classmethod
    def run(cls, model_root: Path) -> Path:
        model_root = FishNF4Assets.validate_model_root(model_root)
        root = Path(__file__).resolve().parents[1]
        spec = FishNF4Assets
        original = model_root / spec.RELATIVE_PATH
        original.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=spec.MODEL_REPO,
            revision=spec.MODEL_REVISION,
            local_dir=original,
            max_workers=2,
            allow_patterns=[*spec.LINKED_FILES, "tokenizer_config.json", "README.md"],
        )
        manifest = ModelManifest(root / "hear/model_manifest.json")
        missing = manifest.validate_local(model_root, WorkerRole.RECONSTRUCTION)
        if missing:
            raise RuntimeError("fish_nf4_asset_validation_failed:" + ",".join(missing))
        runtime = original.with_name(original.name + "-runtime")
        if runtime.exists():
            return spec.runtime_path(model_root)
        url = f"https://huggingface.co/fishaudio/s2-pro/resolve/{spec.TOKENIZER_REVISION}/tokenizer_config.json"
        with urllib.request.urlopen(url, timeout=30) as response:
            metadata = response.read(2_000_001)
        if hashlib.sha256(metadata).hexdigest() != spec.TOKENIZER_METADATA_SHA256:
            raise RuntimeError("official_fish_tokenizer_metadata_mismatch")
        staging = Path(tempfile.mkdtemp(prefix=".nf4-runtime-", dir=original.parent))
        try:
            for name in spec.LINKED_FILES:
                os.link(original / name, staging / name)
            (staging / "tokenizer_config.json").write_bytes(metadata)
            (staging / "hear-runtime.json").write_text(
                json.dumps(
                    {
                        "model_repo": spec.MODEL_REPO,
                        "model_revision": spec.MODEL_REVISION,
                        "source_revision": spec.SOURCE_REVISION,
                        "tokenizer_metadata_revision": spec.TOKENIZER_REVISION,
                        "quantization": "nf4",
                        "weight_bytes_changed": False,
                    },
                    indent=2,
                )
                + "\n"
            )
            staging.rename(runtime)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return spec.runtime_path(model_root)

    @classmethod
    def main(cls) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--model-root", type=Path, default=Path("/root/hear-ai-v11/models"))
        args = parser.parse_args()
        path = cls.run(args.model_root.resolve())
        print(
            json.dumps(
                {
                    "runtime_checkpoint": str(path),
                    "verified": True,
                    "production_license_approval_changed": False,
                }
            )
        )


if __name__ == "__main__":
    ProvisionFishNF4.main()
