"""Provision immutable Sound Cleanup runtime bundles from pinned public sources."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

PANN_URL = (
    "https://zenodo.org/records/3987831/files/"
    "Cnn14_DecisionLevelMax_mAP=0.385.pth?download=1"
)
PANN_MD5 = "70539c43c18b6a289b3199c503a82c5a"
LABEL_URL = (
    "https://raw.githubusercontent.com/qiuqiangkong/"
    "audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
)
LABEL_SHA256 = "cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429"
SILERO_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
AUDIOSEP_REPO = "https://github.com/Audio-AGI/AudioSep.git"
AUDIOSEP_REVISION = "944583f18b84589dc965de3ad77525c945334252"
AUDIOSEP_SPACE_REVISION = "5638854dccfaea5c5fa4f634c00fe74fbb119244"
AUDIOSEP_SHA256 = "f8cda01bfd0ebd141eef45d41db7a3ada23a56568465840d3cff04b8010ce82c"
ROBERTA_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
ANALYSIS_MANIFEST = "f878d14f1d892e142db2c3f582a5092aabc9ac260c9b771b02a09b0ea71389a9"
SEPARATOR_MANIFEST = "e1227365d076eafde534152c75be2c106302705c578eb8f71969c014f2958546"


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)


def verified(path: Path, expected: str, algorithm: str = "sha256") -> None:
    actual = digest(path, algorithm)
    if actual != expected:
        raise RuntimeError(f"asset_digest_mismatch:{path.name}:{actual}")
def silero_asset() -> Path:
    import silero_vad

    root = Path(silero_vad.__file__).resolve().parent
    candidates = [p for p in root.rglob("silero_vad.jit") if p.is_file()]
    if len(candidates) != 1:
        raise RuntimeError("pinned_silero_asset_not_found")
    verified(candidates[0], SILERO_SHA256)
    return candidates[0]


def provision_analysis(root: Path, work: Path) -> None:
    output = root / "sound-cleanup-v1-runtime"
    if output.is_dir() and digest(output / "manifest.json") == ANALYSIS_MANIFEST:
        return
    if output.exists():
        shutil.rmtree(output)
    source = work / "sound-analysis-source"
    source.mkdir()
    download(PANN_URL, source / "panns-checkpoint.pth")
    verified(source / "panns-checkpoint.pth", PANN_MD5, "md5")
    download(LABEL_URL, source / "labels.csv")
    verified(source / "labels.csv", LABEL_SHA256)
    shutil.copyfile(silero_asset(), source / "silero_vad.jit")
    panns_labels = Path.home() / "panns_data/class_labels_indices.csv"
    panns_labels.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "labels.csv", panns_labels)
    subprocess.run(
        [sys.executable, "scripts/provision_sound_cleanup_models.py",
         "--source", str(source), "--output", str(output)],
        check=True,
    )
    verified(output / "manifest.json", ANALYSIS_MANIFEST)
def provision_separator(root: Path, work: Path) -> None:
    output = root / "sound-cleanup-specialist/runtime"
    if output.is_dir() and digest(output / "manifest.json") == SEPARATOR_MANIFEST:
        return
    if output.exists():
        shutil.rmtree(output)
    source = work / "AudioSep"
    subprocess.run(["git", "clone", "--filter=blob:none", AUDIOSEP_REPO, str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", AUDIOSEP_REVISION], check=True)
    checkpoint = Path(
        hf_hub_download(
            repo_id="Audio-AGI/AudioSep",
            repo_type="space",
            revision=AUDIOSEP_SPACE_REVISION,
            filename="checkpoint/audiosep_base_4M_steps.ckpt",
            cache_dir=work / "hf-audiosep",
        )
    )
    verified(checkpoint, AUDIOSEP_SHA256)
    tokenizer = Path(
        snapshot_download(
            repo_id="FacebookAI/roberta-base",
            revision=ROBERTA_REVISION,
            allow_patterns=[
                "config.json", "merges.txt", "vocab.json",
                "tokenizer.json", "tokenizer_config.json",
            ],
            cache_dir=work / "hf-roberta",
            local_dir=work / "roberta-base",
        )
    )
    subprocess.run(
        [
            sys.executable, "scripts/provision_event_separator.py",
            "--source", str(source), "--checkpoint", str(checkpoint),
            "--tokenizer", str(tokenizer), "--output", str(output),
        ],
        check=True,
    )
    verified(output / "manifest.json", SEPARATOR_MANIFEST)
    license_source = source / "LICENSE"
    if license_source.is_file():
        shutil.copyfile(license_source, output / "AUDIOSEP_LICENSE.txt")
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("/models"))
    args = parser.parse_args()
    root = args.model_root.resolve()
    if str(root).startswith("/workspace/"):
        raise RuntimeError("model_assets_must_not_be_written_to_workspace")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hear-sound-models-") as temporary:
        work = Path(temporary)
        provision_analysis(root, work)
        provision_separator(root, work)
    print("SOUND_RELEASE_ASSETS_VERIFIED", root)


if __name__ == "__main__":
    main()
