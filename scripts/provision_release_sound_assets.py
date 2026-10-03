"""Provision immutable Sound Cleanup runtime bundles from pinned public sources."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

PANN_URL = (
    "https://zenodo.org/records/3987831/files/Cnn14_DecisionLevelMax_mAP=0.385.pth?download=1"
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


class ReleaseSoundAssetProvisioner:
    @staticmethod
    def digest(path: Path, algorithm: str = "sha256") -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, algorithm).hexdigest()

    @staticmethod
    def download(url: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name("." + target.name + ".part")
        partial.unlink(missing_ok=True)
        with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(target)

    @staticmethod
    def verified(path: Path, expected: str, algorithm: str = "sha256") -> None:
        actual = ReleaseSoundAssetProvisioner.digest(path, algorithm)
        if actual != expected:
            raise RuntimeError(f"asset_digest_mismatch:{path.name}:{actual}")

    @staticmethod
    def verify_bundle_files(root: Path, names: tuple[str, ...]) -> dict:
        manifest = json.loads((root / "manifest.json").read_text())
        records = manifest.get("files", {})
        for name in names:
            record = records.get(name)
            if not isinstance(record, dict):
                raise RuntimeError(f"bundle_file_missing_from_manifest:{name}")
            if ReleaseSoundAssetProvisioner.digest(root / name) != record.get("sha256"):
                raise RuntimeError(f"bundle_file_digest_mismatch:{name}")
            if (root / name).stat().st_size != record.get("bytes"):
                raise RuntimeError(f"bundle_file_size_mismatch:{name}")
        return manifest

    @staticmethod
    def silero_asset() -> Path:
        silero_vad = importlib.import_module("silero_vad")

        root = Path(silero_vad.__file__).resolve().parent
        matches = [
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file()
            and ReleaseSoundAssetProvisioner.digest(candidate) == SILERO_SHA256
        ]
        if len(matches) != 1:
            raise RuntimeError("pinned_silero_asset_not_found")
        return matches[0]

    @staticmethod
    def provision_analysis(root: Path, work: Path) -> None:
        output = root / "sound-cleanup-v1-runtime"
        if output.exists():
            shutil.rmtree(output)
        source = work / "sound-analysis-source"
        source.mkdir()
        pann = source / "panns-checkpoint.pth"
        labels = source / "labels.csv"
        ReleaseSoundAssetProvisioner.download(PANN_URL, pann)
        ReleaseSoundAssetProvisioner.verified(pann, PANN_MD5, "md5")
        ReleaseSoundAssetProvisioner.download(LABEL_URL, labels)
        ReleaseSoundAssetProvisioner.verified(labels, LABEL_SHA256)
        shutil.copyfile(ReleaseSoundAssetProvisioner.silero_asset(), source / "silero_vad.jit")

        panns_labels = Path.home() / "panns_data" / "class_labels_indices.csv"
        panns_labels.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(labels, panns_labels)
        subprocess.run(
            [
                sys.executable,
                "scripts/provision_sound_cleanup_models.py",
                "--source",
                str(source),
                "--output",
                str(output),
            ],
            check=True,
        )
        manifest = ReleaseSoundAssetProvisioner.verify_bundle_files(
            output, ("silero_vad.jit", "panns_sed.jit", "labels.csv")
        )
        expected = {
            "schema_version": 1,
            "vad_frame_samples": 512,
            "panns_input_frames": 320000,
            "panns_output_frames": 1001,
            "panns_version": "0.1.1",
            "silero_version": "6.2.1",
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise RuntimeError("sound_cleanup_analysis_manifest_contract_mismatch")
        if manifest["files"]["silero_vad.jit"]["sha256"] != SILERO_SHA256:
            raise RuntimeError("sound_cleanup_silero_runtime_mismatch")
        if manifest["files"]["labels.csv"]["sha256"] != LABEL_SHA256:
            raise RuntimeError("sound_cleanup_labels_runtime_mismatch")

    @staticmethod
    def provision_separator(root: Path, work: Path) -> None:
        output = root / "sound-cleanup-specialist" / "runtime"
        if output.parent.exists():
            shutil.rmtree(output.parent)
        source = work / "AudioSep"
        subprocess.run(
            ["git", "clone", "--filter=blob:none", AUDIOSEP_REPO, str(source)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--detach", AUDIOSEP_REVISION],
            check=True,
        )
        actual_revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual_revision != AUDIOSEP_REVISION:
            raise RuntimeError("audiosep_source_revision_mismatch")

        checkpoint = Path(
            hf_hub_download(
                repo_id="Audio-AGI/AudioSep",
                repo_type="space",
                revision=AUDIOSEP_SPACE_REVISION,
                filename="checkpoint/audiosep_base_4M_steps.ckpt",
                cache_dir=work / "hf-audiosep",
            )
        )
        ReleaseSoundAssetProvisioner.verified(checkpoint, AUDIOSEP_SHA256)
        tokenizer = Path(
            snapshot_download(
                repo_id="FacebookAI/roberta-base",
                revision=ROBERTA_REVISION,
                allow_patterns=[
                    "config.json",
                    "merges.txt",
                    "vocab.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ],
                cache_dir=work / "hf-roberta",
                local_dir=work / "roberta-base",
            )
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/provision_event_separator.py",
                "--source",
                str(source),
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                str(tokenizer),
                "--output",
                str(output),
            ],
            check=True,
        )

        manifest = ReleaseSoundAssetProvisioner.verify_bundle_files(
            output, ("audiosep.jit", "queries.json")
        )
        if manifest.get("schema_version") != 1:
            raise RuntimeError("audiosep_runtime_schema_mismatch")
        if manifest.get("input_frames") != 320000 or manifest.get("sample_rate") != 32000:
            raise RuntimeError("audiosep_runtime_shape_contract_mismatch")
        if manifest.get("source_commit") != AUDIOSEP_REVISION:
            raise RuntimeError("audiosep_runtime_source_revision_mismatch")
        if manifest.get("source_checkpoint_sha256") != AUDIOSEP_SHA256:
            raise RuntimeError("audiosep_runtime_checkpoint_mismatch")
        license_source = source / "LICENSE"
        if license_source.is_file():
            shutil.copyfile(license_source, output / "AUDIOSEP_LICENSE.txt")

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-root", type=Path, default=Path("/models"))
        args = parser.parse_args()
        root = args.model_root.resolve()
        if root.is_relative_to(Path(__file__).resolve().parents[1]):
            parser.error("model_storage_must_not_use_source_checkout")
        root.mkdir(parents=True, exist_ok=True)
        release_env = root / "sound-cleanup-release.env"
        release_env.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix="hear-sound-assets-") as raw:
            work = Path(raw)
            ReleaseSoundAssetProvisioner.provision_analysis(root, work)
            ReleaseSoundAssetProvisioner.provision_separator(root, work)
        # Exported TorchScript bytes can differ between builds. Bind this release
        # to its verified exports rather than a hash from an older image.
        digests = {
            "HEAR_SOUND_CLEANUP_BUNDLE_SHA256": ReleaseSoundAssetProvisioner.digest(
                root / "sound-cleanup-v1-runtime" / "manifest.json"
            ),
            "HEAR_SOUND_CLEANUP_SEPARATOR_SHA256": ReleaseSoundAssetProvisioner.digest(
                root / "sound-cleanup-specialist" / "runtime" / "manifest.json"
            ),
        }
        release_env.write_text("".join(f"{key}={value}\n" for key, value in digests.items()))
        print(json.dumps({"model_root": str(root), "manifest_digests": digests, "verified": True}))


if __name__ == "__main__":
    ReleaseSoundAssetProvisioner.main()
