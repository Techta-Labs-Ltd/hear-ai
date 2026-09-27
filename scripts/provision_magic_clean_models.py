from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib import request

from huggingface_hub import snapshot_download

FILES = {
    "sam-audio-base": {
        "LICENSE": "4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be",
        "checkpoint.pt": "b5f3e29ea7a9e80e90a00da495a8aafe890571f371c4bfb88c052c65a5636839",
        "config.json": "b99a0ee6296edaeb8d355d41d365b33faa94b40af00b2c34d643a43617b10fb2",
    },
    "t5-base": {
        "config.json": "46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033",
        "model.safetensors": "a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4",
        "spiece.model": "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
        "tokenizer.json": "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066",
    },
    "laion-clap": {
        "630k-best.pt": "e02951eae3c9955db546c50086059e6457188ec39446858f2bddbcb4b56b1cb3",
    },
    "pe-a-frame-large": {
        "config.json": "382227d331004428a954209d29609046d06db755b347c9232b27271d921f1126",
        "model.safetensors": "cb1b7d596f1765e6fe21707f7c78989b06e08bc5692ffa88868ad266cce65660",
        "preprocessor_config.json": (
            "d68bb68c371d05defe1f07dc25fbf211e865368dd4702b8bb85fd3eb518df20d"
        ),
        "special_tokens_map.json": (
            "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049"
        ),
        "tokenizer.json": "9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1",
        "tokenizer_config.json": (
            "3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2"
        ),
    },
}


class MagicCleanProvisioner:
    def __init__(self, model_root: Path) -> None:
        self.root = model_root / "magic-clean"
        self.root.mkdir(parents=True, exist_ok=True)

    def deepfilter(self) -> None:
        destination = self.root / "DeepFilterNet3"
        if self._deepfilter_valid(destination):
            return
        with tempfile.TemporaryDirectory(prefix="deepfilter-", dir=self.root) as raw:
            staging = Path(raw)
            archive_path = staging / "DeepFilterNet3.zip"
            with (
                request.urlopen(
                    "https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip",
                    timeout=120,
                ) as response,
                archive_path.open("wb") as target,
            ):
                shutil.copyfileobj(response, target)
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    resolved = (staging / member.filename).resolve()
                    if not resolved.is_relative_to(staging.resolve()):
                        raise RuntimeError("invalid_deepfilter_archive")
                archive.extractall(staging)
            candidate = staging / "DeepFilterNet3"
            if not self._deepfilter_valid(candidate):
                raise RuntimeError("deepfilter_asset_verification_failed")
            self._replace(candidate, destination)

    def sam(self, token: str | None) -> None:
        repositories = (
            (
                "facebook/sam-audio-base",
                "81f64008f9f957c2b57a45923fb5299c66e9d186",
                "sam-audio-base",
            ),
            (
                "google-t5/t5-base",
                "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1",
                "t5-base",
            ),
            (
                "facebook/pe-a-frame-large",
                "40187271298f84e2966d4518c88dde698540c9ad",
                "pe-a-frame-large",
            ),
            (
                "lukewys/laion_clap",
                "b3708341862f581175dba5c356a4ebf74a9b6651",
                "laion-clap",
            ),
        )
        if not token and not self._files_valid(
            self.root / "sam-audio-base", FILES["sam-audio-base"]
        ):
            raise RuntimeError("HF_TOKEN is required for facebook/sam-audio-base")
        for repo_id, revision, name in repositories:
            destination = self.root / name
            if self._files_valid(destination, FILES[name]):
                continue
            with tempfile.TemporaryDirectory(prefix=f"{name}-", dir=self.root) as raw:
                candidate = Path(raw) / name
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_dir=candidate,
                    allow_patterns=list(FILES[name]),
                    token=token,
                )
                if not self._files_valid(candidate, FILES[name]):
                    raise RuntimeError(f"{name}_asset_verification_failed")
                self._replace(candidate, destination)
        cache = self.root.parent / ".cache" / "huggingface" / "hub"
        dependencies = (
            (
                "bert-base-uncased",
                "86b5e0934494bd15c9632b12f734a8a67f723594",
                ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"],
            ),
            (
                "roberta-base",
                "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
                [
                    "config.json",
                    "model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                ],
            ),
            (
                "facebook/bart-base",
                "aadd2ab0ae0c8268c7c9693540e9904811f36177",
                ["config.json", "tokenizer.json", "vocab.json", "merges.txt"],
            ),
        )
        for repo_id, revision, patterns in dependencies:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=cache,
                allow_patterns=patterns,
            )
            repository_cache = cache / f"models--{repo_id.replace('/', '--')}"
            references = repository_cache / "refs"
            references.mkdir(parents=True, exist_ok=True)
            (references / "main").write_text(revision)

    @staticmethod
    def _replace(candidate: Path, destination: Path) -> None:
        previous = destination.with_name(destination.name + ".previous")
        if previous.exists():
            shutil.rmtree(previous)
        if destination.exists():
            destination.rename(previous)
        try:
            candidate.rename(destination)
        except BaseException:
            if previous.exists() and not destination.exists():
                previous.rename(destination)
            raise
        if previous.exists():
            shutil.rmtree(previous)

    @classmethod
    def _deepfilter_valid(cls, directory: Path) -> bool:
        return cls._files_valid(
            directory,
            {
                "config.ini": "415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290",
                "checkpoints/model_120.ckpt.best": (
                    "23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003"
                ),
            },
        )

    @classmethod
    def _files_valid(cls, directory: Path, files: dict[str, str]) -> bool:
        try:
            return all(
                cls._sha256(directory / filename) == digest for filename, digest in files.items()
            )
        except OSError:
            return False

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()


class MagicCleanProvisioningCli:
    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-root", type=Path, default=Path("/models"))
        parser.add_argument("--engine", choices=("all", "deepfilter", "sam"), default="all")
        args = parser.parse_args()
        provisioner = MagicCleanProvisioner(args.model_root)
        if args.engine in {"all", "deepfilter"}:
            provisioner.deepfilter()
        if args.engine in {"all", "sam"}:
            provisioner.sam(os.environ.get("HF_TOKEN"))
        return 0


if __name__ == "__main__":
    raise SystemExit(MagicCleanProvisioningCli.main())
