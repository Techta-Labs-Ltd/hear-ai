"""Explicit conversion of verified local PANNs and Silero model files."""

import argparse
import hashlib
import importlib
import json
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import torch


class EventExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio):
        return self.model(audio, None)["framewise_output"]


class SoundBundleProvisioner:
    CHECKPOINT_MD5 = "70539c43c18b6a289b3199c503a82c5a"
    VAD_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"

    @staticmethod
    def digest(path, algorithm="sha256"):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, algorithm).hexdigest()

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--source", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        source = args.source.resolve()
        if args.output.exists():
            parser.error("output already exists; provision a new version")
        if cls.digest(source / "panns-checkpoint.pth", "md5") != cls.CHECKPOINT_MD5:
            raise ValueError("official_panns_checkpoint_mismatch")
        if cls.digest(source / "silero_vad.jit") != cls.VAD_SHA256:
            raise ValueError("pinned_silero_checkpoint_mismatch")
        label_path = Path.home() / "panns_data/class_labels_indices.csv"
        if not label_path.is_file() or cls.digest(label_path) != cls.digest(source / "labels.csv"):
            raise ValueError("provision_panns_labels_before_import")
        if version("panns-inference") != "0.1.1" or version("torchlibrosa") != "0.1.0":
            raise ValueError("use_pinned_provisioning_dependencies")
        module = importlib.import_module("panns_inference.models")
        torch.set_num_threads(2)
        model = module.Cnn14_DecisionLevelMax(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        ).eval()
        state = torch.load(source / "panns-checkpoint.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        wrapper = EventExport(model).eval()
        samples = torch.zeros(1, 320000)
        with torch.inference_mode():
            exported = torch.jit.trace(wrapper, samples, check_trace=True)
            actual = exported(samples)
            torch.testing.assert_close(actual, wrapper(samples))
            if actual.shape[1] < 1000 or actual.shape[2] != 527:
                raise ValueError("panns_export_output_shape")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=args.output.parent, prefix=".sound-bundle-"
        ) as temporary:
            staging = Path(temporary) / "bundle"
            staging.mkdir()
            exported.save(str(staging / "panns_sed.jit"))
            for name in ("silero_vad.jit", "labels.csv"):
                shutil.copyfile(source / name, staging / name)
            manifest = {
                "schema_version": 1,
                "vad_frame_samples": 512,
                "panns_input_frames": 320000,
                "panns_output_frames": actual.shape[1],
                "panns_hop_seconds": 0.01,
                "torch_export_version": version("torch"),
                "panns_version": "0.1.1",
                "silero_version": "6.2.1",
                "files": {
                    name: {
                        "sha256": cls.digest(staging / name),
                        "bytes": (staging / name).stat().st_size,
                    }
                    for name in ("silero_vad.jit", "panns_sed.jit", "labels.csv")
                },
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            staging.rename(args.output)
        print(
            json.dumps(
                {
                    "bundle": str(args.output),
                    "manifest_sha256": cls.digest(args.output / "manifest.json"),
                }
            )
        )


if __name__ == "__main__":
    SoundBundleProvisioner.main()
