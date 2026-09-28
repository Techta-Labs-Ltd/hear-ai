"""Export a pinned AudioSep specialist for explicit, reviewed event repair."""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


class Export(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform, query):
        return self.model({"mixture": waveform, "condition": query})["waveform"]


class EventSeparatorProvisioner:
    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--source", type=Path, required=True)
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--tokenizer", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        SOURCE, CHECKPOINT, TOKENIZER, BUNDLE = (
            args.source,
            args.checkpoint,
            args.tokenizer,
            args.output,
        )
        revision = subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != "944583f18b84589dc965de3ad77525c945334252":
            raise ValueError("use_the_pinned_audiosep_source_revision")
        subprocess.run(
            [
                "git",
                "-C",
                str(SOURCE),
                "diff",
                "--quiet",
                "HEAD",
                "--",
                "models/resunet.py",
                "models/base.py",
            ],
            check=True,
        )
        if BUNDLE.exists():
            parser.error("output exists; choose a new immutable bundle path")
        with CHECKPOINT.open("rb") as stream:
            assert (
                hashlib.file_digest(stream, "sha256").hexdigest()
                == "f8cda01bfd0ebd141eef45d41db7a3ada23a56568465840d3cff04b8010ce82c"
            )
        torch.set_num_threads(2)
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)[
            "state_dict"
        ]
        config = RobertaConfig.from_pretrained(TOKENIZER, local_files_only=True)
        text = RobertaModel(config).eval()
        prefix = "query_encoder.model.text_branch."
        weights = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        weights.pop("embeddings.position_ids", None)
        weights.pop("embeddings.token_type_ids", None)
        text.load_state_dict(weights, strict=True)
        projection = torch.nn.Sequential(
            torch.nn.Linear(768, 512), torch.nn.ReLU(), torch.nn.Linear(512, 512)
        ).eval()
        prefix = "query_encoder.model.text_projection."
        projection.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}, strict=True
        )
        tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
        phrases = {
            "handling": "paper rustling",
            "impact": "a door slamming",
            "animal": "a dog barking",
            "cough": "a person coughing",
            "click": "clicking sounds",
        }
        queries = {}
        with torch.inference_mode():
            for kind, phrase in phrases.items():
                inputs = tokenizer(
                    [phrase, phrase],
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                embedding = torch.nn.functional.normalize(
                    projection(text(**inputs)["pooler_output"]), dim=-1
                )[0]
                queries[kind] = embedding.tolist()
        del text, projection
        sys.path.insert(0, str(SOURCE))
        ResUNet30 = importlib.import_module("models.resunet").ResUNet30
        separator = ResUNet30(input_channels=1, output_channels=1, condition_size=512).eval()
        prefix = "ss_model."
        separator.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}, strict=True
        )
        del state
        model = Export(separator).eval()
        waveform = torch.zeros(1, 1, 320000)
        query = torch.tensor([queries["animal"]])
        print("Tracing fixed 10-second inference window", flush=True)
        with torch.inference_mode():
            exported = torch.jit.trace(model, (waveform, query), check_trace=True)
            torch.manual_seed(0)
            sample = torch.randn_like(waveform) * 0.01
            reference = model(sample, query)
            actual = exported(sample, query)
            torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-4)
            assert actual.shape == waveform.shape and torch.isfinite(actual).all()
        BUNDLE.mkdir(parents=True, exist_ok=False)
        exported.save(str(BUNDLE / "audiosep.jit"))
        (BUNDLE / "queries.json").write_text(
            json.dumps({"phrases": phrases, "embeddings": queries})
        )
        files = {}
        for name in ("audiosep.jit", "queries.json"):
            with (BUNDLE / name).open("rb") as stream:
                files[name] = {
                    "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                    "bytes": (BUNDLE / name).stat().st_size,
                }
        manifest = {
            "schema_version": 1,
            "input_frames": 320000,
            "sample_rate": 32000,
            "source_commit": "944583f18b84589dc965de3ad77525c945334252",
            "source_checkpoint_sha256": "f8cda01bfd0ebd141eef45d41db7a3ada23a56568465840d3cff04b8010ce82c",
            "files": files,
            "export_equivalence": "zero and seeded noise match eager, fixed length",
            "tokenizer_file_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in TOKENIZER.iterdir()
                if p.is_file() and p.suffix in (".json", ".txt")
            },
            "torch_version": torch.__version__,
        }
        (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(
            "SPECIALIST_EXPORT_COMPLETE",
            hashlib.sha256((BUNDLE / "manifest.json").read_bytes()).hexdigest(),
            flush=True,
        )


if __name__ == "__main__":
    EventSeparatorProvisioner.main()
