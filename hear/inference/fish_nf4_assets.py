"""Verify the provisioned NF4 runtime view without modifying model weights."""

import hashlib
import os
from pathlib import Path


class FishNF4Assets:
    SOURCE_REPO = "https://github.com/groxaxo/fish-speech-int4-patch.git"
    SOURCE_REVISION = "fc4e1e24ff3b8d7d28fdd66e6789f23acb63c5bb"
    MODEL_REPO = "groxaxo/s2-pro-BnB-4Bits"
    MODEL_REVISION = "5c09659b9dbea2f64b90c1a611c4824560619ce7"
    TOKENIZER_REVISION = "1de9996b6be38b745688de084d87a5633f714e4e"
    TOKENIZER_METADATA_SHA256 = "b8d149343ae425b0da67e6708686aceb51be7815d9792f265fc12ff04d5e9856"
    RELATIVE_PATH = "fish-speech/s2-pro-nf4"
    LINKED_FILES = (
        "model.pth",
        "codec.pth",
        "config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "LICENSE.md",
    )

    @classmethod
    def runtime_path(cls, model_root: Path) -> Path:
        original = model_root / cls.RELATIVE_PATH
        runtime = original.with_name(original.name + "-runtime")
        if not runtime.is_dir() or not runtime.resolve().is_relative_to(model_root.resolve()):
            raise RuntimeError("fish_nf4_runtime_view_not_provisioned")
        for name in cls.LINKED_FILES:
            # Hardlinks share verified, immutable original bytes; they are not
            # a second unverified model copy or a second disk allocation.
            if (
                not (runtime / name).is_file()
                or (runtime / name).is_symlink()
                or not os.path.samefile(original / name, runtime / name)
            ):
                raise RuntimeError("fish_nf4_runtime_view_mismatch:" + name)
        metadata = runtime / "tokenizer_config.json"
        if metadata.is_symlink() or metadata.stat().st_size > 2_000_000:
            raise RuntimeError("fish_nf4_tokenizer_metadata_invalid")
        if hashlib.sha256(metadata.read_bytes()).hexdigest() != cls.TOKENIZER_METADATA_SHA256:
            raise RuntimeError("fish_nf4_tokenizer_metadata_mismatch")
        return runtime
