"""Construct the pinned Small audio core without importing the multimodal package."""

import ast
import hashlib
import importlib
import json
import math
import sys
import types
import uuid
from pathlib import Path
from typing import Optional

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class SamCoreModules:
    def __init__(self, core, names):
        self.core = core
        self._names = names

    def close(self):
        self.core = None
        for name in reversed(self._names):
            sys.modules.pop(name, None)
        self._names.clear()


class SamCoreBuilder:
    CONFIG_SHA256 = "50a67c841676043afdd3af6484ef6a1b47e300e36ee24eb3e07c31da8216f47b"
    SOURCES = {
        "model.py": "b85217b41f8fcc083908db3dd6c37a211048f52b8e601e5eb4794bef18d56108",
        "config.py": "0fc427191f9f8f0914d47f545448db1ec1f7d3c599d86e242d7b19f2cb422578",
        "transformer.py": "ed25fe2a3ed4466859b45b35e0dfc87d5cd9af044f290ef804c5953a5ead9bf8",
        "rope.py": "c920b965863e7349ad88593e03c0d49734138b614d0569e6d7acc7af179944be",
        "patcher.py": "d4b406dca9a9eb4200b514b7d3c528aca0d5af006ac871320a00f836b0291082",
        "align.py": "8d4d1b61587f71cbc687e73dfcc6226336b242d65e5edc02e6a9ee4b4a43ab9c",
    }

    @staticmethod
    def _read(path: Path, expected: str, guard: ResourceGuard) -> bytes:
        guard.check()
        if path.is_symlink() or not path.is_file():
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM source asset missing")
        with path.open("rb") as source:
            payload = source.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024 or hashlib.sha256(payload).hexdigest() != expected:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM source asset mismatch")
        guard.check()
        return payload

    @staticmethod
    def _definitions(payload, filename, names, namespace):
        tree = ast.parse(payload, filename=filename)
        selected = [
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names
        ]
        if len(selected) != len(names) or {node.name for node in selected} != set(names):
            raise ValueError("pinned SAM definitions missing")
        exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)

    @classmethod
    def build(
        cls, source_directory: Path, config_path: Path, guard: ResourceGuard
    ) -> SamCoreModules:
        names = []
        try:
            source = source_directory / "sam_audio/model"
            # Verify all bytes before executing any of the supplied source code.
            payloads = {
                name: cls._read(source / name, digest, guard)
                for name, digest in cls.SOURCES.items()
            }
            config = json.loads(cls._read(config_path, cls.CONFIG_SHA256, guard))
            torch = importlib.import_module("torch")
            prefix = "_hear_sam_core_" + uuid.uuid4().hex
            package = types.ModuleType(prefix)
            package.__path__ = []  # Relative imports can only resolve the admitted modules.
            sys.modules[prefix] = package
            names.append(prefix)
            cfg = types.ModuleType(prefix + ".config")
            cfg.Optional = Optional
            cls._definitions(
                payloads["config.py"], "sam/config.py", {"TransformerConfig"}, cfg.__dict__
            )
            sys.modules[cfg.__name__] = cfg
            names.append(cfg.__name__)
            modules = {}
            for name in ("rope", "patcher", "align", "transformer"):
                module = types.ModuleType(prefix + "." + name)
                module.__package__ = prefix
                sys.modules[module.__name__] = module
                names.append(module.__name__)
                exec(
                    compile(payloads[name + ".py"], "sam/" + name + ".py", "exec"), module.__dict__
                )
                modules[name] = module
            namespace = {"torch": torch, "math": math, "Optional": Optional, "__name__": prefix}
            cls._definitions(
                payloads["model.py"],
                "sam/model.py",
                {"SinusoidalEmbedding", "EmbedAnchors"},
                namespace,
            )
            with torch.device("meta"):
                core = torch.nn.Module()
                core.transformer = modules["transformer"].DiT(
                    cfg.TransformerConfig(**config["transformer"])
                )
                core.proj = torch.nn.Linear(768, 1536)
                core.memory_proj = torch.nn.Linear(768, 1536)
                core.align_masked_video = modules["align"].AlignModalities(1024, 1536)
                core.embed_anchors = namespace["EmbedAnchors"](3, 128, 1536)
            # Nonpersistent frequency buffers are absent from the checkpoint.
            # Recreate them on CPU using the pinned definitions/formula, never empty storage.
            with torch.device("cpu"):
                core.timestep_emb = namespace["SinusoidalEmbedding"](1536)
                core.transformer.rope_embeddings.reset_parameters()
                embedder = core.transformer.t_embedder
                half = embedder.frequency_embedding_size // 2
                embedder.freqs = torch.exp(
                    -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
                )
            core.eval()
            guard.check()
            return SamCoreModules(core, names)
        except BaseException as error:
            for name in reversed(names):
                sys.modules.pop(name, None)
            if isinstance(error, CleanExecutionError):
                raise
            if isinstance(error, Exception):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "SAM core construction failed"
                ) from None
            raise
