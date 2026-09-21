"""Pinned codec construction without DACVAE download or training-package setup."""

import ast
import importlib
import json
import linecache
import math
import sys
import threading
import types
from pathlib import Path
from typing import Optional, Union

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.sam_core_loader import SamCoreBuilder
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class SamCodecModules:
    def __init__(self, codec, names, filename, lease):
        self.codec, self._names, self._filename, self._lease = codec, names, filename, lease

    def close(self):
        if self._lease is not None:
            self.codec = None
            for name in reversed(self._names):
                sys.modules.pop(name, None)
            linecache.cache.pop(self._filename, None)
            self._lease.release()
            self._lease = None


class SamCodecBuilder:
    _lease = threading.Lock()
    SOURCES = {
        "nn/layers.py": "e4acacab9f6bf34cf87d54cd69ddb380e202e203c73c6c763e54e13cc1336bcb",
        "nn/bottleneck.py": "751687788fe8ea012995f2a5b5603348db994c765982ef0b35d4f759c7c2ee9b",
        "model/dacvae.py": "54191698d6bece6f10398d262157aea510b65619562d98448b425f63698bc36e",
    }

    @classmethod
    def build(cls, source_directory: Path, config_path: Path, guard: ResourceGuard):
        guard.check()
        if not cls._lease.acquire(blocking=False):
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "SAM codec namespace occupied")
        names = []
        filename = "<hear-dacvae-pinned-layers.py>"
        try:
            if any(name == "dacvae" or name.startswith("dacvae.") for name in sys.modules):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "foreign DACVAE modules loaded"
                )
            payloads = {
                name: SamCoreBuilder._read(source_directory / "dacvae" / name, digest, guard)
                for name, digest in cls.SOURCES.items()
            }
            config = json.loads(
                SamCoreBuilder._read(config_path, SamCoreBuilder.CONFIG_SHA256, guard)
            )["audio_codec"]
            torch = importlib.import_module("torch")
            np = importlib.import_module("numpy")
            for name in ("dacvae", "dacvae.nn", "dacvae.model"):
                package = types.ModuleType(name)
                package.__path__ = []
                sys.modules[name] = package
                names.append(name)
            layers = types.ModuleType("dacvae.nn.layers")
            layers.__file__ = filename
            sys.modules[layers.__name__] = layers
            names.append(layers.__name__)
            text = payloads["nn/layers.py"].decode()
            # TorchScript inspects source; supply exactly the bytes already verified.
            linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
            exec(compile(text, filename, "exec"), layers.__dict__)
            bottleneck = types.ModuleType("dacvae.nn.bottleneck")
            bottleneck.__dict__.update(
                torch=torch, nn=torch.nn, Union=Union, NormConv1d=layers.NormConv1d
            )
            sys.modules[bottleneck.__name__] = bottleneck
            names.append(bottleneck.__name__)
            SamCoreBuilder._definitions(
                payloads["nn/bottleneck.py"],
                "dacvae/bottleneck.py",
                {"VAEBottleneck"},
                bottleneck.__dict__,
            )
            model = types.ModuleType("dacvae.model.dacvae")
            model.__dict__.update(
                torch=torch,
                nn=torch.nn,
                np=np,
                math=math,
                List=list,
                Optional=Optional,
                Union=Union,
            )
            for name in (
                "MsgProcessor",
                "NormConv1d",
                "NormConvTranspose1d",
                "Snake1d",
                "activation",
            ):
                model.__dict__[name] = getattr(layers, name)
            sys.modules[model.__name__] = model
            names.append(model.__name__)
            constants = {"default_decoder_convtr_kwargs", "default_wm_encoder_kwargs"}
            for node in ast.parse(payloads["model/dacvae.py"]).body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Name) and target.id in constants:
                        model.__dict__[target.id] = ast.literal_eval(node.value)
            if not constants.issubset(model.__dict__):
                raise ValueError("pinned codec constants missing")
            SamCoreBuilder._definitions(
                payloads["model/dacvae.py"],
                "dacvae/model.py",
                {
                    "ResidualUnit",
                    "EncoderBlock",
                    "LSTMBlock",
                    "Encoder",
                    "DecoderBlock",
                    "WatermarkEncoderBlock",
                    "WatermarkDecoderBlock",
                    "Watermarker",
                    "Decoder",
                },
                model.__dict__,
            )
            with torch.device("meta"):
                codec = torch.nn.Module()
                codec.encoder = model.Encoder(
                    config["encoder_dim"], config["encoder_rates"], config["latent_dim"]
                )
                codec.quantizer = bottleneck.VAEBottleneck(
                    config["latent_dim"], config["codebook_size"], config["codebook_dim"]
                )
                codec.decoder = model.Decoder(
                    config["latent_dim"],
                    config["decoder_dim"],
                    config["decoder_rates"],
                    [8, 5, 4, 2],
                )
            codec.quantizer.dummy_codebook_loss = torch.tensor(0.0, device="cpu")
            codec.eval()
            guard.check()
            return SamCodecModules(codec, names, filename, cls._lease)
        except BaseException as error:
            for name in reversed(names):
                sys.modules.pop(name, None)
            if names:
                linecache.cache.pop(filename, None)
            cls._lease.release()
            if isinstance(error, CleanExecutionError):
                raise
            if isinstance(error, Exception):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "SAM codec construction failed"
                ) from error
            raise
