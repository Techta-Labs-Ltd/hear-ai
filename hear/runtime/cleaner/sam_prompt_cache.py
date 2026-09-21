"""Bounded CPU conditioning snapshots admitted by externally pinned identities."""

import hashlib
import importlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@dataclass(frozen=True)
class SamPromptIdentity:
    prompt_sha256: str
    model_sha256: str
    precision_sha256: str
    embedding_sha256: str
    mask_sha256: str
    tokens: int

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name != "tokens" and (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError("invalid SAM prompt identity digest")
        if type(self.tokens) is not int or not 1 <= self.tokens <= 512:
            raise ValueError("invalid SAM prompt token count")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {"schema": "sam-prompt-cache-v1", **asdict(self)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


class SamPromptCache:
    MAX_ENTRIES = 16

    @classmethod
    def from_files(cls, entries: tuple):
        """Load externally pinned raw conditioning assets, without pickle or T5.

        Each (identity, path) file contains token-major little-endian FP32
        embeddings followed by one 0/1 mask byte per token. Identities come from
        trusted deployment configuration, never from the asset or request.
        """
        if not isinstance(entries, tuple) or not 1 <= len(entries) <= cls.MAX_ENTRIES:
            raise ValueError("invalid SAM prompt cache size")
        np = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        loaded = []
        for identity, path in entries:
            if not isinstance(identity, SamPromptIdentity):
                raise ValueError("SAM prompt identity required")
            size = identity.tokens * (768 * 4 + 1)
            try:
                # Nonblocking open prevents a misconfigured FIFO from hanging startup.
                fd = os.open(Path(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as asset:
                    info = os.fstat(asset.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                        raise ValueError("invalid conditioning file size/type")
                    payload = asset.read(size + 1)
                if len(payload) != size:
                    raise ValueError("conditioning file changed size")
            except (OSError, ValueError):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "SAM conditioning asset unavailable"
                ) from None
            boundary = identity.tokens * 768 * 4
            embedding_bytes, mask_bytes = payload[:boundary], payload[boundary:]
            if (
                hashlib.sha256(embedding_bytes).hexdigest() != identity.embedding_sha256
                or hashlib.sha256(mask_bytes).hexdigest() != identity.mask_sha256
                or any(value not in (0, 1) for value in mask_bytes)
            ):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "SAM conditioning asset identity mismatch"
                )
            embedding = torch.from_numpy(
                np.frombuffer(embedding_bytes, dtype="<f4").astype(np.float32, copy=True)
            ).reshape(1, identity.tokens, 768)
            mask = torch.from_numpy(np.frombuffer(mask_bytes, dtype=np.uint8).astype(bool)).reshape(
                1, identity.tokens
            )
            loaded.append((identity, embedding, mask))
        return cls(tuple(loaded))

    def __init__(self, entries: tuple):
        """Snapshot precomputed CPU embeddings; never choose or approve a prompt.

        Each entry is (expected identity, embedding, mask). The model digest must
        cover encoder/tokenizer assets and runtime, not merely the display name.
        Trusted provisioning must bind those identities to approved presets.
        """
        if not isinstance(entries, tuple) or not 1 <= len(entries) <= self.MAX_ENTRIES:
            raise ValueError("invalid SAM prompt cache size")
        torch = importlib.import_module("torch")
        self._entries = {}
        for identity, embedding, mask in entries:
            if not isinstance(identity, SamPromptIdentity):
                raise ValueError("SAM prompt identity required")
            if (
                not isinstance(embedding, torch.Tensor)
                or not isinstance(mask, torch.Tensor)
                or embedding.device.type != "cpu"
                or mask.device.type != "cpu"
                or embedding.dtype != torch.float32
                or mask.dtype != torch.bool
                or embedding.shape != (1, identity.tokens, 768)
                or mask.shape != (1, identity.tokens)
            ):
                raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "invalid SAM prompt shape")
            frozen = embedding.detach().contiguous().clone()
            frozen_mask = mask.detach().contiguous().clone()
            if (
                not torch.isfinite(frozen).all()
                or not frozen.any()
                or not frozen_mask.any()
                or hashlib.sha256(frozen.numpy().astype("<f4", copy=False).tobytes()).hexdigest()
                != identity.embedding_sha256
                or hashlib.sha256(frozen_mask.numpy().tobytes()).hexdigest() != identity.mask_sha256
            ):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "SAM prompt contents do not match pinned identity"
                )
            key = (identity.prompt_sha256, identity.model_sha256, identity.precision_sha256)
            if key in self._entries:
                raise ValueError("duplicate SAM prompt cache key")
            self._entries[key] = (identity, frozen, frozen_mask)

    def get(self, identity: SamPromptIdentity):
        """Return isolated CPU tensors only for an exact admitted identity."""
        if not isinstance(identity, SamPromptIdentity):
            raise ValueError("SAM prompt identity required")
        key = (identity.prompt_sha256, identity.model_sha256, identity.precision_sha256)
        entry = self._entries.get(key)
        if entry is None or entry[0] != identity:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM prompt is not cached")
        return entry[1].clone(), entry[2].clone()

    def close(self) -> None:
        self._entries.clear()
