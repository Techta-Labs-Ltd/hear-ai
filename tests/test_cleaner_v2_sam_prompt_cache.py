import hashlib
import os
from dataclasses import replace

import pytest
import torch

from hear.runtime.cleaner.sam_prompt_cache import SamPromptCache, SamPromptIdentity
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@pytest.fixture
def entry():
    embedding = torch.arange(1536, dtype=torch.float32).reshape(1, 2, 768)
    mask = torch.tensor([[True, False]])
    identity = SamPromptIdentity(
        "a" * 64,
        "b" * 64,
        "c" * 64,
        hashlib.sha256(embedding.numpy().tobytes()).hexdigest(),
        hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
        2,
    )
    return identity, embedding, mask


def test_cache_snapshot_and_returns_are_isolated(entry):
    identity, embedding, mask = entry
    expected = embedding.clone()
    cache = SamPromptCache((entry,))
    embedding.zero_()
    mask.zero_()
    actual, actual_mask = cache.get(identity)
    assert torch.equal(actual, expected) and actual_mask.tolist() == [[True, False]]
    actual.zero_()
    actual_mask.zero_()
    again, again_mask = cache.get(identity)
    assert torch.equal(again, expected) and again_mask.tolist() == [[True, False]]
    cache.close()
    cache.close()
    with pytest.raises(CleanExecutionError) as error:
        cache.get(identity)
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE


@pytest.mark.parametrize(
    "field",
    ["prompt_sha256", "model_sha256", "precision_sha256", "embedding_sha256", "mask_sha256"],
)
def test_every_identity_component_is_checked(entry, field):
    identity, _, _ = entry
    cache = SamPromptCache((entry,))
    changed = replace(identity, **{field: "d" * 64})
    assert identity.digest != changed.digest
    with pytest.raises(CleanExecutionError):
        cache.get(changed)


@pytest.mark.parametrize(
    "kind", ["nan", "zero", "changed", "empty_mask", "wrong_mask", "dtype", "shape"]
)
def test_invalid_or_mismatched_conditioning_rejected(entry, kind):
    identity, embedding, mask = entry
    if kind == "nan":
        embedding[0, 0, 0] = float("nan")
    elif kind == "zero":
        embedding.zero_()
    elif kind == "changed":
        embedding[0, 0, 0] += 1
    elif kind == "empty_mask":
        mask.zero_()
    elif kind == "wrong_mask":
        mask[:] = True
    elif kind == "dtype":
        embedding = embedding.double()
    else:
        embedding = embedding[:, :1]
    with pytest.raises(CleanExecutionError) as error:
        SamPromptCache(((identity, embedding, mask),))
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE


def test_duplicate_and_unbounded_cache_rejected(entry):
    for entries in ((), (entry, entry), (entry,) * 17):
        with pytest.raises(ValueError):
            SamPromptCache(entries)


@pytest.mark.parametrize("tokens", [True, 0, 513, 1.5])
def test_token_bound_is_strict(entry, tokens):
    with pytest.raises(ValueError):
        replace(entry[0], tokens=tokens)


def test_raw_asset_roundtrip_and_snapshot(entry, tmp_path):
    identity, embedding, mask = entry
    path = tmp_path / "conditioning.bin"
    path.write_bytes(embedding.numpy().astype("<f4").tobytes() + mask.numpy().tobytes())
    cache = SamPromptCache.from_files(((identity, path),))
    path.unlink()
    actual, actual_mask = cache.get(identity)
    assert torch.equal(actual, embedding)
    assert torch.equal(actual_mask, mask)


@pytest.mark.parametrize("kind", ["missing", "short", "extra", "changed", "symlink", "fifo"])
def test_raw_asset_rejects_bad_files(entry, tmp_path, kind):
    identity, embedding, mask = entry
    path = tmp_path / "conditioning.bin"
    payload = embedding.numpy().tobytes() + mask.numpy().tobytes()
    if kind == "short":
        path.write_bytes(payload[:-1])
    elif kind == "extra":
        path.write_bytes(payload + b"x")
    elif kind == "changed":
        path.write_bytes(b"x" + payload[1:])
    elif kind == "symlink":
        target = tmp_path / "target"
        target.write_bytes(payload)
        path.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(path)
    with pytest.raises(CleanExecutionError) as error:
        SamPromptCache.from_files(((identity, path),))
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE


def test_raw_asset_rejects_noncanonical_mask_even_when_digest_matches(entry, tmp_path):
    identity, embedding, _ = entry
    mask = bytes([2, 0])
    identity = replace(identity, mask_sha256=hashlib.sha256(mask).hexdigest())
    path = tmp_path / "conditioning.bin"
    path.write_bytes(embedding.numpy().tobytes() + mask)
    with pytest.raises(CleanExecutionError, match="identity mismatch"):
        SamPromptCache.from_files(((identity, path),))
