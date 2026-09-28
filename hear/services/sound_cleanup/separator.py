"""Optional, explicitly selected overlap-repair previews with pinned local assets."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from hear.runtime.cleaner.asset_probe import PinnedAssetProbe
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class EventSeparator:
    def __init__(self, root: Path, digest: str, device: str):
        if (
            device not in ("cpu", "cuda:0")
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("invalid_separator_configuration")
        payload = PinnedAssetProbe.read_regular(root / "manifest.json", maximum_bytes=65536)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "separator_manifest_mismatch")
        self.manifest = json.loads(payload)
        if self.manifest.get("input_frames") != 320000 or self.manifest.get("sample_rate") != 32000:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "separator_contract_mismatch")
        self.root, self.digest, self.device = root, digest, device
        self.model: Any = None
        self.queries: dict[str, list[float]] | None = None

    def close(self) -> None:
        self.model = None
        self.queries = None

    def _load(self, guard: ResourceGuard) -> None:
        guard.check()
        if self.model is not None:
            return
        for name in ("audiosep.jit", "queries.json"):
            record = self.manifest.get("files", {}).get(name, {})
            expected = record.get("sha256", "")
            if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "separator_asset_digest_missing"
                )
            PinnedAssetProbe.sha256(
                self.root / name, expected, maximum_bytes=2_000_000_000, check=guard.check
            )
        queries = json.loads((self.root / "queries.json").read_text())["embeddings"]
        if set(queries) != {"handling", "impact", "animal", "cough", "click"}:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "separator_query_contract_mismatch"
            )
        for value in queries.values():
            if np.shape(value) != (512,) or not np.isfinite(value).all():
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "separator_query_values_invalid"
                )
        self.queries = queries
        self.model = torch.jit.load(
            str(self.root / "audiosep.jit"), map_location=self.device
        ).eval()
        guard.check()

    def propose(self, audio, region, evidence, analyser, guard):
        self._load(guard)
        assert self.queries is not None and self.model is not None
        kind = region.kind
        if kind not in self.queries:
            return None, "unsupported_separator_event_kind", {}
        left = max(0, ((region.start - 48000) // 1536) * 1536)
        audio.seek(left)
        context = audio.read(480000, dtype="float32", always_2d=True)
        if region.end - left > len(context) or not np.isfinite(context).all():
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "separator_context_invalid")
        estimate = np.zeros_like(context)
        for channel in range(context.shape[1]):
            guard.check()
            samples = signal.resample_poly(context[:, channel], 2, 3)
            samples = np.pad(samples, (0, 320000 - len(samples)))
            waveform = torch.from_numpy(samples).reshape(1, 1, -1).to(self.device)
            query = torch.tensor([self.queries[kind]], dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                isolated = self.model(waveform, query).detach().cpu().numpy()
            guard.check()
            if isolated.shape != (1, 1, 320000) or not np.isfinite(isolated).all():
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid_separator_output")
            estimate[:, channel] = signal.resample_poly(isolated[0, 0], 3, 2)[: len(context)]
        start, end = region.start - left, region.end - left
        wanted = context[start:end].astype("float64")
        removed = estimate[start:end].astype("float64") * 0.85
        base_rms = float(np.sqrt(np.mean(wanted**2)))
        removed_rms = float(np.sqrt(np.mean(removed**2)))
        if removed_rms < max(1e-5, base_rms * 0.05):
            return None, "target_estimate_not_significant", {}
        if removed_rms > max(base_rms * 1.25, 0.001):
            return None, "target_estimate_exceeds_source", {}
        candidate = wanted - removed
        if np.max(np.abs(candidate)) > max(np.max(np.abs(wanted)) * 1.5, 0.01):
            return None, "separator_peak_increase", {}
        # Explicit previews still reject gross speech loss or speech in the
        # removed estimate. VAD agreement is not a word-retention certificate.
        check = context.copy()
        check[start:end] = candidate
        check_path = guard.workspace / "separator-speech-check.wav"
        removed_path = guard.workspace / "separator-removed-check.wav"
        guard.reserve_scratch(2 * len(context) * context.shape[1] * 4 + 4096)
        try:
            for path, samples in ((check_path, check), (removed_path, estimate)):
                with path.open("xb") as stream:
                    sf.write(stream, samples, 48000, format="WAV", subtype="FLOAT")
            candidate_prob, _ = analyser._vad(check_path, guard)
            removed_prob, _ = analyser._vad(removed_path, guard)
        finally:
            check_path.unlink(missing_ok=True)
            removed_path.unlink(missing_ok=True)
        lo, hi = start // 1536, min(len(candidate_prob), (end + 1535) // 1536)
        index = np.minimum((left + np.arange(lo, hi) * 1536) // 1536, len(evidence.speech) - 1)
        anchor = evidence.speech[index] >= 0.8
        lost = anchor & (np.max(candidate_prob[lo:hi], axis=1) < 0.1)
        leakage = anchor & (np.max(removed_prob[lo:hi], axis=1) >= 0.5)
        metrics = {
            "speech_anchor_frames": int(np.count_nonzero(anchor)),
            "speech_loss_frames": int(np.count_nonzero(lost)),
            "removed_speech_frames": int(np.count_nonzero(leakage)),
            "separator_manifest_sha256": self.digest,
        }
        if np.count_nonzero(lost) > 1 or np.count_nonzero(leakage) > 2:
            return None, "possible_speech_loss_or_leakage", metrics
        return candidate.astype("float32"), "", metrics
