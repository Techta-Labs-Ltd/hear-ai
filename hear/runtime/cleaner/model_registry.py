"""Explicit runtime identity and certification admission, without model fallbacks."""

from collections.abc import Callable
from dataclasses import dataclass

from hear.services.magic_clean.contracts import (
    CleanExecutionError,
    CleanPlan,
    ErrorCode,
    RuntimeIdentity,
)
from hear.services.magic_clean.engines.base import CleanEngine


@dataclass(frozen=True)
class CertifiedRuntime:
    identity: RuntimeIdentity
    evidence_sha256: str
    max_frames: int
    max_input_bytes: int
    sample_rates: tuple[int, ...]
    channel_counts: tuple[int, ...]

    def __post_init__(self):
        if len(self.evidence_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.evidence_sha256
        ):
            raise ValueError("certification evidence requires a SHA-256 digest")
        if min(self.max_frames, self.max_input_bytes) < 1:
            raise ValueError("certified input limits must be positive")
        if not self.sample_rates or any(rate < 8000 or rate > 96000 for rate in self.sample_rates):
            raise ValueError("invalid certified sample rates")
        if not self.channel_counts or any(count not in (1, 2) for count in self.channel_counts):
            raise ValueError("invalid certified channel counts")


class EngineRegistry:
    PROFILES = {
        "deepfilternet3": "natural",
        "sam_audio_small": "voice_focus",
        "noise_profile": "music_atmosphere",
    }

    def __init__(
        self,
        runtimes: tuple[CertifiedRuntime, ...],
        loaders: dict[str, Callable[[], CleanEngine]],
        readiness: dict[str, Callable[[RuntimeIdentity], bool]] | None = None,
    ):
        self._runtimes = {runtime.identity.engine: runtime for runtime in runtimes}
        if len(self._runtimes) != len(runtimes):
            raise ValueError("duplicate certified engine")
        if set(loaders) != set(self._runtimes):
            raise ValueError("loaders must exactly match certified runtimes")
        self._loaders = dict(loaders)
        self._readiness = dict(readiness or {})
        if not set(self._readiness).issubset(self._runtimes):
            raise ValueError("readiness probes require certified runtimes")

    def _ready(self, runtime: CertifiedRuntime) -> bool:
        probe = self._readiness.get(runtime.identity.engine)
        if probe is None:
            return False
        try:
            # A probe verifies pinned local assets/dependencies without allocating
            # a model or downloading files. Exceptions never leak into capabilities.
            return probe(runtime.identity) is True
        except Exception:
            return False

    def capabilities(self) -> dict:
        """Internal backend snapshot, not public job state or a resource reservation.

        Ingress authenticates this response. Readiness is checked again at load;
        a healthy snapshot cannot promise a future allocation will succeed.
        """
        profiles = []
        for engine, profile in self.PROFILES.items():
            runtime = self._runtimes.get(engine)
            ready = runtime is not None and self._ready(runtime)
            profiles.append(
                {
                    "profile": profile,
                    "ready": ready,
                    "reason": None
                    if ready
                    else ("not_certified" if runtime is None else "runtime_unavailable"),
                    "runtime": runtime.identity.model_dump(mode="json") if runtime else None,
                    "evidence_sha256": runtime.evidence_sha256 if runtime else None,
                    "max_input_bytes": runtime.max_input_bytes if runtime else None,
                    "channel_counts": list(runtime.channel_counts) if runtime else [],
                    "rate_limits": [
                        {
                            "sample_rate": rate,
                            "max_frames": runtime.max_frames,
                            "max_duration_seconds": runtime.max_frames / rate,
                        }
                        for rate in runtime.sample_rates
                    ]
                    if runtime
                    else [],
                }
            )
        return {"contract_version": "hear.cleaner.capabilities.v2", "profiles": profiles}

    def load(
        self, plan: CleanPlan, *, frames: int, size_bytes: int, sample_rate: int, channels: int
    ) -> CleanEngine:
        runtime = self._runtimes.get(plan.runtime.engine)
        if runtime is None or runtime.identity != plan.runtime:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "requested runtime is not certified"
            )
        if not (
            0 < frames <= runtime.max_frames
            and 0 < size_bytes <= runtime.max_input_bytes
            and sample_rate in runtime.sample_rates
            and channels in runtime.channel_counts
        ):
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "input exceeds certified limits"
            )
        try:
            if not self._ready(runtime):
                raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "runtime is not ready")
            engine = self._loaders[plan.runtime.engine]()
        except CleanExecutionError:
            raise
        except Exception as exc:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "certified engine could not load"
            ) from exc
        if engine.identity != runtime.identity:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "loaded runtime identity mismatch"
            )
        return engine
