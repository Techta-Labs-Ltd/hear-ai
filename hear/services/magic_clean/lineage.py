"""Tenant-scoped Magic Clean lineage and deterministic audio hashing."""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import struct
import subprocess
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

MAX_MAGIC_CLEAN_LINEAGE_DEPTH = 32
MAGIC_CLEAN_ROOT_URL_KEY = "magic_clean_root_url"
MAGIC_CLEAN_PARENT_JOB_ID_KEY = "magic_clean_parent_job_id"
MAGIC_CLEAN_SOURCE_FILE_SHA256_KEY = "magic_clean_source_file_sha256"
MAGIC_CLEAN_SOURCE_PCM_SHA256_KEY = "magic_clean_source_pcm_sha256"
MAGIC_CLEAN_DELIVERED_FILE_SHA256_KEY = "magic_clean_delivered_file_sha256"
MAGIC_CLEAN_DELIVERED_PCM_SHA256_KEY = "magic_clean_delivered_pcm_sha256"
MAGIC_CLEAN_ENGINE_REVISION_KEY = "magic_clean_engine_revision"
_PCM_HASH_DOMAIN = b"hear.magic-clean.decoded-f32le.v1\x00"
_SHA256_HEX_LENGTH = 64
AUDIO_HASH_TIMEOUT_SECONDS = 4 * 60 * 60
type MagicCleanMatchMethod = Literal[
    "new_source", "exact_url", "delivered_file_sha256", "delivered_pcm_sha256"
]


class MagicCleanLineageError(ValueError):
    """Known Magic Clean ancestry cannot be resolved without guessing."""


@dataclass(frozen=True, slots=True)
class MagicCleanLineageResolution:
    """Canonical source selected for one submitted Magic Clean input."""

    submitted_url: str
    root_url: str
    matched_by: MagicCleanMatchMethod
    lineage_job_ids: tuple[str, ...] = ()
    matched_job_ids: tuple[str, ...] = ()
    matched_output_urls: tuple[str, ...] = ()

    @property
    def hops(self) -> int:
        return len(self.lineage_job_ids)

    @property
    def is_known_derivative(self) -> bool:
        return self.matched_by != "new_source"


@dataclass(frozen=True, slots=True)
class _ScopedMagicCleanJob:
    job_id: str
    output_url: str
    parent_url: str
    delivered_file_sha256: str | None
    delivered_pcm_sha256: str | None


class MagicCleanLineageResolver:
    @staticmethod
    def extract_enhanced_audio_url(result_json: object) -> str:
        """Return only the exact persisted ``enhanced_audio.audio_url`` value."""
        if not isinstance(result_json, dict):
            return ""
        enhanced_audio = result_json.get("enhanced_audio")
        if not isinstance(enhanced_audio, dict):
            return ""
        value = enhanced_audio.get("audio_url")
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def resolve_magic_clean_lineage(
        submitted_url: str,
        *,
        backend_id: str,
        track_id: str,
        jobs: Iterable[object],
        max_depth: int = MAX_MAGIC_CLEAN_LINEAGE_DEPTH,
    ) -> MagicCleanLineageResolution:
        """Resolve an exact known Magic Clean output URL to its canonical root.

        Only completed Magic Clean jobs in the supplied backend and track scope are
        eligible. An unrecognized URL, including a known output from another scope,
        is treated as a new source revision.
        """
        clean_url, clean_backend_id, clean_track_id = MagicCleanLineageResolver._validated_identity(
            submitted_url, backend_id, track_id, max_depth
        )
        records = MagicCleanLineageResolver._scoped_jobs(jobs, clean_backend_id, clean_track_id)
        return MagicCleanLineageResolver._resolve_exact_url(
            clean_url, submitted_url=clean_url, records=records, max_depth=max_depth
        )

    @staticmethod
    def resolve_magic_clean_hash_alias(
        submitted_url: str,
        *,
        backend_id: str,
        track_id: str,
        jobs: Iterable[object],
        submitted_file_sha256: str | None = None,
        submitted_pcm_sha256: str | None = None,
        max_depth: int = MAX_MAGIC_CLEAN_LINEAGE_DEPTH,
    ) -> MagicCleanLineageResolution:
        """Resolve a byte-identical or decoded-PCM-identical prior output alias.

        Cryptographic hashes are compared only inside the backend and track scope.
        Multiple matching artifacts are safe only when all of their exact URL
        lineages resolve to the same canonical root.
        """
        clean_url, clean_backend_id, clean_track_id = MagicCleanLineageResolver._validated_identity(
            submitted_url, backend_id, track_id, max_depth
        )
        file_sha256 = MagicCleanLineageResolver._normalize_supplied_sha256(
            submitted_file_sha256, "submitted_file_sha256"
        )
        pcm_sha256 = MagicCleanLineageResolver._normalize_supplied_sha256(
            submitted_pcm_sha256, "submitted_pcm_sha256"
        )
        records = MagicCleanLineageResolver._scoped_jobs(jobs, clean_backend_id, clean_track_id)
        matches: dict[str, tuple[_ScopedMagicCleanJob, set[str]]] = {}
        for record in records:
            methods: set[str] = set()
            if file_sha256 and record.delivered_file_sha256 == file_sha256:
                methods.add("delivered_file_sha256")
            if pcm_sha256 and record.delivered_pcm_sha256 == pcm_sha256:
                methods.add("delivered_pcm_sha256")
            if not methods:
                continue
            identity = record.job_id or f"output:{record.output_url}"
            existing = matches.get(identity)
            if existing is None:
                matches[identity] = (record, methods)
            elif existing[0] != record:
                raise MagicCleanLineageError(
                    "Magic Clean hash lineage contains conflicting records for one job"
                )
            else:
                existing[1].update(methods)
        if not matches:
            return MagicCleanLineageResolver._new_source_resolution(clean_url)
        resolved_matches: list[tuple[_ScopedMagicCleanJob, MagicCleanLineageResolution]] = []
        for record, _methods in matches.values():
            if not record.output_url:
                raise MagicCleanLineageError(
                    "Magic Clean hash matched an artifact without a persisted output URL"
                )
            resolved_matches.append(
                (
                    record,
                    MagicCleanLineageResolver._resolve_exact_url(
                        record.output_url,
                        submitted_url=record.output_url,
                        records=records,
                        max_depth=max_depth,
                    ),
                )
            )
        roots = {resolution.root_url for _, resolution in resolved_matches}
        if len(roots) != 1:
            raise MagicCleanLineageError(
                "Magic Clean hash alias is ambiguous across canonical roots"
            )
        selected_record, selected_resolution = min(
            resolved_matches,
            key=lambda item: (item[1].hops, item[1].lineage_job_ids, item[0].output_url),
        )
        del selected_record
        matched_methods = {method for _record, methods in matches.values() for method in methods}
        matched_by: MagicCleanMatchMethod = (
            "delivered_file_sha256"
            if "delivered_file_sha256" in matched_methods
            else "delivered_pcm_sha256"
        )
        matched_records = [record for record, _methods in matches.values()]
        return MagicCleanLineageResolution(
            submitted_url=clean_url,
            root_url=selected_resolution.root_url,
            matched_by=matched_by,
            lineage_job_ids=selected_resolution.lineage_job_ids,
            matched_job_ids=tuple(sorted(record.job_id for record in matched_records)),
            matched_output_urls=tuple(sorted(record.output_url for record in matched_records)),
        )

    @staticmethod
    def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
        """Return a streaming SHA-256 digest of the file's exact bytes."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            while chunk := source.read(chunk_size):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def sha256_decoded_pcm(
        path: str | os.PathLike[str],
        *,
        ffmpeg_executable: str = "ffmpeg",
        ffprobe_executable: str = "ffprobe",
        chunk_size: int = 1024 * 1024,
        timeout_seconds: float = AUDIO_HASH_TIMEOUT_SECONDS,
    ) -> str:
        """Hash deterministic first-stream f32le PCM without loading it into memory.

        The hash domain includes an explicit version, sample rate, and channel count
        before the interleaved decoded samples. This distinguishes identical sample
        bytes interpreted with different audio stream layouts.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        sample_rate, channels = MagicCleanLineageResolver._probe_audio_format(
            path, ffprobe_executable, timeout_seconds
        )
        command = [
            ffmpeg_executable,
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-threads",
            "1",
            "-i",
            os.fspath(path),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ]
        digest = hashlib.sha256()
        digest.update(_PCM_HASH_DOMAIN)
        digest.update(struct.pack(">II", sample_rate, channels))
        decoded_bytes = 0
        with tempfile.TemporaryFile() as stderr:
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)
            except OSError as exc:
                raise RuntimeError("ffmpeg executable is unavailable") from exc
            if process.stdout is None:
                process.kill()
                process.wait()
                raise RuntimeError("ffmpeg did not expose decoded audio output")
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + timeout_seconds
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout_seconds)
                    events = selector.select(timeout=min(30.0, remaining))
                    if not events:
                        if process.poll() is not None:
                            break
                        continue
                    chunk = os.read(process.stdout.fileno(), chunk_size)
                    if not chunk:
                        break
                    digest.update(chunk)
                    decoded_bytes += len(chunk)
                return_code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait()
                raise RuntimeError("ffmpeg timed out while hashing decoded audio") from exc
            except BaseException:
                process.kill()
                process.wait()
                raise
            finally:
                selector.close()
                process.stdout.close()
            if return_code != 0:
                raise RuntimeError("ffmpeg failed to decode audio for deterministic hashing")
        bytes_per_frame = channels * 4
        if decoded_bytes == 0:
            raise ValueError("decoded audio is empty")
        if decoded_bytes % bytes_per_frame:
            raise RuntimeError("ffmpeg returned a partial decoded PCM frame")
        return digest.hexdigest()

    @staticmethod
    def magic_clean_artifact_hashes(path: str | os.PathLike[str]) -> tuple[str, str]:
        """Hash bytes and decoded PCM sequentially in one contained worker."""
        return (
            MagicCleanLineageResolver.sha256_file(path),
            MagicCleanLineageResolver.sha256_decoded_pcm(path),
        )

    @staticmethod
    def _validated_identity(
        submitted_url: str, backend_id: str, track_id: str, max_depth: int
    ) -> tuple[str, str, str]:
        clean_url = str(submitted_url or "").strip()
        clean_backend_id = str(backend_id or "").strip()
        clean_track_id = str(track_id or "").strip()
        if not clean_url:
            raise ValueError("submitted_url is required")
        if not clean_backend_id or not clean_track_id:
            raise ValueError("backend_id and track_id are required for Magic Clean lineage")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        return (clean_url, clean_backend_id, clean_track_id)

    @staticmethod
    def _scoped_jobs(
        jobs: Iterable[object], backend_id: str, track_id: str
    ) -> tuple[_ScopedMagicCleanJob, ...]:
        records: list[_ScopedMagicCleanJob] = []
        for candidate in jobs:
            if str(getattr(candidate, "backend_id", "") or "") != backend_id:
                continue
            if str(getattr(candidate, "track_id", "") or "") != track_id:
                continue
            if str(getattr(candidate, "status", "") or "") != "completed":
                continue
            job_type = str(getattr(candidate, "job_type", "") or "").replace("-", "_")
            if job_type != "magic_clean":
                continue
            options = getattr(candidate, "job_options", None)
            safe_options = options if isinstance(options, dict) else {}
            root_hint = MagicCleanLineageResolver._clean_string(
                safe_options.get(MAGIC_CLEAN_ROOT_URL_KEY)
            )
            parent_url = root_hint or MagicCleanLineageResolver._clean_string(
                getattr(candidate, "input_url", "")
            )
            records.append(
                _ScopedMagicCleanJob(
                    job_id=MagicCleanLineageResolver._clean_string(getattr(candidate, "id", "")),
                    output_url=MagicCleanLineageResolver.extract_enhanced_audio_url(
                        getattr(candidate, "result_json", None)
                    ),
                    parent_url=parent_url,
                    delivered_file_sha256=MagicCleanLineageResolver._normalize_persisted_sha256(
                        safe_options.get(MAGIC_CLEAN_DELIVERED_FILE_SHA256_KEY)
                    ),
                    delivered_pcm_sha256=MagicCleanLineageResolver._normalize_persisted_sha256(
                        safe_options.get(MAGIC_CLEAN_DELIVERED_PCM_SHA256_KEY)
                    ),
                )
            )
        return tuple(records)

    @staticmethod
    def _resolve_exact_url(
        start_url: str,
        *,
        submitted_url: str,
        records: tuple[_ScopedMagicCleanJob, ...],
        max_depth: int,
    ) -> MagicCleanLineageResolution:
        by_output: dict[str, list[_ScopedMagicCleanJob]] = {}
        for record in records:
            if record.output_url:
                by_output.setdefault(record.output_url, []).append(record)
        current_url = start_url
        visited: set[str] = set()
        lineage_job_ids: list[str] = []
        lineage_output_urls: list[str] = []
        while True:
            if current_url in visited:
                raise MagicCleanLineageError("Magic Clean lineage contains a cycle")
            visited.add(current_url)
            candidates = by_output.get(current_url, [])
            if not candidates:
                if not lineage_job_ids:
                    return MagicCleanLineageResolver._new_source_resolution(submitted_url)
                return MagicCleanLineageResolution(
                    submitted_url=submitted_url,
                    root_url=current_url,
                    matched_by="exact_url",
                    lineage_job_ids=tuple(lineage_job_ids),
                    matched_job_ids=(lineage_job_ids[0],),
                    matched_output_urls=tuple(lineage_output_urls),
                )
            if len(candidates) != 1:
                raise MagicCleanLineageError("Magic Clean exact URL lineage is ambiguous")
            if len(lineage_job_ids) >= max_depth:
                raise MagicCleanLineageError("Magic Clean lineage exceeds the safe depth")
            candidate = candidates[0]
            if not candidate.job_id:
                raise MagicCleanLineageError("Magic Clean lineage job is missing its identifier")
            if not candidate.parent_url:
                raise MagicCleanLineageError(
                    "Known Magic Clean output has no persisted root or parent URL"
                )
            lineage_job_ids.append(candidate.job_id)
            lineage_output_urls.append(current_url)
            current_url = candidate.parent_url

    @staticmethod
    def _new_source_resolution(submitted_url: str) -> MagicCleanLineageResolution:
        return MagicCleanLineageResolution(
            submitted_url=submitted_url, root_url=submitted_url, matched_by="new_source"
        )

    @staticmethod
    def _normalize_supplied_sha256(value: str | None, name: str) -> str | None:
        if value is None or not str(value).strip():
            return None
        normalized = str(value).strip().lower()
        if not MagicCleanLineageResolver._is_sha256(normalized):
            raise ValueError(f"{name} must be a SHA-256 hex digest")
        return normalized

    @staticmethod
    def _normalize_persisted_sha256(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        return normalized if MagicCleanLineageResolver._is_sha256(normalized) else None

    @staticmethod
    def _is_sha256(value: str) -> bool:
        return len(value) == _SHA256_HEX_LENGTH and all(
            character in "0123456789abcdef" for character in value
        )

    @staticmethod
    def _clean_string(value: object) -> str:
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _probe_audio_format(
        path: str | os.PathLike[str], ffprobe_executable: str, timeout_seconds: float
    ) -> tuple[int, int]:
        command = [
            ffprobe_executable,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            os.fspath(path),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, check=False, text=True, timeout=timeout_seconds
            )
        except OSError as exc:
            raise RuntimeError("ffprobe executable is unavailable") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("ffprobe timed out while inspecting audio") from exc
        if completed.returncode != 0:
            raise RuntimeError("ffprobe failed to inspect audio for deterministic hashing")
        try:
            payload = json.loads(completed.stdout)
            stream = payload["streams"][0]
            sample_rate = int(stream["sample_rate"])
            channels = int(stream["channels"])
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("audio has no valid first audio stream") from exc
        if sample_rate <= 0 or channels <= 0:
            raise ValueError("audio stream has invalid sample rate or channels")
        return (sample_rate, channels)
