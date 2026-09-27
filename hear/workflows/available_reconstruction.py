from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from pathlib import Path

from hear.audio.io import AudioIO
from hear.audio.workspace import AudioWorkspace
from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope, ReconstructionOperation
from hear.contracts.outcomes import ExecutionOutcome
from hear.execution.native import NativeExecutor
from hear.storage.b2 import B2StorageFactory


class AvailableReconstructionWorkflow:
    def __init__(
        self,
        audio: AudioIO,
        storage_factory: B2StorageFactory,
        native: NativeExecutor,
        *,
        workspace_root: Path,
        timeout_seconds: float,
    ) -> None:
        self._audio = audio
        self._storage_factory = storage_factory
        self._native = native
        self._workspace_root = workspace_root
        self._timeout_seconds = timeout_seconds

    async def stream(self, envelope: AttemptEnvelope):
        if envelope.operation is None:
            raise ValueError("reconstruction_operation_required")
        workspace = AudioWorkspace(self._workspace_root, envelope.job_id, envelope.attempt_id)
        sequence = 1
        yield self._event(envelope, sequence, "preparing", 0, ExecutionEventType.STARTED)
        sequence += 1
        try:
            source = await self._audio.download_source(str(envelope.source.url), workspace)
            yield self._event(envelope, sequence, "rendering", 20, ExecutionEventType.STAGE)
            sequence += 1
            output = workspace.file("reconstructed.mp3")
            segments = await self._render(envelope, source, output, workspace)
            digest = await self._native.run(self._sha256, output)
            duration = await self._native.run(self._duration, output)
            storage = self._storage_factory.create(envelope.storage)
            artifact = await self._native.run(
                storage.upload_file,
                output,
                storage.key("jobs", envelope.job_id, envelope.attempt_id, "reconstructed.mp3"),
                sha256=digest,
                content_type="audio/mpeg",
            )
            result = {
                "engine": "ffmpeg_timeline_v1",
                "operation": envelope.operation.value,
                "requires_approval": True,
                "duration": duration,
                "segments": segments,
                "reconstructed_audio": artifact.model_dump(mode="json"),
            }
            manifest = await self._native.run(
                storage.upload_json,
                result,
                storage.key("jobs", envelope.job_id, envelope.attempt_id, "reconstruction.json"),
            )
            outcome = ExecutionOutcome(
                job_id=envelope.job_id,
                attempt_id=envelope.attempt_id,
                track_id=envelope.track_id,
                job_type=envelope.job_type,
                source_revision=envelope.source.revision,
                status="completed",
                artifacts=(artifact, manifest),
                result=result,
            )
            yield self._event(
                envelope,
                sequence,
                "completed",
                100,
                ExecutionEventType.OUTCOME,
                {"outcome": outcome.model_dump(mode="json")},
            )
        finally:
            workspace.cleanup()

    async def _render(
        self,
        envelope: AttemptEnvelope,
        source: Path,
        output: Path,
        workspace: AudioWorkspace,
    ) -> list[dict]:
        operation = envelope.operation
        changes: list[dict]
        if operation == ReconstructionOperation.REMOVE_SEGMENTS:
            changes = [
                {
                    "segment_start": envelope.options.get("segment_start"),
                    "segment_end": envelope.options.get("segment_end"),
                    "is_deletion": True,
                }
            ]
        elif operation in {
            ReconstructionOperation.REPLACE_SEGMENTS,
            ReconstructionOperation.EDIT_TRANSCRIPT,
            ReconstructionOperation.PREVIEW,
        }:
            raw_changes = envelope.options.get("changes")
            if not isinstance(raw_changes, list) or not raw_changes:
                raise ValueError("reconstruction_changes_required")
            changes = raw_changes
        elif operation == ReconstructionOperation.REBUILD:
            replacement_url = str(envelope.options.get("rendered_audio_url") or "").strip()
            if not replacement_url:
                raise ValueError("rendered_audio_url_required")
            replacement = await self._audio.download_source(
                replacement_url,
                workspace,
                name="rendered.audio",
            )
            await self._audio.encode_mp3(replacement, output, maximum_kbps=96)
            return []
        else:
            raise ValueError("unsupported_reconstruction_operation")
        return await self._splice(source, output, changes, workspace)

    async def _splice(
        self,
        source: Path,
        output: Path,
        raw_changes: list,
        workspace: AudioWorkspace,
    ) -> list[dict]:
        duration = await self._native.run(self._duration, source)
        changes = []
        for index, raw in enumerate(raw_changes):
            if not isinstance(raw, dict):
                raise ValueError("invalid_reconstruction_change")
            try:
                start = float(raw["segment_start"])
                end = float(raw["segment_end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid_reconstruction_interval") from exc
            if start < 0 or end <= start or end > duration:
                raise ValueError("invalid_reconstruction_interval")
            replacement_url = str(raw.get("replacement_audio_url") or "").strip()
            deletion = bool(raw.get("is_deletion"))
            if not deletion and not replacement_url:
                raise ValueError("replacement_audio_url_required")
            changes.append((start, end, deletion, replacement_url, index))
        changes.sort(key=lambda item: item[0])
        if any(
            current[0] < previous[1]
            for previous, current in zip(changes, changes[1:], strict=False)
        ):
            raise ValueError("overlapping_reconstruction_intervals")
        inputs = [source]
        replacements: dict[int, int] = {}
        for _start, _end, deletion, url, index in changes:
            if deletion:
                continue
            replacement = await self._audio.download_source(
                url,
                workspace,
                name=f"replacement-{index}.audio",
            )
            replacements[index] = len(inputs)
            inputs.append(replacement)
        filters = []
        labels = []
        cursor = 0.0
        part = 0
        result = []
        format_filter = "aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo"
        for start, end, deletion, _url, index in changes:
            if start > cursor:
                label = f"p{part}"
                filters.append(
                    f"[0:a]atrim=start={cursor:.6f}:end={start:.6f},"
                    f"asetpts=PTS-STARTPTS,{format_filter}[{label}]"
                )
                labels.append(label)
                part += 1
            if not deletion:
                label = f"p{part}"
                filters.append(
                    f"[{replacements[index]}:a]asetpts=PTS-STARTPTS,{format_filter}[{label}]"
                )
                labels.append(label)
                part += 1
            result.append(
                {
                    "segment_start": start,
                    "segment_end": end,
                    "is_deletion": deletion,
                }
            )
            cursor = end
        if cursor < duration:
            label = f"p{part}"
            filters.append(
                f"[0:a]atrim=start={cursor:.6f},asetpts=PTS-STARTPTS,{format_filter}[{label}]"
            )
            labels.append(label)
        if not labels:
            raise ValueError("reconstruction_would_remove_all_audio")
        filters.append("".join(f"[{label}]" for label in labels) + f"concat=n={len(labels)}:v=0:a=1[out]")
        await self._native.run(self._run_splice, inputs, output, ";".join(filters))
        return result

    def _run_splice(self, inputs: list[Path], output: Path, graph: str) -> None:
        command = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
        for item in inputs:
            command.extend(("-i", str(item)))
        command.extend(
            (
                "-filter_complex",
                graph,
                "-map",
                "[out]",
                "-b:a",
                "96k",
                str(output),
            )
        )
        try:
            subprocess.run(
                command,
                capture_output=True,
                check=True,
                timeout=self._timeout_seconds,
            )
        except BaseException:
            output.unlink(missing_ok=True)
            raise

    def _duration(self, path: Path) -> float:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=self._timeout_seconds,
        )
        duration = float((json.loads(completed.stdout).get("format") or {}).get("duration") or 0)
        if duration <= 0:
            raise ValueError("invalid_audio_duration")
        return round(duration, 6)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _event(
        envelope: AttemptEnvelope,
        sequence: int,
        stage: str,
        progress: float,
        event_type: ExecutionEventType,
        data: dict | None = None,
    ) -> ExecutionEvent:
        return ExecutionEvent(
            event_id=str(uuid.uuid4()),
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            job_type=envelope.job_type,
            source_revision=envelope.source.revision,
            sequence=sequence,
            event=event_type,
            stage=stage,
            progress_pct=progress,
            data=data or {},
        )
