from __future__ import annotations

import asyncio
import hmac
import json
import os
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.background import BackgroundTask

from hear.contracts.cleaning import CleaningProfiles
from hear.contracts.jobs import AttemptEnvelope
from hear.runtime.gateway import (
    GatewayAttempt,
    GatewayDeadlineExpired,
    GatewayQueueFull,
    GatewayUnavailable,
    RabbitMQGateway,
)
from hear.runtime.ownership import BackendOwnershipPolicy, BackendRegistry
from hear.runtime.simulation import SimulationBoundary


class PodGateway:
    def __init__(
        self,
        runtime: RabbitMQGateway,
        api_key: str,
        *,
        enable_docs: bool = False,
        ownership_policy: BackendOwnershipPolicy | BackendRegistry | None = None,
        cleaning_mode: str = "available",
        sound_cleanup_available: bool = False,
        overlap_preview_available: bool = False,
    ) -> None:
        self._simulation = SimulationBoundary.enabled()
        self._runtime = runtime
        self._api_key = api_key.strip()
        self.enable_docs = enable_docs
        self._ownership = ownership_policy
        self._cleaning_mode = cleaning_mode
        self._sound_cleanup_available = sound_cleanup_available
        self._overlap_preview_available = overlap_preview_available
        self.router = APIRouter(tags=["gateway"])
        self.router.add_api_route(
            "/v1/attempts", self.submit_attempt, methods=["POST"], status_code=202
        )
        self.router.add_api_route("/v1/attempts/stream", self.stream_attempt, methods=["POST"])
        self.router.add_api_route("/healthz", self.healthz, methods=["GET"])
        self.router.add_api_route("/readyz", self.readyz, methods=["GET"])
        self.router.add_api_route("/capabilities", self.capabilities, methods=["GET"])
        self.router.add_api_route("/metrics", self.metrics, methods=["GET"])
        self.router.add_api_route("/drain", self.drain, methods=["POST"])

    async def healthz(self) -> dict[str, str]:
        return {"status": "healthy"}

    async def readyz(self):
        lanes = await self._runtime.lane_status()
        ready = bool(lanes) and all(item["status"] == "ready" for item in lanes.values())
        return JSONResponse(
            {
                "status": "ready" if ready else "loading",
                "runtime_mode": "simulation" if self._simulation else "production",
                "lanes": lanes,
            },
            status_code=200 if ready else 503,
        )

    async def capabilities(self) -> dict:
        lanes = await self._runtime.lane_status()
        ready = bool(lanes) and all(item["status"] == "ready" for item in lanes.values())
        catalogue = CleaningProfiles.catalogue()
        natural_lane = lanes.get("magic_clean_natural")
        available = natural_lane is not None and natural_lane["status"] == "ready"
        for name, profile in catalogue["profiles"].items():
            profile["available"] = available and (
                self._cleaning_mode == "available" or name == "natural"
            )
        catalogue["engine_mode"] = self._cleaning_mode
        catalogue["sound_cleanup"] = {
            "available": available
            and self._cleaning_mode == "available"
            and self._sound_cleanup_available,
            "version": "sound-cleanup-local-v1",
            "enabled_by_default": False,
            "targets": ["handling", "impact", "animal", "cough", "click"],
            "supports_selected_regions": True,
            "preserves_stereo": True,
            "preserves_duration": True,
            "requires_approval": True,
            "overlapping_speech_repair": False,
            "selected_overlap_preview": self._overlap_preview_available
            and self._sound_cleanup_available
            and available
            and self._cleaning_mode == "available",
            "coughs_require_explicit_consent": True,
        }
        return {
            "status": "ready" if ready else "loading",
            "lanes": lanes,
            "runtime_mode": "simulation" if self._simulation else "production",
            "backend_type": "simulated_local" if self._simulation else "external",
            "concurrency": {
                "host_total": int(os.environ.get("HEAR_HOST_MAX_CONCURRENT_JOBS", "1")),
                "per_role_worker": int(os.environ.get("HEAR_POD_MAX_CONCURRENT_JOBS", "1")),
            },
            "magic_clean": catalogue,
            "reconstruction": {
                "engine": "fish_speech_s2_pro",
                "mode": "text_to_speech_editing",
                "available": bool(lanes.get("reconstruction"))
                and lanes["reconstruction"]["status"] == "ready",
                "max_concurrent_jobs_per_worker": 1,
                "requires_approval": True,
                "input_audio_replacement_required": False,
                "source_output_timeline": True,
                "operations": [
                    "replace_segments",
                    "edit_transcript",
                    "rebuild",
                    "remove_segments",
                    "preview",
                ],
            },
        }

    async def drain(self, authorization: str | None = Header(default=None)) -> dict[str, str]:
        self._authenticate(authorization)
        await self._runtime.drain()
        return {"status": "draining"}

    async def metrics(self) -> PlainTextResponse:
        lanes = await self._runtime.lane_status()
        lines = ["# TYPE hear_gateway_ready gauge"]
        ready = bool(lanes) and all(item["status"] == "ready" for item in lanes.values())
        lines.append(f"hear_gateway_ready {int(ready)}")
        lines.append("# TYPE hear_queue_messages gauge")
        for role, item in lanes.items():
            lines.append(f'hear_queue_messages{{role="{role}"}} {int(item["queued"])}')
        lines.append("# TYPE hear_queue_consumers gauge")
        for role, item in lanes.items():
            lines.append(f'hear_queue_consumers{{role="{role}"}} {int(item["consumers"])}')
        scratch = Path(os.environ.get("HEAR_TEMP_DIR", "/tmp"))
        try:
            free_bytes = shutil.disk_usage(scratch).free
        except OSError:
            free_bytes = 0
        lines.extend(
            ("# TYPE hear_scratch_free_bytes gauge", f"hear_scratch_free_bytes {free_bytes}")
        )
        gpu = await self._gpu_memory()
        if gpu is not None:
            used, free = gpu
            lines.extend(
                (
                    "# TYPE hear_gpu_memory_used_bytes gauge",
                    f"hear_gpu_memory_used_bytes {used}",
                    "# TYPE hear_gpu_memory_free_bytes gauge",
                    f"hear_gpu_memory_free_bytes {free}",
                )
            )
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    @staticmethod
    async def _gpu_memory() -> tuple[int, int] | None:
        try:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=2.0)
            if process.returncode != 0:
                return None
            first = stdout.decode().splitlines()[0]
            used_mib, free_mib = (int(value.strip()) for value in first.split(",", 1))
            return used_mib * 1024 * 1024, free_mib * 1024 * 1024
        except (OSError, ValueError, IndexError, TimeoutError):
            return None

    def validate_request(self, envelope: AttemptEnvelope, authorization: str | None) -> None:
        if isinstance(self._ownership, BackendRegistry):
            try:
                self._ownership.authenticate(envelope, authorization)
            except ValueError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
        else:
            self._authenticate(authorization)
        if self._ownership is not None:
            try:
                self._ownership.validate(envelope)
            except ValueError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
        sound = envelope.options.get("sound_cleanup", {})
        if (
            envelope.job_type.value == "magic_clean"
            and sound.get("enabled")
            and (not self._sound_cleanup_available or self._cleaning_mode != "available")
        ):
            raise HTTPException(status_code=503, detail="sound_cleanup_not_provisioned")
        if (
            envelope.job_type.value == "magic_clean"
            and sound.get("preview_overlaps")
            and not self._overlap_preview_available
        ):
            raise HTTPException(status_code=503, detail="overlap_separator_not_provisioned")
        if envelope.options.get("reduce_stationary_noise") and not self._sound_cleanup_available:
            raise HTTPException(status_code=503, detail="background_analyser_not_provisioned")
        if envelope.job_type.value == "magic_clean" and self._cleaning_mode != "available":
            if envelope.options.get("profile") != "natural" or any(
                envelope.options.get(key)
                for key in (
                    "auto_level",
                    "remove_clicks",
                    "trim_silence",
                    "reduce_stationary_noise",
                )
            ):
                raise HTTPException(status_code=422, detail="preset_requires_available_engine_mode")

    async def submit_attempt(
        self, envelope: AttemptEnvelope, authorization: str | None = Header(default=None)
    ) -> JSONResponse:
        self.validate_request(envelope, authorization)
        try:
            await self._runtime.submit(envelope)
        except GatewayQueueFull as exc:
            raise HTTPException(
                status_code=429, detail=str(exc), headers={"Retry-After": "5"}
            ) from exc
        except GatewayDeadlineExpired as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GatewayUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return JSONResponse(
            status_code=202,
            content={
                "schema_version": 1,
                "status": "accepted",
                "backend_id": envelope.backend_id,
                "job_id": envelope.job_id,
                "run_id": envelope.run_id,
                "attempt_id": envelope.attempt_id,
                "track_id": envelope.track_id,
                "source_revision": envelope.source.revision,
                "result_delivery": "owning_backend_callback",
                "duplicate_policy": "backend_claim_is_authoritative",
            },
        )

    async def stream_attempt(
        self, envelope: AttemptEnvelope, authorization: str | None = Header(default=None)
    ) -> StreamingResponse:
        self.validate_request(envelope, authorization)
        try:
            attempt = await self._runtime.enqueue(envelope)
        except GatewayQueueFull as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
                headers={"Retry-After": "5"},
            ) from exc
        except GatewayDeadlineExpired as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except GatewayUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return StreamingResponse(
            self._events(attempt),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
            background=BackgroundTask(attempt.close),
        )

    def _authenticate(self, authorization: str | None) -> None:
        scheme, _, token = (authorization or "").partition(" ")
        if not self._api_key:
            raise HTTPException(status_code=503, detail="pod_api_key_not_configured")
        if scheme.lower() != "bearer" or not token or not hmac.compare_digest(token, self._api_key):
            raise HTTPException(
                status_code=401,
                detail="invalid_pod_api_key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def _events(self, attempt: GatewayAttempt) -> AsyncIterator[str]:
        yield self._encode(
            "queued",
            {
                "job_id": attempt.envelope.job_id,
                "attempt_id": attempt.envelope.attempt_id,
                "track_id": attempt.envelope.track_id,
                "queue": "accepted",
            },
        )
        iterator = self._runtime.stream(attempt)
        next_event = asyncio.ensure_future(iterator.__anext__())
        try:
            while True:
                done, _ = await asyncio.wait({next_event}, timeout=15.0)
                if not done:
                    yield ": keep-alive\n\n"
                    continue
                try:
                    payload = next_event.result()
                except StopAsyncIteration:
                    return
                if payload.get("kind") == "event":
                    event = payload["data"]
                    yield self._encode(event["event"], event, event.get("event_id"))
                else:
                    yield self._encode(payload.get("event", "error"), payload.get("data", {}))
                next_event = asyncio.ensure_future(iterator.__anext__())
        finally:
            if not next_event.done():
                next_event.cancel()
                await asyncio.gather(next_event, return_exceptions=True)
            await iterator.aclose()
            await attempt.close()

    @staticmethod
    def _encode(event_name: str, data: dict, event_id: str | None = None) -> str:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        lines = []
        if event_id:
            lines.append(f"id: {event_id}")
        lines.append(f"event: {event_name}")
        lines.append(f"data: {payload}")
        return "\n".join(lines) + "\n\n"
