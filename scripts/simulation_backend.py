"""Local-only fake backend for end-to-end tests with real Hear model workers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse

from hear.contracts.jobs import AttemptEnvelope
from hear.contracts.outcomes import ExecutionOutcome
from hear.contracts.scope import ExecutionScope
from scripts.simulation_multipart import SimulationMultipart


class SimulationBackend:
    def __init__(self, root: Path):
        self.root = root
        self.multipart = SimulationMultipart(root)
        self.config = json.loads((root / "config.json").read_text())
        self.lock = asyncio.Lock()
        saved = root / "state.json"
        state = json.loads(saved.read_text()) if saved.is_file() else {}
        self.attempts = state.get("attempts", {})
        self.previous_max_active = state.get("max_active", 0)
        self.max_active = sum(self.lease_live(item) for item in self.attempts.values())
        self.app = FastAPI(title="Hear local simulation — not production")
        self.app.add_api_route("/simulation/register", self.register, methods=["POST"])
        self.app.add_api_route("/simulation/summary", self.summary, methods=["GET"])
        self.app.add_api_route("/source/{name}", self.source, methods=["GET"])
        self.app.add_api_route("/api/v1/sse/tracks/{track_id}/events", self.sse, methods=["GET"])
        self.app.add_api_route(
            "/api/v1/internal/ai/runtime/protocol", self.protocol, methods=["GET"]
        )
        self.app.add_api_route(
            "/api/v1/internal/ai/runtime/catalog", self.catalogue, methods=["GET"]
        )
        self.app.add_api_route(
            "/api/v1/internal/ai/attempts/{attempt_id}/{action}", self.callback, methods=["POST"]
        )
        self.app.add_api_route(
            "/s3/{bucket}/{key:path}", self.object, methods=["PUT", "HEAD", "GET", "POST", "DELETE"]
        )

    @staticmethod
    def lease_live(item: dict) -> bool:
        last = item.get("last_heartbeat", item.get("claimed_at", 0))
        return item.get("status") == "running" and time.time() - float(last) < 90

    def authenticate(self, request: Request):
        if not secrets.compare_digest(
            request.headers.get("x-service-key", ""), self.config["service_key"]
        ):
            raise HTTPException(403, "simulation_test_key_required")

    @staticmethod
    def safe(value: str):
        if any(p in ("", ".", "..") for p in value.split("/")) or "\\" in value:
            raise HTTPException(400, "invalid_local_path")
        return value

    async def register(self, request: Request):
        self.authenticate(request)
        envelope = AttemptEnvelope.model_validate(await request.json())
        if envelope.backend_id != "simulation-local":
            raise HTTPException(403, "not_a_simulation_job")
        async with self.lock:
            if envelope.attempt_id in self.attempts:
                raise HTTPException(409, "already_registered")
            self.attempts[envelope.attempt_id] = {
                "envelope": envelope.model_dump(mode="json"),
                "scope": ExecutionScope.digest(envelope.model_dump(mode="json")),
                "status": "pending",
                "events": [],
                "claims": 0,
                "sse": [],
            }
            self.record_sse(self.attempts[envelope.attempt_id], "submitted", {})
        return {"registered": True, "attempt_id": envelope.attempt_id}

    @staticmethod
    def record_sse(item: dict, event: str, data: dict) -> None:
        history = item.setdefault("sse", [])
        history.append({"id": len(history) + 1, "event": event, "data": data})

    async def sse(self, track_id: str, request: Request):
        self.authenticate(request)
        item = next(
            (
                value
                for value in self.attempts.values()
                if value["envelope"]["track_id"] == track_id
            ),
            None,
        )
        if item is None:
            raise HTTPException(404, "simulation_track_not_registered")
        try:
            cursor = int(request.headers.get("last-event-id", "0"))
        except ValueError:
            raise HTTPException(400, "invalid_last_event_id") from None

        async def stream():
            nonlocal cursor
            while not await request.is_disconnected():
                async with self.lock:
                    events = [value.copy() for value in item.get("sse", []) if value["id"] > cursor]
                    terminal = item["status"] in {"completed", "failed", "cancelled"}
                for value in events:
                    cursor = value["id"]
                    yield f"id: {cursor}\nevent: {value['event']}\ndata: {json.dumps(value['data'])}\n\n"
                if terminal:
                    return
                if not events:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.2)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def source(self, name: str):
        path = self.root / "sources" / self.safe(name)
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, "source_not_found")
        return FileResponse(path)

    async def protocol(self, request: Request):
        self.authenticate(request)
        return {
            "schema_version": 1,
            "backend_id": "simulation-local",
            "attempt_control": True,
            "simulation": True,
        }

    async def catalogue(self, request: Request):
        self.authenticate(request)
        return {
            "version": 1,
            "categories": ["Local News", "Community", "Arts", "Sport", "Health"],
            "tags": ["#community", "#local-news"],
            "keyword_rules": {"news": "#local-news"},
            "harm_keywords": [],
            "taxonomy_paths": ["Local News", "Community", "Arts", "Sport", "Health"],
        }

    async def callback(self, attempt_id: str, action: str, request: Request):
        payload = await request.json()
        async with self.lock:
            item = self.attempts.get(attempt_id)
            if item is None:
                raise HTTPException(404, "unregistered_fake_job")
            envelope = item["envelope"]
            if not secrets.compare_digest(
                request.headers.get("x-ai-attempt-grant", ""), envelope["reporting_grant"]
            ):
                raise HTTPException(403, "invalid_test_grant")
            if action == "claim":
                if payload.get("request_scope_sha256") != item["scope"]:
                    raise HTTPException(403, "request_scope_mismatch")
                if item["status"] == "completed":
                    return {"decision": "already_completed"}
                if item["status"] in {"failed", "cancelled"}:
                    return {"decision": "cancelled"}
                if self.lease_live(item):
                    return {"decision": "lease_unavailable"}
                if item["status"] == "running":
                    item["expired_leases"] = item.get("expired_leases", 0) + 1
                item.update(
                    status="running",
                    worker=payload["worker_id"],
                    generation=payload["generation"],
                    claimed_at=time.time(),
                    last_heartbeat=time.time(),
                    claims=item["claims"] + 1,
                )
                self.max_active = max(
                    self.max_active, sum(self.lease_live(x) for x in self.attempts.values())
                )
                self.record_sse(item, "processing", {"attempt_id": attempt_id})
                self.persist()
                return {"decision": "execute", "lease_seconds": 90, "heartbeat_seconds": 15}
            if request.headers.get("x-ai-worker-id") != item.get("worker") or request.headers.get(
                "x-ai-worker-generation"
            ) != item.get("generation"):
                raise HTTPException(403, "worker_identity_mismatch")
            if action == "heartbeat":
                item["last_heartbeat"] = time.time()
                item["heartbeats"] = item.get("heartbeats", 0) + 1
                return {"accepted": True}
            if action == "events":
                item["events"].append(
                    {k: payload.get(k) for k in ("event", "stage", "sequence", "progress_pct")}
                )
                self.record_sse(item, payload.get("event", "progress"), item["events"][-1])
                self.persist()
                return {"accepted": True}
            if action != "outcome":
                raise HTTPException(404, "unknown_callback")
            outcome = ExecutionOutcome.model_validate(payload)
            if (
                any(
                    str(payload.get(k)) != str(envelope.get(k))
                    for k in ("job_id", "attempt_id", "track_id", "job_type")
                )
                or outcome.source_revision != envelope["source"]["revision"]
            ):
                raise HTTPException(403, "outcome_identity_mismatch")
            verified = []
            for artifact in outcome.artifacts:
                if artifact.bucket_name != self.config[
                    "bucket"
                ] or not artifact.object_key.startswith(envelope["artifact_prefix"] + "/"):
                    raise HTTPException(403, "artifact_scope_mismatch")
                path = self.root / "objects" / self.safe(artifact.object_key)
                if not path.is_file():
                    raise HTTPException(422, "missing_uploaded_artifact")
                with path.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != artifact.sha256 or path.stat().st_size != artifact.size_bytes:
                    raise HTTPException(422, "artifact_readback_integrity_failed")
                verified.append(
                    {"key": artifact.object_key, "sha256": actual, "bytes": path.stat().st_size}
                )
            item.update(
                status=outcome.status, completed_at=time.time(), outcome=payload, readback=verified
            )
            self.record_sse(
                item, str(outcome.status), {"attempt_id": attempt_id, "status": str(outcome.status)}
            )
            self.persist()
            return {"accepted": True, "simulation": True}

    def persist(self):
        state = self.root / "state.json.part"
        state.write_text(json.dumps({"attempts": self.attempts, "max_active": self.max_active}))
        state.chmod(0o600)
        state.replace(self.root / "state.json")
        result = {
            "simulation": True,
            "real_models": True,
            "cloud_storage_used": False,
            "max_simultaneously_claimed": self.max_active,
            "active_leases": sum(self.lease_live(x) for x in self.attempts.values()),
            "prior_process_reported_peak": self.previous_max_active,
            "peak_scope": "valid leases since simulation backend startup",
            "attempts": [],
        }
        for identity, item in self.attempts.items():
            result["attempts"].append(
                {
                    "attempt_id": identity,
                    "job_type": item["envelope"]["job_type"],
                    **{k: v for k, v in item.items() if k not in ("envelope", "scope")},
                }
            )
        temporary = self.root / "summary.json.part"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.root / "summary.json")

    async def summary(self, request: Request):
        self.authenticate(request)
        async with self.lock:
            self.persist()
            return json.loads((self.root / "summary.json").read_text())

    async def object(self, bucket: str, key: str, request: Request):
        if bucket != self.config["bucket"]:
            raise HTTPException(403, "simulation_bucket_mismatch")
        # Loopback-only minimal S3 transport emulator, not AWS authentication.
        if (
            request.method != "GET"
            and "Credential=HEAR_SIMULATION_ONLY/" not in request.headers.get("authorization", "")
        ):
            raise HTTPException(403, "fake_s3_credential_required")
        path = self.root / "objects" / self.safe(key)
        if "uploads" in request.query_params or "uploadId" in request.query_params:
            return await self.multipart.handle(request, bucket, key, path)
        metadata = self.root / "metadata" / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        if request.method == "PUT":
            if "aws-chunked" in request.headers.get("content-encoding", ""):
                raise HTTPException(400, "simulation_requires_unencoded_body")
            path.parent.mkdir(parents=True, exist_ok=True)
            size = 0
            with path.open("wb") as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 64 * 1024**2:
                        raise HTTPException(413, "simulation_object_limit")
                    output.write(chunk)
            body = {
                "size": size,
                "content_type": request.headers.get("content-type", "application/octet-stream"),
                "sha256": request.headers.get("x-amz-meta-sha256", ""),
            }
            metadata.parent.mkdir(exist_ok=True)
            metadata.write_text(json.dumps(body))
            return Response(status_code=200, headers={"ETag": '"simulation-object"'})
        if not path.is_file() or not metadata.is_file():
            return Response(status_code=404)
        details = json.loads(metadata.read_text())
        headers = {
            "Content-Length": str(details["size"]),
            "Content-Type": details["content_type"],
            "x-amz-meta-sha256": details["sha256"],
            "ETag": '"simulation-object"',
        }
        if request.method == "HEAD":
            return Response(headers=headers)
        return FileResponse(path, headers=headers)

    @classmethod
    def main(cls):
        root = Path(os.environ["HEAR_SIMULATION_ROOT"])
        uvicorn.run(
            cls(root).app,
            host="127.0.0.1",
            port=18081,
            log_level="warning",
            ssl_keyfile=str(root / "tls/key.pem"),
            ssl_certfile=str(root / "tls/cert.pem"),
        )


if __name__ == "__main__":
    SimulationBackend.main()
