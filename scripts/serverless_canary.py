"""Submit one attempt to a RunPod Serverless endpoint and report what the worker did.

With a backend-issued envelope this proves the full path: claim, execution,
events and outcome callbacks. With a synthetic envelope it proves the image
boots, models preload, the handler runs and the worker reaches the backend
(the backend then rejects the unknown attempt, which is the expected outcome).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values

from hear.contracts.jobs import AttemptEnvelope
from hear.runtime.roles import WorkerRole


class ServerlessCanary:
    """Minimal RunPod client: the backend owns real dispatch (see docs/GO_AI_DISPATCH_PLAN.md)."""

    DEFAULT_BASE_URL = "https://api.runpod.ai/v2"
    TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}

    def __init__(self, api_key: str, *, base_url: str = DEFAULT_BASE_URL) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key.strip()}"}

    async def _get(self, path: str) -> dict:
        response = await self._client.get(f"{self._base_url}/{path}", headers=self._headers)
        response.raise_for_status()
        return response.json()

    async def _post(self, path: str, payload: dict) -> dict:
        response = await self._client.post(
            f"{self._base_url}/{path}", json=payload, headers=self._headers
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def synthetic_envelope(policy: dict[str, Any], job_type: str) -> AttemptEnvelope:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        job_id = f"canary-{stamp}"
        return AttemptEnvelope.model_validate(
            {
                "job_id": job_id,
                "run_id": f"{job_id}-run",
                "attempt_id": f"{job_id}-attempt",
                "job_type": job_type,
                "track_id": f"{job_id}-track",
                "user_id": "canary",
                "source": {
                    "url": f"https://{policy['source_hosts'][0]}/canary/source.mp3",
                    "revision": 1,
                    "file_sha256": "0" * 64,
                },
                "storage": {
                    "endpoint_url": policy["storage_endpoint"],
                    "bucket_name": policy["bucket_name"],
                    "key_id": "canary",
                    "application_key": "canary",
                    "folder_prefix": f"creators/canary/audio/jobs/{job_id}/",
                    "public_base_url": policy["public_base_url"],
                    "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                },
                "operation": "remove_segments" if job_type == "reconstruction" else None,
                "options": {
                    "magic_clean": {"profile": "natural"},
                    "reconstruction": {"same_speaker": False, "segment_start": 0.0, "segment_end": 1.0},
                }.get(job_type, {}),
                "artifact_prefix": f"creators/canary/audio/jobs/{job_id}/{job_id}-attempt",
                "deadline": (datetime.now(UTC) + timedelta(minutes=20)).isoformat(),
                "reporting_grant": "canary-grant-not-issued-by-backend",
                "backend_base_url": policy["backend_base_urls"][0],
                "backend_id": policy["backend_id"],
            }
        )

    async def run(
        self, endpoint_id: str, role: WorkerRole, envelope: AttemptEnvelope, wait_seconds: float
    ) -> dict:
        health_before = await self._get(f"{endpoint_id}/health")
        started = time.perf_counter()
        receipt = await self._post(f"{endpoint_id}/run", {"input": envelope.model_dump(mode="json")})
        provider_job_id = str(receipt.get("id") or "")
        if not provider_job_id:
            raise RuntimeError("serverless_receipt_missing_job_id")
        states: list[str] = []
        stream: list[dict[str, Any]] = []
        final: dict | None = None
        while time.perf_counter() - started < wait_seconds:
            status = await self._get(f"{endpoint_id}/status/{provider_job_id}")
            state = str(status.get("status") or "UNKNOWN")
            if not states or states[-1] != state:
                states.append(state)
            response = await self._client.get(
                f"{self._base_url}/{endpoint_id}/stream/{provider_job_id}",
                headers=self._headers,
            )
            if response.status_code == 200:
                for item in response.json().get("stream", []):
                    stream.append(item.get("output", item))
            if state in self.TERMINAL:
                final = status
                break
            await asyncio.sleep(3)
        await self._client.aclose()
        return {
            "endpoint_id": endpoint_id,
            "role": role.value,
            "provider_job_id": provider_job_id,
            "attempt_id": envelope.attempt_id,
            "states": states,
            "final_state": str(final.get("status")) if final else "timeout",
            "error": str(final.get("error")) if final and final.get("error") else None,
            "worker_events": [item.get("event") for item in stream],
            "stream": stream[:20],
            "seconds": round(time.perf_counter() - started, 1),
            "health_before": health_before,
        }

    @classmethod
    def main(cls) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--endpoint-id", required=True)
        parser.add_argument("--role", choices=[r.value for r in WorkerRole], required=True)
        parser.add_argument("--envelope", type=Path, help="backend-issued AttemptEnvelope JSON")
        parser.add_argument(
            "--policy-env", type=Path, default=Path("/root/hear-ai-config/production.env")
        )
        parser.add_argument(
            "--api-key-file", type=Path, default=Path("/root/hear-ai-config/runpod-api.key")
        )
        parser.add_argument("--timeout", type=float, default=900)
        parser.add_argument("--output", type=Path)
        parser.add_argument(
            "--synthetic",
            action="store_true",
            help="pass when the worker ran and the backend refused the unknown attempt",
        )
        args = parser.parse_args()
        role = WorkerRole(args.role)
        if args.envelope:
            envelope = AttemptEnvelope.model_validate_json(args.envelope.read_text())
        else:
            raw_policy = os.environ.get("HEAR_BACKEND_POLICY_JSON") or (
                dotenv_values(args.policy_env).get("HEAR_BACKEND_POLICY_JSON") if args.policy_env.is_file() else None
            )
            policy = json.loads(raw_policy or "{}")
            job_type = {
                "magic_clean_natural": "magic_clean",
                "reconstruction": "reconstruction",
            }.get(role.value, "pipeline")
            envelope = cls.synthetic_envelope(policy, job_type)
        api_key = os.environ.get("RUNPOD_DEPLOY_API_KEY") or args.api_key_file.read_text()
        result = asyncio.run(cls(api_key).run(args.endpoint_id, role, envelope, args.timeout))
        text = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)
        if args.synthetic:
            reached_backend = "backend_attempt_rejected" in result["worker_events"] or (
                result["error"] is not None and "/claim'" in result["error"]
            )
            return 0 if reached_backend else 1
        return 0 if result["final_state"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(ServerlessCanary.main())
