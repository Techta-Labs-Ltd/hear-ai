"""Submit real-model jobs to the running API, with only backend/storage simulated."""

import concurrent.futures
import hashlib
import json
import os
import time
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import soundfile as sf

from hear.contracts.jobs import AttemptEnvelope


class SimulatedJobs:
    ROOT = Path("/root/hear-ai-v11/simulation-20260928")

    @classmethod
    def main(cls):
        root = cls.ROOT
        config = json.loads((root / "config.json").read_text())
        base = "https://127.0.0.1:18081"
        headers = {"X-Service-Key": config["service_key"]}
        backend = httpx.Client(verify=str(root / "tls/cert.pem"), timeout=30)
        api = httpx.Client(
            base_url="http://127.0.0.1:8000",
            timeout=30,
            headers={"Authorization": "Bearer " + config["ingress_token"]},
        )
        requests = []
        copies = int(os.environ.get("HEAR_CANARY_COPIES", "1"))
        if not 1 <= copies <= 10:
            raise ValueError("canary_copies_must_be_one_to_ten")
        kinds = tuple(
            os.environ.get(
                "HEAR_CANARY_JOB_TYPES", "magic_clean,transcription,pipeline,reconstruction"
            ).split(",")
        )
        if not kinds or any(
            kind not in {"magic_clean", "transcription", "pipeline", "reconstruction"}
            for kind in kinds
        ):
            raise ValueError("invalid_canary_job_types")
        timeout = int(os.environ.get("HEAR_CANARY_TIMEOUT_SECONDS", "600"))
        if not 60 <= timeout <= 10800:
            raise ValueError("invalid_canary_timeout")
        max_active = api.get("/capabilities").json().get("concurrency", {}).get("host_total", 2)
        for kind in kinds * copies:
            job, attempt, track = (str(uuid.uuid4()) for _ in range(3))
            source = (
                "fish-reference.wav"
                if kind == "reconstruction"
                else os.environ.get("HEAR_CANARY_SOURCE", "input.mp3")
            )
            if Path(source).name != source:
                raise ValueError("invalid_canary_source_name")
            file = root / "sources" / source
            prefix = f"creators/evaluation/audio/jobs/{job}"
            expiry = datetime.now(UTC) + timedelta(seconds=max(1800, timeout + 300))
            options = {"profile": "studio_voice"} if kind == "magic_clean" else {}
            if kind == "reconstruction":
                options = {
                    "same_speaker": True,
                    "edited_transcript": "The Hear application can now send a text edit through the job API.",
                    "reference": {
                        "start_seconds": 0,
                        "end_seconds": sf.info(file).duration,
                        "text": "This is a test of the Hear reconstruction service. Fish Speech is generating this sentence using four bit quantized weights.",
                    },
                }
            raw = {
                "schema_version": 1,
                "backend_id": "simulation-local",
                "job_id": job,
                "run_id": str(uuid.uuid4()),
                "attempt_id": attempt,
                "track_id": track,
                "user_id": str(uuid.uuid4()),
                "job_type": kind,
                "source": {
                    "url": base + "/source/" + source,
                    "revision": 1,
                    "file_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                },
                "storage": {
                    "endpoint_url": base + "/s3",
                    "bucket_name": config["bucket"],
                    "key_id": "HEAR_SIMULATION_ONLY",
                    "application_key": "not-a-cloud-credential",
                    "folder_prefix": prefix + "/",
                    "public_base_url": base + "/s3/" + config["bucket"],
                    "expires_at": expiry.isoformat(),
                },
                "options": options,
                "artifact_prefix": prefix + "/" + attempt,
                "deadline": expiry.isoformat(),
                "reporting_grant": str(uuid.uuid4()),
                "backend_base_url": base + "/api/v1",
            }
            if kind == "reconstruction":
                raw["operation"] = "rebuild"
            value = AttemptEnvelope.model_validate(raw).model_dump(mode="json")
            response = backend.post(base + "/simulation/register", json=value, headers=headers)
            response.raise_for_status()
            requests.append(value)

        def submit(value):
            response = api.post("/v1/attempts", json=value)
            print("API_ACCEPTANCE", value["job_type"], response.status_code, flush=True)
            response.raise_for_status()
            assert response.status_code == 202
            assert response.json()["attempt_id"] == value["attempt_id"]
            return response.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(requests))) as pool:
            accepted = list(pool.map(submit, requests))
        identities = {value["attempt_id"] for value in requests}
        previous = None
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            response = backend.get(base + "/simulation/summary", headers=headers)
            response.raise_for_status()
            summary = response.json()
            rows = [r for r in summary["attempts"] if r["attempt_id"] in identities]
            states = dict(Counter(r["job_type"] + ":" + r["status"] for r in rows))
            if states != previous:
                print("JOB_STATES", states, flush=True)
                previous = states
            if all(r["status"] in ("completed", "failed", "cancelled") for r in rows):
                break
            time.sleep(2)
        assert len(rows) == len(requests) and all(r["status"] == "completed" for r in rows), states
        assert summary["max_simultaneously_claimed"] <= max_active
        assert all(r["claims"] == 1 and r["readback"] for r in rows)
        for value in requests:
            assert api.post("/v1/attempts", json=value).status_code == 202
        time.sleep(3)
        report = backend.get(base + "/simulation/summary", headers=headers).json()
        report["acceptance_responses"] = accepted
        report["negative_auth_status"] = httpx.post(
            "http://127.0.0.1:8000/v1/attempts", json=requests[0], timeout=10
        ).status_code
        assert report["negative_auth_status"] == 403
        public_checks = []
        for row in report["attempts"]:
            if row["attempt_id"] not in identities:
                continue
            for artifact in row.get("outcome", {}).get("artifacts", []):
                if not artifact.get("audio_url"):
                    continue
                digest = hashlib.sha256()
                with backend.stream("GET", artifact["audio_url"]) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        digest.update(chunk)
                assert digest.hexdigest() == artifact["sha256"]
                public_checks.append({"key": artifact["object_key"], "sha256_verified": True})
        report["returned_audio_urls_verified"] = public_checks
        report["all_selected_real_job_types_completed"] = True
        report["all_four_real_job_types_completed"] = len(set(kinds)) == 4
        report["attempts"] = [row for row in report["attempts"] if row["attempt_id"] in identities]
        report["source_fixture"] = os.environ.get("HEAR_CANARY_SOURCE", "input.mp3")
        report["real_cloud_backblaze_tested"] = False
        report["updated_api_running"] = True
        Path(os.environ.get("HEAR_CANARY_REPORT", str(root / "verification.json"))).write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print("ALL_SELECTED_REAL_JOB_TYPES_COMPLETED", root / "verification.json", flush=True)


if __name__ == "__main__":
    SimulatedJobs.main()
