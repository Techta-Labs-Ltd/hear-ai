"""Run controlled event/background cleanup through the real-model simulation API."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import soundfile as sf

from hear.contracts.jobs import AttemptEnvelope


class AudioCleanupCanary:
    ROOT = Path("/root/hear-ai-v11/simulation-20260928")
    OUTPUT = Path("/root/hear-ai-v11/performance-20260928/audio-cleanup-canary.json")

    @classmethod
    def source(cls, name: str, origin: Path) -> Path:
        target = cls.ROOT / "sources" / name
        if not target.exists():
            shutil.copyfile(origin, target)
        if (
            hashlib.sha256(target.read_bytes()).hexdigest()
            != hashlib.sha256(origin.read_bytes()).hexdigest()
        ):
            raise RuntimeError("cleanup_canary_source_mismatch")
        return target

    @classmethod
    def envelope(cls, source: Path, options: dict) -> dict:
        config = json.loads((cls.ROOT / "config.json").read_text())
        job, attempt, track = (str(uuid.uuid4()) for _ in range(3))
        prefix = f"creators/evaluation/audio/jobs/{job}"
        expiry = datetime.now(UTC) + timedelta(minutes=20)
        base = "https://127.0.0.1:18081"
        raw = {
            "schema_version": 1,
            "backend_id": "simulation-local",
            "job_id": job,
            "run_id": str(uuid.uuid4()),
            "attempt_id": attempt,
            "track_id": track,
            "user_id": str(uuid.uuid4()),
            "job_type": "magic_clean",
            "source": {
                "url": f"{base}/source/{source.name}",
                "revision": 1,
                "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
            "storage": {
                "endpoint_url": f"{base}/s3",
                "bucket_name": config["bucket"],
                "key_id": "HEAR_SIMULATION_ONLY",
                "application_key": "not-a-cloud-credential",
                "folder_prefix": prefix + "/",
                "public_base_url": f"{base}/s3/{config['bucket']}",
                "expires_at": expiry.isoformat(),
            },
            "options": options,
            "artifact_prefix": prefix + "/" + attempt,
            "deadline": expiry.isoformat(),
            "reporting_grant": str(uuid.uuid4()),
            "backend_base_url": base + "/api/v1",
        }
        return AttemptEnvelope.model_validate(raw).model_dump(mode="json")

    @classmethod
    def main(cls) -> None:
        config = json.loads((cls.ROOT / "config.json").read_text())
        base = "https://127.0.0.1:18081"
        backend = httpx.Client(verify=str(cls.ROOT / "tls/cert.pem"), timeout=30)
        api = httpx.Client(
            base_url="http://127.0.0.1:8000",
            timeout=30,
            headers={"Authorization": "Bearer " + config["ingress_token"]},
        )
        bark = cls.source(
            "bark-overlap.wav",
            Path(
                "/workspace/hear-ai-v11/clean/sound-cleanup-v1-20260928/overlap-fixture/mixture.wav"
            ),
        )
        background = cls.source(
            "background-before.wav",
            Path(
                "/workspace/hear-ai-v11/clean/background-audit-20260928/01_before_background_cleanup.wav"
            ),
        )
        negative = cls.source(
            "bark-negative.wav",
            Path(
                "/workspace/hear-ai-v11/clean/bark-correction-20260928T015550Z/reference_before_bark_was_added.wav"
            ),
        )
        jobs = {
            "bark": cls.envelope(
                bark,
                {
                    "profile": "studio_voice",
                    "sound_cleanup": {
                        "enabled": True,
                        "auto_detect": False,
                        "targets": ["animal"],
                        "preview_overlaps": True,
                        "regions": [
                            {
                                "start_seconds": 5.0,
                                "end_seconds": 10.0,
                                "kind": "animal",
                                "confirmed_no_speech": False,
                            }
                        ],
                    },
                },
            ),
            "bark_auto": cls.envelope(
                bark,
                {
                    "profile": "studio_voice",
                    "sound_cleanup": {"enabled": True, "auto_detect": True, "targets": ["animal"]},
                },
            ),
            "bark_negative": cls.envelope(
                negative,
                {
                    "profile": "studio_voice",
                    "sound_cleanup": {"enabled": True, "auto_detect": True, "targets": ["animal"]},
                },
            ),
            "background": cls.envelope(
                background, {"profile": "studio_voice", "reduce_stationary_noise": True}
            ),
        }
        headers = {"X-Service-Key": config["service_key"]}
        for value in jobs.values():
            backend.post(
                base + "/simulation/register", json=value, headers=headers
            ).raise_for_status()
            response = api.post("/v1/attempts", json=value)
            assert response.status_code == 202, response.text
        identities = {value["attempt_id"] for value in jobs.values()}
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            summary = backend.get(base + "/simulation/summary", headers=headers).json()
            rows = [row for row in summary["attempts"] if row["attempt_id"] in identities]
            if len(rows) == len(jobs) and all(
                row["status"] in {"completed", "failed"} for row in rows
            ):
                break
            time.sleep(1)
        assert len(rows) == len(jobs) and all(row["status"] == "completed" for row in rows), rows
        result = {}
        for label, request in jobs.items():
            row = next(row for row in rows if row["attempt_id"] == request["attempt_id"])
            validation = next(
                a
                for a in row["outcome"]["artifacts"]
                if a["object_key"].endswith("/validation.json")
            )
            payload = json.loads((cls.ROOT / "objects" / validation["object_key"]).read_text())
            master = next(
                a
                for a in row["outcome"]["artifacts"]
                if a["object_key"].endswith("/cleaned_master.flac")
            )
            info = sf.info(cls.ROOT / "objects" / master["object_key"])
            result[label] = {
                "seconds": row["completed_at"] - row["claimed_at"],
                "frames": info.frames,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "sound_cleanup": payload["sound_cleanup"],
                "background_cleanup": payload["background_cleanup"],
                "warnings": payload["warnings"],
            }
        assert result["bark"]["sound_cleanup"]["preview_count"] >= 1
        assert result["bark"]["sound_cleanup"]["audio_changed"] is True
        assert result["bark"]["sound_cleanup"]["outside_repair_regions_unchanged"] is True
        assert result["bark"]["sound_cleanup"]["overlap_separation_available"] is True
        assert result["background"]["background_cleanup"]["enabled"] is True
        assert result["background"]["background_cleanup"]["status"] not in {
            "rejected_speech_activity_loss",
            "failed",
        }
        cls.OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    AudioCleanupCanary.main()
