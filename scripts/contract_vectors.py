"""Write docs/contract-vectors.json: fixed inputs with the digests another language must reproduce.

The Go backend signs reporting grants and hashes execution scopes exactly like the
Python backend did; these vectors pin the byte-level format so both sides can be
tested against the same numbers. Re-run after any change to ExecutionScope.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path

from hear.contracts.jobs import AttemptEnvelope
from hear.contracts.scope import ExecutionScope


class ContractVectors:
    SECRET = "contract-vector-secret-do-not-use-in-production"
    GRANT_PREFIX = b"hear-attempt-v1:"

    @staticmethod
    def envelope() -> dict:
        return {
            "schema_version": 1,
            "job_id": "0192f0a1-7c3e-7d4b-8a11-0000000000aa",
            "run_id": "0192f0a1-7c3e-7d4b-8a11-0000000000bb",
            "attempt_id": "0192f0a1-7c3e-7d4b-8a11-0000000000cc",
            "job_type": "magic_clean",
            "operation": None,
            "track_id": "0192f0a1-7c3e-7d4b-8a11-0000000000dd",
            "user_id": "0192f0a1-7c3e-7d4b-8a11-0000000000ee",
            "source": {
                "url": "https://cdn.hear.media/creators/demo/audio/tracks/t1/audio.mp3",
                "revision": 3,
                "file_sha256": "a" * 64,
            },
            "storage": {
                "endpoint_url": "https://s3.eu-central-003.backblazeb2.com/",
                "bucket_name": "hear-media",
                "key_id": "0030000000000000000000001",
                "application_key": "K003contractvectorkeyvalue",
                "folder_prefix": "creators/demo/audio/jobs/0192f0a1-7c3e-7d4b-8a11-0000000000aa/",
                "public_base_url": "https://cdn.hear.media/",
                "expires_at": "2026-10-05T10:00:00Z",
            },
            "options": {"profile": "studio_voice"},
            "artifact_prefix": "creators/demo/audio/jobs/0192f0a1-7c3e-7d4b-8a11-0000000000aa/0192f0a1-7c3e-7d4b-8a11-0000000000cc",
            "deadline": "2026-10-04T18:30:00Z",
            "reporting_grant": "placeholder",
            "backend_base_url": "https://api.hear.media/api/v1",
            "backend_id": "hear-backend",
        }

    @classmethod
    def grant(cls, job_id: str, attempt_id: str, scope: str, exp: int) -> dict:
        payload = {"v": 1, "backend": "hear-backend", "job": job_id, "attempt": attempt_id, "scope": scope, "exp": exp}
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=False).encode()
        signature = hmac.new(cls.SECRET.encode(), cls.GRANT_PREFIX + raw, hashlib.sha256).digest()
        b64 = lambda data: base64.urlsafe_b64encode(data).rstrip(b"=").decode()  # noqa: E731
        return {"payload": payload, "payload_json": raw.decode(), "signing_input": (cls.GRANT_PREFIX + raw).decode(), "token": f"{b64(raw)}.{b64(signature)}"}

    @classmethod
    def build(cls) -> dict:
        envelope = cls.envelope()
        validated = AttemptEnvelope.model_validate(envelope).model_dump(mode="json")
        scope = ExecutionScope.digest(validated)
        grant = cls.grant(envelope["job_id"], envelope["attempt_id"], scope, 1791221400)
        return {
            "note": "Inputs and expected outputs for the backend's scope digest and reporting grant; secret is a test value.",
            "secret": cls.SECRET,
            "envelope": envelope,
            "scope_sha256": scope,
            "grant": grant,
            "scope_rules": "ExecutionScope.digest in hear/contracts/scope.py: canonical JSON, sorted keys, no spaces, application_key hashed, timestamps as POSIX seconds.",
        }

    @classmethod
    def main(cls) -> int:
        target = Path(__file__).resolve().parents[1] / "docs" / "contract-vectors.json"
        target.write_text(json.dumps(cls.build(), indent=2) + "\n")
        print(target)
        return 0


if __name__ == "__main__":
    raise SystemExit(ContractVectors.main())
