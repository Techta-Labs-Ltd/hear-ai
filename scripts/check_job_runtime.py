"""Read-only deployment checks. Never declares model execution or cloud uploads tested."""

import argparse
import json
import os
from pathlib import Path

import httpx
from dotenv import dotenv_values

from hear.config import RuntimeSettings
from hear.inference.manifest import ModelManifest
from hear.runtime.ownership import BackendRegistry, DeploymentOwnership
from hear.runtime.roles import WorkerRole
from hear.runtime.simulation import SimulationBoundary


class JobRuntimeCheck:
    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--env-file", type=Path)
        parser.add_argument("--output", type=Path)
        args = parser.parse_args()
        if args.env_file:
            for key, value in dotenv_values(args.env_file).items():
                if value is not None:
                    os.environ[key] = value
        root = Path(__file__).resolve().parents[1]
        simulation = SimulationBoundary.enabled()
        blockers = []
        roles = []
        try:
            roles = [
                WorkerRole(x.strip())
                for x in os.environ.get("HEAR_POD_STACK_ROLES", "").split(",")
                if x.strip()
            ]
        except ValueError:
            blockers.append("unsupported_or_retired_worker_role")
        if not roles:
            blockers.append("no_worker_roles_configured")
        settings = RuntimeSettings.from_environment(dict(os.environ))
        model_root = settings.model_root
        manifest = ModelManifest(root / "hear/model_manifest.json")
        checks = {}
        for role in roles:
            if role == WorkerRole.MAGIC_CLEAN_NATURAL:
                files = [
                    model_root / "magic-clean/DeepFilterNet3/config.ini",
                    model_root / "magic-clean/DeepFilterNet3/checkpoints/model_120.ckpt.best",
                ]
                missing = [str(p.relative_to(model_root)) for p in files if not p.is_file()]
                licences = []
            else:
                role_root = (
                    (settings.fish_speech_model_root or model_root)
                    if role == WorkerRole.RECONSTRUCTION
                    else model_root
                )
                missing = list(manifest.validate_local(role_root, role))
                licences = list(manifest.license_blockers(role))
            checks[role.value] = {"missing_models": missing, "license_blockers": licences}
            if missing:
                blockers.append("role_missing_models:" + role.value)
            if licences and not simulation:
                blockers.append("role_license_review_required:" + role.value)
        callbacks = []
        try:
            owner = DeploymentOwnership.load()
            if owner is None:
                blockers.append("backend_ownership_policy_missing")
            elif isinstance(owner, BackendRegistry):
                callbacks = [(key, value[1]) for key, value in owner.entries.items()]
            else:
                callbacks = [(owner.backend_id, os.environ.get("HEAR_BACKEND_INTERNAL_URL", ""))]
        except (ValueError, KeyError, TypeError):
            blockers.append("invalid_backend_ownership_configuration")
        backend_checks = {}
        for backend_id, base in callbacks:
            if not base.startswith("https://"):
                blockers.append("invalid_backend_callback_base:" + backend_id)
                continue
            try:
                with httpx.Client(timeout=5, follow_redirects=False) as client:
                    response = client.get(
                        base.rstrip("/") + "/internal/ai/runtime/protocol",
                        headers={"X-Service-Key": os.environ.get("HEAR_BACKEND_SERVICE_KEY", "")},
                    )
                    data = response.json() if response.status_code == 200 else {}
                ok = (
                    response.status_code == 200
                    and data.get("schema_version") == 1
                    and data.get("backend_id") == backend_id
                    and data.get("attempt_control") is True
                )
                backend_checks[backend_id] = {
                    "http_status": response.status_code,
                    "protocol_matches": ok,
                }
                if not ok:
                    blockers.append("backend_protocol_unavailable:" + backend_id)
            except (httpx.HTTPError, ValueError):
                backend_checks[backend_id] = {"protocol_matches": False}
                blockers.append("backend_protocol_unavailable:" + backend_id)
        report = {
            "status": "blocked" if blockers else "configuration_checked",
            "runtime_mode": "simulation" if simulation else "production",
            "production_approval_claimed": False,
            "blockers": blockers,
            "roles": checks,
            "backends": backend_checks,
            "admission": {
                "per_worker_limit": settings.pod_max_concurrent_jobs,
                "host_total_limit": int(os.environ.get("HEAR_HOST_MAX_CONCURRENT_JOBS", "1")),
                "mechanism": "process_shared_global_and_role_slot_locks",
                "configured_roles": [r.value for r in roles],
                "configured_is_not_load_tested_capacity": True,
            },
            "live_inference_verified": False,
            "cloud_roundtrip_verified": False,
            "note": "Configuration inspection only. Require an authenticated canary before enabling app dispatch.",
        }
        text = json.dumps(report, indent=2)
        if args.output:
            with args.output.open("x") as file:
                file.write(text + "\n")
        print(text)
        return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(JobRuntimeCheck.main())
