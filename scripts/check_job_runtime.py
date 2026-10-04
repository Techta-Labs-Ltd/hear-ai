"""Read-only deployment checks. Never declares model execution or cloud uploads tested."""

import argparse
import json
import os
from pathlib import Path

import httpx
from dotenv import dotenv_values
from pydantic import ValidationError

from hear.config import RuntimeSettings
from hear.inference.manifest import ModelManifest
from hear.runtime.ownership import BackendRegistry, DeploymentOwnership
from hear.runtime.roles import WorkerRole
from hear.runtime.simulation import SimulationBoundary
from hear.services.pipeline.catalog import PipelineCatalogClient


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
        try:
            settings = RuntimeSettings.from_environment(dict(os.environ))
        except ValidationError as exc:
            # Pydantic's rendered error contains input values, including secrets.
            blockers.extend(
                "invalid_runtime_setting:" + ".".join(map(str, error["loc"]))
                for error in exc.errors(include_input=False)
            )
            return JobRuntimeCheck.write_report(args, {
                "status": "blocked", "blockers": blockers,
                "live_inference_verified": False, "cloud_roundtrip_verified": False,
                "production_approval_claimed": False,
            })
        if settings.backend_internal_url is None:
            blockers.append("backend_callback_url_missing")
        if WorkerRole.PIPELINE in roles and (
            settings.backend_service_key is None
            or not settings.backend_service_key.get_secret_value().strip()
        ):
            blockers.append("backend_service_key_missing")
        model_root = settings.model_root
        manifest = ModelManifest(root / "hear/model_manifest.json")
        checks = {}
        for role in roles:
            if role == WorkerRole.MAGIC_CLEAN_NATURAL:
                cleaner_dir = settings.magic_clean_model_directory
                files = [
                    cleaner_dir / "config.ini",
                    cleaner_dir / "checkpoints/model_120.ckpt.best",
                ]
                missing = [str(p) for p in files if not p.is_file()]
                licences = []
            else:
                role_root = (
                    (settings.fish_speech_model_root or model_root)
                    if role == WorkerRole.RECONSTRUCTION
                    else model_root
                )
                features = settings.model_features if role == WorkerRole.PIPELINE else frozenset()
                missing = list(
                    manifest.validate_local(
                        role_root, role, enabled_features=features, overrides=settings.model_paths
                    )
                )
                licences = list(manifest.license_blockers(role, enabled_features=features))
            checks[role.value] = {"missing_models": missing, "license_blockers": licences}
            if missing:
                blockers.append("role_missing_models:" + role.value)
            if licences and not simulation and not settings.fish_license_approved:
                blockers.append("role_license_review_required:" + role.value)
            checks[role.value]["license_acknowledged"] = settings.fish_license_approved
        callbacks = []
        try:
            owner = DeploymentOwnership.load()
            if owner is None:
                blockers.append("backend_ownership_policy_missing")
            elif isinstance(owner, BackendRegistry):
                callbacks = [(key, value[1]) for key, value in owner.entries.items()]
            else:
                base = os.environ.get("HEAR_BACKEND_INTERNAL_URL", "")
                owner.require_reporting_origin(base)
                callbacks = [(owner.backend_id, base)]
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
        catalog_check = None
        if WorkerRole.PIPELINE in roles and settings.backend_internal_url and settings.backend_service_key:
            try:
                catalog = PipelineCatalogClient(
                    str(settings.backend_internal_url), settings.backend_service_key.get_secret_value()
                ).fetch()
                catalog_check = {
                    "version": catalog.configuration.version,
                    "categories": len(catalog.categories), "tags": len(catalog.configuration.tags),
                }
                if not catalog.categories:
                    blockers.append("pipeline_catalog_has_no_categories")
            except (httpx.HTTPError, ValueError):
                blockers.append("pipeline_catalog_unavailable")
        report = {
            "status": "blocked" if blockers else "configuration_checked",
            "runtime_mode": "simulation" if simulation else "production",
            "production_approval_claimed": False,
            "blockers": blockers,
            "roles": checks,
            "backends": backend_checks,
            "pipeline_catalog": catalog_check,
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
        return JobRuntimeCheck.write_report(args, report)

    @staticmethod
    def write_report(args, report: dict) -> int:
        text = json.dumps(report, indent=2)
        if args.output:
            with args.output.open("x") as file:
                file.write(text + "\n")
        print(text)
        return 2 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(JobRuntimeCheck.main())
