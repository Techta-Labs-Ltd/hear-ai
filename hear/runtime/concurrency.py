"""Validated per-type admission separate from model-process concurrency."""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass

from hear.runtime.roles import WorkerRole


@dataclass(frozen=True)
class RoleConcurrency:
    process_jobs: int
    role_jobs: int

    @staticmethod
    def limits(value: str) -> dict[str, int]:
        parsed = json.loads(value or "{}")
        names = {role.value for role in WorkerRole}
        if not isinstance(parsed, dict) or any(
            key not in names or type(number) is not int or not 1 <= number <= 16
            for key, number in parsed.items()
        ):
            raise ValueError("invalid_role_concurrency_limits")
        return parsed

    @classmethod
    def load(
        cls, role: WorkerRole, default: int, host: int, environment: Mapping[str, str] | None = None
    ):
        env = os.environ if environment is None else environment
        processes = cls.limits(env.get("HEAR_POD_PROCESS_LIMITS", "{}"))
        roles = cls.limits(env.get("HEAR_POD_ROLE_LIMITS", "{}"))
        process_limit = processes.get(role.value, default)
        role_limit = roles.get(role.value, process_limit)
        if not 1 <= process_limit <= role_limit <= host <= 16:
            raise ValueError("inconsistent_host_role_process_limits")
        if role in {WorkerRole.RECONSTRUCTION, WorkerRole.MAGIC_CLEAN_NATURAL}:
            if process_limit != 1:
                raise ValueError("stateful_audio_engine_requires_one_job_per_process")
        if role == WorkerRole.RECONSTRUCTION and role_limit > 2:
            raise ValueError("fish_reconstruction_requires_at_most_two_isolated_workers")
        return cls(process_limit, role_limit)
