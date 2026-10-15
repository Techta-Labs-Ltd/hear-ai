"""Read-only fleet check for the RunPod workers, with an explicit repair mode.

Compares every Serverless endpoint with its committed deploy plan and reports
the conditions that silently take the fleet down: a worker cap at 0, unhealthy
workers, jobs stuck in a queue with nobody to take them, workers scheduled on
GPU types the plan never listed, a low account balance, and a Pod gateway that
does not answer. `--repair` restores the plan's worker cap and GPU exclusions;
nothing else is ever changed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx

from hear.tools.runpod_api import RunPodApi


class FleetHealth:
    def __init__(
        self,
        api: RunPodApi,
        plans: dict[str, dict[str, Any]],
        *,
        min_balance: float = 25.0,
        pod_url: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._api = api
        self._plans = plans
        self._min_balance = min_balance
        self._pod_url = pod_url.rstrip("/") if pod_url else None
        self._client = client or httpx.Client(timeout=httpx.Timeout(30.0))

    @staticmethod
    def load_plans(directory: Path) -> dict[str, dict[str, Any]]:
        plans = {}
        for path in sorted(directory.glob("*.json")):
            endpoint = json.loads(path.read_text())["endpoint"]
            plans[endpoint["name"]] = endpoint
        return plans

    def snapshot(self) -> dict[str, Any]:
        account = self._api.account()
        endpoints = []
        for record in account.get("endpoints") or []:
            endpoints.append({**record, "health": self._api.health(record["id"])})
        pod = None
        if self._pod_url:
            try:
                response = self._client.get(f"{self._pod_url}/readyz")
                pod = {"url": self._pod_url, "status": response.status_code}
            except httpx.HTTPError as exc:
                pod = {"url": self._pod_url, "status": None, "error": type(exc).__name__}
        return {"balance": account.get("clientBalance"), "endpoints": endpoints, "pod": pod}

    def findings(self, snapshot: dict[str, Any]) -> list[str]:
        problems: list[str] = []
        by_name = {item["name"]: item for item in snapshot["endpoints"]}
        for name, plan in self._plans.items():
            record = by_name.get(name)
            if record is None:
                problems.append(f"endpoint_missing:{name}")
                continue
            if (record.get("workersMax") or 0) < plan["workersMax"]:
                problems.append(
                    f"workers_max_below_plan:{name}:{record.get('workersMax')}<{plan['workersMax']}"
                )
            workers = (record.get("health") or {}).get("workers") or {}
            jobs = (record.get("health") or {}).get("jobs") or {}
            if workers.get("unhealthy"):
                problems.append(f"unhealthy_workers:{name}:{workers['unhealthy']}")
            alive = sum(
                workers.get(key) or 0 for key in ("ready", "running", "initializing", "idle")
            )
            if (jobs.get("inQueue") or 0) > 0 and alive == 0:
                problems.append(f"queue_stuck:{name}:{jobs['inQueue']}")
            allowed = set(plan.get("gpuTypeIds") or [])
            seen = {(pod.get("machine") or {}).get("gpuTypeId") for pod in record.get("pods") or []}
            for gpu in sorted(filter(None, seen)):
                if gpu not in allowed:
                    problems.append(f"worker_on_unplanned_gpu:{name}:{gpu}")
            pool = record.get("gpuIds") or ""
            for excluded in plan.get("gpuExcludes") or []:
                if f"-{excluded}" not in pool.split(","):
                    problems.append(f"gpu_exclusion_missing:{name}:{excluded}")
            plan_image = plan.get("imageName")
            if plan_image and (record.get("template") or {}).get("imageName") != plan_image:
                problems.append(f"template_image_differs:{name}")
        balance = snapshot.get("balance")
        if balance is not None and balance < self._min_balance:
            problems.append(f"balance_low:{balance:.2f}<{self._min_balance:.2f}")
        pod = snapshot.get("pod")
        if pod is not None and pod.get("status") != 200:
            problems.append(f"pod_not_ready:{pod.get('status') or pod.get('error')}")
        return problems

    def repair(self, snapshot: dict[str, Any]) -> list[str]:
        """Restore each endpoint's plan cap and GPU exclusions. Nothing else."""
        actions: list[str] = []
        by_name = {item["name"]: item for item in snapshot["endpoints"]}
        for name, plan in self._plans.items():
            record = by_name.get(name)
            if record is None:
                continue
            excludes = list(plan.get("gpuExcludes") or [])
            allowed = set(plan.get("gpuTypeIds") or [])
            for pod in record.get("pods") or []:
                gpu = (pod.get("machine") or {}).get("gpuTypeId")
                if gpu and gpu not in allowed and gpu not in excludes:
                    excludes.append(gpu)
            saved = self._api.save_endpoint_gpu_exclusions(
                record, excludes, workers_max=plan["workersMax"]
            )
            if saved is not None:
                actions.append(
                    f"saved:{name}:workersMax={saved.get('workersMax')}:gpuIds={saved.get('gpuIds')}"
                )
        return actions

    @classmethod
    def main(cls) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plans-dir", type=Path, default=Path("deploy/runpod"))
        parser.add_argument(
            "--api-key-file", type=Path, default=Path("/root/hear-ai-config/runpod-api.key")
        )
        parser.add_argument("--pod-url", default=os.environ.get("HEAR_POD_GATEWAY_URL", ""))
        parser.add_argument("--min-balance", type=float, default=25.0)
        parser.add_argument("--repair", action="store_true")
        parser.add_argument("--output", type=Path)
        args = parser.parse_args()
        api_key = os.environ.get("RUNPOD_DEPLOY_API_KEY") or args.api_key_file.read_text()
        checker = cls(
            RunPodApi(api_key),
            cls.load_plans(args.plans_dir),
            min_balance=args.min_balance,
            pod_url=args.pod_url or None,
        )
        snapshot = checker.snapshot()
        report: dict[str, Any] = {"findings": checker.findings(snapshot), "actions": []}
        if args.repair and report["findings"]:
            report["actions"] = checker.repair(snapshot)
            snapshot = checker.snapshot()
            report["findings_after_repair"] = checker.findings(snapshot)
        report["summary"] = {
            "balance": snapshot.get("balance"),
            "pod": snapshot.get("pod"),
            "endpoints": [
                {
                    "name": item["name"],
                    "workersMax": item.get("workersMax"),
                    "gpuIds": item.get("gpuIds"),
                    "health": item.get("health"),
                    "worker_gpus": sorted(
                        {
                            str((pod.get("machine") or {}).get("gpuTypeId"))
                            for pod in item.get("pods") or []
                        }
                    ),
                }
                for item in snapshot["endpoints"]
            ],
        }
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)
        remaining = report.get("findings_after_repair", report["findings"])
        return 1 if remaining else 0


if __name__ == "__main__":
    raise SystemExit(FleetHealth.main())
