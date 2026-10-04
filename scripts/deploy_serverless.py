"""Roll a RunPod Serverless template and endpoint to a published image digest.

Reads a deployment plan (the ``template`` and ``endpoint`` sections recorded for
each role), updates the template image, creates or updates the endpoint by name,
and prints the endpoint health. Secrets come from files or the environment only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx


class RunPodServerlessDeployer:
    REST_BASE = "https://rest.runpod.io/v1"
    RUN_BASE = "https://api.runpod.ai/v2"

    def __init__(
        self,
        api_key: str,
        *,
        rest_base: str = REST_BASE,
        run_base: str = RUN_BASE,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("runpod_api_key_required")
        self._rest = rest_base.rstrip("/")
        self._run = run_base.rstrip("/")
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0))
        self._headers = {"Authorization": f"Bearer {api_key.strip()}"}

    def _call(self, method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
        response = self._client.request(method, url, json=payload, headers=self._headers)
        if response.status_code >= 400:
            raise RuntimeError(f"runpod_api_{response.status_code}:{method}:{url}")
        return response.json() if response.content else {}

    @staticmethod
    def require_digest(image: str) -> str:
        if "@sha256:" not in image or len(image.rsplit("@sha256:", 1)[1]) != 64:
            raise ValueError("image_must_be_pinned_by_sha256_digest")
        return image

    TEMPLATE_KEYS = (
        "name",
        "env",
        "containerDiskInGb",
        "dockerEntrypoint",
        "dockerStartCmd",
        "containerRegistryAuthId",
    )

    def update_template(self, template_id: str, template: dict[str, Any], image: str) -> dict:
        payload = {key: template[key] for key in self.TEMPLATE_KEYS if key in template}
        payload["imageName"] = self.require_digest(image)
        return self._call("PATCH", f"{self._rest}/templates/{template_id}", payload)

    def create_template(self, template: dict[str, Any], image: str) -> dict:
        payload = {key: template[key] for key in self.TEMPLATE_KEYS if key in template}
        payload.update({"imageName": self.require_digest(image), "isServerless": True})
        return self._call("POST", f"{self._rest}/templates", payload)

    def ensure_endpoint(self, endpoint: dict[str, Any], template_id: str) -> dict:
        existing = {
            item["name"]: item for item in self._call("GET", f"{self._rest}/endpoints") or []
        }
        payload = {**endpoint, "templateId": template_id}
        current = existing.get(endpoint["name"])
        if current is not None:
            payload.pop("computeType", None)
            return self._call("PATCH", f"{self._rest}/endpoints/{current['id']}", payload)
        return self._call("POST", f"{self._rest}/endpoints", payload)

    def health(self, endpoint_id: str) -> dict:
        return self._call("GET", f"{self._run}/{endpoint_id}/health")

    def deploy(self, plan: dict[str, Any], template_id: str | None, image: str) -> dict:
        if template_id:
            template = self.update_template(template_id, plan["template"], image)
        else:
            template = self.create_template(plan["template"], image)
            template_id = str(template["id"])
        endpoint = self.ensure_endpoint(plan["endpoint"], template_id)
        return {
            "template_id": template_id,
            "image": image,
            "endpoint_id": endpoint.get("id"),
            "endpoint_name": endpoint.get("name"),
            "base_url": f"{self._run}/{endpoint.get('id')}",
            "health": self.health(endpoint["id"]) if endpoint.get("id") else None,
            "template_name": template.get("name"),
        }

    @classmethod
    def main(cls) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", type=Path, required=True, help="recorded deployment JSON")
        parser.add_argument("--template-id", help="omit to create a new template from the plan")
        parser.add_argument("--image", required=True, help="registry/image@sha256:<digest>")
        parser.add_argument(
            "--api-key-file", type=Path, default=Path("/root/hear-ai-config/runpod-api.key")
        )
        parser.add_argument("--output", type=Path)
        parser.add_argument("--dry-run", action="store_true")
        args = parser.parse_args()
        plan = json.loads(args.plan.read_text())
        cls.require_digest(args.image)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "template_id": args.template_id or "<create>",
                        "image": args.image,
                        "endpoint": plan["endpoint"]["name"],
                    }
                )
            )
            return 0
        api_key = os.environ.get("RUNPOD_DEPLOY_API_KEY") or args.api_key_file.read_text()
        result = cls(api_key).deploy(plan, args.template_id, args.image)
        text = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)
        return 0


if __name__ == "__main__":
    raise SystemExit(RunPodServerlessDeployer.main())
