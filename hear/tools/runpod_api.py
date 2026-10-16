"""Small RunPod client shared by the deploy and fleet-health scripts.

REST (`rest.runpod.io/v1`) owns templates and the plain endpoint fields. The
GraphQL API is the only place that exposes the endpoint's GPU *pool* string,
which carries negative entries such as ``-NVIDIA L4`` to keep a type out of the
pool. RunPod adds new card types to its pools (the 24 GB pool gained Blackwell
MIG slices in October 2026), so an endpoint that names specific cards can still
be scheduled onto hardware the image was never built for. Exclusions written
through `save_endpoint_gpu_exclusions` stop that.
"""

from __future__ import annotations

from typing import Any

import httpx

REST_BASE = "https://rest.runpod.io/v1"
RUN_BASE = "https://api.runpod.ai/v2"
GRAPHQL_BASE = "https://api.runpod.io/graphql"

ENDPOINTS_QUERY = """{
  myself {
    clientBalance
    endpoints {
      id name gpuIds idleTimeout locations networkVolumeId scalerType scalerValue
      templateId workersMin workersMax workersStandby
      template { id imageName }
      pods { id desiredStatus lastStatusChange imageName machine { gpuTypeId } }
    }
  }
}"""

SAVE_ENDPOINT_MUTATION = """mutation($input: EndpointInput!) {
  saveEndpoint(input: $input) { id name gpuIds workersMin workersMax }
}"""


class RunPodApi:
    def __init__(
        self,
        api_key: str,
        *,
        rest_base: str = REST_BASE,
        run_base: str = RUN_BASE,
        graphql_base: str = GRAPHQL_BASE,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("runpod_api_key_required")
        self.rest_base = rest_base.rstrip("/")
        self.run_base = run_base.rstrip("/")
        self.graphql_base = graphql_base
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0))
        self._headers = {"Authorization": f"Bearer {api_key.strip()}"}

    def call(self, method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
        response = self._client.request(method, url, json=payload, headers=self._headers)
        if response.status_code >= 400:
            raise RuntimeError(f"runpod_api_{response.status_code}:{method}:{url}")
        return response.json() if response.content else {}

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self._client.request(
            "POST",
            self.graphql_base,
            json={"query": query, "variables": variables or {}},
            headers=self._headers,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"runpod_graphql_{response.status_code}")
        body = response.json()
        if body.get("errors"):
            messages = "; ".join(str(item.get("message")) for item in body["errors"])
            raise RuntimeError(f"runpod_graphql_error:{messages}")
        return body.get("data") or {}

    def account(self) -> dict[str, Any]:
        """Balance plus every endpoint with its GPU pool and worker pods."""
        return self.graphql(ENDPOINTS_QUERY).get("myself") or {}

    def health(self, endpoint_id: str) -> dict[str, Any]:
        return self.call("GET", f"{self.run_base}/{endpoint_id}/health")

    @staticmethod
    def pool_with_exclusions(gpu_ids: str, excludes: list[str]) -> str:
        parts = [part for part in gpu_ids.split(",") if part]
        for name in excludes:
            if f"-{name}" not in parts:
                parts.append(f"-{name}")
        return ",".join(parts)

    def save_endpoint(self, endpoint: dict[str, Any], **changes: Any) -> dict[str, Any]:
        """Re-save a GraphQL endpoint record with `changes` applied; other fields kept."""
        payload = {
            "id": endpoint["id"],
            "name": endpoint["name"],
            "templateId": endpoint["templateId"],
            "gpuIds": endpoint["gpuIds"],
            "idleTimeout": endpoint.get("idleTimeout"),
            "scalerType": endpoint.get("scalerType"),
            "scalerValue": endpoint.get("scalerValue"),
            "workersMin": endpoint.get("workersMin"),
            "workersMax": endpoint.get("workersMax"),
            # EndpointInput has no workersStandby; RunPod derives it from workersMax.
            "networkVolumeId": endpoint.get("networkVolumeId") or None,
            "locations": endpoint.get("locations") or None,
        }
        payload.update(changes)
        payload = {key: value for key, value in payload.items() if value is not None}
        return self.graphql(SAVE_ENDPOINT_MUTATION, {"input": payload}).get("saveEndpoint") or {}

    def save_endpoint_gpu_exclusions(
        self, endpoint: dict[str, Any], excludes: list[str], *, workers_max: int | None = None
    ) -> dict[str, Any] | None:
        """Add `excludes` to the endpoint's GPU pool (and optionally fix workersMax).

        Returns the saved record, or None when nothing needed changing.
        """
        remaining = list(excludes)
        skipped: list[str] = []
        while True:
            pool = self.pool_with_exclusions(endpoint["gpuIds"], remaining)
            changes: dict[str, Any] = {}
            if pool != endpoint["gpuIds"]:
                changes["gpuIds"] = pool
            if workers_max is not None and endpoint.get("workersMax") != workers_max:
                changes["workersMax"] = workers_max
            if not changes:
                return None
            try:
                saved = self.save_endpoint(endpoint, **changes)
            except RuntimeError as exc:
                # A type outside every selected pool can never be scheduled here, so
                # RunPod refuses to list it; drop that one and keep the others.
                name = self._rejected_exclusion(str(exc))
                if name is None or name not in remaining:
                    raise
                remaining.remove(name)
                skipped.append(name)
                continue
            if skipped:
                saved = {**saved, "skipped_exclusions": skipped}
            return saved

    @staticmethod
    def _rejected_exclusion(message: str) -> str | None:
        marker = 'Invalid exclusion "-'
        if marker not in message:
            return None
        rest = message.split(marker, 1)[1]
        return rest.split('"', 1)[0] or None
