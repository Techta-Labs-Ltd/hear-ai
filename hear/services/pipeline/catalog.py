from __future__ import annotations

import httpx

from hear.runtime.ownership import BackendRegistry, DeploymentOwnership
from hear.services.pipeline.configuration import PipelineCatalog, PipelineConfiguration


class PipelineCatalogClient:
    def __init__(
        self,
        backend_url: str,
        service_key: str,
        *,
        timeout_seconds: float = 20.0,
    ) -> None:
        ownership = DeploymentOwnership.load()
        if isinstance(ownership, BackendRegistry):
            entry = ownership.entries.get(ownership.pipeline_backend_id)
            if entry is None:
                raise ValueError("pipeline_catalog_owner_not_configured")
            entry[0].require_reporting_origin(backend_url)
        elif ownership is not None:
            ownership.require_reporting_origin(backend_url)
        self._url = f"{backend_url.rstrip('/')}/internal/ai/runtime/catalog"
        self._headers = {"X-Service-Key": service_key}
        self._timeout = timeout_seconds

    def fetch(self) -> PipelineCatalog:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(self._url, headers=self._headers)
            response.raise_for_status()
            configuration = PipelineConfiguration.model_validate(response.json())
        return PipelineCatalog(configuration)
