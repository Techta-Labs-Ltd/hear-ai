from __future__ import annotations

import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from hear.api.body_limit import RequestBodyLimitMiddleware
from hear.api.gateway import PodGateway
from hear.config import RuntimeSettings
from hear.runtime.cleaner.asset_probe import PinnedAssetProbe
from hear.runtime.gateway import RabbitMQGateway
from hear.runtime.ownership import DeploymentOwnership
from hear.runtime.roles import WorkerRole
from hear.services.registration import WorkerRegistrar
from hear.services.sound_cleanup.assets import SoundCleanupAssets


class GatewayEntrypoint:
    @staticmethod
    def roles() -> set[WorkerRole]:
        configured = os.environ.get("HEAR_POD_STACK_ROLES", WorkerRole.PIPELINE.value)
        names = {item.strip() for item in configured.split(",") if item.strip()}
        if "magic_clean_sam_audio" in names:
            logging.getLogger(__name__).warning(
                "Ignoring retired SAM worker role; separation jobs are not remapped"
            )
            names.remove("magic_clean_sam_audio")
        roles = {WorkerRole(item) for item in names}
        if not roles:
            raise RuntimeError("pod_stack_roles_required")
        return roles

    @staticmethod
    def sound_cleanup_ready(settings: RuntimeSettings) -> bool:
        bundle = settings.sound_cleanup_bundle
        digest = settings.sound_cleanup_bundle_sha256
        if bundle is None or not digest:
            return False
        try:
            SoundCleanupAssets.load(bundle, digest)
            return True
        except Exception:
            logging.getLogger(__name__).warning("Sound Cleanup bundle is not ready")
            return False

    @staticmethod
    def overlap_preview_ready(settings: RuntimeSettings) -> bool:
        root = settings.sound_cleanup_separator_bundle
        digest = settings.sound_cleanup_separator_sha256
        if root is None or not digest:
            return False
        try:
            payload = PinnedAssetProbe.read_regular(root / "manifest.json", maximum_bytes=65536)
            if hashlib.sha256(payload).hexdigest() != digest:
                return False
            manifest = json.loads(payload)
            for name in ("audiosep.jit", "queries.json"):
                PinnedAssetProbe.sha256(
                    root / name, manifest["files"][name]["sha256"], maximum_bytes=2_000_000_000
                )
            return True
        except Exception:
            return False

    @classmethod
    def registrar(cls, settings: RuntimeSettings, runtime, roles: set[WorkerRole]):
        """The self-registration loop, or None with a logged reason."""
        logger = logging.getLogger(__name__)
        if settings.registration_interval_seconds <= 0:
            logger.warning("worker_registration disabled: HEAR_REGISTRATION_INTERVAL_SECONDS=0")
            return None
        gateway_url = WorkerRegistrar.derive_gateway_url(
            {**os.environ, "HEAR_PUBLIC_GATEWAY_URL": str(settings.public_gateway_url or "")}
        )
        if gateway_url is None:
            logger.warning(
                "worker_registration disabled: set HEAR_PUBLIC_GATEWAY_URL (no RUNPOD_POD_ID)"
            )
            return None
        if settings.backend_internal_url is None or settings.backend_service_key is None:
            logger.warning("worker_registration disabled: backend URL or service key missing")
            return None
        return WorkerRegistrar(
            backend_url=str(settings.backend_internal_url),
            service_key=settings.backend_service_key.get_secret_value(),
            gateway_url=gateway_url,
            worker_id=settings.worker_id or f"pod-{os.environ.get('RUNPOD_POD_ID', 'local')}",
            worker_generation=settings.worker_generation or settings.image_revision or "unknown",
            roles=[role.value for role in roles],
            image_revision=settings.image_revision,
            engine_revision=settings.engine_revision,
            lane_status=runtime.lane_status,
            interval_seconds=settings.registration_interval_seconds,
        )

    @classmethod
    def create_app(cls) -> FastAPI:
        settings = RuntimeSettings.from_environment(dict(os.environ))
        roles = cls.roles()
        runtime = RabbitMQGateway(
            settings.rabbitmq_url or "amqp://guest:guest@127.0.0.1:5672/%2F",
            roles,
        )
        registrar = cls.registrar(settings, runtime, roles)
        gateway = PodGateway(
            runtime,
            settings.pod_api_key.get_secret_value() if settings.pod_api_key else "",
            sound_cleanup_available=cls.sound_cleanup_ready(settings),
            overlap_preview_available=cls.overlap_preview_ready(settings),
            ownership_policy=DeploymentOwnership.load(),
            enable_docs=os.environ.get("HEAR_ENABLE_DOCS", "false").lower() == "true",
        )

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            await runtime.start()
            if registrar is not None:
                registrar.start()
            try:
                yield
            finally:
                if registrar is not None:
                    await registrar.close()
                await runtime.close()

        app = FastAPI(
            title="Hear AI Pod API",
            version="1.0.0",
            docs_url="/docs" if gateway.enable_docs else None,
            redoc_url="/redoc" if gateway.enable_docs else None,
            openapi_url="/openapi.json" if gateway.enable_docs else None,
            lifespan=lifespan,
        )
        app.add_middleware(
            RequestBodyLimitMiddleware,
            max_bytes=int(os.environ.get("HEAR_API_MAX_BODY_BYTES", "2097152")),
        )
        app.include_router(gateway.router)
        return app

    @classmethod
    def main(cls) -> None:
        uvicorn.run(
            cls.create_app(),
            host=os.environ.get("HTTP_HOST", "0.0.0.0"),
            port=int(os.environ.get("HTTP_PORT", "8000")),
            log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        )


if __name__ == "__main__":
    GatewayEntrypoint.main()
