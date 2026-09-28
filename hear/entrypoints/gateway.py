from __future__ import annotations

import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from hear.api.body_limit import RequestBodyLimitMiddleware
from hear.api.gateway import PodGateway
from hear.runtime.cleaner.asset_probe import PinnedAssetProbe
from hear.runtime.gateway import RabbitMQGateway
from hear.runtime.ownership import DeploymentOwnership
from hear.runtime.roles import WorkerRole
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
    def sound_cleanup_ready() -> bool:
        bundle = os.environ.get("HEAR_SOUND_CLEANUP_BUNDLE", "").strip()
        digest = os.environ.get("HEAR_SOUND_CLEANUP_BUNDLE_SHA256", "").strip()
        if not bundle or not digest:
            return False
        try:
            SoundCleanupAssets.load(Path(bundle), digest)
            return True
        except Exception:
            logging.getLogger(__name__).warning("Sound Cleanup bundle is not ready")
            return False

    @staticmethod
    def overlap_preview_ready() -> bool:
        path = os.environ.get("HEAR_SOUND_CLEANUP_SEPARATOR_BUNDLE", "")
        digest = os.environ.get("HEAR_SOUND_CLEANUP_SEPARATOR_SHA256", "")
        if not path or not digest:
            return False
        try:
            root = Path(path)
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
    def create_app(cls) -> FastAPI:
        runtime = RabbitMQGateway(
            os.environ.get("HEAR_RABBITMQ_URL", "amqp://guest:guest@127.0.0.1:5672/%2F"),
            cls.roles(),
        )
        gateway = PodGateway(
            runtime,
            os.environ.get("HEAR_POD_API_KEY", ""),
            sound_cleanup_available=cls.sound_cleanup_ready(),
            overlap_preview_available=cls.overlap_preview_ready(),
            ownership_policy=DeploymentOwnership.load(),
            cleaning_mode=os.environ.get("HEAR_OPTIONAL_ENGINE_MODE", "available"),
            enable_docs=os.environ.get("HEAR_ENABLE_DOCS", "false").lower() == "true",
        )

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            await runtime.start()
            try:
                yield
            finally:
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
