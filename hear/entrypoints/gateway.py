from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from hear.api.body_limit import RequestBodyLimitMiddleware
from hear.api.gateway import PodGateway
from hear.runtime.gateway import RabbitMQGateway
from hear.runtime.roles import WorkerRole


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

    @classmethod
    def create_app(cls) -> FastAPI:
        runtime = RabbitMQGateway(
            os.environ.get("HEAR_RABBITMQ_URL", "amqp://guest:guest@127.0.0.1:5672/%2F"),
            cls.roles(),
        )
        gateway = PodGateway(
            runtime,
            os.environ.get("HEAR_POD_API_KEY", ""),
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
