import asyncio
import logging

from ray import serve

from hear.config import settings
from hear.proto import resolver_pb2
from hear.resolver import main as resolver_main
from hear.resolver.version_coordinator import get_version_coordinator

logger = logging.getLogger(__name__)


def _entity(e):
    return resolver_pb2.Entity(
        id=e.id or "",
        name=e.name or "",
        slug=e.slug or "",
        city=e.city or "",
        lat=e.lat or "",
        lng=e.lng or "",
        post_code=e.post_code or "",
        confidence=e.confidence or 0.0,
        resolution=e.resolution or "",
    )


def _to_reply(resp):
    reply = resolver_pb2.ResolveReply(
        version=resp.version,
        action=resp.action or "",
        freetext=resp.freetext or "",
        cache_hit=bool(resp.cache_hit),
        resolved_in_ms=resp.resolved_in_ms or 0.0,
    )
    if resp.category:
        reply.category.CopyFrom(_entity(resp.category))
    if resp.creator:
        reply.creator.CopyFrom(_entity(resp.creator))
    if resp.organisation:
        reply.organisation.CopyFrom(_entity(resp.organisation))
    if resp.location:
        reply.location.CopyFrom(_entity(resp.location))
    for t in resp.tags:
        reply.tags.append(_entity(t))
    for c in resp.candidates:
        reply.candidates.append(_entity(c))
    if resp.temporal:
        reply.temporal.CopyFrom(resolver_pb2.Temporal(
            type=resp.temporal.get("type", ""),
            value=resp.temporal.get("value", ""),
            date=resp.temporal.get("date") or "",
        ))
    return reply


@serve.deployment(
    name="resolver",
    ray_actor_options={
        "num_cpus": 0.3,
        "num_gpus": settings.RESOLVER_NUM_GPUS,
    },
    num_replicas=settings.RESOLVER_REPLICA_COUNT,
    health_check_period_s=10,
    health_check_timeout_s=30,
    max_ongoing_requests=10,
)
class ResolverDeployment:
    """Own the resolver model, index, cache, and rebuild lifecycle."""

    async def __init__(self):
        self._bootstrap_error: Exception | None = None
        self._version_coordinator = None
        self._version_sync_task: asyncio.Task | None = None
        resolver_main.manager.on_ready = resolver_main._on_ready
        await self._bootstrap()

    async def _bootstrap(self):
        try:
            logger.info("bootstrapping...")
            resolver_main.semantic.load_model()
            await resolver_main.cache.connect()
            self._version_coordinator = get_version_coordinator()
            await resolver_main.manager.startup()
            current = int(resolver_main.manager.get_active().get("version", 0))
            await self._version_coordinator.publish_version.remote(current)
            self._version_sync_task = asyncio.create_task(self._sync_versions())
            logger.info("ready version=%s", current)
        except Exception as exc:
            self._bootstrap_error = exc
            logger.exception("resolver bootstrap failed")
            raise

    async def _sync_versions(self) -> None:
        while True:
            try:
                desired = int(
                    await self._version_coordinator.get_desired_version.remote()
                )
                current = int(resolver_main.manager.get_active().get("version", 0))
                if desired > 0 and desired != current:
                    await resolver_main.manager.load_version(desired)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("resolver version synchronization failed")
            await asyncio.sleep(settings.RESOLVER_VERSION_SYNC_SECONDS)

    async def Resolve(self, request, grpc_context=None):
        if not resolver_main.manager.is_ready():
            raise RuntimeError("resolver index is not ready")

        resp = await resolver_main.resolve_utterance(
            request.utterance, request.country_code or "gb"
        )
        return _to_reply(resp)

    async def Health(self, request, grpc_context=None):
        ready = resolver_main.manager.is_ready()
        return resolver_pb2.HealthReply(
            status="ok" if ready else "loading",
            version=int(resolver_main.manager.get_active().get("version", 0)),
            ready=ready,
        )

    async def check_health(self) -> None:
        if self._bootstrap_error is not None:
            raise RuntimeError("resolver bootstrap failed") from self._bootstrap_error

    async def Rebuild(self, request):
        target = request.version if request.HasField("version") else None
        if target is None:
            target = await resolver_main.manager.latest_version()
        if target and self._version_coordinator is not None:
            await self._version_coordinator.publish_version.remote(target)
        domain_request = resolver_main.RebuildRequest(
            version=target
        )
        result = await resolver_main.rebuild(domain_request)
        return resolver_pb2.RebuildReply(
            status=result.status,
            current_version=result.current_version,
            detail=result.detail or "",
        )

    async def Apply(self, request):
        if not request.HasField("version"):
            return resolver_pb2.RebuildReply(
                status="failed",
                current_version=int(resolver_main.manager.get_active().get("version", 0)),
                detail="version is required",
            )
        if self._version_coordinator is not None:
            await self._version_coordinator.publish_version.remote(request.version)
        result = await resolver_main.apply(resolver_main.RebuildRequest(version=request.version))
        return resolver_pb2.RebuildReply(
            status=result.status,
            current_version=result.current_version,
            detail=result.detail or "",
        )


resolver_app = ResolverDeployment.bind()
