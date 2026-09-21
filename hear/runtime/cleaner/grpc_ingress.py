import math
import threading
import time
from dataclasses import dataclass

import grpc

from hear.proto import cleaner_v2_pb2 as wire
from hear.proto.cleaner_v2_pb2_grpc import CleanerExecutionServicer
from hear.runtime.cleaner.executor import PublishedExecutionError
from hear.runtime.cleaner.wire import CleanerWireCodec
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@dataclass(frozen=True)
class CleanerPrincipal:
    """Trusted authenticator output, not values copied from the request ticket."""

    backend_id: str
    tenant_scopes: frozenset[str]

    def __post_init__(self):
        if (
            not isinstance(self.backend_id, str)
            or not self.backend_id
            or not isinstance(self.tenant_scopes, frozenset)
            or not self.tenant_scopes
            or any(not isinstance(scope, str) or not scope for scope in self.tenant_scopes)
        ):
            raise ValueError("invalid authenticated cleaner principal")


class IngressRejection(Exception):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail


class CleanerGrpcIngress(CleanerExecutionServicer):
    # Pass these to the dedicated grpc.server. Post-deserialization ByteSize checks
    # alone do not bound allocation of an incoming protobuf message.
    SERVER_OPTIONS = (
        ("grpc.max_receive_message_length", CleanerWireCodec.MAX_REQUEST_BYTES),
        ("grpc.max_send_message_length", CleanerWireCodec.MAX_REFERENCE_BYTES),
    )

    def __init__(self, authenticate, dispatch):
        """Inject transport authentication and the existing executor/provider adapter.

        dispatch(decoded, cancelled, deadline_monotonic) must honor cancellation,
        tighten the ticket deadline, and enforce current fences/scoped grants via
        the executor authorizer. It returns a PublishedBundle. No durable job state
        or automatic retry is owned here. TLS and actual credentials remain the
        dedicated server's responsibility; never expose this route to browsers.
        """
        self.authenticate, self.dispatch = authenticate, dispatch
        self._slot = threading.Lock()

    def ExecuteAttempt(self, request, context):
        try:
            return self._execute(request, context)
        except IngressRejection as error:
            return context.abort(error.status, error.detail)
        except CleanExecutionError as error:
            status = {
                ErrorCode.CANCELLED: grpc.StatusCode.CANCELLED,
                ErrorCode.DEADLINE_EXCEEDED: grpc.StatusCode.DEADLINE_EXCEEDED,
                ErrorCode.RESOURCE_EXHAUSTED: grpc.StatusCode.RESOURCE_EXHAUSTED,
                ErrorCode.ENGINE_UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
                ErrorCode.ARTIFACT_CONFLICT: grpc.StatusCode.ABORTED,
                ErrorCode.INVALID_AUDIO: grpc.StatusCode.INVALID_ARGUMENT,
            }.get(error.code, grpc.StatusCode.INTERNAL)
            return context.abort(status, "cleaner attempt rejected")
        except Exception:
            # Native errors/validation inputs can contain credentials or paths.
            return context.abort(grpc.StatusCode.INTERNAL, "cleaner execution failed")

    def _execute(self, request, context):
        try:
            principal = self.authenticate(context)
        except Exception:
            principal = None
        if not isinstance(principal, CleanerPrincipal):
            raise IngressRejection(grpc.StatusCode.UNAUTHENTICATED, "authentication required")
        if request.ByteSize() > CleanerWireCodec.MAX_REQUEST_BYTES:
            raise IngressRejection(grpc.StatusCode.RESOURCE_EXHAUSTED, "request too large")
        decoded = CleanerWireCodec.decode(request.SerializeToString(deterministic=True))
        if (
            decoded.ticket.backend_id != principal.backend_id
            or decoded.ticket.tenant_scope not in principal.tenant_scopes
        ):
            raise IngressRejection(grpc.StatusCode.PERMISSION_DENIED, "attempt scope denied")
        cancelled = threading.Event()
        if not context.add_callback(cancelled.set) or not context.is_active():
            raise IngressRejection(grpc.StatusCode.CANCELLED, "request is no longer active")
        remaining = context.time_remaining()
        deadline = None
        if remaining is not None:
            if not math.isfinite(remaining) or remaining <= 0:
                raise IngressRejection(grpc.StatusCode.DEADLINE_EXCEEDED, "request expired")
            deadline = time.monotonic() + remaining
        if not self._slot.acquire(blocking=False):
            raise IngressRejection(grpc.StatusCode.RESOURCE_EXHAUSTED, "cleaner worker occupied")
        try:
            if cancelled.is_set():
                raise IngressRejection(grpc.StatusCode.CANCELLED, "request cancelled")
            try:
                bundle = self.dispatch(decoded, cancelled, deadline)
            except PublishedExecutionError as error:
                bundle = error.bundle
            payload = CleanerWireCodec.encode_result(decoded.ticket, bundle)
            return wire.ManifestReference.FromString(payload)
        finally:
            self._slot.release()
