import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest

from hear.proto import cleaner_v2_pb2 as wire
from hear.proto import cleaner_v2_pb2_grpc as rpc
from hear.runtime.cleaner.executor import PublishedExecutionError
from hear.runtime.cleaner.grpc_ingress import CleanerGrpcIngress, CleanerPrincipal
from hear.runtime.cleaner.wire import CleanerWireCodec
from hear.services.magic_clean.artifacts import ArtifactWriter
from hear.services.magic_clean.contracts import ErrorCode
from tests.test_cleaner_v2_artifacts import MemoryStore
from tests.test_cleaner_v2_artifacts import bundle as bundle_fixture
from tests.test_cleaner_v2_wire import envelope as envelope_fixture
from tests.test_cleaner_v2_wire import ticket as ticket_fixture

bundle, envelope, ticket = bundle_fixture, envelope_fixture, ticket_fixture


class Aborted(Exception):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail


class Context:
    active = True
    remaining = 30

    def abort(self, status, detail):
        raise Aborted(status, detail)

    def add_callback(self, callback):
        self.callback = callback
        return self.active

    def is_active(self):
        return self.active

    def time_remaining(self):
        return self.remaining


@pytest.fixture
def request_and_principal(envelope):
    return (
        wire.ExecuteAttemptRequest.FromString(CleanerWireCodec.encode(envelope)),
        CleanerPrincipal(envelope.ticket.backend_id, frozenset((envelope.ticket.tenant_scope,))),
    )


@pytest.mark.parametrize("failure", ["auth", "backend", "tenant", "inactive", "deadline", "busy"])
def test_rejected_request_never_dispatches(request_and_principal, failure):
    request, principal = request_and_principal
    context = Context()
    expected = grpc.StatusCode.PERMISSION_DENIED
    if failure == "auth":
        principal = None
        request.Clear()  # Authentication precedes semantic decoding.
        expected = grpc.StatusCode.UNAUTHENTICATED
    elif failure == "backend":
        principal = CleanerPrincipal("foreign", principal.tenant_scopes)
    elif failure == "tenant":
        principal = CleanerPrincipal(principal.backend_id, frozenset(("foreign",)))
    elif failure == "inactive":
        context.active = False
        expected = grpc.StatusCode.CANCELLED
    elif failure == "deadline":
        context.remaining = 0
        expected = grpc.StatusCode.DEADLINE_EXCEEDED
    else:
        expected = grpc.StatusCode.RESOURCE_EXHAUSTED
    ingress = CleanerGrpcIngress(lambda context: principal, lambda *args: pytest.fail("dispatched"))
    if failure == "busy":
        ingress._slot.acquire()
    try:
        with pytest.raises(Aborted) as error:
            ingress.ExecuteAttempt(request, context)
        assert error.value.status == expected
    finally:
        if failure == "busy":
            ingress._slot.release()


def test_callback_propagates_and_native_details_are_redacted(request_and_principal):
    request, principal = request_and_principal
    context = Context()

    def dispatch(decoded, cancelled, deadline):
        assert time.monotonic() < deadline <= time.monotonic() + 30
        assert not cancelled.is_set()
        context.callback()
        assert cancelled.is_set()
        raise RuntimeError("private-token-never-log")

    ingress = CleanerGrpcIngress(lambda context: principal, dispatch)
    with pytest.raises(Aborted) as error:
        ingress.ExecuteAttempt(request, context)
    assert error.value.status == grpc.StatusCode.INTERNAL
    assert "private-token" not in error.value.detail
    assert not ingress._slot.locked()


@pytest.mark.parametrize("oversized", [False, True])
def test_bad_envelope_is_rejected_before_dispatch(request_and_principal, oversized):
    request, principal = request_and_principal
    if oversized:
        request.source_read_grant.token = "x" * (CleanerWireCodec.MAX_REQUEST_BYTES + 1)
    else:
        request.ClearField("ticket")
    ingress = CleanerGrpcIngress(lambda context: principal, lambda *args: pytest.fail("dispatched"))
    with pytest.raises(Aborted) as error:
        ingress.ExecuteAttempt(request, Context())
    assert error.value.status == (
        grpc.StatusCode.RESOURCE_EXHAUSTED if oversized else grpc.StatusCode.INVALID_ARGUMENT
    )
    assert not ingress._slot.locked()


def test_real_grpc_returns_verified_published_failure(envelope, bundle, request_and_principal):
    request, principal = request_and_principal
    published = ArtifactWriter(MemoryStore()).publish_failure(
        envelope.ticket, ErrorCode.ENGINE_UNAVAILABLE, bundle[3]
    )

    def dispatch(decoded, cancelled, deadline):
        assert decoded == envelope and not cancelled.is_set() and deadline is not None
        raise PublishedExecutionError(ErrorCode.ENGINE_UNAVAILABLE, published)

    ingress = CleanerGrpcIngress(lambda context: principal, dispatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        server = grpc.server(pool, options=CleanerGrpcIngress.SERVER_OPTIONS)
        rpc.add_CleanerExecutionServicer_to_server(ingress, server)
        port = server.add_insecure_port("127.0.0.1:0")  # Loopback fixture only, not deployment.
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                response = rpc.CleanerExecutionStub(channel).ExecuteAttempt(request, timeout=10)
                reference = CleanerWireCodec.decode_result(
                    response.SerializeToString(), envelope.ticket
                )
                assert reference.outcome == "failed"
                assert reference.error_code == ErrorCode.ENGINE_UNAVAILABLE
                assert reference.sha256 == published.reference.sha256
        finally:
            server.stop(0).wait()
    assert not ingress._slot.locked()
