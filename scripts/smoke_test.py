#!/usr/bin/env python3
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HEAR_SERVICE_KEY = os.environ.get("HEAR_SERVICE_KEY")
if not HEAR_SERVICE_KEY:
    sys.exit("HEAR_SERVICE_KEY not set — source the .env used by the running cluster first")

import ray

os.environ.pop("RAY_ADDRESS", None)
from ray import serve

ray.init(address="auto", ignore_reinit_error=True)

status = serve.status()
print("Ray Serve deployments:")
for app_name, app_status in sorted(status.applications.items()):
    for dep_name, dep in sorted(app_status.deployments.items()):
        print(f"  {app_name}/{dep_name}: {dep.status} {dep.replica_states}")

import grpc as _grpc

from hear.proto.pipeline_pb2 import GetResultRequest, SubscribeRequest
from hear.proto.pipeline_pb2_grpc import PipelineStub
from hear.proto.resolver_pb2 import HealthRequest, ResolveRequest
from hear.proto.resolver_pb2_grpc import ResolverStub

ch = _grpc.insecure_channel("localhost:50051")
_grpc.channel_ready_future(ch).result(timeout=10)
print("gRPC channel: CONNECTED")

metadata = (("x-api-key", HEAR_SERVICE_KEY), ("application", "hear"))

r_stub = ResolverStub(ch)
reply = r_stub.Resolve(
    ResolveRequest(utterance="play jazz music", country_code="US"),
    timeout=10,
    metadata=metadata,
)
print(f"Resolve: cat={reply.category.name if reply.category else 'none'}")
print(
    "Resolver health:",
    r_stub.ResolverHealth(HealthRequest(), timeout=5, metadata=metadata),
)

p_stub = PipelineStub(ch)
try:
    for evt in p_stub.Subscribe(
        SubscribeRequest(job_id="smoke-test"),
        timeout=10,
        metadata=metadata,
    ):
        print(f"Subscribe: event={evt.event} status={evt.status}")
        break
except _grpc.RpcError as exc:
    print(f"Subscribe: {exc.code()} {exc.details()}")

try:
    res = p_stub.GetResult(
        GetResultRequest(job_id="smoke-test"),
        timeout=10,
        metadata=metadata,
    )
    print(f"GetResult: status={res.status or res.error}")
except _grpc.RpcError as exc:
    print(f"GetResult: {exc.code()} {exc.details()}")

print("\nSmoke test complete — Ray Serve gRPC:50051")
