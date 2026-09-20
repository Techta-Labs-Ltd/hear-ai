import os
import sys

import grpc as _grpc
import ray
from ray import serve

from hear.proto.pipeline_pb2 import GetResultRequest, SubscribeRequest
from hear.proto.pipeline_pb2_grpc import PipelineStub


class SmokeTestCommand:
    @staticmethod
    def main():
        HEAR_SERVICE_KEY = os.environ.get("HEAR_SERVICE_KEY")
        if not HEAR_SERVICE_KEY:
            sys.exit("HEAR_SERVICE_KEY not set — source the .env used by the running cluster first")
        os.environ.pop("RAY_ADDRESS", None)
        ray.init(address="auto", ignore_reinit_error=True)
        status = serve.status()
        print("Ray Serve deployments:")
        for app_name, app_status in sorted(status.applications.items()):
            for dep_name, dep in sorted(app_status.deployments.items()):
                print(f"  {app_name}/{dep_name}: {dep.status} {dep.replica_states}")
        ch = _grpc.insecure_channel("localhost:50051")
        _grpc.channel_ready_future(ch).result(timeout=10)
        print("gRPC channel: CONNECTED")
        metadata = (("x-api-key", HEAR_SERVICE_KEY), ("application", "hear"))
        p_stub = PipelineStub(ch)
        try:
            for evt in p_stub.Subscribe(
                SubscribeRequest(job_id="smoke-test"), timeout=10, metadata=metadata
            ):
                print(f"Subscribe: event={evt.event} status={evt.status}")
                break
        except _grpc.RpcError as exc:
            print(f"Subscribe: {exc.code()} {exc.details()}")
        try:
            res = p_stub.GetResult(
                GetResultRequest(job_id="smoke-test"), timeout=10, metadata=metadata
            )
            print(f"GetResult: status={res.status or res.error}")
        except _grpc.RpcError as exc:
            print(f"GetResult: {exc.code()} {exc.details()}")
        print("\nSmoke test complete — Ray Serve gRPC:50051")


if __name__ == "__main__":
    SmokeTestCommand.main()
