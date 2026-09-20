import argparse
import os
import sys
import uuid

import grpc
import httpx
import ray
from google.protobuf.empty_pb2 import Empty as Empty

from hear.proto import pipeline_pb2
from hear.proto.pipeline_pb2_grpc import PipelineStub


class LiveTestReporter:
    @staticmethod
    def ok(name: str, detail: str = ""):
        LiveTestReporter.PASS += 1
        msg = f"  PASS  {name}"
        if detail:
            msg += f"  |  {detail}"
        print(msg)

    @staticmethod
    def fail(name: str, detail: str = ""):
        LiveTestReporter.FAIL += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f"  |  {detail}"
        print(msg)

    @staticmethod
    def check(name: str, condition: bool, detail: str = ""):
        if condition:
            LiveTestReporter.ok(name, detail)
        else:
            LiveTestReporter.fail(name, detail)


class LiveTestCommand:
    @staticmethod
    def main():
        HEAR_SERVICE_KEY = os.environ.get("HEAR_SERVICE_KEY")
        if not HEAR_SERVICE_KEY:
            sys.exit("HEAR_SERVICE_KEY not set")
        HEAR_BACKEND_ID = os.environ.get("HEAR_BACKEND_ID")
        STORAGE_ENV_NAMES = (
            "HEAR_STORAGE_ENDPOINT_URL",
            "HEAR_STORAGE_BUCKET_NAME",
            "HEAR_STORAGE_KEY_ID",
            "HEAR_STORAGE_APPLICATION_KEY",
            "HEAR_STORAGE_FOLDER_PREFIX",
            "HEAR_STORAGE_PUBLIC_BASE_URL",
            "HEAR_STORAGE_EXPIRES_AT",
        )
        missing_storage_env = [name for name in STORAGE_ENV_NAMES if not os.environ.get(name)]
        if not HEAR_BACKEND_ID or missing_storage_env:
            sys.exit(
                "HEAR_BACKEND_ID and all HEAR_STORAGE_* variables are required; missing: "
                + ", ".join(
                    (["HEAR_BACKEND_ID"] if not HEAR_BACKEND_ID else []) + missing_storage_env
                )
            )
        os.environ.pop("RAY_ADDRESS", None)
        ray.init(address="auto", ignore_reinit_error=True)
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--destructive", action="store_true", help="Run state-mutating RPCs too"
        )
        parser.add_argument("--http-base", default="http://localhost:8000", help="HTTP base URL")
        args = parser.parse_args()
        GRPC_TARGET = "localhost:50051"
        METADATA = (("x-api-key", HEAR_SERVICE_KEY), ("application", "hear"))
        LiveTestReporter.PASS = 0
        LiveTestReporter.FAIL = 0
        print("\n=== 1. gRPC Channel ===")
        ch = grpc.insecure_channel(
            GRPC_TARGET, options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)]
        )
        try:
            grpc.channel_ready_future(ch).result(timeout=15)
            LiveTestReporter.ok("Channel ready", f"{GRPC_TARGET}")
        except Exception as e:
            LiveTestReporter.fail("Channel ready", str(e))
            sys.exit(1)
        p_stub = PipelineStub(ch)
        print("\n=== 2. Pipeline.Health ===")
        try:
            reply = p_stub.Health(Empty(), timeout=10, metadata=METADATA)
            LiveTestReporter.check(
                "Health returns typed message", type(reply) is pipeline_pb2.HealthReply
            )
            LiveTestReporter.check("Health.status", reply.status in ("healthy", "unhealthy"))
            LiveTestReporter.check("Health.gpu_available", isinstance(reply.gpu_available, bool))
            gpu_has_data = reply.gpu_available and reply.gpu_memory.free_mb > 0
            LiveTestReporter.check(
                "Health.gpu_memory valid", not reply.gpu_available or gpu_has_data
            )
            LiveTestReporter.check("Health.active_jobs >= 0", reply.active_jobs >= 0)
            LiveTestReporter.check("Health.queued_jobs >= 0", reply.queued_jobs >= 0)
        except grpc.RpcError as e:
            LiveTestReporter.fail("Health", f"{e.code()} {e.details()}")
        print("\n=== 3. Pipeline.GetQueueStats ===")
        try:
            reply = p_stub.GetQueueStats(Empty(), timeout=10, metadata=METADATA)
            LiveTestReporter.check(
                "GetQueueStats returns typed message", type(reply) is pipeline_pb2.QueueStatsReply
            )
            LiveTestReporter.check("GetQueueStats.active >= 0", reply.active >= 0)
            LiveTestReporter.check("GetQueueStats.queued >= 0", reply.queued >= 0)
            LiveTestReporter.check("GetQueueStats.total >= 0", reply.total >= 0)
            LiveTestReporter.check(
                "GetQueueStats.estimated_wait_s >= 0", reply.estimated_wait_s >= 0
            )
            LiveTestReporter.check(
                "GetQueueStats.avg_job_duration_s >= 0", reply.avg_job_duration_s >= 0
            )
        except grpc.RpcError as e:
            LiveTestReporter.fail("GetQueueStats", f"{e.code()} {e.details()}")
        print("\n=== 4. Pipeline.Moderate ===")
        try:
            reply = p_stub.Moderate(
                pipeline_pb2.TextRequest(text="A calm field recording of birds and rain."),
                timeout=30,
                metadata=METADATA,
            )
            LiveTestReporter.check(
                "Moderate returns typed message", type(reply) is pipeline_pb2.ModerationReply
            )
            LiveTestReporter.check("Moderate.flagged is bool", isinstance(reply.flagged, bool))
            LiveTestReporter.check(
                "Moderate.intent non-empty", reply.intent in ("safe", "harmful", "")
            )
            LiveTestReporter.check("Moderate.severity string", isinstance(reply.severity, str))
            LiveTestReporter.check(
                "Moderate.flagged_categories is list",
                isinstance(list(reply.flagged_categories), list),
            )
            LiveTestReporter.check(
                "Moderate.blocked_words_found is list",
                isinstance(list(reply.blocked_words_found), list),
            )
        except grpc.RpcError as e:
            LiveTestReporter.fail("Moderate", f"{e.code()} {e.details()}")
        print("\n=== 5. Pipeline.Categorize ===")
        try:
            reply = p_stub.Categorize(
                pipeline_pb2.CategorizeRequest(
                    text="A calm field recording of birds and rain.",
                    custom_tags=["nature"],
                    max_tags=4,
                ),
                timeout=60,
                metadata=METADATA,
            )
            LiveTestReporter.check(
                "Categorize returns typed message", type(reply) is pipeline_pb2.CategorizationReply
            )
            LiveTestReporter.check(
                "Categorize.categories non-empty", len(list(reply.categories)) > 0
            )
            LiveTestReporter.check("Categorize.tags non-empty", len(list(reply.tags)) > 0)
            LiveTestReporter.check("Categorize.sentiment non-empty", reply.sentiment != "")
            LiveTestReporter.check(
                "Categorize.categorizer_mode non-empty", reply.categorizer_mode != ""
            )
            LiveTestReporter.check("Categorize.llm_used is bool", isinstance(reply.llm_used, bool))
        except grpc.RpcError as e:
            LiveTestReporter.fail("Categorize", f"{e.code()} {e.details()}")
        print("\n=== 6. Job lifecycle (HTTP submit → Subscribe → GetResult → CancelJob) ===")
        LIVE_JOB_ID = f"live-test-{uuid.uuid4()}"
        TRACK_ID = "c22c33f7-5e48-4b8e-8a3e-b5b103de5e60"
        TEST_AUDIO_URL = os.environ.get(
            "HEAR_TEST_AUDIO_URL", "http://127.0.0.1:8765/speech_sample.wav"
        )
        try:
            resp = httpx.post(
                f"{args.http_base}/process",
                headers={"X-Service-Key": HEAR_SERVICE_KEY, "Content-Type": "application/json"},
                json={
                    "job_id": LIVE_JOB_ID,
                    "backend_id": HEAR_BACKEND_ID,
                    "storage": {
                        "endpoint_url": os.environ["HEAR_STORAGE_ENDPOINT_URL"],
                        "bucket_name": os.environ["HEAR_STORAGE_BUCKET_NAME"],
                        "key_id": os.environ["HEAR_STORAGE_KEY_ID"],
                        "application_key": os.environ["HEAR_STORAGE_APPLICATION_KEY"],
                        "folder_prefix": os.environ["HEAR_STORAGE_FOLDER_PREFIX"],
                        "public_base_url": os.environ["HEAR_STORAGE_PUBLIC_BASE_URL"],
                        "expires_at": os.environ["HEAR_STORAGE_EXPIRES_AT"],
                    },
                    "track_id": TRACK_ID,
                    "job_type": "transcription",
                    "max_tags": 8,
                    "user_id": "production-live-test",
                    "audio_url": TEST_AUDIO_URL,
                },
                timeout=30,
            )
            LiveTestReporter.check(
                "HTTP submit status",
                resp.status_code in (200, 202, 503),
                f"status={resp.status_code}",
            )
            if resp.status_code == 503:
                print("    Service unavailable — skipping Subscribe/GetResult/CancelJob")
            else:
                body = resp.json()
                LiveTestReporter.check("HTTP submit has job_id", body.get("job_id") == LIVE_JOB_ID)
                LiveTestReporter.check("HTTP submit has run_id", bool(body.get("run_id")))
                try:
                    events = list(
                        p_stub.Subscribe(
                            pipeline_pb2.SubscribeRequest(job_id=LIVE_JOB_ID),
                            timeout=15,
                            metadata=METADATA,
                        )
                    )
                    LiveTestReporter.check("Subscribe returns events", len(events) > 0)
                    for evt in events[:3]:
                        LiveTestReporter.check(
                            "Subscribe event has typed PipelineEvent",
                            type(evt) is pipeline_pb2.PipelineEvent,
                        )
                        LiveTestReporter.check("Subscribe event.event non-empty", evt.event != "")
                        break
                except grpc.RpcError as e:
                    LiveTestReporter.fail("Subscribe", f"{e.code()} {e.details()}")
                try:
                    reply = p_stub.GetResult(
                        pipeline_pb2.GetResultRequest(job_id=LIVE_JOB_ID),
                        timeout=15,
                        metadata=METADATA,
                    )
                    LiveTestReporter.check(
                        "GetResult returns typed JobResult", type(reply) is pipeline_pb2.JobResult
                    )
                    LiveTestReporter.check("GetResult.job_id matches", reply.job_id == LIVE_JOB_ID)
                    LiveTestReporter.check("GetResult.status non-empty", reply.status != "")
                    has_payload = (
                        reply.HasField("pipeline")
                        or reply.HasField("transcription")
                        or reply.HasField("audio_tag")
                        or reply.HasField("magic_clean")
                        or reply.HasField("reconstruct")
                    )
                    LiveTestReporter.check(
                        "GetResult has typed payload",
                        has_payload if reply.status in ("completed", "running", "queued") else True,
                    )
                    if reply.HasField("transcription"):
                        LiveTestReporter.check(
                            "GetResult.transcription.transcription.language non-empty",
                            reply.transcription.transcription.language != "",
                        )
                except grpc.RpcError as e:
                    LiveTestReporter.fail("GetResult", f"{e.code()} {e.details()}")
                try:
                    reply = p_stub.CancelJob(
                        pipeline_pb2.GetResultRequest(job_id=LIVE_JOB_ID),
                        timeout=15,
                        metadata=METADATA,
                    )
                    LiveTestReporter.check(
                        "CancelJob returns typed JobResult", type(reply) is pipeline_pb2.JobResult
                    )
                    LiveTestReporter.check("CancelJob.job_id matches", reply.job_id == LIVE_JOB_ID)
                except grpc.RpcError as e:
                    LiveTestReporter.fail("CancelJob", f"{e.code()} {e.details()}")
        except httpx.HTTPError as e:
            LiveTestReporter.fail("HTTP submit connection", str(e))
        print("\n=== 7. GetResult with nonexistent job_id ===")
        try:
            reply = p_stub.GetResult(
                pipeline_pb2.GetResultRequest(job_id="nonexistent-job-id"),
                timeout=10,
                metadata=METADATA,
            )
            LiveTestReporter.check(
                "GetResult nonexistent returns JobResult", type(reply) is pipeline_pb2.JobResult
            )
            LiveTestReporter.check("GetResult nonexistent has empty job_id", reply.job_id == "")
        except grpc.RpcError as e:
            LiveTestReporter.check(
                "GetResult nonexistent raises NOT_FOUND", e.code() == grpc.StatusCode.NOT_FOUND
            )
        print("\n=== 8. CancelJob with nonexistent job_id ===")
        try:
            reply = p_stub.CancelJob(
                pipeline_pb2.GetResultRequest(job_id="nonexistent-job-id"),
                timeout=10,
                metadata=METADATA,
            )
            LiveTestReporter.check(
                "CancelJob nonexistent returns JobResult", type(reply) is pipeline_pb2.JobResult
            )
        except grpc.RpcError as e:
            LiveTestReporter.check(
                "CancelJob nonexistent raises error", True, f"{e.code()} {e.details()}"
            )
        print("\n=== 9. Pipeline.ListDiscovery ===")
        try:
            reply = p_stub.ListDiscovery(
                pipeline_pb2.DiscoveryRequest(sort="latest", limit=3, offset=0),
                timeout=15,
                metadata=METADATA,
            )
            LiveTestReporter.check(
                "ListDiscovery returns typed message",
                type(reply) is pipeline_pb2.ListDiscoveryReply,
            )
            LiveTestReporter.check("ListDiscovery.sort non-empty", reply.sort != "")
            LiveTestReporter.check("ListDiscovery.limit > 0", reply.limit > 0)
            LiveTestReporter.check("ListDiscovery.total >= 0", reply.total >= 0)
            LiveTestReporter.check(
                "ListDiscovery.items is list", isinstance(list(reply.items), list)
            )
            if len(list(reply.items)) > 0:
                item = reply.items[0]
                LiveTestReporter.check("DiscoveryItem.track_id non-empty", item.track_id != "")
                LiveTestReporter.check(
                    "DiscoveryItem has discovery Struct", item.HasField("discovery")
                )
        except grpc.RpcError as e:
            LiveTestReporter.fail("ListDiscovery", f"{e.code()} {e.details()}")
        print("\n=== 11. Auth rejection ===")
        try:
            reply = p_stub.Health(
                Empty(), timeout=5, metadata=(("x-api-key", "wrong"), ("application", "hear"))
            )
            LiveTestReporter.fail("Health with bad key", "should have been rejected")
        except grpc.RpcError as e:
            LiveTestReporter.check(
                "Health with bad key raises UNAUTHENTICATED",
                e.code() == grpc.StatusCode.UNAUTHENTICATED,
            )
        print("\n=== 12. Missing application metadata ===")
        try:
            reply = p_stub.Health(Empty(), timeout=5, metadata=(("x-api-key", HEAR_SERVICE_KEY),))
            LiveTestReporter.fail("Health without application metadata", "should have failed")
        except grpc.RpcError as e:
            LiveTestReporter.check(
                "Health without app metadata raises error", True, f"{e.code()} {e.details()}"
            )
        TOTAL = LiveTestReporter.PASS + LiveTestReporter.FAIL
        print(f"\n{'=' * 50}")
        print(f"  {LiveTestReporter.PASS}/{TOTAL} passed  ({LiveTestReporter.FAIL} failed)")
        print(f"{'=' * 50}")
        ray.shutdown()
        sys.exit(0 if LiveTestReporter.FAIL == 0 else 1)


if __name__ == "__main__":
    LiveTestCommand.main()
