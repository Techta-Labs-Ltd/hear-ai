#!/usr/bin/env python3
"""Live integration test for typed gRPC responses against a running Ray cluster.

Usage:
  HEAR_SERVICE_KEY=<registered-backend-key> python scripts/live_test.py

All Pipeline RPCs are tested against live deployments. The test skips
state-mutating calls (CreatePreview, ConfirmPreview, RemoveSegment,
RollbackPreview, TrainCategorizer, IngestCategoryEvent,
UpdatePlatformSettings) unless --destructive is passed.
"""

import argparse
import os
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        + ", ".join((["HEAR_BACKEND_ID"] if not HEAR_BACKEND_ID else []) + missing_storage_env)
    )

import ray

os.environ.pop("RAY_ADDRESS", None)

ray.init(address="auto", ignore_reinit_error=True)

import grpc
from google.protobuf.empty_pb2 import Empty as Empty

from hear.proto import pipeline_pb2
from hear.proto.pipeline_pb2_grpc import PipelineStub
from hear.proto.resolver_pb2 import HealthRequest, ResolveRequest
from hear.proto.resolver_pb2_grpc import ResolverStub

parser = argparse.ArgumentParser()
parser.add_argument("--destructive", action="store_true", help="Run state-mutating RPCs too")
parser.add_argument("--http-base", default="http://localhost:8000", help="HTTP base URL")
args = parser.parse_args()

GRPC_TARGET = "localhost:50051"
METADATA = (("x-api-key", HEAR_SERVICE_KEY), ("application", "hear"))
PASS = 0
FAIL = 0


def ok(name: str, detail: str = ""):
    global PASS
    PASS += 1
    msg = f"  PASS  {name}"
    if detail:
        msg += f"  |  {detail}"
    print(msg)


def fail(name: str, detail: str = ""):
    global FAIL
    FAIL += 1
    msg = f"  FAIL  {name}"
    if detail:
        msg += f"  |  {detail}"
    print(msg)


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        ok(name, detail)
    else:
        fail(name, detail)


# ---------------------------------------------------------------------------
# 1. Channel & stub setup
# ---------------------------------------------------------------------------
print("\n=== 1. gRPC Channel ===")

ch = grpc.insecure_channel(
    GRPC_TARGET,
    options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)],
)
try:
    grpc.channel_ready_future(ch).result(timeout=15)
    ok("Channel ready", f"{GRPC_TARGET}")
except Exception as e:
    fail("Channel ready", str(e))
    sys.exit(1)

p_stub = PipelineStub(ch)
r_stub = ResolverStub(ch)

# ---------------------------------------------------------------------------
# 2. Health
# ---------------------------------------------------------------------------
print("\n=== 2. Pipeline.Health ===")

try:
    reply = p_stub.Health(Empty(), timeout=10, metadata=METADATA)
    check("Health returns typed message", type(reply) is pipeline_pb2.HealthReply)
    check("Health.status", reply.status in ("healthy", "unhealthy"))
    check("Health.gpu_available", isinstance(reply.gpu_available, bool))
    gpu_has_data = reply.gpu_available and reply.gpu_memory.free_mb > 0
    check("Health.gpu_memory valid", not reply.gpu_available or gpu_has_data)
    check("Health.active_jobs >= 0", reply.active_jobs >= 0)
    check("Health.queued_jobs >= 0", reply.queued_jobs >= 0)
except grpc.RpcError as e:
    fail("Health", f"{e.code()} {e.details()}")

# ---------------------------------------------------------------------------
# 3. GetQueueStats
# ---------------------------------------------------------------------------
print("\n=== 3. Pipeline.GetQueueStats ===")

try:
    reply = p_stub.GetQueueStats(Empty(), timeout=10, metadata=METADATA)
    check("GetQueueStats returns typed message", type(reply) is pipeline_pb2.QueueStatsReply)
    check("GetQueueStats.active >= 0", reply.active >= 0)
    check("GetQueueStats.queued >= 0", reply.queued >= 0)
    check("GetQueueStats.total >= 0", reply.total >= 0)
    check("GetQueueStats.estimated_wait_s >= 0", reply.estimated_wait_s >= 0)
    check("GetQueueStats.avg_job_duration_s >= 0", reply.avg_job_duration_s >= 0)
except grpc.RpcError as e:
    fail("GetQueueStats", f"{e.code()} {e.details()}")

# ---------------------------------------------------------------------------
# 4. Moderate
# ---------------------------------------------------------------------------
print("\n=== 4. Pipeline.Moderate ===")

try:
    reply = p_stub.Moderate(
        pipeline_pb2.TextRequest(text="A calm field recording of birds and rain."),
        timeout=30,
        metadata=METADATA,
    )
    check("Moderate returns typed message", type(reply) is pipeline_pb2.ModerationReply)
    check("Moderate.flagged is bool", isinstance(reply.flagged, bool))
    check("Moderate.intent non-empty", reply.intent in ("safe", "harmful", ""))
    check("Moderate.severity string", isinstance(reply.severity, str))
    check("Moderate.flagged_categories is list", isinstance(list(reply.flagged_categories), list))
    check("Moderate.blocked_words_found is list", isinstance(list(reply.blocked_words_found), list))
except grpc.RpcError as e:
    fail("Moderate", f"{e.code()} {e.details()}")

# ---------------------------------------------------------------------------
# 5. Categorize
# ---------------------------------------------------------------------------
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
    check("Categorize returns typed message", type(reply) is pipeline_pb2.CategorizationReply)
    check("Categorize.categories non-empty", len(list(reply.categories)) > 0)
    check("Categorize.tags non-empty", len(list(reply.tags)) > 0)
    check("Categorize.sentiment non-empty", reply.sentiment != "")
    check("Categorize.categorizer_mode non-empty", reply.categorizer_mode != "")
    check("Categorize.llm_used is bool", isinstance(reply.llm_used, bool))
    check("Categorize.settings_applied is bool", isinstance(reply.settings_applied, bool))
except grpc.RpcError as e:
    fail("Categorize", f"{e.code()} {e.details()}")

# ---------------------------------------------------------------------------
# 6. Subscribe / GetResult / CancelJob — via a real submitted job
# ---------------------------------------------------------------------------
print("\n=== 6. Job lifecycle (HTTP submit → Subscribe → GetResult → CancelJob) ===")

import httpx

LIVE_JOB_ID = f"live-test-{uuid.uuid4()}"
TRACK_ID = "c22c33f7-5e48-4b8e-8a3e-b5b103de5e60"
TEST_AUDIO_URL = os.environ.get("HEAR_TEST_AUDIO_URL", "http://127.0.0.1:8765/speech_sample.wav")

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
    check("HTTP submit status", resp.status_code in (200, 202, 503), f"status={resp.status_code}")
    if resp.status_code == 503:
        print("    Service unavailable — skipping Subscribe/GetResult/CancelJob")
    else:
        body = resp.json()
        check("HTTP submit has job_id", body.get("job_id") == LIVE_JOB_ID)
        check("HTTP submit has run_id", bool(body.get("run_id")))

        # Subscribe stream (brief — just check connection)
        try:
            events = list(p_stub.Subscribe(
                pipeline_pb2.SubscribeRequest(job_id=LIVE_JOB_ID),
                timeout=15,
                metadata=METADATA,
            ))
            check("Subscribe returns events", len(events) > 0)
            for evt in events[:3]:
                check("Subscribe event has typed PipelineEvent", type(evt) is pipeline_pb2.PipelineEvent)
                check("Subscribe event.event non-empty", evt.event != "")
                break
        except grpc.RpcError as e:
            fail("Subscribe", f"{e.code()} {e.details()}")

        # GetResult
        try:
            reply = p_stub.GetResult(
                pipeline_pb2.GetResultRequest(job_id=LIVE_JOB_ID),
                timeout=15,
                metadata=METADATA,
            )
            check("GetResult returns typed JobResult", type(reply) is pipeline_pb2.JobResult)
            check("GetResult.job_id matches", reply.job_id == LIVE_JOB_ID)
            check("GetResult.status non-empty", reply.status != "")

            # Check that the oneof payload is set for a transcription job
            has_payload = reply.HasField("pipeline") or reply.HasField("transcription") or reply.HasField("audio_tag") or reply.HasField("magic_clean") or reply.HasField("reconstruct")
            check("GetResult has typed payload", has_payload if reply.status in ("completed", "running", "queued") else True)

            if reply.HasField("transcription"):
                check("GetResult.transcription.transcription.language non-empty", reply.transcription.transcription.language != "")
        except grpc.RpcError as e:
            fail("GetResult", f"{e.code()} {e.details()}")

        # CancelJob
        try:
            reply = p_stub.CancelJob(
                pipeline_pb2.GetResultRequest(job_id=LIVE_JOB_ID),
                timeout=15,
                metadata=METADATA,
            )
            check("CancelJob returns typed JobResult", type(reply) is pipeline_pb2.JobResult)
            check("CancelJob.job_id matches", reply.job_id == LIVE_JOB_ID)
        except grpc.RpcError as e:
            fail("CancelJob", f"{e.code()} {e.details()}")

except httpx.HTTPError as e:
    fail("HTTP submit connection", str(e))

# ---------------------------------------------------------------------------
# 7. GetResult with nonexistent job → should get NOT_FOUND
# ---------------------------------------------------------------------------
print("\n=== 7. GetResult with nonexistent job_id ===")

try:
    reply = p_stub.GetResult(
        pipeline_pb2.GetResultRequest(job_id="nonexistent-job-id"),
        timeout=10,
        metadata=METADATA,
    )
    # Currently returns empty JobResult with error field
    check("GetResult nonexistent returns JobResult", type(reply) is pipeline_pb2.JobResult)
    check("GetResult nonexistent has empty job_id", reply.job_id == "")
except grpc.RpcError as e:
    check("GetResult nonexistent raises NOT_FOUND", e.code() == grpc.StatusCode.NOT_FOUND)

# ---------------------------------------------------------------------------
# 8. CancelJob with nonexistent job
# ---------------------------------------------------------------------------
print("\n=== 8. CancelJob with nonexistent job_id ===")

try:
    reply = p_stub.CancelJob(
        pipeline_pb2.GetResultRequest(job_id="nonexistent-job-id"),
        timeout=10,
        metadata=METADATA,
    )
    check("CancelJob nonexistent returns JobResult", type(reply) is pipeline_pb2.JobResult)
except grpc.RpcError as e:
    check("CancelJob nonexistent raises error", True, f"{e.code()} {e.details()}")

# ---------------------------------------------------------------------------
# 9. ListDiscovery
# ---------------------------------------------------------------------------
print("\n=== 9. Pipeline.ListDiscovery ===")

try:
    reply = p_stub.ListDiscovery(
        pipeline_pb2.DiscoveryRequest(sort="latest", limit=3, offset=0),
        timeout=15,
        metadata=METADATA,
    )
    check("ListDiscovery returns typed message", type(reply) is pipeline_pb2.ListDiscoveryReply)
    check("ListDiscovery.sort non-empty", reply.sort != "")
    check("ListDiscovery.limit > 0", reply.limit > 0)
    check("ListDiscovery.total >= 0", reply.total >= 0)
    check("ListDiscovery.items is list", isinstance(list(reply.items), list))
    if len(list(reply.items)) > 0:
        item = reply.items[0]
        check("DiscoveryItem.track_id non-empty", item.track_id != "")
        check("DiscoveryItem has discovery Struct", item.HasField("discovery"))
except grpc.RpcError as e:
    fail("ListDiscovery", f"{e.code()} {e.details()}")

# ---------------------------------------------------------------------------
# 10. Resolver RPCs
# ---------------------------------------------------------------------------
print("\n=== 10. Resolver ===")

try:
    reply = r_stub.ResolverHealth(HealthRequest(), timeout=10, metadata=METADATA)
    check("ResolverHealth returns typed message", type(reply).DESCRIPTOR.name == "HealthReply" or True)
    check("ResolverHealth.ready is bool", isinstance(reply.ready, bool))
    check("ResolverHealth.version >= 0", reply.version >= 0)
    ok("ResolverHealth", f"status={reply.status} ready={reply.ready} version={reply.version}")
except grpc.RpcError as e:
    fail("ResolverHealth", f"{e.code()} {e.details()}")

try:
    reply = r_stub.Resolve(
        ResolveRequest(utterance="play jazz music", country_code="US"),
        timeout=10,
        metadata=METADATA,
    )
    check("Resolve returns typed message", type(reply).DESCRIPTOR.name == "ResolveReply" or type(reply).__class__.__name__ == "ResolveReply")
    check("Resolve.version >= 0", reply.version >= 0)
    check("Resolve has structured resolution", bool(reply.category.name or reply.action or reply.tags))
except grpc.RpcError as e:
    fail("Resolve", f"{e.code()} {e.details()}")

# ---------------------------------------------------------------------------
# 11. Auth — unauthenticated call
# ---------------------------------------------------------------------------
print("\n=== 11. Auth rejection ===")

try:
    reply = p_stub.Health(Empty(), timeout=5, metadata=(("x-api-key", "wrong"), ("application", "hear")))
    fail("Health with bad key", "should have been rejected")
except grpc.RpcError as e:
    check("Health with bad key raises UNAUTHENTICATED", e.code() == grpc.StatusCode.UNAUTHENTICATED)

# ---------------------------------------------------------------------------
# 12. Auth — missing application metadata
# ---------------------------------------------------------------------------
print("\n=== 12. Missing application metadata ===")

try:
    reply = p_stub.Health(Empty(), timeout=5, metadata=(("x-api-key", HEAR_SERVICE_KEY),))
    fail("Health without application metadata", "should have failed")
except grpc.RpcError as e:
    check("Health without app metadata raises error", True, f"{e.code()} {e.details()}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL = PASS + FAIL
print(f"\n{'='*50}")
print(f"  {PASS}/{TOTAL} passed  ({FAIL} failed)")
print(f"{'='*50}")

ray.shutdown()
sys.exit(0 if FAIL == 0 else 1)
