# Hear backend job integration

This document describes the required backend changes for submitting work to
Hear AI after backend-bound authentication and per-job storage routing.

## Required backend configuration

Each backend deployment needs:

- A registered `backend_id`. The current local registration is `hear-backend`.
- The original service key whose SHA-256 hash is registered in Hear AI.
- The Hear AI HTTP URL and gRPC target.
- A mechanism for issuing job-scoped B2 credentials.

The raw service key stays in the owning backend. Send it as
`X-Service-Key` over HTTP and `x-api-key` over gRPC. Never send the stored
SHA-256 hash.

## Job storage context

Every new job requires this storage object:

```json
{
  "endpoint_url": "https://s3.example.com",
  "bucket_name": "backend-bucket",
  "key_id": "temporary-key-id",
  "application_key": "temporary-application-key",
  "folder_prefix": "users/user-456/jobs/job-123",
  "public_base_url": "https://media.example.com/backend-bucket",
  "expires_at": "2026-07-27T18:00:00Z"
}
```

The endpoint, bucket, and public base URL must be allowlisted for the submitted
backend. The folder prefix must be relative and traversal-free. Credentials
must remain valid for the complete queue, processing, retry, preview, and
cleanup lifecycle.

Do not send storage credentials to browser or mobile clients. Prefer restricted
credentials scoped to the supplied folder, and revoke them after the job
reaches a terminal state.

## Backend persistence

Create the backend job record before submission and persist:

- `job_id`
- `backend_id`
- `run_id` after acceptance
- `track_id`
- `user_id`
- `job_type`
- status and current stage
- storage bucket and folder prefix, but not plaintext credentials
- terminal artifact metadata

Use a UUID or ULID for `job_id`. It is also the idempotency key.

## REST submission

Send new asynchronous jobs to `POST /process`:

```http
POST /process
Content-Type: application/json
X-Service-Key: <registered backend service key>
```

```json
{
  "job_id": "01JBACKENDGENERATEDULID",
  "backend_id": "hear-backend",
  "track_id": "track-123",
  "user_id": "user-456",
  "job_type": "pipeline",
  "audio_url": "https://media.example.com/source/track-123.mp3",
  "max_tags": 5,
  "storage": {
    "endpoint_url": "https://s3.example.com",
    "bucket_name": "backend-bucket",
    "key_id": "temporary-key-id",
    "application_key": "temporary-application-key",
    "folder_prefix": "users/user-456/jobs/01JBACKENDGENERATEDULID",
    "public_base_url": "https://media.example.com/backend-bucket",
    "expires_at": "2026-07-27T18:00:00Z"
  }
}
```

`POST /discovery` requires the same `backend_id` and `storage` object.

Successful acceptance returns:

```json
{
  "job_id": "01JBACKENDGENERATEDULID",
  "backend_id": "hear-backend",
  "run_id": "generated-run-id",
  "track_id": "track-123",
  "job_type": "pipeline",
  "status": "queued",
  "replayed": false
}
```

Handle responses as follows:

| Status | Backend action |
| --- | --- |
| `202` | Store `run_id`, mark queued, and subscribe over gRPC. |
| `200` with `replayed: true` | Reuse the original run and continue recovery. |
| `401` | Fix the service key or backend mismatch. Do not retry blindly. |
| `409` | The `job_id` was reused with a different payload or destination. |
| `422` | Fix missing or invalid backend/storage fields. |
| `503` or timeout | Retry the identical request with the same `job_id`. |

Old submissions without `backend_id` and `storage` are intentionally rejected.

## gRPC contract changes

The source contract is `hear/proto/pipeline.proto`. Regenerate the backend's
client stubs from that file before deploying the backend changes.

All gRPC calls require:

```text
application: hear
x-api-key: <registered backend service key>
```

The service key resolves to one backend. For RPCs that submit storage work, the
resolved backend must exactly match the request `backend_id`.

### New `StorageContext`

```proto
message StorageContext {
  string endpoint_url = 1;
  string bucket_name = 2;
  string key_id = 3;
  string application_key = 4;
  string folder_prefix = 5;
  string public_base_url = 6;
  string expires_at = 7;
}
```

### Changed job submission messages

`SubmitJobRequest` adds:

```proto
string backend_id = 24;
StorageContext storage = 25;
```

`SubmitJobResponse` adds:

```proto
string backend_id = 5;
```

Example:

```python
from hear.proto.pipeline_pb2 import StorageContext, SubmitJobRequest

metadata = (
    ("application", "hear"),
    ("x-api-key", service_key),
)

accepted = client.SubmitJob(
    SubmitJobRequest(
        job_id=job_id,
        backend_id="hear-backend",
        track_id=track_id,
        user_id=user_id,
        job_type="pipeline",
        audio_url=audio_url,
        max_tags=5,
        storage=StorageContext(
            endpoint_url=storage.endpoint_url,
            bucket_name=storage.bucket_name,
            key_id=storage.key_id,
            application_key=storage.application_key,
            folder_prefix=storage.folder_prefix,
            public_base_url=storage.public_base_url,
            expires_at=storage.expires_at,
        ),
    ),
    metadata=metadata,
    timeout=30,
)
```

Use either REST submission or `SubmitJob`, not both for the same job.

### Changed event and result messages

`PipelineEvent` adds:

```proto
string backend_id = 15;
```

`JobResult` adds:

```proto
string backend_id = 9;
```

The backend must verify this value before applying an event or result to its
own job record.

`Subscribe`, `GetResult`, and `CancelJob` keep their existing request messages,
but Hear AI now checks that the authenticated backend owns the requested job.
Cross-backend reads and cancellations are rejected as not found.

### Changed direct reconstruction messages

`ReconstructRequest`, used by `CreatePreview`, adds:

```proto
string backend_id = 8;
StorageContext storage = 9;
```

`RemoveSegmentRequest` adds:

```proto
string backend_id = 6;
StorageContext storage = 7;
```

`ConfirmPreview`, `RollbackPreview`, and `GetPreview` do not resend credentials.
They use the encrypted storage context saved when the preview was created and
enforce backend ownership.

### Changed artifact responses

Uploaded artifact messages now identify their owner and destination:

| Message | Added fields |
| --- | --- |
| `EnhancedAudio` | `bucket_name = 3`, `backend_id = 4` |
| `RebuiltAudio` | `bucket_name = 4`, `backend_id = 5` |
| `SegmentAudio` | `bucket_name = 7`, `backend_id = 8` |
| `CreatePreviewReply` | `b2_key = 9`, `bucket_name = 10`, `backend_id = 11` |
| `ConfirmPreviewReply` | `bucket_name = 9`, `backend_id = 10` |
| `RemoveSegmentReply` | `bucket_name = 11`, `backend_id = 12` |
| `Preview` | `bucket_name = 12`, `backend_id = 13` |

Artifact payloads contain `backend_id`, `bucket_name`, `b2_key`, and
`audio_url`, but never `key_id` or `application_key`.

## Progress and result delivery

Hear AI does not call backend webhooks. The backend must use gRPC:

```python
for event in client.Subscribe(
    SubscribeRequest(job_id=job_id),
    metadata=metadata,
    timeout=60 * 60,
):
    persist_event(event)
    if event.event in {"job_completed", "job_failed", "job_cancelled"}:
        break
```

Persist `stage_changed`, `stage_result`, and `job_retrying`. Only
`job_completed`, `job_failed`, and `job_cancelled` are terminal.

After a terminal event, or whenever a stream reconnects, request the
authoritative durable result:

```python
result = client.GetResult(
    GetResultRequest(job_id=job_id),
    metadata=metadata,
    timeout=30,
)
```

Read final artifacts from:

- Pipeline: `result.pipeline.compressed_audio`
- Magic clean: `result.magic_clean.enhanced_audio`
- Reconstruction: `result.reconstruct.rebuilt_audio`
- Reconstructed segments: `result.reconstruct.segments`

## Object layout

Hear AI creates controlled suffixes beneath the supplied folder:

```text
<folder_prefix>/source/<job_id>.mp3
<folder_prefix>/enhanced/<job_id>.mp3
<folder_prefix>/reconstructed/<job_id>.mp3
<folder_prefix>/previews/<preview_id>.mp3
<folder_prefix>/segments/<job_id>/<segment_id>.mp3
```

The backend must treat returned `b2_key` as authoritative. Do not reconstruct
paths independently.

## Error and retry policy

Stable storage failures include:

- `missing_storage_context`
- `storage_credentials_expired`

Never fall back to another backend, another bucket, or global B2 credentials.
For expired credentials, issue fresh credentials and submit a new job with a
new `job_id`.

An identical retry may reuse the same `job_id`. Changing `backend_id`, bucket,
endpoint, public base URL, or folder prefix changes the idempotency fingerprint
and produces `409`.

## Deployment sequence

1. Generate backend client stubs from the updated protobuf.
2. Add backend configuration for service key, backend ID, HTTP URL, and gRPC target.
3. Implement job-scoped B2 credential creation.
4. Persist backend jobs before submission.
5. Add required REST and gRPC storage fields.
6. Add gRPC metadata to every call.
7. Consume `Subscribe` and recover with `GetResult`.
8. Persist returned artifact metadata and verify `backend_id`.
9. Revoke temporary storage credentials after terminal completion.
10. Reject or migrate old queued backend jobs that lack storage context.

## Test procedure

First run Hear AI's local contract suite:

```bash
uv run pytest -q \
  tests/test_backend_storage.py \
  tests/test_job_submission.py \
  tests/test_fastapi_ingress.py \
  tests/test_grpc_contracts.py \
  tests/test_typed_grpc.py \
  tests/test_audio_delivery.py
```

For an end-to-end test, export credentials locally without committing them:

```bash
export HEAR_SERVICE_KEY='...'
export HEAR_BACKEND_ID='hear-backend'
export HEAR_STORAGE_ENDPOINT_URL='...'
export HEAR_STORAGE_BUCKET_NAME='...'
export HEAR_STORAGE_KEY_ID='...'
export HEAR_STORAGE_APPLICATION_KEY='...'
export HEAR_STORAGE_FOLDER_PREFIX='users/test/jobs/test-run'
export HEAR_STORAGE_PUBLIC_BASE_URL='...'
export HEAR_STORAGE_EXPIRES_AT='2026-07-27T18:00:00Z'
export HEAR_TEST_AUDIO_URL='https://.../test-audio.mp3'
```

With Hear AI running, execute the non-destructive live contract test:

```bash
uv run python scripts/live_test.py
```

Do not add `--destructive` unless preview confirmation, deletion, training, and
other state-changing operations are intentionally being tested.
