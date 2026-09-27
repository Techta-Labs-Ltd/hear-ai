# Audio jobs backend contract and end-to-end test runbook

## Incident diagnosis (2026-08-25)

Job `1233507d-1a32-43f7-affe-b5dded76f6eb` and the surrounding magic-clean
jobs did not fail because of their audio or requested levels. The Ray Serve
`magic_clean` replica failed its constructor three times and was marked
unhealthy. The constructor error was:

```text
FileNotFoundError: MossFormer2 checkpoint is incomplete:
/models/mossformer2-se-48k
```

The required file is
`/models/mossformer2-se-48k/last_best_checkpoint`. The orchestrator
then received `DeploymentUnavailableError` at the `enhancing` stage. The prior
runtime check accepted an empty model directory, so startup appeared valid.

Before starting or deploying Hear AI, run:

```bash
uv run --no-project python main.py --validate-only
```

The Supervisor-managed Ray server provisions missing model files automatically
before it starts Serve. Artifacts are stored only under the repository-root
`/models` directory. The manual script can prewarm that same cache, but does
not use a second model location. Validate the workspace again after a manual
prewarm.

## Backend interaction contract

Submit all asynchronous job families to `POST /process` with `X-Service-Key`. Hear AI
does not deliver HTTP callbacks. Consume progress with gRPC `Subscribe` and
recover terminal state with `GetResult`, using metadata `application: hear`
and `x-api-key: <service key>`.

Every submission requires these top-level fields:

- `job_id`: backend-generated idempotency key; use a new value for a deliberate rerun
- `backend_id`, `track_id`, and `user_id`
- `job_type` and (for these jobs) `audio_url`
- complete job-scoped `storage`: `endpoint_url`, `bucket_name`, `key_id`,
  `application_key`, `folder_prefix`, `public_base_url`, and `expires_at`

`bucket_name` and `folder_prefix` alone are not a valid storage context.
Credentials must remain valid for queue time, processing, upload, retries, and
cleanup reconciliation. `magic_clean` requires at least
`MAGIC_CLEAN_STORAGE_CREDENTIAL_MIN_TTL_SECONDS` remaining when a new job is
admitted and again when an attempt starts (24 hours by default). If queue time
erodes that reserve, the job stays queued without consuming the attempt. Replay
the same semantic request and `job_id` with refreshed credentials; the backend
should issue a longer lifetime when its supported recording or recovery window
requires one. `Subscribe` reports the parked state as `job_queued` with
`error: storage_credentials_expiring`; an exact unchanged replay returns the
same queued run but cannot resume dispatch.

### Complete submission field reference

| Field | Type | Required/behavior |
|---|---|---|
| `job_id` | string | Required idempotency key; use a new id for a deliberate rerun. |
| `backend_id` | string | Required; must match the backend selected by the service key. |
| `track_id` | string | Required; it need not already exist in Hear AI. |
| `user_id` | string | Required and non-blank; used by fair scheduling. |
| `job_type` | string | Required; hyphens normalize to underscores. |
| `audio_url` | string | Required for `magic_clean`, `audio_tag`, `edit_transcript`, and `reconstruct`. |
| `max_tags` | integer | Defaults to 8; `audio_tag` currently always requests at most two suggestions. |
| `edited_transcript` | string | Required for `edit_transcript`; send the corrected full transcript. |
| `changes` | array | Required and non-empty for `reconstruct`. |
| `same_speaker` | boolean | Defaults to true. |
| `speech`, `music`, `background` | integer | Magic Clean percentages 0-100; supply all three or none. |
| `cut_silence` | boolean | Magic Clean control; defaults to false. |
| `storage` | object | Required complete job-scoped storage context. |

All storage fields are required: `endpoint_url`, `bucket_name`, `key_id`,
`application_key`, `folder_prefix`, `public_base_url`, and `expires_at`.
Credentials must be temporary and must never be logged or persisted in
plaintext. REST returns `202` for a new job and `200` for an identical replay.
A changed non-secret payload with the same `job_id` returns HTTP `409` / gRPC
`ALREADY_EXISTS`; invalid input returns HTTP `422` / gRPC `INVALID_ARGUMENT`.
A durably saved job whose Ray dispatch was not acknowledged returns HTTP `503`
/ gRPC `UNAVAILABLE`; retry the identical request.

Magic Clean has one credential-only idempotency carve-out. While the job is
queued, an authenticated replay may replace its storage credentials without
changing the destination or audio semantics. The refresh must restore the full
minimum lifetime, and changing key material requires a strictly later
`expires_at` so a delayed retry cannot restore older credentials. Once status is
`running`, changed credentials or a later expiry are rejected with `422`; an
unchanged replay may still recover current status. Terminal replays return the
original run/result and do not start a replacement job.

### gRPC `Subscribe` stream

`SubscribeRequest` contains `job_id`. `PipelineEvent` contains `event`, job/run/
backend/track/type identity fields, `status`, `current_stage`, display-only
`label` and `description`, best-effort progress/timing, sanitized `error`, and
an untyped `google.protobuf.Struct result`. Always use typed `GetResult` for the
authoritative terminal business result.

Events currently emitted are:

- `job_queued`: accepted by fair scheduling, or a Magic Clean job parked for
  credential refresh. In the latter case `error` is
  `storage_credentials_expiring`; otherwise `result` contains queue details and
  normalized Magic Clean controls when applicable.
- `stage_changed`: entered `current_stage`; progress is the stage midpoint.
- `queue_position`: queue update; do not depend on internal fields that are not
  represented in `PipelineEvent`.
- `stage_result`: `{ "stage": "<id>", "data": { ... } }`; not every stage
  emits one.
- `job_retrying`: `result.report` contains `stage`, `error`, `attempt`, and
  `retryable`.
- `heartbeat`: emitted after 120 seconds without an event; do not transition
  backend state.
- `job_completed`, `job_failed`, `job_cancelled`: terminal; the stream closes.

If completion happens before subscription or during a disconnect, reconnecting
replays the terminal event from PostgreSQL. Follow it with `GetResult`.

#### Known stream contract blockers

- `audio_tag` declares `audio_tagging` but currently invokes `transcribing`, so
  streamed label/progress do not match its declared flow.
- `edit_transcript` declares `downloading`, `transcribing`,
  `diffing_transcript`, and `reconstructing_edits`, but currently emits only
  `reconstructing`, which is not in its declared flow and has zero progress.
- Direct `reconstruct` declares `downloading` then `reconstructing`, but only
  emits the latter after download.
- Magic Clean emits all five stage changes but no intermediate `stage_result`.

Backend code must tolerate the current behavior, but the ordered-stream release
gate is not passed until these mismatches are fixed and contract-tested.

### Magic clean

```json
{
  "job_id": "job-clean-001",
  "backend_id": "backend-a",
  "track_id": "track-001",
  "user_id": "user-001",
  "job_type": "magic_clean",
  "audio_url": "https://media.example/source.mp3",
  "speech": 100,
  "music": 25,
  "background": 20,
  "cut_silence": true,
  "storage": {
    "endpoint_url": "https://s3.example.com",
    "bucket_name": "OldAlexa",
    "key_id": "temporary-key-id",
    "application_key": "temporary-application-key",
    "folder_prefix": "localtns/the-news-foundation/audio/jobs/job-clean-001",
    "public_base_url": "https://media.example/OldAlexa",
    "expires_at": "2099-01-01T00:00:00Z"
  }
}
```

Read the result from `JobResult.magic_clean.enhanced_audio`. Persist its
returned `bucket_name`, `b2_key`, and `audio_url`; never derive an object key.

Defaults are `speech=100`, `music=10`, `background=10`, and
`cut_silence=false`. The declared stream is `downloading (0-10)`, `separating
(10-35)`, `enhancing (35-80)`, `mixing (80-95)`, and `finalizing (95-100)`.
Each Magic Clean `stage_changed.result` contains the four normalized controls.

Require `JobResult.payload == magic_clean` and validate every field:

```text
magic_clean
  enhanced: bool
  enhanced_audio { audio_url, b2_key, bucket_name, backend_id }
  quality { quality_score, snr_db, peak_db, lufs, clipping_detected }
  stage_times: google.protobuf.Struct
  transcription {
    transcript, segments[] { start, end, text, speaker, words[] },
    language, confidence
  }
  moderation {
    flagged, severity, intent, reason,
    flagged_categories[], blocked_words_found[]
  }
```

Magic Clean currently returns empty/default transcription and moderation.
Proto3 renders missing numeric quality fields as zero, so zero is not proof of
a passing live measurement.

### Audio tag

Submit `job_type: "audio_tag"` with the common required fields and `audio_url`.
This job transcribes a short utterance and returns at most two suggestions. It
does not run moderation or discovery and does not upload an artifact.

Require `JobResult.payload == audio_tag`:

```text
audio_tag
  source_audio_url: string
  transcription: string
  suggestions: repeated string (maximum two)
```

Current result construction does not populate `source_audio_url`, so it reads
as an empty proto3 string. This is a release blocker if the backend requires a
source URL echo. Test an utterance with two obvious topics, silent/non-speech
audio, zero-to-two trimmed non-empty suggestions, and reconnect recovery.

### Audio regeneration (`edit_transcript`)

Use `edit_transcript` when the backend has the corrected full transcript and
wants Hear AI to detect and regenerate only changed speech. Include non-empty
`edited_transcript`. Read `JobResult.reconstruct.rebuilt_audio` and `segments`.

### Reconstruction (`reconstruct`)

Use `reconstruct` when the backend already knows exact timed replacements.
Include a non-empty `changes` array; every item requires
`segment_end > segment_start` and non-empty `new_text`. Read
`JobResult.reconstruct.rebuilt_audio` and `segments`.

Do not confuse `rebuild` with audio regeneration: in the current contract,
`rebuild` runs the text/pipeline flow and returns `JobResult.pipeline`; it does
not return newly synthesized audio.

The full typed result shared by `edit_transcript` and `reconstruct` is:

```text
reconstruct
  edited_transcript: optional string
  rebuilt_audio { audio_url, b2_key, duration, bucket_name, backend_id }
  is_regenerated: bool
  transcription {
    transcript, segments[] { start, end, text, speaker, words[] },
    language, confidence
  }
  moderation {
    flagged, severity, intent, reason,
    flagged_categories[], blocked_words_found[]
  }
  segments[] {
    segment_start, segment_end, b2_key, audio_url, duration,
    is_deletion, bucket_name, backend_id
  }
```

For `edit_transcript`, require the returned edited transcript and source
transcription. Direct `reconstruct` currently leaves those fields empty/default.
Both must set `is_regenerated=true` and return the rebuilt track. Submission
validation rejects empty `new_text`, so direct deletion through an empty timed
replacement is not currently supported.

### Preview-based audio editing RPCs

These synchronous gRPC operations are separate from asynchronous jobs:

1. `CreatePreview(ReconstructRequest)` accepts `audio_url`, `track_id`,
   `backend_id`, complete `storage`, `same_speaker`, and either `changes[]` or
   the optional single `segment_start`/`segment_end`/`new_text` fields.
2. `GetPreview(PreviewRequest)` recovers a pending preview.
3. `ConfirmPreview(PreviewRequest)` promotes it and returns the final artifact.
4. `RollbackPreview(PreviewRequest)` abandons it.
5. `RemoveSegment(RemoveSegmentRequest)` removes a known range and returns a
   new complete artifact.

Test create/get/confirm, create/get/rollback, expiry, wrong-backend access,
confirmation replay, invalid ranges, removal at the start/middle/end, artifact
validation, and cleanup. Never use a production track or bucket.

## Backend state and retries

Persist `job_id`, accepted `run_id`, job type, track/user/backend identities,
status, current stage, and returned artifact metadata. Do not persist plaintext
storage credentials.

- Retry a failed or timed-out submission with the identical `job_id` and body.
- A replay returns the original run. A changed non-secret payload with the same
  `job_id` returns `409`.
- If Magic Clean reports or persists `storage_credentials_expiring`, keep the
  existing `job_id` and replay its unchanged semantic request with credentials
  that restore the full configured lifetime. Do this only while the job is
  queued; key rotation must use a strictly later `expires_at`.
- Reconnect `Subscribe` after transport loss and call `GetResult` until a
  terminal state is recovered.
- Retry transient processing failures only while the storage credentials remain
  valid. Configuration/model-unavailable errors require an operator fix; job
  retries cannot repair them.
- Treat `b2_key` as authoritative and update the track only after a completed
  typed result is persisted.

## Throughput and latency plan

The fastest safe improvement is to separate admission from scarce model
capacity. Increasing orchestrator concurrency alone does not increase GPU
throughput and can create a larger in-memory/audio queue.

1. Keep independent per-type limits. Set each limit no higher than its actual
   Ray replica capacity unless queueing at the orchestrator is intentional.
2. Remove cold starts for latency-sensitive workloads by using a non-zero
   minimum replica only when GPU memory budgeting proves it can coexist with
   resident transcription/LLM models. Otherwise use a dedicated GPU worker pool
   for Fish Speech and magic clean.
3. Scale stateless pipeline deployments independently. Add nodes/replicas before
   raising `ORCHESTRATOR_MAX_CONCURRENT_JOBS`; retain per-user fairness and
   PostgreSQL as durable state.
4. Avoid downloading the source twice. The backend should provide one stable,
   region-local object URL; Hear downloads once and uploads directly to the
   submitted destination. Do not proxy large audio through the backend.
5. Track queue latency, stage duration, cold-start time, upload time, GPU memory,
   and failures by `job_type`/stage. Scale from measured bottlenecks, not only
   total request count.
6. Apply backpressure: return queued state, cap pending work per user/type, and
   reject or defer admission when temporary disk, database connections, or GPU
   queue depth cross configured limits.

For horizontal scale, every node needs the same immutable model artifacts and
native dependencies. Temporary audio must remain job/run scoped and be cleaned
on success and failure. Storage credentials must be folder-scoped, encrypted at
rest, and long-lived enough for the worst-case queue plus retry duration.

## End-to-end release gate

Use short, non-private fixtures with known speech, music, noise, and silence.
The target must have all model checkpoints, a compatible GPU, an isolated
PostgreSQL database, a registered non-production backend, disposable scoped
storage credentials, and a client generated from the deployed proto.

Before submission:

```bash
uv run --no-project python main.py --validate-only
```

Also confirm HTTP/gRPC health, model replica health, database access, source URL
readability, destination write/read/delete access, temp space, and GPU memory.

For every case:

1. Generate a unique job id and folder prefix; retain a secret-free request.
2. Submit and record the accepted run id.
3. Record every `PipelineEvent` with a UTC timestamp.
4. Disconnect once, reconnect, and verify terminal recovery.
5. Call `GetResult`; assert all identity/status fields and the expected oneof.
6. HEAD/download each returned artifact using the returned key/URL. Verify
   non-zero size, `audio/mpeg`, decoder readability, duration, loudness, peak,
   clipping, and bucket/key/URL consistency.
7. Confirm job/run temp cleanup on success and failure.
8. Replay the identical request and require the original run id.
9. Change one non-secret field with the same job id and require conflict.

### Functional matrix

| Family | Required cases |
|---|---|
| Magic Clean | Defaults; 100/0/0 voice focus; 100/100/100 preserve mix; silence cut on/off; partial controls; -1/101 controls; human A/B check for speech damage. |
| Audio tag | Two obvious topics; silence/non-speech; no more than two suggestions; no moderation/discovery/artifact; reconnect. |
| Edit transcript | Replacement; insertion; deletion; punctuation/case-only; first/last word; two distant edits; unchanged transcript; source without valid timed segments. |
| Reconstruct | One/multiple replacements; boundaries; speaker matching on/off; invalid, overlapping, and out-of-bounds ranges; duration expansion/contraction. |
| Preview editing | Create/get/confirm; create/get/rollback; expiry; wrong backend; confirmation replay; remove first/middle/last range. |

Expected overlap and out-of-bounds reconstruction behavior must be defined
before release; submission validation currently checks only time ordering and
non-empty replacement text.

For every asynchronous family also test unreadable source, expired/unauthorized
storage, prefix denial, malformed/empty audio, unavailable model, transient
retry, cancellation while queued/running, restart recovery, and wrong-backend
access. Require sanitized errors, correct failed stage/retry report, durable
terminal recovery, no false artifact, and cleanup.

Record release evidence:

| Case/job id | Run id | Stream order | Reconnect | Typed result | Audio/artifact | Idempotency | Cleanup | Pass/fail |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

Run the repository contract tests first:

```bash
uv run pytest \
  tests/test_job_submission.py \
  tests/test_grpc_contracts.py \
  tests/test_typed_grpc.py \
  tests/test_audio_tag_contract.py \
  tests/test_magic_clean_stages.py \
  tests/test_magic_clean_pipeline.py \
  tests/test_diff_engine.py \
  tests/test_audio_delivery.py
```

Repository unit/contract tests mock external storage and model effects. A true
live audio assertion additionally requires provisioned checkpoints, GPU, an
isolated PostgreSQL database, and disposable scoped object-storage credentials;
it must not be run against production by default.

A live run performs inference, database mutations, downloads, and storage
writes. It requires explicit authorization and verified non-production targets.
Do not run `scripts/live_test.py --destructive` without separate review and
approval of its exact target and scope.
