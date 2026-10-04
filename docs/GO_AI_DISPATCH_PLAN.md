# Go AI dispatch service: implementation plan

Scope: the backend side of AI processing moves from the Python backend (`hear-backend`)
to a Go service that owns AI jobs end to end: queueing, choosing Pod or Serverless,
sending, receiving worker callbacks, applying results, approvals, quotas, health,
and the status stream the frontend shows. The GPU workers (`hear-ai`, this repo) do
not change their protocol. Everything below is written against the code as it is on
2026-10-04: `hear-ai` commit `0736003`, `hear-backend` `1ca1e50`,
`hear-catalog-sync-go` `a9b560a`, `hear-frontend` `90b97a8`.

The Go service follows the conventions of `hear-catalog-sync-go`: one binary with
roles, Postgres as the only source of truth, RabbitMQ carrying row IDs only with
publisher confirms, relay/worker/reconciler loops, quorum queues with dead-letter
queues, hand-written `/healthz` and `/metrics`, `pgx`, `amqp091-go`, stdlib logging.

## 1. What is wrong today (why the logic changes)

| Today (Python backend) | Consequence | Plan |
| --- | --- | --- |
| Attempt state is a JSON blob in `processing_jobs.callback_payload["_runtime_v1"]`; no runs/attempts/events tables | Nothing to query, replay or audit; progress events are dropped after updating two columns | Dedicated tables for jobs, attempts, events, dispatch outbox, quota ledger, transport health |
| No lease-expiry sweep | A worker that dies leaves the job "processing" forever (the 15-minute cron re-defers it indefinitely) | Reconciler fails the attempt when its lease expires; the user is told |
| Queues are arq on Redis, in-process with the API | Lost on Redis restart, invisible, no dead-letter | RabbitMQ with durable quorum queues, outbox relay, dead-letter queues |
| Static Backblaze key sent to every worker | One leaked envelope exposes the whole bucket | Per-job scoped B2 key with a name prefix and expiry |
| Dispatch retries submissions up to 3 times and a stuck-job cron re-queues jobs | Silent retries, duplicated GPU spend, users not informed | A job that fails is failed once and reported; only a submission that was never accepted may be re-sent |
| No quota, no Pod/Serverless split, no health gating | Cannot control Serverless spend, jobs are sent to dead endpoints | Admin-configurable daily Serverless budget, routing policy, health prober, circuit breaker |
| AI SSE events carry no `id:`; replay only works for `stream_events` rows | A reconnecting browser misses events | Every AI event gets a monotonic `id` and is replayable with `Last-Event-ID` |

## 2. Service shape

No new repository. The code goes into the existing Go project
(`hear-catalog-sync-go`, which becomes the Go backend) as package `internal/ai/`
with new roles on the same binary, selected like the others by `CATALOG_ROLE`
(renamed `ROLE`). The Python backend keeps only the business API the frontend
calls (auth, tracks, publications, billing, …); every AI worker, queue, cron,
callback and dispatch path in Python is removed (§15). Roles:

| Role | Replicas | Responsibility |
| --- | --- | --- |
| `api` | N | Creator endpoints (start job, approve, reject, retry, list), internal worker callbacks (`/internal/ai/attempts/...`), RunPod webhook, SSE for AI streams, admin settings and stats |
| `relay` | 1 | Polls `ai_dispatch_outbox` and `ai_result_outbox`, publishes row IDs to RabbitMQ with confirms (singleton, lease-based like the catalog relay) |
| `dispatcher` | 1 | Consumes `hear.ai.dispatch`: picks the transport, builds the envelope, sends it, records the receipt |
| `applier` | N | Consumes `hear.ai.results` and `hear.ai.approvals`: validates artifacts, applies outcomes, swaps audio, replaces transcripts, stages deletions |
| `prober` | 1 | Probes Pod `/readyz` + `/capabilities` and every Serverless endpoint `/health`, keeps `ai_transport_health` current |
| `reconciler` | 1 | Expired leases, expired approvals, Serverless status polling for attempts with no heartbeat, quota day rollover, stuck-row repair |

All roles share `signal.NotifyContext`, graceful shutdown, `/healthz` (503 until the
role is consuming or polling) and `/metrics`.

## 3. Data model (Postgres)

Migrations go through the `hear-backend` alembic chain, as the catalog service does,
in a new schema `ai`. Timestamps are `timestamptz`; IDs are UUIDv7 (time-ordered).

```sql
ai.jobs
  id uuid pk, track_id uuid, user_id uuid, org_id uuid null,
  job_type text  -- pipeline | transcription | magic_clean | reconstruction
  operation text null,            -- reconstruction only
  options jsonb,                  -- exactly what is sent to the worker
  status text,                    -- see state machine
  failure_code text null, failure_message text null,
  requested_transport text null,  -- 'auto' | 'pod' | 'serverless' (admin/debug override)
  transport text null,            -- where it actually ran
  source_media_file_id uuid, source_revision int, source_sha256 char(64),
  current_attempt_id uuid null,
  candidate jsonb null,           -- approval candidate (delivery artifact, report) for clean/reconstruction
  result jsonb null,              -- applied result summary (what the UI shows)
  retry_of_job_id uuid null,      -- manual retry lineage
  created_at, queued_at, dispatched_at, started_at, finished_at, applied_at, expires_at
  index (track_id, status), index (status, queued_at), index (user_id, created_at desc)

ai.attempts
  id uuid pk, job_id uuid fk, run_id uuid,
  transport text, provider_job_id text null, endpoint_id text null,
  envelope jsonb,                 -- the exact AttemptEnvelope sent (application_key redacted)
  scope_sha256 char(64), grant_expires_at timestamptz,
  storage_key_id text, storage_key_expires_at timestamptz,   -- scoped B2 key, revoked at the end
  status text,                    -- pending | submitted | running | completed | failed | cancelled | lost
  worker_id text null, generation text null, image_revision text null, engine_revision text null,
  lease_until timestamptz null, last_sequence int default 0, heartbeats int default 0,
  claimed_at, last_heartbeat_at, finished_at,
  outcome jsonb null, outcome_sha256 char(64) null, failure_code text null
  unique (job_id, run_id)

ai.attempt_events
  id bigserial pk,                -- the SSE id
  attempt_id uuid fk, job_id uuid, track_id uuid,
  sequence int, event text, stage text null, progress_pct numeric null,
  message text null, data jsonb null, received_at timestamptz
  unique (attempt_id, sequence)

ai.dispatch_outbox   -- "send this job" commands (relay → hear.ai.dispatch)
  id uuid pk, job_id uuid, status text (pending|queued|processing|done|failed),
  attempts int, next_attempt_at, lease_expires_at, last_error text

ai.result_outbox     -- "apply this outcome / approval / rejection / expiry" commands (relay → hear.ai.results, hear.ai.approvals)
  id uuid pk, job_id uuid, kind text (outcome|approve|reject|expire|cleanup), payload jsonb,
  status, attempts, next_attempt_at, lease_expires_at, last_error

ai.transport_health
  transport text, endpoint_id text (pod: 'pod'), role text,
  healthy bool, checked_at, detail jsonb,      -- lanes / workers / jobs from the probe
  consecutive_failures int, circuit_open_until timestamptz null
  pk (transport, endpoint_id)

ai.quota_ledger
  day date, transport text, job_type text, dispatched int, completed int, failed int,
  gpu_seconds numeric,                          -- from attempt started→finished
  pk (day, transport, job_type)

ai.settings           -- admin-editable, versioned
  key text pk, value jsonb, updated_by uuid, updated_at
```

Keys and defaults in `ai.settings` (exposed in the admin UI, group `ai`):

| Key | Default | Meaning |
| --- | --- | --- |
| `serverless.enabled` | true | Allow Serverless at all |
| `serverless.daily_job_limit` | 500 | Jobs per calendar day (timezone `quota.timezone`) across all types |
| `serverless.daily_limit_by_type` | `{}` | Optional per-type limits, e.g. `{"reconstruction": 100}` |
| `serverless.max_inflight_by_role` | `{"pipeline":3,"magic_clean_natural":3,"reconstruction":2}` | Must not exceed each endpoint's `workersMax` |
| `pod.enabled` | true | Allow the Pod |
| `pod.reserve_inflight` | 0 | In-flight slots kept free on the Pod for priority jobs |
| `routing.policy` | `pod_first` | `pod_first`, `serverless_first`, `balanced` (alternate), `pod_only`, `serverless_only` |
| `routing.pod_queue_wait_seconds` | 120 | Under `pod_first`, if the Pod cannot accept within this wait, use Serverless (if budget remains) |
| `quota.timezone` | `Europe/London` | Day boundary for the ledger |
| `approval.expiry_hours` | 24 | Candidate approval window |
| `job.deadline_minutes_by_type` | `{"pipeline":30,"transcription":30,"magic_clean":30,"reconstruction":30}` | Envelope `deadline`; the worker refuses to start after it |

## 4. State machines

Job (`ai.jobs.status`):

```
queued ──► dispatching ──► submitted ──► processing ──► received ──► applying ──► completed
   │            │              │             │              │            │
   │            │              │             │              │            └──► apply_failed (operator action)
   │            │              │             │              └──► awaiting_approval ──► applying ──► completed
   │            │              │             │                                      └──► rejected / expired
   │            └──► queued (transport unavailable or submission not accepted; bounded, see §7)
   └──► cancelled (user, before processing starts)
   any of dispatching/submitted/processing ──► failed (worker outcome failed, lease lost, deadline, provider failure)
```

Attempt: `pending → submitted → running → completed | failed | cancelled | lost`.
`lost` means the lease expired without an outcome (worker died, Pod rebooted,
Serverless worker killed). A `lost` attempt fails its job; it is never re-run
automatically.

Every transition is one Postgres transaction with `SELECT ... FOR UPDATE` on the job
row, writes an `ai.attempt_events` row (event `status_changed`) and, where the user
should see it, publishes the SSE event (§9) after commit.

## 5. Dispatch: choosing Pod or Serverless

The dispatcher runs per job, inside one transaction:

1. Lock the job; it must be `queued`. Load the source media, verify it still has the
   same `media_file_id` and `audio_revision`; otherwise fail the job with
   `source_changed_before_dispatch` (the user re-runs it).
2. Compute capacity:
   - Pod: `ai.transport_health` row `pod` is healthy (probe ≤ 30 s old, `/readyz` 200,
     lane for the job's role `ready`) and `inflight_pod(role) < per_type[role]` from
     `/capabilities.concurrency` minus `pod.reserve_inflight`.
   - Serverless: the role's endpoint is healthy (`/health` reachable, no `unhealthy`
     workers growing), `inflight_serverless(role) < max_inflight_by_role[role]`, and
     `ledger(today, serverless, *) < daily_job_limit` (and the per-type limit).
3. Apply `routing.policy`. If no transport can take the job, leave it `queued` with
   `next_attempt_at = now + 15 s` and record the reason in
   `ai.attempt_events` as `waiting_for_capacity` (shown to the user as "Waiting for a
   free processor"). Nothing is sent anywhere.
4. Create the attempt: new `run_id`/`attempt_id`, scoped B2 key (§6), reporting grant,
   envelope. Increment `ai.quota_ledger.dispatched` for the chosen transport in the
   same transaction. Commit.
5. Send (outside the transaction, with a 25 s timeout):
   - Pod: `POST {POD_BASE_URL}/v1/attempts`, `Authorization: Bearer {POD_API_KEY}`.
     Only `202` with an echoed `attempt_id` counts as accepted.
   - Serverless: `POST https://api.runpod.ai/v2/{endpoint}/run`,
     `Authorization: Bearer {RUNPOD_API_KEY}`, body
     `{"input": envelope, "webhook": "{CALLBACK_BASE}/internal/ai/providers/runpod/{attempt_id}?sig=…", "policy": {"executionTimeout": deadline_ms}}`.
     Accepted when the response carries an `id` and status `IN_QUEUE`/`IN_PROGRESS`.
6. Record the receipt (`attempt.status = submitted`, `provider_job_id`). Not accepted
   (connection refused, 5xx, 429): revert to `queued`, decrement the ledger, mark the
   transport's `consecutive_failures` (3 in a row opens the circuit for 60 s), and let
   the next pass choose again. A `4xx` other than 429 is a bug in our envelope: fail
   the job with `dispatch_rejected:{detail}`; do not resend.

The counter the backend needs ("how many jobs went where") is `ai.quota_ledger`,
exposed as `GET /api/v1/admin/ai/stats?from=&to=` and as Prometheus gauges
`hear_ai_jobs_total{transport,job_type,status}` and
`hear_ai_serverless_budget_remaining`.

Pod concurrency is read from `/capabilities`: `concurrency.host_total`,
`concurrency.per_type` (e.g. `{"pipeline":7,"magic_clean_natural":2,"reconstruction":1}`)
and the lanes in `/readyz`. Serverless endpoint IDs come from config
(`RUNPOD_ENDPOINTS_JSON`, role → endpoint); `workersMax` is read once a day from the
RunPod REST API and stored in `transport_health.detail` so the admin limit cannot
exceed it.

## 6. The envelope we send (unchanged worker contract)

`AttemptEnvelope` (`hear/contracts/jobs.py`), `extra="forbid"`:

```json
{
  "schema_version": 1,
  "job_id": "0192f0a1-…",            "run_id": "0192f0a1-…",     "attempt_id": "0192f0a1-…",
  "job_type": "magic_clean",          "operation": null,
  "track_id": "…", "user_id": "…",
  "source": {"url": "https://cdn.hear.media/creators/x/audio/tracks/…/audio.mp3",
             "revision": 3, "file_sha256": "<sha256 of the exact object>"},
  "storage": {"endpoint_url": "https://s3.eu-central-003.backblazeb2.com/",
              "bucket_name": "hear-media", "key_id": "<scoped key id>",
              "application_key": "<scoped key>", "folder_prefix": "creators/x/audio/jobs/<job_id>/",
              "public_base_url": "https://cdn.hear.media/", "expires_at": "2026-10-05T10:00:00Z"},
  "options": {"profile": "studio_voice"},
  "artifact_prefix": "creators/x/audio/jobs/<job_id>/<attempt_id>",
  "deadline": "2026-10-04T18:30:00Z",
  "reporting_grant": "<token>",
  "backend_base_url": "https://api.hear.media/api/v1",
  "backend_id": "hear-backend"
}
```

Per-type `operation`/`options` (validated by the worker; validate the same in Go):

| job_type | operation | options |
| --- | --- | --- |
| `pipeline` | — | `{"max_tags": 8, "track_name": "...", "source": "...", "speaker": "...", "content_description": "..."}` (all optional) |
| `transcription` | — | `{}` |
| `magic_clean` | — | `{"profile": "natural|studio_voice|outdoor_mobile|clean_raw", "auto_level"?, "remove_clicks"?, "trim_silence"?, "attenuation_limit_db"?: 12|18|24|36|60, "sound_cleanup"?: {...}}` |
| `reconstruction` | `replace_segments` | `{"same_speaker": true, "changes": [{"segment_start": 84.866, "segment_end": 93.662, "original_text": "…", "new_text": "…"}]}` |
| `reconstruction` | `edit_transcript` | same as `replace_segments` |
| `reconstruction` | `rebuild` | `{"same_speaker": true, "edited_transcript": "…", "reference": {"start_seconds": 0, "end_seconds": 15, "text": "…"}}` |
| `reconstruction` | `remove_segments` | `{"segment_start": 10.0, "segment_end": 12.5}` |

Scoped storage key: created with B2 `b2_create_key` (`capabilities: [listBuckets, readFiles, writeFiles]`,
`bucketId`, `namePrefix = folder_prefix`, `validDurationInSeconds = deadline + 1 h`);
the key id is stored on the attempt and deleted (`b2_delete_key`) when the attempt
reaches a terminal state. Workers upload only under `artifact_prefix`.

Reporting grant: keep the existing format so `hear-ai` tests still apply:
`base64url(json) "." base64url(HMAC-SHA256(secret, "hear-attempt-v1:" + json))` with
`{"v":1,"backend":"hear-backend","job":job_id,"attempt":attempt_id,"scope":scope_sha256,"exp":epoch}`.
`scope_sha256` is `ExecutionScope.digest(envelope)` (`hear/contracts/scope.py`);
implement the same canonical JSON in Go and add a cross-language test vector.

## 7. What we receive: the four callbacks

All four are `POST {backend_base_url}/internal/ai/attempts/{attempt_id}/{claim|heartbeat|events|outcome}`
with headers `X-AI-Attempt-Grant`, `X-AI-Worker-ID`, `X-AI-Worker-Generation`.
The worker treats HTTP 401/403/404/409/410/422 as "not mine, stop" (never retried)
and 5xx/429 as transient (retried with backoff). Respond `2xx` only when the row is
committed.

| Suffix | Body | Response | Rules |
| --- | --- | --- | --- |
| `claim` | `{"worker_id","generation","image_revision","engine_revision","request_scope_sha256"}` | `{"decision": "execute", "lease_seconds": 120, "heartbeat_seconds": 20}` | Decision `already_completed` if the attempt is terminal; `cancelled` if the job was cancelled; `stale` if the source media changed; `lease_unavailable` if another worker holds a live lease; `409 execution_request_scope_mismatch` if the scope hash differs. `execute` stores worker identity, sets `lease_until`, job `processing`. |
| `heartbeat` | `{"worker_id","generation","sequence"}` | `{"accepted": true}` | Owner must match; extend `lease_until`; `409 attempt_no_longer_running` otherwise |
| `events` | `ExecutionEvent` (below) | `{"accepted": true}` or `{"accepted": true, "duplicate": true}` | Dedup on `(attempt_id, sequence)` **and** `event_id`; store every event; update job `stage`/`progress`; publish SSE |
| `outcome` | `ExecutionOutcome` (below) | `{"accepted": true}` | Idempotent by `outcome_sha256`; a different outcome for the same attempt is `409 conflicting_duplicate_outcome`; validate artifacts; set attempt terminal, job `received`, write `ai.result_outbox(kind=outcome)`; respond; the applier does the rest |

`ExecutionEvent`:

```json
{"schema_version": 1, "event_id": "uuid", "job_id": "…", "attempt_id": "…", "track_id": "…",
 "job_type": "pipeline", "backend_id": "hear-backend", "source_revision": 3, "sequence": 4,
 "event": "stage", "stage": "transcribing", "progress_pct": 27.5, "message": null, "data": {}}
```

`event` ∈ `started | stage | progress | warning | artifact_prepared | outcome`.
Stages per job type, in order (the SSE label map in §9 covers them):

| job_type | stages |
| --- | --- |
| pipeline | `preparing` 0 → `transcribing` 5–50 (progress events) → `moderating` 55 → `categorizing` 65 → `discovering` 75 → `completed` 100 |
| transcription | `preparing` 0 → `transcribing` 5–95 (progress) → `completed` 100 |
| magic_clean | `preparing` 0 → `processing` 20 → `denoising` 30 → `sound_cleanup` 55 (optional) → `mastering` 75 → `uploading` 85 → `completed` 100 / `failed` 100 |
| reconstruction | `preparing` 0 → `generating_speech` 15–80 (progress) → `uploading` 85 → `completed` 100 |

`artifact_prepared` events on the pipeline carry intermediate data
(`data.transcription` summary, `data.moderation`, `data.categorization`); store them,
do not act on them before the outcome.

`ExecutionOutcome`:

```json
{"schema_version": 1, "job_id": "…", "attempt_id": "…", "track_id": "…", "job_type": "magic_clean",
 "backend_id": "hear-backend", "source_revision": 3, "status": "completed",
 "artifacts": [{"bucket_name": "hear-media", "object_key": "creators/x/audio/jobs/<job>/<attempt>/delivery_audio.mp3",
                "size_bytes": 1518336, "sha256": "…", "content_type": "audio/mpeg",
                "audio_url": "https://cdn.hear.media/creators/x/audio/jobs/<job>/<attempt>/delivery_audio.mp3"}],
 "result": {...}, "error_code": null}
```

`status` ∈ `completed | failed | cancelled`. A failed outcome has `error_code`
(e.g. `resource_exhausted`, `invalid_audio`, `deadline_exceeded`, `source_mismatch`,
`invalid_request`, `source_unavailable`, `process_failed`) and `result.message`.

`result` per job type (as produced by `hear-ai` on 2026-10-04):

| job_type | `result` | artifacts |
| --- | --- | --- |
| pipeline | `transcription{transcript, segments[{id,start,end,text,words[{word,start,end,prob}]}], language, confidence, word_confidence_available, duration, audio_duration, silent, performance}`, `moderation{flagged, severity, intent, reason, flagged_categories, blocked_words_found}`, `categorization{categories, tags, sentiment, llm_used, …}`, `discovery{title, summary_short, summary_long, freeform_tags, controlled_tags, search_phrases, speaker, …}`, `content_description`, `flagged`, `silent` | none |
| transcription | `transcription{…same shape…}` | none |
| magic_clean | `profile, engine, requires_approval: true, source_sha256, delivery_audio{bucket_name, object_key, size_bytes, sha256, content_type, audio_url, duration_seconds}, report{duration_seconds, channels, sample_rate, gain_db, target_lufs, delivery_measurement, warnings[], speech_preservation, sound_cleanup, background_cleanup, timeline?}` | the delivery MP3 |
| reconstruction | `operation, engine, requires_approval: true, reconstructed_audio{sample_rate, source_frames, output_frames, channels, duration, duration_delta_seconds, timeline_policy, delivery_measurement, final_gain_db, source_sha256, delivery{…artifact…, duration_seconds}, segments[{segment_start, segment_end, source_start_frame, source_end_frame, output_start_frame, output_end_frame, output_start_seconds, output_end_seconds, duration, duration_delta_seconds, new_text, is_deletion, b2_key, audio_url, bucket_name, sha256}]}` | the full MP3 plus one MP3 per generated segment (`rebuild` reuses the delivery) |

Artifact validation before anything is applied: bucket equals the attempt's bucket,
every key starts with `artifact_prefix/` with no `.`/`..`/empty segments, keys are
unique, `audio_url == public_base_url + key`, and the object's size and SHA-256 are
re-read from B2 (`HEAD` + streaming hash) and must match the manifest. Any mismatch
fails the job with `artifact_validation_failed` and the artifacts are deleted.

Transport-level signals that are **not** callbacks: the Pod's optional SSE preview
and RunPod's `output` can contain `worker_waiting`, `worker_retrying`,
`attempt_dead_lettered`, `attempt_deadline_exceeded`, `backend_attempt_rejected`,
`capability_rejected`, `claim_rejected`, `worker_execution_failed`. Record them when
seen (webhook or status poll) as `ai.attempt_events` with `event = provider`; they
never change job state on their own except as described in §8.

## 8. Failures: no automatic retry

Rules, in priority order:

1. A job that **started processing and failed** (`outcome.status = failed`, or `lost`
   lease, or `deadline`) becomes `failed` with `failure_code` and a human message. The
   user sees it in the SSE stream (`failed` event), in the job list, and gets a
   notification. Nothing is re-sent. The UI offers **Retry**, which creates a new job
   (`retry_of_job_id` set) that goes through the normal queue.
2. A **submission that was never accepted** (connection refused, 503, 429, circuit
   open) is not a failure: the job stays `queued`, the dispatcher waits for capacity
   or a healthy transport, and the user sees "Waiting for a free processor". This is
   bounded by `job.deadline_minutes`: after that, the job fails with
   `no_processing_capacity` and is reported.
3. **Lost lease**: the reconciler runs every 15 s;
   `attempt.status = running AND lease_until < now()` → attempt `lost`, job `failed`
   (`worker_lost`). Before failing a Serverless attempt it reads
   `GET /v2/{endpoint}/status/{provider_job_id}` once; `COMPLETED` with an outcome
   already received means the callback raced the sweep and nothing changes.
4. **Provider failure** (RunPod webhook or status `FAILED`, `TIMED_OUT`, `CANCELLED`
   while our attempt is not terminal): job `failed` with `provider_{status}` and the
   provider `error` text; `cancel` is also called to be safe.
5. **Apply failure** (our side cannot apply a received outcome: B2 copy error, DB error):
   the result row goes to `apply_failed` after 5 bounded retries of the *apply* step
   only (the GPU work is done and valid; re-applying is safe and idempotent). The job
   shows "Result received, applying failed — support notified"; an operator can re-run
   the apply from the admin UI. This is the one place a retry is automatic, and it
   never touches the GPU.
6. **Cancellation** is only possible while `queued` or `dispatching`; a running attempt
   is told `cancelled` at its next heartbeat/claim and its outcome is discarded.

Every failure writes `failure_code`, `failure_message`, the attempt id, the transport
and the worker identity, so support can see exactly which machine and image produced
it.

## 9. Status stream to the frontend

Keep the frontend's contract (`src/@core/lib/sse/ai-events.ts`): endpoint
`GET /api/v1/sse/tracks/{track_id}/events`, named events `enqueued`, `submitted`,
`stage`, `stage_result`, `preview_ready`, `complete`, `failed`, `job_cancelled`, plus
the group variant with `group_complete`.

Phase 1 (no frontend change): the Go service publishes the same event names and
fields to Redis `sse:track:{id}` and writes `sse:last:track:{id}`; the existing SSE
app keeps serving browsers. Phase 2: the Go `api` role serves the endpoint itself,
with `id:` = `ai.attempt_events.id`, replay from the table for `Last-Event-ID`, a
snapshot of the latest job on connect, and a 20 s `: keepalive` comment. The Python
SSE app is then removed for the AI streams.

Mapping from worker events to SSE:

| Trigger | SSE event | Fields |
| --- | --- | --- |
| job created | `enqueued` | `job_id, job_type, track_id, status: queued, queue_position, transport: null` |
| waiting for capacity | `stage` | `stage: waiting_for_capacity, label: "Waiting for a free processor", progress_pct: 0` |
| accepted by a transport | `submitted` | `run_id, transport: pod|serverless` |
| worker `started`/`stage`/`progress` | `stage` | `stage, label, progress_pct, elapsed_seconds, estimated_remaining` |
| pipeline `artifact_prepared` | `stage_result` | `stage, result` (moderation/categorization summary) |
| outcome completed, approval required | `preview_ready` | `status: awaiting_approval, requires_approval: true, preview_audio_url, preview_id, expires_at, result` |
| outcome applied | `complete` | `status: completed, progress_pct: 100, result` (tags/category/discovery for pipeline; `applied: true, audio_url, duration, audio_revision` for audio) |
| job failed (any reason in §8) | `failed` | `error, error_code, stage, transport, retryable: true` |
| cancelled / rejected / expired | `job_cancelled` | `reason` |

Labels (`label`) are produced server-side from the stage names in §7 so the frontend
stays a renderer. New frontend work: show `transport` and `error_code` in the job
card, a **Retry** button on failed jobs, the "Waiting for a free processor" state, and
the admin page described in §11.

## 10. Applying results

All applies run in the `applier` role from `ai.result_outbox` rows, one transaction
per step, idempotent (a replayed row finds the work done and acks).

**pipeline** (no approval): write `transcriptions` (replace the row for the track:
`raw_text`, `word_timestamps = segments`, `language`, `model_used = "qwen3-asr"`),
tags (`audio_track_tags`, source `ai`), `track.category_id`, `track.discovery` /
`description` / `short_description`, a `content_flags` row if `moderation.flagged`,
`content_insights` summary, `track.pipeline_completed_at`; track status `ready`
(or `flagged`). Then `complete`.

**transcription** (no approval): replace the `transcriptions` row (the previous
transcript is not kept), bump `track.state_version`, `complete`.

**magic_clean**: the outcome makes the job `awaiting_approval` with
`candidate = {delivery_audio, report}` and `expires_at = now + approval.expiry_hours`;
`preview_ready` is sent with `preview_audio_url = delivery_audio.audio_url`.
On **approve** (`POST /api/v1/creator/tracks/{id}/magic-clean/{job_id}/apply`):

1. Lock job and track; the track's `audio_revision` must equal the attempt's
   `source_revision`, otherwise `409 source_changed` and the candidate is marked stale.
2. Server-side copy the delivery object to the owner namespace
   `{owner_prefix}/tracks/{track_id}/audio/{uuid}.mp3` (B2 `b2_copy_file`), verify
   size and SHA-256 again.
3. Create the new `media_files` row, swap `track.media_file_id`, bump
   `audio_revision` and `state_version`, reset speed layers, journal
   `audio_source_changed` (same semantics as `replace_source` today).
4. Stage the **old** media file and the job folder (`folder_prefix`) for deletion in
   `media_file_deletions` (the existing Go storage-cleanup worker deletes them); the old
   file is gone as soon as that worker runs, after the swap has committed and the CDN
   cache for the old URL has been purged.
5. Queue a new `transcription` job for the track (the audio changed), `complete`
   with `{applied: true, audio_url, duration, audio_revision}`.

On **reject**: delete the candidate artifacts (deletion outbox), job `rejected`,
`job_cancelled{reason: rejected}`. On **expiry** (reconciler): same with
`reason: expired` and a notification.

**reconstruction**: same approval flow. On **confirm**
(`POST /api/v1/creator/tracks/{id}/edit-recording/{job_id}/confirm`): copy
`reconstructed_audio.delivery` (the worker already renders the full file; the backend
no longer joins segments itself), swap media as above, delete old media, the job
folder and the segment MP3s, then update the transcript from `options.changes` and
`reconstructed_audio.segments` (output timeline), and queue a new `transcription`
job so word timings are regenerated. `rollback` = reject.

Deletion is durable and ordered: nothing is deleted until the replacing row is
committed; deletions are rows in `media_file_deletions`, never inline calls.

## 11. Admin and observability

- `GET/PATCH /api/v1/admin/settings/ai` → the keys in §3, with validation
  (Serverless in-flight limits capped by `workersMax`; policy enum).
- `GET /api/v1/admin/ai/stats?from&to` → per day × transport × job type: dispatched,
  completed, failed, GPU seconds, average and p95 duration; budget remaining today.
- `GET /api/v1/admin/ai/transports` → current health of the Pod and each endpoint,
  circuit state, last probe detail, in-flight counts.
- `GET /api/v1/admin/ai/jobs?status&transport&user&track` and
  `GET /api/v1/admin/ai/jobs/{id}` → job, attempts, the full event timeline, the
  outcome, the envelope (secrets redacted). Buttons: cancel (if queued), re-run apply
  (if `apply_failed`), force-fail (if stuck).
- Frontend: a settings group "AI processing" (budget, policy, toggles, timezone) and a
  dashboard card "Jobs today: Pod N / Serverless M of limit L".
- Metrics (`/metrics`): `hear_ai_jobs_total{transport,job_type,status}`,
  `hear_ai_job_duration_seconds{transport,job_type}` histogram,
  `hear_ai_inflight{transport,role}`, `hear_ai_queue_depth{status}`,
  `hear_ai_transport_healthy{transport,endpoint}`, `hear_ai_serverless_budget_remaining`,
  `hear_ai_outbox_backlog{outbox}`, `hear_ai_lease_lost_total`, RabbitMQ queue depths.
- Alerts: transport unhealthy > 5 min, queue depth growing with zero dispatches for
  10 min, lease lost > 3/h, apply_failed > 0, budget exhausted before 18:00 local.
- Logs: one line per transition, `job_id attempt_id transport worker_id stage`.

## 12. RabbitMQ topology

Dedicated vhost `hear-ai` on the existing broker stack. Direct exchange
`hear.ai.commands`; quorum queues `hear.ai.dispatch`, `hear.ai.results`,
`hear.ai.approvals`, `hear.ai.cleanup`, each with `x-delivery-limit: 5` and a
dead-letter exchange to the quorum queue `hear.ai.dead`. Messages are
`{"id": "<outbox row id>"}` only, persistent, `mandatory`, publisher confirms,
manual ack after the row is committed. Prefetch 8 for dispatch, 4 for results. A
dead-lettered row is visible in the admin job view and in `hear_ai_outbox_backlog`.

## 13. Security

- Pod: `POD_API_KEY` (bearer) in a Swarm secret; the Pod only accepts HTTPS from the
  backend network or through a Cloudflare tunnel with an access policy.
- Serverless: `RUNPOD_API_KEY` secret; RunPod webhook URL carries an HMAC signature of
  `attempt_id` and expiry; the handler verifies it and ignores unknown attempts.
- Callbacks: grant HMAC with `AI_SERVICE_SECRET` (≥ 32 bytes), constant-time compare,
  `exp` enforced, attempt id bound; worker identity fenced by generation; callback
  bodies limited to 16 MB (`extra=forbid`).
- Storage: scoped B2 key per attempt (§6); public URLs only through the CDN base.
- Secrets never enter `ai.attempts.envelope` (application key redacted on write).

## 14. Crash and restart guarantees

- API handlers only write rows (job + outbox) in one transaction; nothing is sent
  from a request handler.
- Relay leases outbox rows (`queued`, `lease_expires_at`); a crashed relay's rows
  return to `pending` after the lease.
- Workers claim rows with `UPDATE ... WHERE status='queued' RETURNING`; a redelivered
  message that finds the row done is acked without work.
- Sending to a transport is the only non-transactional side effect; if the process
  dies between "sent" and "recorded", the reconciler finds `attempt.status = pending`
  older than 60 s, asks the Pod (`/capabilities` has no job lookup, so the Pod case is
  resolved by the worker's own `claim`, which will arrive and be accepted) or RunPod
  (`/status`) and repairs the row.
- Worker callbacks are idempotent; the worker retries 5xx.
- Approval, rejection and expiry are outbox rows; deletions are outbox rows.
- A restart of any role loses nothing; a restart of RabbitMQ loses nothing (quorum
  queues); a restart of Postgres pauses everything and resumes.

## 15. What to remove once the Go roles are live

Principle: the Python backend keeps no AI logic at all. It keeps the frontend-facing
business endpoints; the ones that start or resolve AI jobs are moved to the Go `api`
role with unchanged paths, and the Python router for them is deleted, not proxied.

In `hear-backend` (Python), the current AI workers and all of their logic: the arq
AI worker process and its handlers (`core/worker/handlers/ai.py`, the `backend` and
`ai-result` queues, worker settings), `services/ai/{runtime_dispatch, runtime_attempts,
runtime_grant, runtime_scope, serverless_submission, storage, scheduler, handlers,
media (apply paths), job_result_processor, cleanup, sse_publisher}.py`,
`api/v1/ai_runtime_callbacks.py`, `api/v1/internal/ai_runtime.py`,
`core/worker/handlers/ai.py`, the arq AI queues (`backend`, `ai-result`) and crons
(`dispatch_ai_jobs`, `sync_stuck_ai_jobs`, `replay_stored_ai_outcomes`,
`purge_stale_magic_clean_previews`), `grpc_client/` and `services/ai/client.py`
(legacy `/process` + gRPC path), the `HEAR_AI_*`, `AI_MAX_*`, `AI_DISPATCH_*` settings.
The creator endpoints (`/tracks/{id}/magic-clean`, `/transcribe`, `/edit-recording`,
`/tag`, `/jobs/pending`) move to the Go `api` role with unchanged paths and bodies.

In `hear-ai` (this repo): `hear/dispatch/` and `tests/test_dispatch_clients.py`
(the Python backend-side clients), the README section "From the backend, use
`hear.dispatch`", and the `hear.dispatch` import in `scripts/serverless_canary.py`
(the canary keeps a small RunPod client of its own). The worker protocol modules
(`hear/contracts/`, `hear/execution/reporter.py`) stay: they are the contract the Go
service implements.

## 16. Delivery order

1. Schema + settings + `api` role with the four callbacks and the job/attempt model,
   tested against `hear-ai`'s `scripts/simulation_backend.py` cases and the contract
   tests in `hear-ai/tests` (port the grant and scope test vectors).
2. `relay`, `dispatcher`, `prober`, `reconciler`; Pod first, then Serverless with the
   webhook; quota ledger; admin settings and stats.
3. `applier`: pipeline and transcription (no approval), then magic clean and
   reconstruction approvals with media swap and deletions.
4. SSE phase 1 (Redis publish, same event names); frontend: transport/error/Retry,
   waiting state, admin AI settings page.
5. Cutover: run Go and Python side by side with `routing.policy = pod_only` and a
   zero Serverless budget; dispatch one real job per type from Go; compare outcomes;
   switch the creator endpoints; remove the Python paths (§15); SSE phase 2.
6. Load test: 50 queued jobs, Pod unplugged mid-run, RunPod key revoked mid-run,
   RabbitMQ restarted mid-run, Go roles restarted mid-run; assert no lost, duplicated
   or silently retried job.
