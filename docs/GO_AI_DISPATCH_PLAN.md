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

## 0. Verdict and placement

**Verdict: build it this way.** The design is the one pattern already proven in
production here (Postgres as the only truth, RabbitMQ carrying row IDs, relay /
worker / reconciler loops), it removes the failure classes the current Python code
actually exhibits (jobs that hang forever, silent retries, published tracks flipped
by AI), and it keeps the GPU worker protocol untouched so nothing on the pod or
Serverless side changes. It is a full replacement of the AI side of the backend,
roughly 3–4 weeks for one engineer who knows the Go codebase; the parallel run in
§17 step 3 is what makes it safe and must not be skipped.

**What lives where (final):**

| Concern | Owner | Why |
| --- | --- | --- |
| AI jobs: queue, Pod/Serverless choice, quota, health, callbacks, apply, approvals, deletions, job SSE, admin stats | **Go** (`internal/ai/` roles in the existing Go project) | All of it is queue and state-machine work; Go's event-driven consumers do it with a fraction of the CPU of the current arq/gRPC/cron setup |
| **Publishing**: publish jobs, speed-layer rendering, catalog confirmation, scheduling | **Python** (as today) | It is not AI work and touches scheduling, speed layers and catalog confirmation that already work. Go only flips a gated publish row to `pending` when a pipeline result lands (§10.4). Moving it would double the migration for no runtime gain |
| Catalog / Meilisearch indexing | **Go catalog service** (as today) | The AI applier stages outbox rows (§10.3); it never calls Meilisearch |
| Creator business API (tracks, publications, billing, auth), content SSE | **Python** | Unchanged; reads AI jobs only through the `ai.v_track_jobs` view |
| GPU workers | `hear-ai` (this repo) | Protocol unchanged |

**CPU and memory.** The Python side today runs, per environment: an arq `backend`
worker (8 slots) and an `ai-result` worker, a gRPC subscription per active job, a
dispatch cron every minute plus four more AI crons, Redis-based queues polled by
arq, and request handlers that re-parse multi-megabyte JSON blobs
(`callback_payload`, `result_metadata`) on every callback and every track read. The
Go roles replace all of that with blocking consumers and a single relay; the design
rules that keep the footprint minimal are:

- No polling loops in the hot path: the relay wakes on Postgres `LISTEN/NOTIFY`
  (`NOTIFY ai_outbox`) raised by the trigger that inserts outbox rows, with a 5 s
  fallback tick; consumers block on RabbitMQ with prefetch 8/4; the prober runs every
  30 s; the reconciler every 15 s with indexed range queries only.
- Callbacks do no heavy work in the request: a claim or heartbeat is one indexed
  `UPDATE`; an event is one `INSERT` plus a Redis publish; an outcome is validated
  for shape and written to the outbox, and the artifact hashing (the only CPU-bound
  step, ~0.5 s per 170 MB file, streamed in 1 MB buffers) runs in the `applier`.
- Payloads are stored once, as columns, not re-parsed: `ai.jobs.result` holds the
  compact summary the UI needs; transcripts live in `transcriptions`, not in a job
  blob.
- One static binary per role, distroless image, no interpreter, no per-request
  allocation of ORMs or Pydantic models.

Expected footprint, to be confirmed by measurement in §17 step 5: each Go role idles
at a few MB of RSS and effectively 0 CPU; the `api` role serves callbacks at well
under 1 ms of CPU each; the `applier` is the only role whose CPU is proportional to
job volume (hashing and B2 copies), and it scales horizontally. The two Python AI
worker services, the gRPC channel in every Python worker type and the five AI crons
are removed outright, which is where the current CPU bloat comes from. Both numbers
(Python before, Go after) are recorded from cgroup CPU seconds and RSS over a week
of equal job volume and go into the admin stats page, so the saving is a measured
fact rather than an expectation.

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
  intent text null,               -- 'publish' | 'background_enrichment' | null (pipeline only)
  batch_id uuid null, group_id uuid null,
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

### 10.1 Job status and track status are different things

Today AI code writes `audio_tracks.status` in fifteen places (`TrackStateManager`
`mark_processing/mark_pending/mark_publish_pending/mark_retry_pending/mark_ready/
mark_ready_after_job/mark_draft`, the bulk `UPDATE audio_tracks SET status='ready'` in
`sync_stuck_ai_jobs`, the bulk `status='processing'` in `enqueue_group_tagging`). The
result is that a queued or failed AI job flips a track between `processing`, `pending`,
`draft` and `ready`, published tracks are touched by group tagging, and the UI
reads the job state off the track. All of that goes.

Rules in the Go service:

1. `ai.jobs.status` is the only job state. The Go service never writes
   `audio_tracks.status`. The track payload the frontend gets carries `jobs` and
   `latest_job` read from `ai.jobs` (a read-only SQL view `ai.v_track_jobs` the Python
   track serializer selects from), not derived from the track's status.
2. Pipeline and transcription results change **metadata only**: transcript, tags,
   category, discovery, descriptions, insights, moderation flags. They bump
   `audio_tracks.state_version` (today they do not, which is why catalog sync skips
   them) and never change `status`. A published track stays published and is
   re-indexed (§10.3).
3. Track status changes only when the **audio is replaced** by an approved Magic Clean
   or reconstruction (§10.2), by the user's own publish/unpublish/archive actions, or by
   an admin moderation decision. Nothing else.
4. The `processing`/`pending` "waiting for AI" track states are retired. Whether a job
   is running is a property of the job, shown from `ai.jobs`.
5. A failed, cancelled, expired or stale job leaves the track exactly as it was.

### 10.2 Replacing audio requires publishing again

Today `replace_source` runs with `preserve_publication=True`: a published track keeps
`published` while its audio, speed layers and catalog document are silently rebuilt.
That changes. When an approved Magic Clean or reconstruction replaces the audio of a
track whose status is `published` or `scheduled`:

1. In the same transaction as the media swap: `status = ready`, `published_at`
   kept as `last_published_at` (new column) and cleared, `publish_requested_at`
   cleared, `speed_layers = NULL`, `pipeline_completed_at = NULL`,
   `audio_revision + 1`, `state_version + 1`.
2. A `catalog_index_outbox` row with `action = 'delete'` for the track (and `upsert`
   rows for its publications/groups so their listings drop it); `content.track.
   unpublished` and `content.track.audio_changed` stream events; a
   `job_applied` notification whose text says the track must be published again.
3. A new `transcription` job is queued (word timings for the new audio).
4. The user publishes again through the normal publish flow; speed layers render,
   the track returns to `published`, the catalog row is upserted. No AI code is
   involved in publishing.

For a track that was `draft` or `ready`, the swap keeps its status. Pipeline
enhancement (`PIPELINE_ENHANCEMENT`, the pipeline replacing audio on its own) is
removed: the pipeline no longer uploads audio, so there is nothing to apply.

### 10.3 Meilisearch and catalog sync

The Go catalog service already owns indexing (`catalog_index_outbox` → relay →
worker → Meilisearch → confirmer). The AI applier only has to stage rows correctly,
in the same transaction as the write that changed the data, and it must do so for
every AI write (today failed jobs, tag-only writes and metadata writes without a
`state_version` bump can skip the queue):

| AI write | Rows staged in the same transaction |
| --- | --- |
| pipeline applied (tags, category, discovery, descriptions, transcript, flag) | `catalog_index_outbox(entity_kind='track', action='upsert', desired_state_version = new state_version, desired_audio_revision = audio_revision, priority 60)`; `audio_tracks.catalog_sync_status='pending'`, `catalog_sync_requested_at=now()`; one `upsert` row per publication and group that contains the track; `stream_events(stream_type='track', event_type='content.track.updated', payload.changes=[...])` |
| transcription applied | same as above (transcript is part of the document) |
| audio replaced on a published track | `delete` for the track, `upsert` for its publications/groups, `content.track.unpublished` + `content.track.audio_changed` |
| audio replaced on an unpublished track | `content.track.audio_changed` only (unpublished tracks are not in the index) |
| job failed / rejected / expired | nothing (the data did not change) |
| group tagging of N tracks | N track `upsert` rows (coalesced by the partial unique index `(entity_kind, entity_id) WHERE status='pending'`) plus one row for the group and each publication |

Rows use the existing `ON CONFLICT ... WHERE status='pending' DO UPDATE` with
`GREATEST(desired_state_version)`, `GREATEST(priority)`, `attempts = 0`, exactly as
the catalog worker's own inserts do, so a burst of 500 group tags produces 500
coalesced rows, not 500 × stages. The existing relay (`FOR UPDATE SKIP LOCKED`,
batch publish with confirms), worker replicas and reconciler (10-minute drift
repair) give the throughput and the self-healing; the AI service adds no sync path
of its own and never calls Meilisearch. `meili_synced` on the SSE `complete` event
is removed; the frontend already reacts to `content.track.synced` from the
confirmer, which is the only truthful signal.

### 10.4 Publishing and the pipeline

The Python publish flow today creates pseudo `job_type='publish'` processing jobs,
parks pipeline jobs at `speed_layers_queued 70%`, and gates publishing on
`has_completed_pipeline`. Publishing stays in Python (publish jobs, speed layers,
catalog confirmation) but loses every AI dependency:

- There is no `publish` AI job. `publish_jobs.pipeline_job_id` becomes
  `ai_job_id uuid null` referencing `ai.jobs`.
- `start_publish` modes collapse to two: if `track.pipeline_completed_at` is set,
  stage the publish job; otherwise create a pipeline job through the Go API with
  `intent='publish'` and stage the publish job as `pipeline_gated`. When Go applies a
  pipeline result it sets `pipeline_completed_at` and, for `intent='publish'`, flips
  the waiting `publish_jobs` row from `pipeline_gated` to `pending` in the same
  transaction. Python's publish worker picks it up from the queue it already
  consumes; the 5-minute `reconcile_publish_jobs` cron becomes a safety net only.
- `pipeline.generate_on_publish_only` keeps its meaning (auto-run the pipeline on
  upload or only on publish) and is read by the Python upload/publish code when it
  decides whether to create a pipeline job; the Go service does not know about it.

### 10.5 Group and batch jobs

Group tagging creates one `pipeline` job per track with `batch_id` set (new column on
`ai.jobs`). The reconciler emits `group_complete` on the group SSE stream when every
job of the batch is terminal (advisory lock on the batch id), with per-track
status. No track status is touched. Upload batches with `intent=publish` work the
same way with `intent` stored on the job.

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

## 15. Migration inventory: everything AI leaves the Python backend

Principle: the Python backend keeps no AI logic. It keeps the frontend-facing
business API; endpoints that start or resolve AI jobs move to the Go `api` role with
unchanged paths and bodies and the Python routers are deleted, not proxied. Paths are
under `hear-backend/src/app`.

### 15.1 Delete (logic moves to Go)

| Area | Files / symbols |
| --- | --- |
| AI service package | `services/ai/` entirely: `__init__`, `service`, `handlers`, `media`, `tagging`, `callbacks`, `job_result_processor`, `cleanup`, `notifications`, `submission_policy`, `client` (HTTP `/process`), `runtime_dispatch`, `runtime_attempts`, `runtime_grant`, `runtime_scope`, `serverless_submission`, `storage`, `scheduler`, `sse_publisher`, `b2_validator`, `audio_joiner` (the worker renders full files; the backend never joins segments), `track_state`, `errors`, `constants` |
| Worker handlers | `core/worker/handlers/ai.py` (all of it: submit, result apply, stuck-job sync, replay, apply/confirm, platform-settings push), `handlers/ai_dispatch.py`, `handlers/cleanup.py` branches `ai_old_media`, `ai_audio_tag`, `ai_job_folder` (deletions become plain `media_file_deletions` rows written by Go), `handlers/taxonomy.py::sync_ai_training_data` (calls gRPC methods that do not exist) |
| arq queues and crons | queues `backend`, `ai-result`; functions `process_transcription`, `process_magic_clean`, `process_rebuild`, `process_magic_clean_apply`, `process_reconstruct_confirm`, `process_ai_result`, `push_ai_platform_settings`, `dispatch_ai_jobs`; crons `dispatch_ai_jobs`, `sync_stuck_ai_jobs`, `replay_stored_ai_outcomes`, `purge_stale_magic_clean_previews`, `reconcile_ai_job_folder_cleanups`, `compact_processing_job_payloads`; `enqueue.py` helpers `enqueue_ai_*`, `enqueue_transcription/magic_clean/rebuild`, `enqueue_ai_tag_category_sync`; `core/utils/queue.py` re-exports; `lifespan.py` gRPC channel; worker services `worker-ai-result`, `worker-backend` in every compose/stack file |
| gRPC | `grpc_client/` (client, proto, generated stubs); `GRPC_*` settings |
| Internal HTTP | `api/v1/ai_runtime_callbacks.py`, `api/v1/internal/ai_runtime.py`, `api/v1/internal/ai_jobs.py`, `api/v1/internal/pipeline.py` (`/internal/process|enhance|transcribe|rebuild`), `api/v1/internal/tracks.py` (`/for-ai`), `api/v1/internal/platform_settings.py`; their mounts in `api/v1/__init__.py`, `internal_app.py`, `internal_api.py`; `dependencies.verify_service_key` if nothing else uses it |
| Settings | `core/config.py::AIServiceSettings` (`HEAR_HTTP_URL`, `HEAR_GRPC_TARGET`, `HEAR_AI_*`, `HEAR_RUNPOD_*`, `HEAR_STORAGE_*`, `AI_MAX_*`, `AI_DISPATCH_*`, `AI_RESULT_*`, `AI_HTTP_*`, `AI_SERVICE_SECRET` except the Alexa HMAC use), `PROCESSING_JOB_PAYLOAD_*`; `settings_service.py` `AI_PLATFORM_SETTING_GROUPS` push (`update_settings` no longer enqueues `push_ai_platform_settings`; the Go `api` role serves `GET /internal/ai/runtime/catalog` from the same tables); `.env.production.example` AI block; README AI sections (stale) |
| SSE | `sse_publisher.py`; the AI fallback snapshot built from `processing_jobs` in `api/v1/sse.py` (phase 1: the SSE app relays Redis only; phase 2: endpoint moves to Go) |
| Track state coupling | every `TrackStateManager` call listed in §10.1 and the two bulk `UPDATE audio_tracks` statements; `audio_source/service.py` policy `preserve_publication` for AI reasons (§10.2); `PIPELINE_ENHANCEMENT` reason and `_apply_tracks/_apply_master/is_enhanced/quality_score/snr_db` writes (the pipeline no longer returns audio) |
| Publish coupling | pseudo `publish` processing jobs (`service.py::_start_publish_with_existing_pipeline`), `speed_layers_pending_publish`, the 70 % `speed_layers_queued` stage, `_emit_publish_success` writes to `processing_jobs`, `publish_jobs.pipeline_job_id` → `ai_job_id` (§10.4) |
| Notifications and email | `services/ai/notifications.py`; `handlers/notify.py::send_ai_terminal_email`, `send_pipeline_started_email`, `send_pipeline_completion_email`; `email_backlog_service.py` scanning `processing_jobs`; markers `failure_email_sent_at`, `flagged_email_sent_at`, `_pipeline_*_email_sent`. Go stages rows in the existing notification outbox (same deterministic `ai:{job_id}:{phase}` keys the Go FCM worker already honours) and in a new `email_outbox` table that the Python email worker sends from (templates `recording-processing`, `recording-ready`, `recording-processing-failed`, `recording-flagged`) |
| Websocket `ai_progress` | all `ws_manager.send_to_user(... "ai_progress" ...)` calls; the SSE track stream is the single progress channel |
| Models and schemas | `models/processing.py::ProcessingJob` (kept read-only until §17 step 6), `schemas/ai_job.py`, `schemas/ai_runtime.py`, `schemas/recording.py::MagicCleanRequest/TrackTagJobRequest/TrackAudioTagRequest/DiscoveryRequest`, `AIJobSummaryRead`, `schedule.py::ProcessingJobRead` |
| Tests and docs | the `tests/test_ai_*`, `test_grpc_*`, `test_processing_job_*`, `test_audio_joiner`, `test_audio_tag_submission` files; `docs/AI_RUNTIME_V1.md`; k6 helpers referencing AI endpoints |

### 15.2 Move to the Go `api` role (same paths, same bodies)

`POST /api/v1/creator/tracks/{id}/magic-clean` (body becomes `{"profile", "auto_level"?, "remove_clicks"?, "trim_silence"?, "sound_cleanup"?}`; the legacy speech/music/background sliders are gone),
`.../magic-clean/{job_id}/apply|reject`, `.../transcribe`, `.../pipeline`, `.../tag`
(a pipeline job), `.../discovery` (a pipeline job), `.../regenerate` (manual retry →
new job), `.../edit-recording` and `.../edit-recording/{job_id}/status|confirm|rollback`,
`PATCH .../transcription` with `changes` (creates a reconstruction `edit_transcript`
job; the transcript-only save stays in Python), `GET .../jobs`, `GET .../jobs/{job_id}`,
`GET .../jobs/pending`, `POST /groups/{id}/tag`, the batch creation's job submission
(`POST /batches` keeps the upload logic in Python and calls Go once per track),
admin `GET/PATCH /admin/settings/ai`, `GET /admin/ai/*` (§11),
`POST /internal/ai/attempts/{id}/*`, `GET /internal/ai/runtime/catalog` and
`/protocol`, `POST /internal/ai/providers/runpod/{attempt_id}` (webhook).
Retired: `POST .../moderate` (pipeline), `POST .../audio-tag` and the `audio_tag`,
`categorization`, `tagging`, `discovery` job types (all are the worker's `pipeline`
type; `max_tags` is an option), `POST /internal/ai-jobs/{id}/replay-apply` (admin
re-apply in §11), `POST /internal/enhance|transcribe|rebuild|process`.

### 15.3 Stays in Python, reading `ai.jobs` only through views

Track, group and publication serializers (`recording_service.py` `_build_active_jobs_map`,
`latest_jobs_for_tracks`, `_job_to_sse_event`, `list_jobs_for_track`; `publication_service.py`
`_batch_enrich_tracks`, `get_group`; `track_state/service.py::snapshot`;
`schedule_service.py` `ai_jobs`; `content_insight_service._list_from_completed_jobs`)
select from `ai.v_track_jobs` (`job_id, track_id, job_type, status, stage, progress_pct,
transport, error_code, requires_approval, preview_audio_url, created_at, finished_at`).
`start_publish`, `PublishJob`, speed layers, `generate_on_publish_only`, catalog sync
services, content events, moderation reports, track deletion (which now also deletes
`ai.jobs` rows for the track via `ON DELETE CASCADE`), `content_insights`,
`content_flags` and `ai_insights` (creator insights, not job-driven) stay as they are.

### 15.4 Database

New schema `ai` (§3) plus `audio_tracks.last_published_at`, `publish_jobs.ai_job_id`,
`email_outbox`. `processing_jobs` and its indexes, `deferred_cleanups` of types
`ai_job_folder/ai_old_media/ai_audio_tag`, the `media_files.metadata_json.
ai_runtime_verified_sha256` key, `speed_render_jobs.publish_job_id` (→ `publish_jobs.id`)
and `audio_tracks.is_enhanced/quality_score/snr_db` are dropped in the final step of
§17 after the history has been copied.

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

## 17. Migrating correctly: data, cutover and proof

1. **Freeze.** Set `pipeline.generate_on_publish_only` on and disable the creator AI
   endpoints in Python (503 with a maintenance message) so no new `processing_jobs`
   are created. Let active ones finish; after 30 minutes fail the rest with
   `migration_abandoned` and notify their users (they can retry on the new system).
2. **Copy history.** One migration copies terminal `processing_jobs` rows into
   `ai.jobs` (status, type, timestamps, `error_message`, `result_metadata` summary,
   `retry_of_job_id`) so job lists and insights keep their history; `publish_jobs.
   pipeline_job_id` is rewritten to `ai_job_id`. Rows are kept read-only in
   `processing_jobs` until step 6.
3. **Verify in parallel.** With the Go roles running against the same database and
   `routing.policy = pod_only`, dispatch one real job of each type from Go for a test
   creator and check: callbacks accepted (`ai.attempt_events` complete), outcome
   applied, `audio_tracks.state_version` bumped, `catalog_index_outbox` row staged and
   confirmed by the Go catalog confirmer, `content.track.updated` seen on the content
   SSE, notification and email rows staged, the track status unchanged for pipeline
   and transcription, `ready` after an approved clean on a published track, Retry
   creating a new job, a failed outcome producing `failed` with no retry, a killed
   worker producing `worker_lost` within 15 s of lease expiry.
4. **Switch.** Point the frontend proxy routes for the AI endpoints at the Go `api`
   role, enable `serverless.enabled` with the agreed budget, remove the Python
   routers and workers (§15.1), redeploy Python without the AI worker services.
5. **Watch for a week:** `hear_ai_*` metrics, the admin job view, catalog
   reconciler drift report (should stay at zero), `media_file_deletions` backlog.
6. **Drop** the legacy tables and columns (§15.4) and the `hear-ai` dispatch package.

Verification queries kept in the Go repo as integration tests (gated by a test
database URL, like the catalog tests): no job in a non-terminal state older than its
deadline; every `ai.jobs.status='completed'` row with an audio artifact has a
`media_file_deletions` row for the previous media; every applied pipeline/transcript
has a `catalog_index_outbox` row at the track's current `state_version` or a confirmed
`catalog_synced_state_version` equal to it; every published track has
`pipeline_completed_at` set; no `audio_tracks.status` value of `processing` or
`pending` remains.
