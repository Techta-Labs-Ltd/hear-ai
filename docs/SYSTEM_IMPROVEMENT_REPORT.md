# Hear AI system improvement report

**Review date:** 2026-08-28  
**Scope:** production entry point, Ray Serve application graph, HTTP/gRPC
transport, authentication and authorization, PostgreSQL job state, scheduling,
audio acquisition and delivery, model deployments, reconstruction, Magic
Clean, observability, tests, and engineering quality.

## Executive summary

Hear AI has a sensible core architecture: one Python 3.12 project, Ray Serve
ownership of ingress and model deployments, PostgreSQL-backed job state,
backend-scoped service-key authentication, encrypted job storage credentials,
idempotent submission, fair scheduling, typed terminal gRPC results, and
job-scoped temporary audio.

The system nevertheless has several production-readiness blockers. The most
important are:

1. Discovery results are not filtered by authenticated backend.
2. Any authenticated backend can invoke global settings mutations.
3. User-supplied audio URLs lack SSRF protection, byte/duration limits, and a
   finite read deadline.
4. Readiness does not verify PostgreSQL or required model deployments.
5. Database schema changes run from every gateway replica during startup.
6. Blocking database, filesystem, boto3, and model work occurs in async actors.
7. Cancellation is not propagated into long downloads and model operations.
8. Several job stage streams contradict their declared contracts.
9. Static analysis is not clean: Ruff reports 771 violations and Mypy reports
   153 errors in 22 source files.

The configured non-live test suite passes, but it does not exercise a real Ray
cluster, PostgreSQL recovery, object storage, GPU inference, cancellation during
inference, or real audio quality. Production acceptance should remain blocked
until the P0 security items and P1 readiness/reliability items are addressed.

## Review method and evidence

The review traced repository behavior from these sources:

- `main.py` and `hear/deployments/app.py`
- `hear/deployments/gateway.py`
- `hear/services/transport/grpc.py` and `operations.py`
- `hear/services/jobs/submission.py` and `scheduler.py`
- `hear/orchestrator.py` and `hear/models/stages.py`
- `hear/models/database.py` and `hear/core/db_gate.py`
- `hear/core/backend_registry.py`, `storage.py`, `downloader.py`, and
  `hear_temp.py`
- model, transcription, Magic Clean, reconstruction
  services
- protobuf contracts and the configured tests

Checks executed against the current worktree:

| Check | Result |
|---|---|
| `uv run pytest` | 256 passed, 49 warnings |
| `uv run ruff check hear main.py tests` | Failed with 771 violations; 414 reported automatically fixable |
| `uv run mypy hear` | Failed with 153 errors in 22 files |
| Live inference/storage/database E2E | Not run; requires an authorized isolated target |

The worktree already contained modified and untracked user work. This report
does not assume those changes are committed or deployed.

## Current architecture assessment

### Strengths to preserve

- Ray Serve owns HTTP, gRPC, model lifecycle, and deployment handles.
- `main.py` validates immutable runtime artifacts before starting Ray.
- Asynchronous jobs use backend-generated idempotency keys.
- Durable job and result state is stored in PostgreSQL.
- Storage credentials are encrypted at rest and omitted from result payloads.
- Storage destinations are checked against a per-backend allow-list.
- Object keys are constrained to a submitted folder prefix.
- Uploads verify remote object size.
- Terminal results are backend-scoped and recoverable after stream reconnect.
- Job scheduling includes per-user round-robin fairness and per-type limits.
- Temporary paths are job/run scoped and cleaned on normal terminal paths.
- Protobuf results use typed payloads instead of returning arbitrary structs.

### Target architecture

The existing architecture should be evolved, not replaced:

```text
Backend
  |-- HTTPS POST /process ----> authenticated admission + durable PostgreSQL job
  `-- gRPC -------------------> authorized typed operations and event replay
                                      |
                                      v
                             durable scheduler/lease owner
                                      |
                 +--------------------+--------------------+
                 |                    |                    |
          transcription         Magic Clean          Fish Speech
                 |                    |                    |
                 +--------------------+--------------------+
                                      |
                         scoped object storage + verified artifact
```

PostgreSQL should remain the durable source of truth. Ray should continue to
own execution and model lifecycle. The improvements below focus on making the
boundary between them explicit, recoverable, authorized, and observable.

## Findings

## P0 — security and tenant isolation

### SYS-001: discovery listing crosses backend boundaries

**Evidence**

- `PipelineGrpcService.ListDiscovery` authenticates a service key but does not
  pass the resolved backend identity to the operation.
- `Operations.list_discovery` queries all completed `AiTrackJob` rows containing
  discovery data.
- `AiTrackJob` does not contain `backend_id`.

**Impact**

One registered backend may receive track ids, job ids, timestamps, and
discovery metadata produced for another backend. This violates the isolation
guarantee documented for gRPC traffic.

**Required change**

1. Add non-null `backend_id` to `AiTrackJob` for new rows.
2. Backfill it from the owning `AiJob` in a controlled migration.
3. Add an index supporting `(backend_id, status, completed_at)`.
4. Pass authenticated `backend_id` from the gRPC layer.
5. Filter discovery before sorting and pagination.
6. Decide whether discovery is private-per-backend or intentionally public. If
   public, expose that through a separate explicit public contract.

**Acceptance criteria**

- Backend A cannot list Backend B data.
- Missing backend ownership never falls back to global visibility.
- Unit and PostgreSQL integration tests cover two backends and pagination.

### SYS-002: privileged RPCs have authentication but no authorization

**Evidence**

`UpdatePlatformSettings` uses the
same service-key authentication as ordinary job consumers. The registry stores
identity and storage allow-lists but no roles or capabilities.

**Impact**

Any backend credential can mutate global moderation keywords, global auto-tag
behavior. A compromised ordinary
backend key becomes a control-plane credential.

**Required change**

- Add explicit backend capabilities, for example:
  `jobs.submit`, `jobs.read`, `discovery.read`, and `platform.settings.write`.
- Default registrations to least privilege.
- Require the relevant capability before invoking an operation.
- Separate service identities for control-plane automation and customer/backend
  job traffic.
- Write an audit record containing actor backend, operation, timestamp, request
  fingerprint, and result. Never store secrets or full sensitive content.
- Add rate limits and concurrency limits for settings mutations.

**Acceptance criteria**

- An ordinary backend receives `PERMISSION_DENIED` for all global mutations.
- A specifically authorized identity succeeds.
- Every mutation is auditable and idempotent where applicable.

### SYS-003: arbitrary audio URL downloading enables SSRF and exhaustion

**Evidence**

`download_audio` follows redirects, accepts any caller-supplied URL, uses
`read=None`, streams until EOF, and applies no byte limit. FFmpeg then decodes
the downloaded object without a process timeout or decoded-duration limit.

**Impact**

- Access to private, loopback, link-local, or cloud metadata endpoints.
- Redirect-based bypass of initial URL validation.
- Worker disk exhaustion and indefinitely occupied jobs.
- Very long or decompression-bomb audio consuming excessive CPU/GPU.
- Reduced availability for every tenant sharing the worker.

**Required change**

1. Require HTTPS for remote source URLs.
2. Resolve every request and redirect target; reject loopback, private,
   link-local, multicast, reserved, and unspecified IP ranges.
3. Prefer per-backend source-host allow-lists or signed object URLs from known
   storage providers.
4. Add finite connect, read, pool, and overall deadlines.
5. Reject excessive declared `Content-Length` and enforce a streaming byte cap.
6. Limit redirect count and revalidate each destination.
7. Validate permitted media types while still treating content as untrusted.
8. Run `ffprobe` with a timeout and reject excessive duration/channels/sample
   rate before inference.
9. Run FFmpeg with a timeout and resource limits.
10. Add per-backend/user queued-byte, duration, and concurrent-download quotas.

**Acceptance criteria**

- Tests reject loopback, private IPs, metadata IPs, redirect-to-private,
  oversized chunked bodies, stalled responses, and over-duration audio.
- Partial files are removed for every rejection path.

## P1 — availability, durability, and correctness

### SYS-004: readiness can report success while dependencies are unusable

**Evidence**

`/ready` trusts `health_data().status`. That operation returns `healthy` based
on local CUDA visibility plus orchestrator queue counts. It does not query
PostgreSQL or required model handles.

**Impact**

Traffic can be admitted while PostgreSQL is down, a model constructor is
failing, a checkpoint is incomplete, or temporary storage is exhausted. The
2026-08-25 Magic Clean incident is an example of readiness not representing
job-type capability.

**Required change**

- Make liveness cheap and local.
- Make readiness verify PostgreSQL with a bounded query, orchestrator response,
  schema version, and required deployment health.
- Report per-capability readiness: transcription, Magic Clean, regeneration,
  categorization, storage/temp capacity.
- Reject only affected job types when an optional capability is unavailable.
- Include stable machine-readable failure codes but not paths containing
  secrets or infrastructure credentials.

**Acceptance criteria**

- PostgreSQL loss makes readiness fail within a bounded interval.
- Magic Clean unavailability rejects or defers Magic Clean without falsely
  rejecting unrelated healthy operations.
- Health checks themselves cannot block indefinitely.

### SYS-005: application replicas perform schema migration at startup

**Evidence**

Every gateway replica calls `init_db()`, which attempts extension creation,
`Base.metadata.create_all`, repeated `ALTER TABLE ADD COLUMN`, and data updates.

**Impact**

- Concurrent DDL and lock contention during rollout.
- Requirement for elevated database privileges in the application identity.
- No ordered schema version or reliable downgrade path.
- Deployment health becomes coupled to mutation privileges.
- A partially applied migration may be difficult to diagnose.

**Required change**

- Adopt a versioned migration tool such as Alembic.
- Run migrations once in a controlled deployment step using a migration
  identity.
- Make application startup read-only with respect to schema.
- Check and expose the expected schema version in readiness.
- Convert legacy nullable ownership fields to non-null after a verified
  backfill.

**Acceptance criteria**

- Gateway replicas start concurrently without issuing DDL.
- Application credentials cannot create extensions or alter tables.
- Upgrade and rollback procedures are documented and tested.

### SYS-006: blocking operations run inside async Ray actors

**Evidence**

Synchronous SQLAlchemy queries, filesystem reads/writes, boto3 operations,
Transformers inference, and some HTTP calls occur in async methods. Ruff's
`ASYNC` findings identify several filesystem cases, while database and model
calls are not all detected automatically.

**Impact**

Blocking one actor event loop delays health responses, gRPC stream events,
cancellation, queue dispatch, and unrelated requests. High `max_ongoing_requests`
does not provide concurrency if calls block the loop.

**Required change**

- Use bounded `asyncio.to_thread`/executors for blocking file, database, boto3,
  and CPU-bound library operations where async replacements are not adopted.
- Do not create an unbounded task per request.
- Move heavy synchronous model inference behind Ray deployment methods that
  intentionally serialize or batch inference.
- Reuse bounded HTTP clients instead of creating one client per download.
- Measure event-loop lag per deployment.
- Reassess the process-wide `db_write_lock`, which serializes commits within an
  actor while other replicas/processes remain unsynchronized.

**Acceptance criteria**

- A slow download or upload does not delay heartbeat/health responses.
- Cancellation latency remains bounded during every long stage.
- Load tests demonstrate predictable queueing rather than event-loop stalls.

### SYS-007: cancellation is stage-boundary only

**Evidence**

`_run_is_current` protects stage transitions and terminal writes, but active
downloads, remote inference, reconstruction batches, and uploads do not share a
cooperative cancellation token.

**Impact**

A cancelled job may continue consuming GPU, CPU, bandwidth, and disk. It may
also upload an artifact that is never returned, increasing storage leakage and
cost.

**Required change**

- Introduce a durable cancellation flag/token keyed by job and run.
- Check it during download chunks, Magic Clean chunks, transcription chunks,
  reconstruction groups, and before/after every remote model call and upload.
- Where Ray permits it safely, cancel outstanding object references.
- Track artifacts created during a run and delete uncommitted artifacts after
  cancellation/failure.
- Define cancellation semantics: best effort, terminal state priority, and
  response when completion races cancellation.

**Acceptance criteria**

- Queued cancellation releases the scheduler entry immediately.
- Running cancellation stops work within an agreed duration.
- No terminal completion overwrites cancellation.
- No unreferenced artifact or temporary file remains.

### SYS-008: stage stream contract is inconsistent with processing

**Confirmed mismatches**

- `audio_tag` declares `audio_tagging` but invokes `transcribing`.
- `edit_transcript` declares `downloading`, `transcribing`,
  `diffing_transcript`, and `reconstructing_edits`, but invokes only
  `reconstructing` after earlier work.
- Direct reconstruction does not emit its declared download stage.
- Magic Clean emits stage changes but no intermediate stage-result events.
- Some internal queue fields are discarded because `PipelineEvent` has no typed
  fields for them.

**Impact**

Backend progress displays are incorrect, progress may remain zero, monitoring
cannot attribute latency correctly, and client assumptions diverge from server
behavior.

**Required change**

- Create one typed stage transition helper used by every job flow.
- Emit the declared stage before work begins and a corresponding stage result
  after it completes where useful.
- Make progress monotonic and reserve 100 for terminal success.
- Either add typed queue fields to protobuf or put a documented shape in
  `result`; do not silently discard producer fields.
- Add golden contract tests for every job family, reconnect, retry, failure,
  cancellation, and terminal replay.

**Acceptance criteria**

- Exact ordered stages match `hear/models/stages.py`.
- Unknown stage ids fail tests.
- Terminal typed `GetResult` and terminal stream result agree.

### SYS-009: scheduling is durable only at the job-state level

**Evidence**

PostgreSQL stores queued/running status, but fair-order queues, active counters,
and type occupancy are held in one orchestrator replica. Recovery scans rebuild
an in-memory queue without a durable enqueue sequence or worker lease.

**Impact**

- Fair ordering and queue position change after restart.
- A lost worker cannot be distinguished from a legitimately long operation.
- Horizontal orchestrator scaling would risk duplicate claims without a lease
  protocol.
- Queue statistics are process-local estimates.

**Required change**

- Persist enqueue time/sequence, priority, user id, lease owner, lease expiry,
  and heartbeat.
- Claim rows atomically with PostgreSQL locking (`FOR UPDATE SKIP LOCKED`) or an
  equivalent lease mechanism.
- Renew leases during work and reclaim expired jobs with a bounded attempt
  policy.
- Calculate durable queue position and age.
- Add maximum total and per-user queued jobs.
- Align type admission with actual Ray replica capacity.

**Acceptance criteria**

- Restart preserves ordering within the documented fairness policy.
- A killed worker is recovered exactly once.
- Two potential dispatchers cannot execute the same run concurrently.

### SYS-010: configured concurrency exceeds scarce model capacity

**Evidence**

The orchestrator defaults to eight concurrent jobs and allows three Magic Clean
jobs, while Magic Clean defaults to one replica. Reconstruction permits two
jobs while Fish Speech defaults to one replica. Multiple GPU deployments also
request fractional shares on the same device.

**Impact**

The application moves waiting from a visible admission queue to hidden Ray
handle/model queues, increases temporary-file lifetime, and risks GPU memory
contention without increasing throughput.

**Required change**

- Benchmark each stage and build an explicit GPU memory/capacity budget.
- Set orchestrator type limits no higher than effective replica concurrency
  unless intentional downstream queueing is measured and bounded.
- Expose downstream handle queue depth and cold-start latency.
- Consider separate GPU worker pools for resident transcription/LLM and bursty
  Magic Clean/Fish Speech workloads.
- Apply admission backpressure before downloading audio.

**Acceptance criteria**

- Load tests identify stable saturation throughput and bounded p95 queue delay.
- No OOM or model eviction occurs at configured concurrency.

## P2 — maintainability, contracts, and operations

### SYS-011: static-analysis gates are substantially failing

**Evidence**

- Ruff: 771 violations.
- Mypy: 153 errors across 22 files.
- Production findings include undefined `Optional` and `Any` in
  `hear/deployments/language_models.py`, blocking async operations, implicit
  optional types, tensor/NumPy type confusion, and untyped SQLAlchemy models.

**Impact**

Real defects are obscured by formatting debt, CI cannot serve as a regression
gate, and refactoring audio/model paths is riskier.

**Required change**

1. Immediately fix `F821`, `F401`, `F811`, `ASYNC`, and other correctness
   classes in production source.
2. Introduce SQLAlchemy 2 `Mapped[]` models so type checking understands ORM
   attributes.
3. Define typed dictionaries or Pydantic models for internal model requests and
   results.
4. Separate source lint from legacy script/integration-test cleanup if needed,
   but do not permanently exclude production paths.
5. Ratchet CI: no new violations, then reduce the baseline to zero.

**Acceptance criteria**

- `uv run ruff check hear main.py tests` passes.
- `uv run mypy hear` passes with meaningful coverage.
- CI runs both before unit tests are accepted.

### SYS-012: database integrity relies heavily on application convention

**Evidence**

Ownership fields remain nullable, `AiTrackJob` has no foreign keys or unique
constraint for job/run/track, timestamps are generally naive UTC, and statuses
are unconstrained strings.

**Required change**

- Add foreign keys and deliberate cascade behavior.
- Add unique `(job_id, run_id, track_id)`.
- Make ownership and required identities non-null.
- Use timezone-aware UTC columns and values.
- Constrain status/job type or validate them in a single domain layer.
- Add retention policies for jobs, events, and previews.

### SYS-013: intermediate event history is not durable

**Evidence**

Only current job state and terminal result are persisted. Live events exist in
an in-memory queue. Reconnect after completion can replay a terminal event but
cannot replay the stage history.

**Required change**

Add a bounded append-only job-event table with `(job_id, run_id, sequence)`,
event type, stage, status, progress, safe result metadata, and timestamp. Let
`Subscribe` accept/reuse a sequence cursor if the protobuf contract is extended.
Apply retention and avoid storing raw transcript/audio content unnecessarily.

### SYS-014: observability is insufficient for production diagnosis

**Evidence**

Some failure paths use `print()` and traceback output. Queue estimates are
mostly placeholders. There is no consistent evidence of metrics covering
stage latency, GPU model queues, event-loop lag, storage transfer, and temp
capacity.

**Required change**

- Structured logging with job/run/backend/type/stage correlation.
- Redaction rules for URLs, credentials, transcripts, and storage contexts.
- Metrics for admission wait, stage duration, model calls, retries,
  cancellation latency, download/upload bytes, temp disk, database pool, Ray
  queue depth, GPU memory/OOM, and artifact verification.
- Traces spanning HTTP submission, durable claim, model calls, upload, and gRPC
  result recovery.
- Alerts tied to user impact and per-job-type readiness.

### SYS-016: typed payload completeness is inconsistent

**Examples**

- `AudioTagPayload.source_audio_url` exists but is not populated.
- Magic Clean exposes transcription and moderation fields but currently returns
  defaults.
- Direct reconstruction and transcript editing share a payload although several
  fields are meaningful only for one path.
- Proto3 scalar defaults make “missing measurement” indistinguishable from a
  real numeric zero.

**Required change**

- Populate contracted fields or remove/deprecate them deliberately.
- Use `optional` scalar fields or explicit `measurement_available`/status when
  absence matters.
- Add result-schema golden tests for every job type.
- Version breaking contract changes and regenerate stubs only during a
  controlled build step.

### SYS-017: real audio quality needs objective and human release gates

Unit tests validate transformations and contracts but do not establish that
real model output is acceptable.

Add a versioned non-private fixture corpus covering:

- clean speech, noise, reverberation, music beds, overlapping speakers;
- silence boundaries and very short/long clips;
- multiple speakers, accents, genders, and speaking rates;
- transcript replacements, insertions, deletions, and duration changes.

Measure intelligibility/transcription preservation, DNSMOS or equivalent,
signal-to-noise improvement, LUFS, peak/clipping, duration drift, splice
discontinuity, speaker similarity, and artifact readability. Pair objective
thresholds with blinded human A/B listening for Magic Clean and regenerated
speech.

## Recommended delivery plan

### Phase 0: immediate containment (1-3 days)

- Restrict global mutation RPCs to a dedicated trusted identity or disable them.
- Disable or restrict `ListDiscovery` until backend filtering exists.
- Apply a conservative source-host allow-list and proxy-level body/time limits.
- Fix undefined names and other production `F`-class Ruff errors.
- Align Magic Clean admission with current replica capacity.

**Exit gate:** no known cross-tenant read or ordinary-backend control-plane
mutation is possible.

### Phase 1: security and readiness (1-2 weeks)

- Implement backend capabilities and audit events.
- Add backend ownership to track jobs and discovery filtering.
- Add SSRF-safe bounded downloading and media limits.
- Implement dependency-aware readiness and per-job-type admission.
- Introduce controlled versioned database migrations.

**Exit gate:** security integration tests pass with two tenants; dependency
failures produce correct readiness and admission behavior.

### Phase 2: job correctness and recovery (2-4 weeks)

- Correct all stage flows and typed result completeness.
- Implement cooperative cancellation and artifact rollback.
- Add durable leases, enqueue ordering, queue limits, and worker recovery tests.
- Persist bounded job event history.
- Move blocking work off actor event loops.

**Exit gate:** restart, retry, reconnect, and cancellation E2E tests pass
without duplicate execution or leaked artifacts.

### Phase 3: performance and quality (2-4 weeks)

- Benchmark deployment capacity and GPU memory.
- Tune replica counts and admission limits from measured data.
- Add full metrics, dashboards, traces, and alerts.
- Establish objective and human audio-quality gates.

**Exit gate:** target load is sustained with bounded p95/p99 latency, no OOM,
and approved audio-quality results.

### Phase 4: engineering baseline (parallel, completed before release)

- Reduce Ruff and Mypy to zero.
- Replace naive UTC usage and address dependency deprecations.
- Add typed ORM and internal request/result models.
- Run unit, PostgreSQL integration, Ray recovery, storage, security, and live
  non-production audio suites in CI/release automation at appropriate stages.

## Prioritized backlog

| Priority | Item | Primary owner | Verification |
|---|---|---|---|
| P0 | Backend-filter discovery | API/data | Two-tenant integration test |
| P0 | Capability-based admin RPC authorization | Security/API | Permission and audit tests |
| P0 | SSRF-safe bounded audio downloader | Platform/security | Malicious URL test suite |
| P1 | Dependency-aware readiness | Platform | Dependency fault injection |
| P1 | Versioned DB migrations | Data/platform | Upgrade/rollback test |
| P1 | Cooperative cancellation | Orchestrator/audio | Live cancellation E2E |
| P1 | Correct stage streams | API/orchestrator | Golden stream tests |
| P1 | Durable leases and queue limits | Orchestrator/data | Kill/recovery/load tests |
| P1 | Remove async blocking | Service owners | Event-loop lag/load test |
| P1 | GPU capacity alignment | ML platform | Memory and throughput benchmark |
| P2 | Static-analysis cleanup | All owners | Ruff and Mypy pass |
| P2 | Typed payload completeness | API/ML | Protobuf golden tests |
| P2 | Durable event history | Data/orchestrator | Cursor replay tests |
| P2 | Observability | Platform | Dashboard/alert drill |
| P2 | Audio-quality corpus and gates | Audio/QA | Objective + human evaluation |

## Production release checklist

### Security

- [ ] Discovery is backend-filtered or intentionally exposed through a separate
  public contract.
- [ ] Global mutations require explicit capabilities.
- [ ] Source URL SSRF, redirects, size, duration, and timeouts are controlled.
- [ ] Storage credentials and sensitive audio/text are redacted from telemetry.
- [ ] Cross-backend tests cover jobs, results, previews, discovery, and
  cancellation.

### Reliability

- [ ] Readiness verifies PostgreSQL, schema, temp capacity, and required models.
- [ ] Schema migration is not performed by application replicas.
- [ ] Cancellation stops active work and removes orphan artifacts.
- [ ] Durable lease recovery prevents duplicate execution.
- [ ] Queue limits and admission backpressure are configured.
- [ ] Stage streams and typed results match the protobuf contract.

### Quality and performance

- [ ] Ruff and Mypy pass.
- [ ] Unit and focused integration suites pass.
- [ ] Real Ray/PostgreSQL/storage restart and reconnect tests pass.
- [ ] Load tests pass at expected concurrency without OOM or disk exhaustion.
- [ ] Magic Clean and regeneration pass objective and human audio evaluation.
- [ ] Operational dashboards and alerts have been exercised.

## Final assessment

The system has a strong architectural foundation and good unit-level contract
coverage, but passing unit tests should not be interpreted as end-to-end
production readiness. The immediate risk is concentrated at trust boundaries
and runtime dependency boundaries: tenant authorization, arbitrary URL intake,
global control-plane operations, readiness, schema ownership, and cancellation.

Addressing the three P0 findings first materially reduces security exposure.
The P1 work then converts the existing durable-job design into a system that is
actually recoverable, observable, and predictable under failure and load. Only
after those gates pass should real Magic Clean and regeneration quality results
be used to approve production release.
