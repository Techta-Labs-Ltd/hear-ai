# Hear backend: direct file-by-file implementation plan

Version 2.0 — 20 September 2026  
Baseline: `60387821fab069d4ea80cc0910ec09779b221cde`  
Companion AI baseline: `a89ee0ff9b231351034d612b7c4dfa6f8e42b091`  
Follow the packages and T01–T44 tests in `00_IMPLEMENTATION_ORDER.md`.

## B00. Scope and fixed owners

Implement in the existing modules below. All 18 files in `src/app/services/ai` have a disposition. Specific known methods are named; remaining per-file directions are implementation targets against the inspected directory inventory, not a claim of exhaustive production testing. New files/methods are explicitly identified. Keep unrelated backend features untouched except for their named integration boundary.

Keep ProcessingJob, StreamEvent/EventJournal, AudioTrack.audio_revision, AudioSourceMutationService, the existing approval metadata, SystemHealthService and SystemIncidentService. Extend these owners. Do not create another job service tree, another source revision on AudioTrack, another event-journal database or a second approval implementation.

The main new service modules allowed are `services/ai/repository.py` for cohesive job/attempt queries and `services/ai/reconciler.py` for durable recovery. Reuse an existing exact equivalent if it is already present in a later commit. Put the internal control router in the proposed `api/v1/internal/ai_execution.py`; add only the required model/migration/delivery fields. Reuse current dependency/lifespan composition rather than inventing an application-wide service-locator framework.

Write constructor-injected classes for owners of resources. Keep pure policy, formatting and normalization functions as functions. Remove empty inheritance shells, hidden globals, explanatory code-comment blocks, commented-out old versions, TODO placeholders, debug prints and swallowed failures. Preserve legal/generated/tool directives. Tests and these plans carry the explanation.

## B01. Shared execution API — new `src/app/api/v1/internal/ai_execution.py`

**Action: add one authenticated internal control adapter. Packages P02, P03.**

Implement the operations below as thin routes to the injected AI job/repository/grant/result owners. The route names are proposed new contract paths, not existing endpoints.

| Route under `/api/v1/internal/ai/jobs/{job_id}/attempts/{attempt_id}` | Required operation |
|---|---|
| `POST /claim` | Verify service identity, current fence, deadline and backend/source scope; atomically assign execution_id and worker_epoch. Duplicate claim cannot start a second writer. |
| `POST /heartbeat` | Check current execution/fence, record liveness and return continue/cancel/drain plus a bounded lease renewal. |
| `POST /grants/refresh` | Issue real refreshed access for the same authorised source/destination; reject scope/semantic changes. |
| `POST /events` | Validate/deduplicate a bounded batch of worker progress/failure events; append through existing EventJournal in the same transaction as the allowed state change. |
| `POST /result` | Validate the known manifest and current identity, persist the result receipt/journal atomically and schedule application. Duplicate identical receipt acknowledges the existing result. |

Use TLS on untrusted links, scoped service authentication, payload limits and explicit deadlines. Bind backend identity to the authenticated service, not only the submitted JSON. Do not expose this router as a user-authorised arbitrary job mutation API. Redact grant bodies and source query strings in logs/traces.

Do not hold a transaction open while fetching a large manifest/object. Read bounded metadata first, perform remote verification outside the transaction, then lock/recheck current attempt/source before committing receipt. The time-of-check/time-of-use boundary must be closed by that final CAS.

**Done when:** T08–T11/T14–T19 pass on actual route/service integration.

## B02. Job and attempt model — `src/app/models/processing.py`

**Action: extend ProcessingJob; do not replace it. Packages P02, P06.**

Retain current fields including result_received_at, result_apply_started_at, result_applied_at, result_apply_attempts, result_apply_failures, next_attempt_at, dispatch_started_at, callback_payload, result_payload and result_metadata. Do not add duplicate timestamps/counters under slightly different names. Keep ScheduledContent, ContentFlag, ContentInsight and AiInsight unchanged unless a specific migrated caller requires a targeted change.

Add the missing durable identity/ownership fields to ProcessingJob: authenticated backend identity, semantic request hash, execution protocol version, current attempt ID/fence, pinned input media/source identity, expected audio revision, blocked reason and last accepted durable event cursor as needed. The transport `source_revision` maps to the snapshot of existing `AudioTrack.audio_revision`; do not add a competing canonical revision to the track.

Add a ProcessingJobAttempt record in this existing model module, or the project's established model split if already required. It must hold job/fence identity, claimed execution/worker epoch, accepted/started/heartbeat/lease/deadline times, output prefix/manifest key, execution state, consumed failure count/error classification and finish time. Enforce uniqueness of `(job_id, fence)` and current execution claim. Distinguish delivery attempts from actual inference failures; current `attempt_count` must not be silently repurposed without migration.

Persist immutable request/control/source snapshots. Keep renewable secrets outside callback_payload/result_metadata. Preserve compatible run_id mapping for legacy jobs. Result/preview metadata includes the producing attempt/fence/source and policy/model versions.

Index due queued jobs by eligibility/order, leased attempts by expiration/state, owner/source lineage by identity/revision and result-application work by due state. Use partial indexes where supported by the actual query plan. Plan retention for history without deleting a pending preview, accepted manifest or unresolved cleanup obligation.

**Done when:** migration and repository tests prove current-attempt uniqueness, rollback compatibility and no loss of existing job/result/approval data; T08–T13/T40.

## B03. Existing event storage — `models/stream_event.py`, `services/events/journal.py`, `services/events/transport.py`

**Action: extend the existing durable journal, not another event store. Packages P02, P07.**

Use StreamEvent for job stream events. Retain its event UUID/ID and cursor-based replay. Treat the stored event ID as the durable replay cursor; global event-ID gaps are normal across different streams and are not proof of missing job events. Store worker event identity/ordinal and current attempt/fence in validated payload/columns; add an indexed unique producer key where needed for deduplication.

In EventJournal, require an explicitly injected EventTransport in production composition rather than silently constructing a Redis client in the constructor. Keep `append` inside the caller's transaction. Require durable journaling enabled for the new execution protocol; do not claim replay durability when the feature flag turns storage off.

Keep sanitisation, but extend it to the actual AI secret names including application_key, key_id where sensitive, storage grants, service headers, signed URL query tokens and nested credential structures. Prefer typed allow-listed event payloads. Do not truncate a result into an apparently valid partial business payload; large results belong in their manifest/result store, with bounded metadata in the journal.

Use the existing event delivery mechanism for pending realtime events; make undelivered events discoverable after restart through published_at/due delivery state. For each additional notification destination, use or extend its durable delivery receipt with a unique event/destination key and bounded backoff. Do not create multiple independent copies of a terminal event. Do not mark email delivered merely because realtime publication succeeded.

Move commit control out of lower-level delivery helpers where a caller still has domain mutations pending. Publish only committed events. Make replay tenant/stream-authorised and paginate until caught up. Return snapshot/retention boundary when a cursor is too old. Pruning must not erase still-pending delivery or required approval/audit records.

**Done when:** T04/T10/T11/T35/T36 pass through existing journal/transport; Redis interruption cannot lose the authoritative result or create duplicate canonical mutations.

## B04. Cohesive data access — new `src/app/services/ai/repository.py`

**Action: add one repository for jobs/attempts, not one file per query. Package P02.**

Constructor receives the current transaction's AsyncSession. Instantiate repositories per transaction; long-lived workers hold a session factory, not one shared session. Keep SQL out of gRPC event handlers and HTTP transport clients.

Implement target operations for: create/find semantic job, fetch current attempt, claim eligible job/attempt, acquire execution, renew current lease, apply allowed progress, record result receipt, claim result application, schedule retry, cancel, query expired work and retrieve latest job by track/type. Each mutating operation accepts the expected attempt/fence/revision/state and uses conditional writes or row locks; return an explicit lost-ownership result.

Use aggregate counts for active execution by family/user instead of loading all active ORM rows. Use bounded keyset queries for queued jobs, stale attempts, result application and cleanup. Never keep a database transaction open over HTTP/gRPC inference, object download or email sending.

Reuse existing transaction helpers; do not add a generic retry wrapper that repeats arbitrary side effects. Retry only transaction-safe idempotent operations according to the backend's existing database policy.

**Done when:** T08–T13 and query-plan tests pass with large historical/queued datasets.

## B05. AI service construction — `services/ai/__init__.py`

**Action: remove service construction and empty inheritance from package imports. Package P04.**

Move the actual AIJobService definition into `service.py`. Remove `class AIJobService(AIJobServiceHandlers): pass` after its inherited responsibilities are placed in their real owners. Keep a stable export from `__init__.py` for callers, not an empty subclass.

Remove global `ai_service = AIService()`. Construct the pooled transport client once in backend lifespan/worker startup and inject it into consumers. Change `get_ai_job_service` into the existing dependency-composition path returning a correctly constructed service; it must not hide new network clients per request.

Move `submit_job_for_track` behaviour into an explicit AIJobService method or a retained thin compatibility function that delegates to it. Normalize the request once, preserve all optional controls and return actual current state for duplicate jobs. Remove the swallowed duplicate-job SSE exception: durable events/delivery handle it instead.

Keep only exports/constants and pure public normalization imports here. Do not construct network/model/Redis clients or service graphs at import.

**Done when:** T01/T20 and all caller import tests pass; no production no-argument transport singleton remains.

## B06. Job lifecycle — `services/ai/service.py`

**Action: make this the explicit business-job service, not a mixin chain. Packages P02, P04, P06.**

Define AIJobService here by consolidating current AIJobServiceCore lifecycle responsibilities. Constructor receives transaction-scoped repository/session, submission policy, source/owner resolver and EventJournal, with explicit service collaborators for result/approval when needed. Do not have it own long-lived transport streams or GPU execution.

Keep `create`, `submit`, cancellation/failure orchestration and user-facing job lookup here. Create/reuse durable jobs using semantic idempotency and pinned source revision. Create the dispatch intent in the same backend transaction; do not call AI while constructing the creator HTTP response. A deliberate rerun is a new job; ambiguous transport delivery reuses the current attempt.

Retain `start_publish`, `_start_publish_first`, `_start_ai_gated_publish` and `_start_publish_with_existing_pipeline` semantics through the current publication workflow. Preserve existing `generate_on_publish_only` configuration and already-published handling. Do not duplicate PublishJob or introduce another publish-mode flag. Keep `publish` jobs distinct from AI inference families; the scheduler must not send non-inference publish jobs to ExecuteAttempt.

Move raw SQL eligibility/attempt updates into the repository, execution delivery to AIJobDispatcher, result application to JobResultProcessor and notifications to committed-event delivery. Retain cohesive pure publish-policy helpers rather than creating a new class for each branch.

When failure occurs, update the current ProcessingJob and its error; preserve published source/status and preview candidates according to policy. Job-state changes should append durable events in the same transaction. Do not call catalog synchronisation or send email inside a still-open AI job transaction.

**Done when:** T02/T08/T13/T19/T23/T33 pass for publish-first, AI-gated, existing-pipeline and background enhancement requests.

## B07. Scheduler — `services/ai/scheduler.py`

**Action: refactor the existing scheduler into AIJobDispatcher. Packages P02, P03.**

Constructor dependencies: session factory/repository factory, existing task enqueue/outbox adapter, AI transport client, storage grant service, cached capability reader and immutable dispatch policy. Keep `round_robin_jobs` as a pure function and retain fairness tests. Do not add another queue framework.

Replace full active-job ORM loading with bounded/aggregate counts. Count actual execution/resource use separately from dispatch in flight and result application. `received`/`awaiting_approval` records must not block GPU capacity. Make per-family and per-user limits explicit; allow short reconstruction work without permanently starving long pipeline/Magic Clean jobs.

Retain the working advisory-lock protection until an equivalent tested claim/CAS strategy replaces it. Claim jobs and attempts in short transactions using current-state/skip-locked or the established equivalent. The first-1000 window must not permanently starve later users: preserve an eligibility cursor/fairness state or implement bounded per-user candidates with ageing. Queue position is an estimate, not a globally exact promise.

Atomically store claimed attempt/fence and durable dispatch intent. After commit, send execution work through the existing queue/worker mechanism. If enqueue fails, revert only the exact still-unaccepted dispatch claim using CAS; do not blindly set queued on an ORM object that another worker may already have started. Reconciliation repairs lost wake-ups from the DB.

Check actual bounded AI admission and actual grant validity. Cold/busy/unavailable capability defers the affected family without consuming inference-failure allowance. A request timeout is ambiguous, not proof of rejection. Query current attempt/manifest before replacing it. Keep backoff with jitter in the backend, not multiplied across GPU services.

**Done when:** T08/T09/T12–T14/T29 pass under multiple backend workers and sustained mixed queues.

## B08. HTTP client — `services/ai/client.py`

**Action: make AIService a transport adapter only. Packages P01, P03, P04.**

Constructor requires immutable endpoint/auth/protocol settings and an existing pooled httpx client. Remove module `_http_client` lazy global and `_get_http_client`. Lifespan/worker startup owns client close. Keep connection/keepalive limits and per-operation deadlines explicit.

Keep v1 submit_job during legacy drain. Centralize `_build_payload` normalization through typed schemas. Preserve backend/job identity validation and HTTP 409 semantics. Interpret acknowledgement status rather than always reporting processing. An identical terminal replay returns its existing result state; different semantics conflict.

Remove redundant per-key mutation helpers where typed discriminated job payloads replace them. Do not create a class per payload branch. Retain clear serializers when they encapsulate actual differences. Preserve false/zero/omitted controls and same_speaker across the HTTP path.

New ExecuteAttempt belongs to the gRPC adapter; this client must not become a second durable submission scheduler. Return structured transport/admission errors without deciding business retries. Never log credential bodies or signed audio URLs. Keep endpoint cleanup explicit, not broad string manipulation that masks bad configuration.

**Done when:** T03/T06/T14/T19/T20 pass and each transport has one owner/lifecycle.

## B09. Storage grants — `services/ai/storage.py`, `services/ai/b2_validator.py`

**Action: replace timestamp-only context building with actual scoped access. Packages P01, P03.**

Refactor AIStorageContext into a constructor-injected grant service in the same `storage.py` file. Dependencies: existing B2/provider adapter, immutable storage policy, credential source and clock. Keep a legacy build adapter only while v1 jobs need it; delete it after drain.

For the immediate repair, validate the effective issuer lifetime against AI's legacy reserve plus permitted wait/transit/skew margin. Actual provider key validity is authoritative. Never extend a JSON timestamp to simulate a new credential. Renew parked v1 Magic Clean jobs through their authenticated credential-only replay rules without changing destination or source semantics.

For new jobs, resolve the owner and immutable source first. Issue genuine restricted provider access just before an execution claim, with separate source-read and attempt-output permissions. Pin destination prefix to backend/job/attempt/execution. Use the provider adapter's supported restricted-key or signed-operation mechanism; implement and integration-test the selected mechanism before enabling v2, rather than treating configured master keys as temporary grants. Keep root credentials in backend only.

Record actual grant identifier/scope/expiry securely; refresh only the current authorised execution and never broaden scope. Make grant expiry/insufficient validity a typed blocked state and wake the existing dispatcher/reconciler after successful refresh. Define bounded cleanup access for expired candidate grants through backend-owned credentials, not leaked long-lived worker keys.

In B2KeyValidator, keep source owner and destination validation. Expose a typed authorised source/destination resolution instead of reconstructing keys independently in each handler. Retain `job_folder` legacy mapping for migrated records; new prefixes include attempt/execution identity. Prevent traversal, cross-owner object references and arbitrary remote URLs.

**Done when:** T03/T10/T17/T21/T41 pass with the actual provider, expiry and wrong-prefix tests. A mocked expiry object alone is not sufficient.

## B10. Submission policy and constants — `services/ai/submission_policy.py`, `services/ai/constants.py`

**Action: keep policy pure and state sets explicit. Packages P02, P04.**

Keep AISubmissionPolicy as the single place for whether an operation is permitted/required under current source/job/publication state. Inject policy configuration or pass a typed snapshot; do not issue network requests or mutate jobs while evaluating eligibility. Retain separate Magic Clean/reconstruct/pipeline admission and existing publish-only rules.

Define allowed state transitions once. Separate active delivery, active inference, result application, awaiting approval and final states; do not reuse one broad ACTIVE_STATUSES set for every capacity calculation. Keep TASK_MAP for existing worker entry points but map new execution profiles without duplicating functions for identical submission behaviour. Keep non-AI publish jobs out of model dispatch.

Add typed retry/error categories: invalid request/ownership, conflict, credentials, capacity, warming/unavailable model, transport uncertainty, worker loss, invalid audio, model execution failure and backend application failure. Do not encode policy by searching error-message strings across modules.

**Done when:** T02/T03/T12/T19/T23/T33 pass and no contradictory state-set definitions drive different schedulers.

## B11. Worker entry points — `core/worker/handlers/ai.py`, `core/worker/enqueue.py`

**Action: keep queue adapters thin; move actual work to the owners above. Packages P02–P04.**

Keep existing registered function names while queue producers migrate. Resolve injected lifespan/worker-context services once; call dispatcher, result processor, reconciler or notification dispatcher. Do not create an AIService/GrpcPipelineClient or database session singleton at module import.

Move `_submit_to_ai` delivery/state logic into AIJobDispatcher with short transactions. Move `_record_result_apply_failure` policy into JobResultProcessor/repository. Replace repeated `_apply_*_kwargs` conversion with the typed request serializer used by both transports. Keep the established result-application retry counters/backoff, but never trigger GPU resubmission for an application-only failure.

Do not hold an AsyncSession open for a long ExecuteAttempt stream. Read/claim and commit, release DB resources, perform network execution, then open short transactions for accepted progress/result. Map queued, accepted, started and terminal responses honestly. Sending SSE or opening a subscription is not proof that processing started.

Give durable queue tasks idempotent identities tied to job/attempt/fence and operation. Redis/task loss must leave a due DB action discoverable. Do not multiply ARQ Retry and application retry budgets without distinguishing delivery retries from inference attempts. Publish/cleanup/email queue names remain owned by their existing features.

**Done when:** T12–T15/T19/T35/T42 pass and handlers contain no copied lifecycle branches or raw canonical media updates.

## B12. gRPC transport — `src/app/grpc_client/client.py`

**Action: transport and stream lifecycle only; no independent business state machine. Packages P01, P03, P04, P07.**

Constructor dependencies: immutable target/TLS/auth settings, channel factory and injected event/result ingestion collaborator. Remove global backend service access and inline raw SQL from the client. Move `_store_result`, `_store_failure`, `_store_cancelled`, `_store_retrying`, `_sync_stage_progress` and identity validation into the repository/result/event owners; a temporary adapter can retain these method names while delegating.

Keep v1 Subscribe/GetResult recovery until legacy drain. Make `ready` distinguish a configured channel from a successful bounded health call; channel existence alone is not remote readiness. Make connection lifecycle idempotent and close/await all subscription tasks; remove completed entries without race-prone cancellation of a replacement task.

Add ExecuteAttempt(server-streaming) with strict new-protocol envelope and bounded concurrency. Authenticate peer/backend and validate event/result job/attempt/fence/source before handing it to durable ingestion. Do not permit missing identity under v2 just because legacy messages allowed it. Preserve explicit field presence, including zero progress and false controls.

Keep stream retries bounded. When the stream stops, wake the durable reconciler; do not let exhaustion remove the only knowledge of an unfinished job. Paginate active subscription recovery beyond the first 500 records. Prefer current DB job/attempt state over process-local `_subscriptions` as the work list.

Use TLS/mTLS or the verified protected network channel for untrusted links. Configure keepalive/deadlines against the actual server and ingress; do not assume a client setting forces server stream capacity. Capture transport errors without secrets or private payload dumps.

**Done when:** T04/T06/T10/T11/T14/T19/T35/T36/T42 pass.

## B13. Result application owner — `services/ai/job_result_processor.py`

**Action: expand the existing JobResultProcessor instead of adding another result-applier service. Packages P02, P04, P06.**

Constructor dependencies: session/repository factory, typed domain result handlers, AudioSourceMutationService factory, EventJournal factory, immutable result/approval policy and clock. Keep actual transactions scoped per operation, not on a concurrently reused instance.

Use one result receipt and application path for stream recovery, manifest reconciliation and direct report. Validate manifest schema, current attempt/fence/execution, expected AudioTrack.audio_revision, model/policy revision and artifact ownership before acceptance. Persist receipt before scheduling application; duplicate receipt cannot erase applied metadata.

Retain `received` and bounded result_apply_attempts/failures/backoff. Claim application with CAS, perform required external validation outside long transactions, then lock/recheck current state and apply. On application-only failure keep the immutable result and retry application; do not run the model again.

Keep `finish` as part of the mutation transaction. For approved workflows, set `awaiting_approval` and deadline once. For no-approval completion, preserve track publication rules and pipeline_completed_at semantics. Append the corresponding durable event in that same transaction.

Make `_preview_details` a pure formatter/reader. The existing mutation of status/deadline must move into explicit candidate creation, not remain callable during SSE emission. Make `_send_preview_ready`, `_send_track_complete` and `emit_terminal_event` delivery of committed event payloads only. Replayed notification cannot reset approval expiry, change a job's status or allocate a new preview ID.

**Done when:** T10–T12/T15/T30/T31/T33/T35/T42 pass, including a crash after domain commit before notification.

## B14. Domain result routing — `services/ai/handlers.py`

**Action: remove inherited service sprawl; keep explicit result handlers. Packages P04, P06.**

Move AIJobServiceHandlers public lifecycle methods into the real AIJobService where they are lifecycle operations. Place result-kind routing in JobResultProcessor with explicit collaborators. Do not retain an inheritance ladder whose subclasses exist only to assemble another class across files.

Use one result-kind dispatch table or a clear match statement for transcription, pipeline/categorization/discovery, Magic Clean and reconstruction results. Keep each domain action cohesive: transcription/tagging persistence delegates to tagging/media services; candidate staging delegates to media/approval; canonical replacement delegates only to AudioSourceMutationService.

Remove inline source URL updates, track-status resets, nested commits and direct notification sends from result handlers. Return typed mutation outcomes for the transaction owner to commit. Preserve moderation/flag records and current public response shapes.

When handlers.py no longer contains a meaningful unit, move its remaining small pure result conversions into callbacks.py and delete handlers.py. Do not leave an empty service shell to satisfy old imports; migrate the imports in the same package.

**Done when:** T02/T23/T30/T33 and constructor/call-path tests show one result application route, not multiple inherited implementations.

## B15. Result normalization — `services/ai/callbacks.py`

**Action: keep a pure compatibility/normalization module. Packages P03, P04.**

Retain `build_callback_result`, `coerce_result_for_track` and `coerce_transcription_block` where they perform actual required wire conversion. Rename the module only if needed later; its legacy name does not justify adding another callback delivery mechanism.

Normalize each legacy result variant into one typed internal result schema at entry. Preserve all authoritative IDs and distinguish transcript arrays, objects and strings according to actual old clients. Reject inconsistent identity instead of coercing it away. Limit payload size and validate numeric finiteness/optional presence.

Remove database, transport, storage and notification side effects. Do not keep multiple coercion copies in grpc_client, handlers, media and service. Do not convert any missing quality field into a passing measurement or assume result.success because an object exists.

**Done when:** legacy/new oneof fixtures and malformed-result/ownership tests pass; T02/T06/T10/T19.

## B16. Media and previews — `services/ai/media.py`

**Action: preserve domain media handling; centralise preview/approval here with existing job metadata. Packages P04, P06.**

Use the existing media/result responsibilities in this module rather than create an unrelated preview database. Constructor-inject the transaction repository/session, source mutation service, media/storage adapter, event journal and approval policy. Move durable preview CRUD imported from AI RegenerationService into the current pending_magic_clean/pending_reconstruct result metadata or the existing backend preview representation.

Stage a candidate with its immutable media/object identity, exact content digest, current source revision, producing attempt/fence, validation report and fixed approval expiry. Register the candidate MediaFile through the existing media lifecycle. Do not set it as canonical just because the worker returned a URL.

Implement explicit create/get/confirm/reject/expire candidate operations. Confirm verifies owner, nonexpired approval-ready state, expected source revision and candidate validation. Apply the exact rendered candidate through AudioSourceMutationService. Do not call AI ConfirmPreview to synthesize again. If the previous preview was only isolated segments, assemble from those exact segments first; user approval must bind to what was actually previewed under the documented contract.

Make repeated confirm idempotent; a current-source mismatch returns stale without overwriting media. Rejection/expiry schedules owned cleanup only after confirming no canonical/reference use. Do not extend expiry when a preview is fetched or its ready notification repeats.

**Done when:** T30–T33/T40 pass across backend and AI restart with no second preview authority left in AI SQL.

## B17. Audio joining — `services/ai/audio_joiner.py`

**Action: keep only backend-owned media assembly that is actually required. Packages P04–P06.**

Separate canonical/group/approved-segment assembly from AI speech synthesis. Remove duplicate reconstruction inference/splice algorithms if the new AI service already produces the complete verified candidate. Keep joining of distinct media/group outputs where the backend remains its product owner.

Inject storage/read references, AudioIO/FFmpeg runner and resource limits rather than construct clients per join. Stream inputs and output with bounded local disk/memory; do not download every source into one bytes list. Validate sample rate/channel/timebase and output duration. Never treat arbitrary URL input as ownership proof.

Use immutable job/attempt/candidate identities and backend-owned cleanup. Let the caller transaction register/apply the result through the source service. No private notification/retry scheduler belongs here. Keep pure segment-order/timing helpers as functions rather than another chain of wrapper classes.

**Done when:** bounded join fixtures and exact-approved-segment preservation pass; no duplicate TTS run occurs during confirmation.

## B18. Tagging — `services/ai/tagging.py`

**Action: retain backend taxonomy/transcript/tag persistence, not inference. Packages P04, P08.**

Inject the transaction session/repositories and immutable taxonomy policy. Apply validated transcription/category/tag/discovery proposals idempotently by current result/source revision. Keep audio-tag suggestions separate from automatic canonical tag application according to current product rules.

Remove duplicated text/JSON coercion now owned by callbacks.py. Do not invoke the GPU, mutate global keyword loaders or send notification directly from persistence helpers. Preserve maximum suggestion counts, label normalization and ownership of custom categories/tags.

Publish taxonomy/model-training proposals as committed events/data for backend review/versioning; do not let every inference write directly to all workers' taxonomy. Reindex only after an accepted canonical metadata change, with the existing catalog mechanism.

**Done when:** T02/T23/T24/T38 pass and repeated result application cannot duplicate tags/training examples.

## B19. Track projection — `services/ai/track_state.py`

**Action: preserve publication-aware TrackStateManager and separate projections. Packages P02, P06.**

Keep useful pure transition predicates such as existing publish/pipeline readiness checks. Do not turn this into another database or notification service. Make operation intent and current track state explicit to transition methods.

For already published tracks, AI queued/running/failed/awaiting-approval must not change published status or published_at. Project active/latest job status separately by track and job_type, using the existing API shape or explicit job fields. A failed Magic Clean job does not erase a successful pipeline state or a newer reconstruction result.

Keep true flag/archive/publish rules intact. Do not universally force ready on every job success; the caller's publication/approval policy decides. Source replacement invalidates derived projections according to existing revision policy, not incidental AI status.

**Done when:** T23/T31/T33/T39 pass for unpublished and published tracks with overlapping job families.

## B20. Cleanup — `services/ai/cleanup.py`

**Action: use this existing backend owner for remote candidate cleanup and legacy tombstones. Packages P03, P06, P08.**

Inject transaction repository/session, provider storage adapter and retention policy. Keep approval_deadline/set_approval_deadline semantics, but set deadlines only on the state transition, not every notification/read. Preserve staged folder cleanup/deferred cleanup integration.

Move AI Magic Clean cleanup tombstones and preview cleanup obligations here. Persist exact job/attempt/execution/object/version identity, reason and not-before time. A cleanup task must recheck current canonical/preview/manifest references and active ownership before deletion. Never delete a newer attempt's object using a shared job key or unverified prefix.

Use bounded paginated batches and separate cleanup retry budget. Record failures for reconciliation rather than swallowing them. Use backend-held least-privilege cleanup access when the worker grant expired. Do not enumerate the entire bucket or all job history to find one candidate.

Migration must import unresolved old tombstones and retain lookup mapping for legacy compatibility keys. Do not remove the AI cleanup tables before every pending obligation is acknowledged by this owner.

**Done when:** T16/T27/T30/T40/T41 pass, especially old cleanup racing a newer successful result.

## B21. AI notifications — `services/ai/notifications.py`

**Action: keep JobNotificationDispatcher; make delivery independent of result application. Packages P04, P07.**

Constructor dependencies: existing notification/provider queue adapter, durable delivery repository, templates/policy and clock. Receive committed job event IDs and bounded metadata, not mutable ORM objects owned by an active inference transaction.

Keep completed, flagged, failure and preview-ready notification semantics. Add idempotent delivery key per event/recipient/channel. Preserve failure_email_sent_at/flagged_email_sent_at compatibility where useful, but do not treat one timestamp as proof that every channel was delivered.

Send outage/recovery through SystemIncidentService, not one AI failure email for every waiting job. Distinguish temporary unavailable/credential/capacity waits from permanent failure. Keep email/provider failures out of GPU retry and track publication state.

Do not send secrets, signed media URLs or raw traceback/model prompts to recipients. Use authenticated stable links/IDs for previews and appropriately scoped admin details. Record permanent delivery failure separately without pretending the audio job failed.

**Done when:** T33–T35/T42 pass with provider timeout and duplicate terminal events.

## B22. SSE publishing — `services/ai/sse_publisher.py`

**Action: adapt committed events to realtime payloads; no domain mutation. Packages P04, P07.**

Retain current TrackEnqueued/TrackSubmitted/TrackProgress/TrackComplete/TrackFailed/TrackPreviewReady payload semantics where clients depend on them. Consolidate duplicate event formatting while preserving names and optional presence. Use the existing EventTransport via explicit injection rather than unrelated Redis globals.

Publish the durable event cursor/UUID so clients can deduplicate and replay through the backend journal. UI subscription/reconnect must retrieve a current snapshot plus authorised missed events; it must not depend on the same backend process or the AI pod's queue.

Remove status changes, preview deadline resets, source updates and callback-payload mutation from publishing. Keep zero progress valid. Coalesce high-frequency progress before durable ingestion under policy; never coalesce away terminal/approval/source events. Batch ID should come from durable job/event metadata, not only a Redis lookup.

**Done when:** T04/T11/T30/T35/T36 pass with Redis unavailable and multiple simultaneous consumers.

## B23. Durable recovery — new `services/ai/reconciler.py`

**Action: one backend watchdog over durable state. Packages P02, P03, P07.**

Target class: AIJobReconciler. Constructor dependencies: session/repository factory, transport client, manifest/storage reader, grant service, dispatcher and clock/policy. Do not add another message broker or GPU retry engine.

Run bounded keyset scans for expired dispatch claims, ambiguous/unacknowledged submissions, missed leases, pending manifests/results, received-but-unapplied results and pending cleanup/delivery. Process more than one page; store progress/eligibility correctly so early rows cannot starve later jobs.

Before creating a new attempt after an ambiguous failure, inspect the exact known output manifest, validate it and lock/recheck current attempt/fence. Reuse an acceptable current result. If a newer attempt already owns the job, reject the old result. Missing/partial manifest and an expired/lost worker permit retry only under the backend's failure/deadline policy.

Wake application/notification/cleanup retries without rerunning inference. Renew credentials for the current authorised execution or defer it safely. Reconnect old-protocol subscriptions during drain but never give the old AI recovery loop ownership of a new-protocol job.

Persist decisions and next_attempt_at/block reason before releasing a claim. Keep bounded per-cycle work and typed metrics; a local subscription dictionary is not the pending-job inventory.

**Done when:** T13–T19/T35/T36/T40–T42 pass, including whole-backend restart.

## B24. Canonical source owner — every `services/audio_source` file

**Action: preserve and integrate existing revision-checked mutation. Packages P04, P06.**

| File | Direct implementation changes |
|---|---|
| `service.py` | Keep AudioSourceMutationService.replace_source and its expected_audio_revision check. Make transaction ownership explicit so result/approval state, source mutation and journal append can commit atomically. Remove hidden nested commits from the domain operation after callers are migrated; the top-level use case owns commit. Persist post-commit waveform/speed/catalog/cleanup intents before delivery so a process crash cannot lose them. |
| `repository.py` | Keep lock_track and current revision validation; add only the missing canonical/idempotency queries. Use the same transaction as approval/result application and require current source revision. |
| `policies.py` | Retain preserve_publication and derived-asset invalidation/scheduling policies. Keep generation-on-publish rules in the current publish policy, not another flag. Do not schedule speed layers for unapproved previews. |
| `types.py` | Keep AudioSourceChangeReason/Policy/Result; include producing job/attempt/source identity where needed without duplicating AudioTrack.audio_revision. |
| `factory.py` | Compose concrete transaction-scoped collaborators once. Pass the real EventJournal, waveform scheduler, invalidator and cleanup/catalog collaborators; avoid creating unrelated hidden global clients. |
| `__init__.py` | Export the stable service/factory/types only. |

The current replace_source already preserves publication when policy allows and uses expected_audio_revision. Retain those safeguards; do not reimplement them separately inside AI handlers. Keep actual updated duration from the accepted candidate. Repeated approval must not increment audio_revision twice.

Do not wait for waveform generation during publish/source mutation. Schedule waveform and speed work independently through existing owners. A waveform event must never become a trigger for speed layers. Use an idempotent canonical-revision key for derivative work and stable 1x source reuse.

**Done when:** T30–T33/T39/T42 pass with failures between source commit and task delivery.

## B25. Health and incidents — `services/system_health/` and `services/system_incidents/`

**Action: reuse existing health state and incident deduplication. Packages P01, P07.**

| File/owner | Exact action |
|---|---|
| `system_health/probes/runpod.py` | Replace the unknown-only RunpodHealthProbe with an injected actual Hear-AI liveness/control/capability client and bounded deadline. The business capability is Hear-AI; do not equate a provider account API response with successful model readiness. |
| `system_health/types.py` | Extend HealthResult metadata for service epoch, capability, ready/cold/warming/busy/draining/unavailable state, last valid heartbeat and sanitised failure code. Map to existing high-level up/degraded/down/unknown without losing the precise reason. |
| `system_health/policies.py` | Centralise consecutive-failure/recovery thresholds, stale status, cooldown and severity. Cold/busy is not automatically down. |
| `system_health/repository.py` | Persist component/capability state and last seen epoch with proper transaction/uniqueness. Use bounded queries, not an all-job scan to determine service availability. |
| `system_health/service.py` | Keep SystemHealthService.record, inject repository/incident/journal dependencies where needed and make health transition plus incident/delivery intent recoverable atomically. Do not lose an outage/recovery notification between separate commits. |
| `system_incidents/service.py` and existing delivery owner | Reuse existing dedupe key/resolve behaviour. Open one incident per affected service/capability/environment, not per queued job. Deliver after commit, with retry/receipt; resolve after the configured recovery evidence. |

Keep the existing probe registration/scheduler and other providers unchanged except required interface additions. Check AI from backend even when no jobs run. Planned drain stops admission, reports intended downtime and allows bounded current work. Unplanned pod death is inferred from missing probes/heartbeats; do not wait for a notification from the dead process.

Expose a user-safe processing availability message and saved/waiting job state. Admin events include affected capability, epoch and safe error code, never credentials/private audio. Retry waiting jobs only under normal family/user capacity after recovery.

**Done when:** T05/T18/T29/T34–T37 pass with individual model failure and whole-pod failure.

## B26. Creator routes and API schemas — `api/v1/creator/tracks.py`, `schemas/ai_job.py`, existing preview/job routes

**Action: keep public APIs stable and delegate to the explicit services. Packages P02–P06.**

Keep the current Magic Clean route and existing reconstruction/edit/tagging/submission routes. Resolve creator/organisation/source permissions first, build the typed job request, call AIJobService and return its durable identity/current state. Do not require earlier AI-side track/transcript records. Do not wait for GPU completion in the creator request.

Preserve existing optional control semantics: all stem levels or defaults, zero values, cut_silence, same_speaker, changes, edited_transcript, source and media identity. Validate no-op/deletion/overlap deliberately rather than accepting contradictory payloads. Add request schema fields for source/timeline revision internally without making user-controlled IDs authoritative.

For job reads and UI updates, return latest job state by job_type plus canonical track state separately. Preserve source availability even when a background job fails. Approval routes call backend media/source owner; they must work while the AI service is offline after the candidate exists.

Add current attempt/result/blocked reason to compatible responses where needed. Keep deliberate rerun and idempotent duplicate semantics clear. Route model-training/catalogue-policy operations to backend ownership, not old AI persistence RPCs. Preserve deprecated routes only with explicit known clients and a removal gate.

**Done when:** T02/T03/T06/T21/T22/T30–T33 and existing client response fixtures pass.

## B27. Protocol, dependency composition and configuration

**Files:** `grpc_client/proto/pipeline.proto`, generated protobuf modules, `core/config.py`, `src/.env.production.example`, existing app lifespan/worker startup and migration directory. **Packages P00–P04, P08, P09.**

Generate both AI/backend descriptors from the same reviewed schema revision during build. Preserve existing fields/oneofs while migrating, add explicit optional presence for missing metrics/controls, and reserve removed identifiers. Never hand-edit generated stubs. Test server-streaming ExecuteAttempt with the locked Ray/gRPC client/server pair and actual ingress/network path.

Construct pooled HTTP/gRPC/provider clients once per owning process in existing lifespan/worker composition. Inject transaction-scoped service factories and close them explicitly. Do not share an AsyncSession across streams/tasks. Route all no-argument singleton imports to this composition.

Keep current backend DB/Redis config. Add only missing execution lease/deadline, family limits, health thresholds, grant, manifest, protocol and delivery settings. Correct the actual cross-service TTL defaults. Consolidate legacy AI_SERVICE/HEAR endpoint aliases into one documented precedence and reject inconsistent values rather than silently sending traffic to the wrong server.

Create additive migrations in the project's actual migration directory after checking current heads. Reuse audio_revision and StreamEvent. Backfill new job identity/version without changing old job semantics. Import required AI preview/lineage/training/cleanup records with an auditable mapping. Do not drop AI state or legacy fields while jobs/readers still depend on them. Keep rollback readers until the retention/rollback window closes.

**Done when:** clean database migration/rollback, legacy drain, descriptor-equivalence and lifespan tests pass; T01/T19/T20/T38/T40/T44.

## B28. Adjacent services to preserve, not rewrite

**Package P08 integration only.**

Preserve backend `services/resolver/`; remove calls to the AI resolver after verifying equivalent current backend resolver behaviour. Do not redesign Alexa recognition in this migration.

Preserve `services/speed_render/`, including its canonical revision/projection behaviour, and `services/waveform/service.py`. Use these existing owners after AI speed removal. Do not render speeds for previews, regenerate 1x unnecessarily, require waveform completion for publish or schedule speed work from waveform events. Audit the producing task/event keys and use canonical audio_revision for idempotency.

Preserve existing catalog indexing, settings, content flagging, publication, email and user/organisation permission services. Move only the named AI-owned data/operations into their appropriate existing owner. Do not refactor payment/authentication/other providers merely because they share BaseService or worker infrastructure.

**Done when:** T23/T33/T39 and existing unrelated integration tests pass without scope creep.

## B29. Final file disposition and release gate

| Existing AI-service file | Final disposition |
|---|---|
| `__init__.py` | Exports only; no empty service subclass or ai_service singleton. |
| `service.py` | Real constructor-injected AIJobService, durable request/lifecycle/publication coordination. |
| `scheduler.py` | Existing fair scheduler refactored into AIJobDispatcher. |
| `client.py` | Injected HTTP compatibility transport only. |
| `storage.py` | Real scoped grant owner, legacy build removed after drain. |
| `b2_validator.py` | Source/destination/owner validation. |
| `submission_policy.py` | Pure eligibility/publication policy. |
| `constants.py` | One set of job kinds/state groups/error mappings. |
| `handlers.py` | Explicit domain result routing during consolidation; delete once fully owned by JobResultProcessor/media/tagging. |
| `job_result_processor.py` | Sole result receipt/application coordinator; no notification-time mutation. |
| `callbacks.py` | Pure typed legacy/result normalization. |
| `media.py` | Candidate/preview/approval media lifecycle via canonical source service. |
| `audio_joiner.py` | Required bounded backend assembly only; no duplicate synthesis. |
| `tagging.py` | Idempotent validated metadata/tag persistence. |
| `track_state.py` | Publication-aware pure state projection rules. |
| `cleanup.py` | Remote candidate/legacy-tombstone retention/deletion owner. |
| `notifications.py` | Durable deduplicated notification delivery. |
| `sse_publisher.py` | Committed-event realtime adapter only. |

New repository and reconciler modules fill missing responsibilities; they must not duplicate a later-existing equivalent. All durable events use the existing StreamEvent/EventJournal. All canonical source mutations use AudioSourceMutationService. No second business queue is introduced in the AI pod.

Run T01–T44 across both repositories, existing backend tests and real Ray/GPU/B2 integration. Record actual results, query/memory/latency measurements and skipped tests. Release with additive readers first, then enable new execution for an explicit cohort, drain legacy ownership, remove obsolete AI persistence/callers and increase traffic only after failure drills. Do not simultaneously run old AI recovery and new backend retries for the same job.

The implementation report must name changed/deleted files, migrated constructors/callers, schema/protocol changes, actual test commands/results, retained legacy exceptions and rollback state. Do not substitute a general “cleaned services” summary for this evidence.

## Baseline source references

The actions are prescribed changes. Current owner/method observations are grounded in these sources:

- [AI service package exports](https://github.com/Techta-Labs-Ltd/hear-backend/blob/60387821fab069d4ea80cc0910ec09779b221cde/src/app/services/ai/__init__.py), [lifecycle/publish service](https://github.com/Techta-Labs-Ltd/hear-backend/blob/60387821fab069d4ea80cc0910ec09779b221cde/src/app/services/ai/service.py), [scheduler](https://github.com/Techta-Labs-Ltd/hear-backend/blob/60387821fab069d4ea80cc0910ec09779b221cde/src/app/services/ai/scheduler.py).
- [ProcessingJob model](https://github.com/Techta-Labs-Ltd/hear-backend/blob/60387821fab069d4ea80cc0910ec09779b221cde/src/app/models/processing.py), [current result processor](https://github.com/Techta-Labs-Ltd/hear-backend/blob/60387821fab069d4ea80cc0910ec09779b221cde/src/app/services/ai/job_result_processor.py), [worker adapter](https://github.com/Techta-Labs-Ltd/hear-backend/blob/60387821fab069d4ea80cc0910ec09779b221cde/src/app/core/worker/handlers/ai.py).
- [HTTP transport](https://github.com/Techta-Labs-Ltd/hear-backend/blob/60387821fab069d4ea80cc0910ec09779b221cde/src/app/services/ai/client.py), [storage context](https://github.com/Techta-Labs-Ltd/hear-backend/blob/60387821fab069d4ea80cc0910ec09779b221cde/src/app/services/ai/storage.py), [gRPC client](https://github.com/Techta-Labs-Ltd/hear-backend/blob/60387821fab069d4ea80cc0910ec09779b221cde/src/app/grpc_client/client.py).
- [Existing EventJournal](https://github.com/Techta-Labs-Ltd/hear-backend/blob/60387821fab069d4ea80cc0910ec09779b221cde/src/app/services/events/journal.py), [existing canonical source mutation](https://github.com/Techta-Labs-Ltd/hear-backend/blob/60387821fab069d4ea80cc0910ec09779b221cde/src/app/services/audio_source/service.py).
- [System health](https://github.com/Techta-Labs-Ltd/hear-backend/blob/60387821fab069d4ea80cc0910ec09779b221cde/src/app/services/system_health/service.py), [RunPod probe](https://github.com/Techta-Labs-Ltd/hear-backend/blob/60387821fab069d4ea80cc0910ec09779b221cde/src/app/services/system_health/probes/runpod.py).
- Inventory: [all AI-service files](https://github.com/Techta-Labs-Ltd/hear-backend/tree/60387821fab069d4ea80cc0910ec09779b221cde/src/app/services/ai), [audio-source files](https://github.com/Techta-Labs-Ltd/hear-backend/tree/60387821fab069d4ea80cc0910ec09779b221cde/src/app/services/audio_source).
