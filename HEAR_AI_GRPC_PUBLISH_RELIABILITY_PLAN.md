# HEAR: AI, gRPC, publishing and crash-recovery implementation plan

**Audit date:** 21 September 2026  
**Status:** AI-repository stabilization implemented on 21 September 2026. Backend deployment, schema migration, load-test and hard-kill acceptance work remains a separate cross-repository rollout.
**Backend snapshot:** `Techta-Labs-Ltd/hear-backend@a117a70113cc88046f1ac3f360d1cb365bb3dd42`
**AI snapshot:** `Techta-Labs-Ltd/hear-ai@53407fb81603bdd8b049f2ca57afdc5505a4b96b`

### Implementation note

This checkout is the AI repository; the backend paths named `B01`–`B13` are
not present here and cannot be safely changed from this workspace. The changes
made here cover the AI-side contract/runtime issues and deployment controls:

- `.env` is loaded by `pydantic-settings`, exported to Supervisor-managed Ray
  processes, and explicit `RAY_ADDRESS=auto` overrides remain intact.
- Job routing is declarative, `tagging` is an explicit categorization alias,
  and unknown persisted job types fail as unsupported instead of falling into
  another workflow.
- Interrupted jobs re-enter the queued path with a new fenced `run_id`; their
  track execution is also queued rather than being incorrectly terminalized.
- The AI protobuf already contains additive compressed-output and structured
  terminal-error fields; the generated bindings remain checked in with the
  source contract.
- Credential refresh remains destination-scoped, and
  `scripts/generate_service_key.py` generates a high-entropy plaintext key
  while installing only its SHA-256 digest in `BACKEND_REGISTRY_JSON`.

The backend attempt/inbox/outbox models, Redis result intake, publication
revalidation, catalog obligations and cross-repository mixed-version tests
still require the backend checkout and deployment environment described by the
remaining PRs below. They are deliberately not marked complete by changes in
this AI-only repository.

This is a static audit of the submission, transport, orchestration, result application, approval and publishing paths inspected in those snapshots. Code defects and recovery gaps are distinguished from performance hypotheses: CPU attribution still requires profiling the deployed processes. The checked repository configuration is not proof of the live Dokploy configuration or the host's capacity.

## 1. Decisions

Keep backend PostgreSQL authoritative for business jobs, execution attempts, approval decisions, source revisions and publication state. Use Redis/ARQ for bounded work delivery and live progress, not as the only copy of accepted work. Use Backblaze for durable immutable outputs and recovery manifests. Keep Cloudflare CDN and the existing speed-render architecture.

Preserve four independent core workflows: pipeline, standalone transcription, Magic Clean and reconstruction. Support every existing job name during migration; do not delete queued legacy work or silently change its requested output. Keep audio-tag suggestions as an explicit inference operation, rather than routing them into a full publication pipeline.

Keep the current HTTP submission transport initially. Fix the shared protocol before considering gRPC submission. In the target design, a Pod/Ray adapter and a future RunPod Serverless adapter use the same workflow inputs, attempt identity, outputs and result-application path. A provider selection is pinned per attempt.

Preserve the reconstruction patcher, audio chunking, channel handling, WAV conversion, same-speaker options, quality checks and lineage protections. Move orchestration and persistence boundaries; do not replace functioning audio algorithms merely to simplify the directory structure.

Increase the AI-result container's CPU allowance, but reduce uncontrolled concurrency. Resource increases are subject to a measured host budget. They are not a substitute for removing blocking calls, repeated conversions and per-progress SQL writes.

## 2. Current end-to-end flow

```text
Backend creates ProcessingJob in PostgreSQL
  -> fair dispatcher claims work
  -> backend ARQ worker calls HTTP POST /process
  -> AI saves AiJob in its own PostgreSQL database
  -> Ray orchestrator schedules models/audio work
  -> backend subscribes through gRPC Subscribe
  -> terminal handling normally calls gRPC GetResult again
  -> backend stores full result in ProcessingJob payloads
  -> backend enqueues arq:queue:ai-result
  -> result service applies transcript/media/metadata
  -> approval where required, or downstream PublishJob
  -> speed-render workflow
  -> publication state, catalog and notifications
```

AI supports `SubmitJob` over gRPC, but `AIService.submit_job()` in the backend currently uses HTTP `/process`. Do not describe this system as gRPC submission end-to-end. Sources: B01, B02, A02, A03, A04.

## 3. Audit findings and required corrections

### F01 — Protobuf contracts have diverged: release blocker

The two repositories define different `SubmitJobRequest` layouts under the same service/package. Examples: field 3 is backend `user_id` but AI `job_type`; field 4 is backend `job_type` but AI `max_tags`. `SubmitJobResponse` field 4 is backend `replayed` but AI `error`. HTTP submissions avoid that particular request decoder today; switching to gRPC before repairing the contract is unsafe.

AI `PipelinePayload` includes `compressed_audio=8`, `report=9`, and `flagged=10`; the backend proto and generated type stub inspected stop at field 7. The backend prefers typed `GetResult` over the streamed result, then converts that payload to a dictionary. Fields unknown to its descriptor are not available in that dictionary. This can prevent the result applicator from seeing the pipeline's compressed output and report.

`same_speaker` is implicit-presence on the backend's preview request and optional on AI. An explicit false value must survive transport, rather than being interpreted as absent and defaulted to true. Test all zero/false/empty-but-valid values.

The backend advertises `TrainCategorizer`, `IngestCategoryEvent` and `UpdatePlatformSettings`; the inspected AI service no longer declares those RPCs. Trace their callers, then either provide a versioned supported replacement or disable/remove the obsolete caller after migration. Do not recreate unwanted training systems just to satisfy stale stubs.

**Fix:** add missing compatible output fields to the backend first and regenerate all bindings. Introduce one shared, pinned contract and a new versioned service for the incompatible request layout. Preserve v1 while old clients drain. Add descriptor and binary round-trip compatibility tests in both repositories. Never hand-edit generated files. Sources: B01, B03, A01, A02.

### F02 — Unknown jobs fall into the Magic Clean handler

Backend `_get_handler()` returns `_process_magic_clean` for unrecognised types. AI accepts `discovery`, but the backend registry has no discovery entry. A discovery result can therefore enter the wrong approval logic.

**Fix:** explicit registry coverage, separate discovery compatibility applicator, and a typed unsupported-contract failure for unknown jobs. Do not mark unsupported work successful or give it a fabricated preview. Sources: B04, A04.

### F03 — Required transcription failures can be swallowed

Standalone and pipeline transcription loops roll back and log exceptions without re-raising. The surrounding result processor can continue to completion.

**Fix:** required output failures abort application and reach the existing retry mechanism. Missing required tracks or transcript outputs are explicit errors, unless the business job was intentionally cancelled/deleted. Optional pipeline stages use explicit partial outcomes rather than broad exception swallowing. Source: B04.

### F04 — Stale protection covers only part of pipeline application

The pipeline source-generation check guards media/transcription, but categorization, discovery, descriptions and moderation execute after the stale branch as well.

**Fix:** validate backend, logical job, attempt, run, source media, audio revision and requested operation before any output is applied. Revalidate after external preparation and before committing a source mutation. Quarantine stale results without changing current metadata or triggering publication. Sources: B04, B05.

### F05 — Progress reception still performs database work

The gRPC receiver converts result data before classifying events, can query job identity, publishes SSE, and selects/commits a ProcessingJob for progress events including heartbeats. The result is converted again in stage synchronization. Zero-valued progress is treated as missing in places.

**Fix:** cache subscription identity; classify first; update a Redis progress record; coalesce ordinary updates; persist bounded checkpoints. Handle `job_snapshot` and `stream_reset` explicitly. Do not discard actual transcript/output chunks under a progress-coalescing rule. Source: B01.

### F06 — Queueing is not necessarily SQL-free

The generic enqueue helper normally creates and commits a TaskLog after Redis enqueueing. Terminal receipt writes full result payloads before the queue. Existing buffering helps some callers but not the whole intake path.

**Fix:** introduce a pure-Redis result-intake enqueue method with explicit enqueue outcomes. Keep general TaskLog auditing elsewhere. Persist intake audit information through the result transaction or a bounded later batch. Sources: B01, B06.

### F07 — AI async transport still runs synchronous database calls

AI `Subscribe`, `GetResult` and `CancelJob` query synchronous SQLAlchemy sessions inside async methods. Result-to-Struct conversion also performs a JSON dump/load round trip. The orchestrator makes additional synchronous database and audio-probe calls.

**Fix:** while the AI database remains, place complete synchronous units of work in a bounded executor with sessions created and closed there, or migrate those units to an async repository. Do not share sessions or gRPC objects between threads. Extract CPU-heavy preparation into bounded processes where profiling supports it. Sources: A02, A03, A05.

### F08 — AI restart recovery fails running work

`recover_jobs()` requeues queued work, but changes running jobs to failed with `service_restarted`. Periodic recovery scans queued jobs. This does not resume an interrupted running workflow.

**Fix:** backend-owned attempts, renewable leases and stage manifests. An interrupted execution attempt is retryable within policy, not automatically a terminal failure of the logical user job. Resume from a validated checkpoint when supported; otherwise rerun only the interrupted stage or attempt against immutable inputs. Never promise continuation of GPU RAM after a killed process. Source: A05.

### F09 — Failure detail can be lost across stream/GetResult

The AI failure event contains a report with stage/attempt/retryability, while the inspected failure update does not persist that report as the canonical result. Typed result payloads do not share a uniform error report, and the backend normally prefers GetResult.

**Fix:** place structured terminal errors in a common versioned envelope and a durable terminal manifest for every job type. Keep error code, failed stage and retry classification independent of the output oneof. Source: A01, A02, A05, B01.

### F10 — Reconstruction approval can rebuild the preview

Backend `_apply_reconstructed_audio()` prefers joining original audio and segments when they are supplied. `_join_reconstructed_segments()` downloads audio into memory and invokes a synchronous joiner. Missing segment downloads can be skipped. Approval may therefore create a different artifact from the preview.

**Fix:** produce the complete patched preview before requesting approval, with a content digest and source revision. Approval promotes that exact artifact. Keep the patcher in AI reconstruction or a dedicated bounded media worker for legacy segment-only responses. A missing required segment must fail preview preparation, not produce a partial edit. Source: B05.

### F11 — Publishing has a revalidation gap

The speed finalizer checks source revision, commits speed projections, calls catalog synchronization, then obtains locks again without repeating all source/deletion/winner checks. A source update during that external interval must not allow the old candidate to finalize the new revision.

**Fix:** validate outputs outside long database locks, then recheck the attempt fence, winning attempt, source revision, source media, deletion and moderation/publication permission in the final transaction. Repeat these checks at every subsequent state transition after external work. Source: B08.

### F12 — Publication side effects can be left unfinished

The speed finalizer sets an attempt ready before later publication completion actions. Retrying a ready attempt returns immediately. Publication success also commits core records before listener notifications, SSE, batch counters and email enqueueing. A crash between these steps leaves obligations that cannot depend on the original stack resuming.

Catalog synchronization is called before the finalizer sets a newly published track's status. The catalog helper can delete non-indexable tracks. Publication indexing must be driven by the final authoritative publication revision rather than an earlier state.

**Fix:** commit publication state and durable outgoing obligations together. Give batch completion, catalog indexing, listener notification and email their own idempotent completion records. A retry of a ready render must reconcile pending finalization obligations, not rerender. Do not emit publication success when a flag/archive guard actually blocked publishing. Sources: B08, B09, B10.

### F13 — Magic Clean lineage queries grow with history

One lineage ownership helper loads all completed Magic Clean jobs with results, then filters scope in Python. That is a scaling candidate, not a measured CPU percentage.

**Fix:** index artifact ownership and output/source hashes; query bounded relevant lineage by backend/track/key/hash. Preserve root-source, engine-version and parameter checks when moving lineage out of the AI database. Source: A05.

### F14 — Credential refresh is not a uniform protocol

Submission fingerprints normally include storage expiration. Special semantic replay/refresh handling exists for Magic Clean, not uniformly for every job type.

**Fix:** immutable business-request fingerprint excludes rotatable credentials. Add an authenticated credential-refresh operation scoped to the existing attempt and unchanged storage destination. Refresh signed input access and output upload credentials without silently changing content or making a new logical job. Source: A04.

## 4. Every job and RPC has an explicit destination

| Existing job/operation | Target route | Required application behavior |
|---|---|---|
| `pipeline` | Pipeline workflow | Apply requested transcript/metadata/moderation and validated compressed media. Publishing is a separate backend-controlled workflow. |
| `transcription` | Standalone transcription | Persist transcript/timing data for the correct source revision. No implicit enhancement, tagging or publication. |
| `magic_clean` | Magic Clean workflow | Preserve controls, lineage and quality checks; produce a complete immutable preview; await approval before source replacement. |
| `reconstruct` | Reconstruction workflow, segment-edit operation | Preserve patcher and same-speaker behavior; create full preview and output manifest; promote exact approved output. |
| `edit_transcript` | Reconstruction workflow, transcript-edit operation | Preserve intended text-edit semantics and timing alignment. Do not substitute transcription-only processing. |
| `rebuild` | Legacy pipeline compatibility operation initially | Preserve its existing contract until callers are classified. It currently uses the AI pipeline route, not the same route as reconstruction. Do not blindly rename it. |
| `categorization` | Explicit metadata operation / selected pipeline stages | Apply only requested metadata. Do not regenerate audio or publish just because it shares workflow infrastructure. |
| Backend `tagging` | Explicit compatibility alias | Normalize to the supported categorization operation before submission. AI's allowed-job list does not contain `tagging`. |
| `discovery` | Explicit discovery compatibility operation | Apply discovery/description with source fencing. No Magic Clean fallback and no accidental approval. |
| `audio_tag` | Dedicated short inference operation | Preserve the short-upload-to-two-suggestions contract. Do not replace the main transcript or start a full publishing pipeline. |
| `CreatePreview`, `RemoveSegment` | Durable reconstruction operation | Long-running work receives a backend job/attempt ID and recoverable output; not only a transient unary response. |
| `ConfirmPreview`, `RollbackPreview` | Idempotent approval/rejection commands | Persist the decision; approval promotes exactly the preview digest; rejection schedules safe candidate cleanup. |
| `Subscribe`, `GetResult`, `GetPreview` | Observation/recovery | Read-only, scoped identity, versioned payloads; resume/reconcile after disconnect. |
| `CancelJob` | Durable cancellation intent | Stop future stage admission, signal execution, fence late results and preserve already-published source media. |
| `Moderate`, `Categorize`, `ListDiscovery`, `Health`, `GetQueueStats` | Bounded query/inference APIs | Timeouts, concurrency limits and no accidental publication side effects. |
| Removed training/settings RPC callers | Capability-checked migration | Move required behavior to an explicitly supported API or retire unused calls; never silently ignore unimplemented responses. |

Sources for existing behavior: A01–A05, B03–B05. Target mappings are proposals, not changes already made.

Create one declarative registry describing canonical workflow, legacy aliases, operation, input schema, result schema, required outputs, approval policy, source-mutation rights and retry class. Use it for validation, dispatch, result application and parameterized tests.

## 5. Target data and transport design

### Durable records

Extend `ProcessingJob` rather than creating another business-job owner. Add an execution-attempt model containing backend/job identity, attempt number, execution provider and provider execution ID, run ID, lease/fencing token, input media ID/revision/checksum, workflow/contract/engine versions, request fingerprint, heartbeat and lease expiry, terminal manifest key/hash, retry classification, and application timestamps.

Add a small result inbox with a unique terminal identity, for example `(backend_id, job_id, attempt_id, terminal_sequence)`. Duplicate identities with a different digest are conflicts to quarantine. Store large result data in immutable B2 manifests, not repeatedly in callback/result/metadata columns.

Reuse the existing EventJournal/StreamEvent, CatalogIndexOutbox and notification-outbox infrastructure. Add missing dedupe keys and explicit staging APIs that participate in the caller's transaction. Do not call an existing helper that commits independently and assume the operation became atomic.

### Manifest-before-notification

Before reporting successful execution, the executor uploads all required artifacts, validates them and writes a terminal manifest last under the backend-assigned attempt prefix. The manifest identifies schema, workflow, operation, source revision, output keys, sizes, media types and checksums; failure and cancellation use the same envelope with structured errors. No credentials are stored in public manifests or SSE payloads.

The backend can reconstruct lost Redis work by examining its unfinished attempts and their known manifest keys. Recovery must not depend on scanning the whole B2 bucket or listing every recording. Keep manifests while application, approval, publication or replay obligations remain unresolved.

### Receive, prepare and apply separately

```text
AI terminal signal
  -> lightweight authenticated intake
  -> compact Redis/ARQ job containing attempt and manifest reference
  -> bounded result worker downloads/validates/normalizes once
  -> short SQL transaction rechecks identity, attempt and source
  -> result inbox + business changes + outgoing obligations commit
  -> independent reliable publishers execute outgoing obligations
```

For the initial compatibility period, retain the current database-backed legacy result receipt. Enable reference-only intake only for attempts whose durable manifest protocol is supported. Queueing only a job ID before durable output exists is not an acceptable migration shortcut.

Redis enqueue acceptance is not the same as confirmed application. ARQ job uniqueness is not sufficient for business idempotency. A worker killed after a commit may execute again; the inbox, source fence and already-applied state must make that replay harmless.

### Progress

Use backend/job/attempt-scoped Redis records with monotonically increasing progress sequence and liveness timestamps. Proposed starting settings: ordinary UI progress at most once per second per attempt; database checkpoint at most once per ten seconds for an active dirty record; stage transitions emitted promptly. Heartbeats do not perform a SQL update or UI broadcast individually. Terminal events and meaningful output chunks are never dropped as disposable progress.

Support snapshots, stream resets, stale sequences and final-state precedence. An old progress update cannot replace completion or the progress of a newer attempt. Pipeline full transcripts are delivered through result references, not repeatedly with every stage event.

### Subscription ownership

Move subscriptions out of generic backend job tasks into a dedicated intake service. Persist a lease per subscription shard/attempt, acquired with an atomic conditional update and renewed by its owner. An expired owner loses authority through a fencing token. Only one active owner handles an attempt; a standby may recover it. Bound waiting subscriptions and reconcile pages continuously instead of only resuming the first N at startup.

## 6. Publishing and approval

Retain three explicit publication modes: AI-gated, publish-first/background-enrichment, and reuse-current-pipeline. Persist the selected policy/configuration version on the publication intent. Preserve the existing meaning during migration; do not casually invert `generate_on_publish_only` based on its name.

```text
AI-gated:
  publish intent -> current-revision AI result applied -> moderation decision
  -> current-revision speed assets ready -> publication transaction
  -> catalog projection acknowledged -> completion notifications

Publish-first:
  publish intent -> current source + speed assets -> publication transaction
  -> catalog/notification completion
  -> independent enrichment for the same pinned source revision

Reuse:
  matching pipeline/config/source revision -> reuse verified outputs
  -> missing speed work only -> normal publication finalization
```

A pipeline timestamp alone should not be the long-term reuse identity. Store the source generation and relevant configuration/engine versions it certifies. Speed layers must match the current source generation and include the original 1.0 plus the four required derived speeds, not merely a nonempty list.

The publication transaction updates authoritative track/job state and records outgoing obligations. PostgreSQL, Redis and Meilisearch are not a single atomic transaction: track the catalog state separately and show publication as indexing/finalizing until the required revision is acknowledged. Synchronize the published document after the authoritative publication transition. Catalog or email outages retry those steps without rerunning AI or audio rendering.

Every publication entry point—creator publish, admin/bulk status changes, scheduled publish, live-publication track additions and retry—must converge on the same publication orchestration service. Preserve the distinction between a publication container's lifecycle and each constituent track. Make group/batch aggregation idempotent per track and publication attempt.

An already-published track keeps its live source while replacement processing or approval is pending. Ordinary AI/transport failure must not unpublish it. Explicit moderation decisions follow a separate documented policy and audit trail, not a generic failure handler.

Approving edited audio records the preview digest, expected source revision, decision actor and decision time. Prepare the complete media before approval. Serialize approval/rejection/expiry races with a conditional state transition. Promotion and cleanup obligations commit together. Repeating approval returns the previous successful decision rather than rendering again.

Cloudflare remains the primary speed renderer; retain the existing local emergency fallback cap of two active workers/jobs. Do not introduce RunPod as the speed-render fallback.

## 7. Crash and update behavior

| Failure boundary | Required behavior |
|---|---|
| Submission accepted but reply lost | Query/retry the same logical request and pinned attempt idempotently. Do not create a second user job. |
| AI process dies during work | Lease expires; reconcile provider and manifest; resume a validated checkpoint or create a fenced retry attempt. Old execution cannot commit current results. |
| Artifact uploaded, event not delivered | Backend reconciler finds the terminal manifest at the known key and restores intake work. |
| Redis restarts or loses recent queue writes | Rebuild dispatch/result delivery from backend durable records and B2 manifests. Preserve AOF/noeviction but do not treat them as zero-loss guarantees. |
| Backend dies before result commit | Transaction rolls back; retry applies the unchanged manifest. |
| Backend dies after result commit | Duplicate intake detects the committed receipt. Outboxes resume remaining actions; no duplicate transcript/source replacement. |
| Crash after render ready, before publication completion | Retry pending publication/outbox phases without rerendering ready audio. |
| Source changes while validation/indexing is underway | Final transaction rejects the old revision/fence and schedules cleanup only for unreferenced candidate artifacts. |
| Cancel overlaps late completion | Backend cancellation state and attempt fence determine whether the late result is retained only for audit/cleanup. |
| Storage credentials expire | Pause/retry credential acquisition for the same scoped attempt; do not alter semantic request identity. |
| Rolling deployment mixes versions | Old and new readers coexist under explicit contract support; queued legacy payloads remain readable; no destructive schema cutover. |
| Approval expiry overlaps a crash | Persisted decision/expiry state resolves the outcome; cleanup cannot delete a successfully promoted source. |

Graceful shutdown stops new claims, makes readiness false, closes/drains streams and records recoverable work. Long audio processing cannot rely on shutdown grace alone. Hard-kill recovery is the acceptance criterion. Health checks must measure worker/event-loop health and lease renewal, not merely whether Redis responds.

## 8. Container CPU and concurrency plan

The inspected production file sets `worker-ai-result` to one replica, 2 CPU, 2G and 12 concurrent jobs. `worker-backend` has 4 CPU, 6G and 50 concurrent jobs; its detached gRPC subscriptions have a separate limit. Source: B13.

| Component | Proposed first deployment | Subsequent condition |
|---|---|---|
| AI-result container | Increase 2 -> 4 CPU; 2G -> 4G; reduce max_jobs 12 -> 2 | Raise concurrency only after realistic load/crash tests. If Python preparation saturates one core, use bounded subprocesses or split the same four-CPU budget across two processes. |
| Backend dispatch container | Keep 4 CPU / 6G initially; bound active submissions and detach intake ownership | Increase to 6 CPU only if measured CPU throttling remains and the host has capacity; avoid paying for hot-path defects with blanket extra quota. |
| Dedicated gRPC intake | New 1 CPU / 1G service, one active subscription owner initially | Add standby/shards only with distributed ownership/fencing implemented. |
| AI Ray gateway/orchestrator | Budget 1 logical CPU each instead of 0.5, after Pod capacity review | Ray logical scheduling resources do not enforce container CPU isolation. Keep GPU/native-thread budgets separate. |
| Local speed fallback | Preserve the two-active-worker/job limit | Do not increase fallback parallelism as a side effect of the gRPC refactor. |

Existing settings for the stabilization deployment:

```dotenv
AI_RESULT_WORKER_MAX_JOBS=2
AI_MAX_GLOBAL_SUBMISSIONS=10
```

These are proposed initial values, not measured optimal capacity. Keep receive capacity high enough for already-running attempts during draining. Do not assume lowering ARQ submission concurrency limits detached subscriptions.

New settings to implement explicitly, not paste into an old build expecting behavior:

```dotenv
AI_PROGRESS_PUBLISH_INTERVAL_MS=1000
AI_PROGRESS_CHECKPOINT_INTERVAL_SECONDS=10
AI_RESULT_INTAKE_MAX_INFLIGHT=4
AI_RESULT_REFERENCE_MAX_BYTES=65536
AI_EXECUTION_LEASE_SECONDS=120
AI_EXECUTION_HEARTBEAT_SECONDS=20
AI_RECONCILE_INTERVAL_SECONDS=30
```

Lease and heartbeat values are starting configuration proposals. They need validation against model stalls, network delay, provider cancellation and update drains. A lease expiry fences an execution; it does not prove the old GPU computation stopped.

Before deployment, measure host cores/RAM, total service reservations, peak working sets, database connections and rolling-update overlap. Leave a proposed 25% operational headroom where feasible. A CPU cap increase cannot create physical capacity. Confirm effective container limits after rollout rather than trusting YAML alone.

Update Compose sources and the swarm-generation workflow together; regenerate stacks using `script/generate_swarm_stacks.py`. Never edit only a generated stack. Do not automatically give every API worker, result worker and FFmpeg subprocess the full host's thread count.

## 9. Implementation work packages

### PR1 — Immediate correctness and stabilization

**Backend:** `src/app/grpc_client/proto/pipeline.proto`, generated bindings, `services/ai/handlers.py`, `core/config.py`, `core/worker/settings.py`, production deployment configuration.

Add missing compatible result fields; cover unknown handlers explicitly; propagate required transcription failures; reject stale metadata application. Add contract regression fixtures before other changes. Increase result CPU/memory and lower its concurrency through the actual deployment environment. Do not switch HTTP submission to gRPC.

**Exit:** all legacy result types have a tested handler, compressed output reaches the applicator, and required failure does not yield successful completion.

### PR2 — Shared contract and compatibility

**Both:** existing proto sources, generated bindings, transport mappers, dependency pins and contract tests. Add a shared pinned contract package or generated artifact workflow with one source of truth.

Introduce the versioned job/attempt/result envelope, structured errors, optional-value semantics, capability discovery and manifest references. Establish v1/v2 coexistence; prevent incompatible field renumbering. Audit removed RPC callers.

**Exit:** binary round-trip and mixed-version tests cover all existing payload kinds, zero/false values, failed/cancelled results and unknown compatible fields.

### PR3 — Backend durable attempts and transactional obligations

**Backend existing:** `models/processing.py`, `models/publish_job.py`, `services/ai/service.py`, `services/ai/scheduler.py`, `core/worker/enqueue.py`, `services/events/journal.py`, catalog/notification outbox services.  
**Proposed additions:** `models/ai_job_attempt.py`, `models/ai_result_inbox.py`, `services/ai/attempt_repository.py`, `services/ai/reconciler.py` and additive migrations.

Use constructor-injected repositories, storage and transports. Establish atomic attempt claims, leases, source fencing, result receipts, and obligation staging without hidden commits. Preserve fair scheduling and count unapplied received results in capacity. Use narrow columns/counts for scheduler and recovery scans rather than loading complete result JSON. Paginate recovery and index due-status/lease/revision lookups.

**Exit:** accepted business work remains reconstructible with Redis empty; duplicates and stale attempts cannot mutate current state.

### PR4 — Durable AI outputs and execution separation

**AI existing:** `hear/orchestrator.py`, `hear/services/jobs/submission.py`, `hear/services/transport/grpc.py`, `hear/deployments/gateway.py`, storage/model-client integration.  
**Proposed additions:** `hear/services/jobs/executor.py`, workflow registry, `result_manifest.py`, `checkpoint_store.py` and provider-neutral execution context.

Extract the four workflows without changing DSP/model semantics. Write output/checkpoint manifests and preserve pipeline configuration, source hash and engine/patcher version. Move blocking units out of async serving. Make recovery a backend-controlled execution policy. Replace unbounded historical lineage scans with indexed scoped artifact lineage.

Keep the AI database during compatibility migration. Export/import the lineage and active attempt information that currently lives there; drain legacy active jobs. Remove AI application-database authority only after backend records/manifests support the whole active lifecycle and replay tests pass.

**Exit:** kill an AI process during each core workflow; the logical job remains visible and recoverable with a valid retry or checkpoint path.

### PR5 — Lightweight gRPC intake and progress

**Backend existing:** `grpc_client/client.py`, `services/ai/sse_publisher.py`, `core/worker/handlers/ai.py`, enqueue/worker configuration.  
**Proposed additions:** `services/ai/result_ingress.py`, `progress_buffer.py`, `subscription_supervisor.py`, `result_preparer.py` and a dedicated intake worker entry point.

Queue compact references without SQL TaskLog writes. Split progress from outputs, support snapshot/reset/reconnect, add jitter and distributed ownership. Fetch large results outside the receive loop. Consolidate terminal gRPC, compatibility callbacks and future provider completion inputs through the same validated inbox/application path.

Enable reference-only intake per capability only after PR4 supplies durable manifests. Keep legacy inline receipt available while old jobs drain.

**Exit:** heartbeat reception performs no per-heartbeat SQL; CPU-heavy result application does not block intake; lost stream events reconcile from durable state.

### PR6 — Exact preview approval and revision-safe publishing

**Backend:** `services/ai/media.py`, `services/ai/job_result_processor.py`, `services/ai/track_state.py`, `core/worker/handlers/publish.py`, `core/worker/handlers/speed_layers.py`, `services/speed_render/finalizer.py`, `services/catalog_index_service.py`, publication/content/scheduling entry points.

Promote the approved artifact without rejoining it. For old segment-only results, prepare a complete preview in an isolated media worker first. Unify publication entry points and persist explicit mode/revision. Recheck source and winner after external work. Record catalog, batch, listener and email obligations transactionally. Make ready-attempt replays finish incomplete publication phases.

Keep the existing source-mutation service, speed-render validation, publish reconciliation and publication-preservation guards; strengthen their boundaries rather than deleting them.

**Exit:** source changes during indexing never publish an old candidate; repeated approval preserves digest; failure at each publish boundary eventually converges without duplicate render or notifications.

### PR7 — Runtime adapters, update procedure and controlled rollout

Implement Pod/Ray and future Serverless adapters against the same execution/result contract. Serverless completion may use provider callbacks or status retrieval rather than assuming a persistent per-job gRPC stream. Keep provider choice on the attempt.

Use additive schema migrations and dual readers. Deploy compatible consumers before enabling new producers. Canary by backend/job capability and job type. Drain legacy queues/active approvals before removing old message readers or AI persistence. Do not clear Redis, reset attempts or delete job artifacts during updates. Preserve a documented rollback window and versioned configuration.

**Exit:** rolling upgrade/rollback tests pass with queued, executing, applying, awaiting-approval and publishing work.

## 10. Test and observability gates

Parameterize the same lifecycle tests across all nine current AI job names, backend `tagging`, and the four canonical workflows. Cover success, validation failure, execution failure, retry exhaustion, cancellation, duplicate delivery, expired credentials, stale revision and unsupported contract. Test preview controls and all existing RPC categories separately.

Inject hard process termination before/after acceptance, upload, manifest publication, Redis enqueue, result commit, approval commit, speed finalization, catalog acknowledgement, batch completion and notification staging. Redis unavailable, empty restored queues, PostgreSQL unavailable, B2 timeout and Meilisearch outage are separate cases. Do not settle for testing graceful shutdown only.

Required assertions:

- 100 duplicate deliveries produce one applied business result and no repeated source mutation.
- A required transcription exception cannot produce a completed result.
- A rejected/stale attempt cannot update audio, transcript, tags, descriptions or moderation on the current revision.
- The approved media digest equals the previewed digest.
- Missing reconstruction segments fail preparation rather than creating partial audio.
- A ready speed attempt still recovers unprocessed publication obligations.
- Batch counts are derived idempotently from unique per-track outcomes.
- A catalog outage does not trigger AI/audio rerendering.
- Already-published content remains live on ordinary background-AI failure.
- No active/pending-approval/referenced artifact is removed by cleanup.
- Reconnect and duplicate ownership do not multiply effective subscription work.

Measure per-container CPU and throttling, event-loop lag, resident memory, result bytes, conversion time, DB statements/transaction duration, active intake/application counts, oldest queue age, stale/duplicate/conflict counts, lease expiry, recovery lag, outbox age and actual catalog revision. Log identifiers and classified errors, never full transcripts or credentials by default.

Load-test recorded production-like transcripts and audio durations at observed peak and a proposed 2x peak burst, not tiny synthetic payloads alone. Increase worker concurrency only when sustained drain rate improves without unacceptable CPU, memory, DB latency or intake lag. No percentage improvement is promised before that measurement.

## Source registry

Repository entries identify inspected code locations at the pinned snapshots above, not unverified live deployment state.

| ID | Repository | Inspected source |
|---|---|---|
| B01 | hear-backend | `src/app/grpc_client/client.py` — event handling, GetResult recovery, receipt and subscription |
| B02 | hear-backend | `src/app/services/ai/client.py` — HTTP submission |
| B03 | hear-backend | `src/app/grpc_client/proto/pipeline.proto`; `generated/pipeline_pb2.pyi` — protocol and generated payload shape |
| B04 | hear-backend | `src/app/services/ai/handlers.py` — registry, required results, stale handling |
| B05 | hear-backend | `src/app/services/ai/media.py` — source fences, approval and reconstruction joining |
| B06 | hear-backend | `src/app/core/worker/enqueue.py` — ARQ queues and TaskLog commits |
| B07 | hear-backend | `src/app/core/worker/handlers/publish.py` — durable publish jobs, watchdogs and reconciliation |
| B08 | hear-backend | `src/app/services/speed_render/finalizer.py` — output validation and publication ordering |
| B09 | hear-backend | `src/app/core/worker/handlers/speed_layers.py` — publication side effects and batch outcomes |
| B10 | hear-backend | `src/app/services/catalog_index_service.py` — synchronization and catalog outbox |
| B11 | hear-backend | `src/app/services/events/journal.py` — existing event journal/replay |
| B12 | hear-backend | `src/app/services/ai/service.py`; `scheduler.py`; `job_result_processor.py`; `track_state.py` — lifecycle and capacity |
| B13 | hear-backend | `docker-compose.prod.yml` — checked worker limits, not verified live limits |
| A01 | hear-ai | `hear/proto/pipeline.proto` — service and output contract |
| A02 | hear-ai | `hear/services/transport/grpc.py` — auth, sync DB in async RPC and typed output mapping |
| A03 | hear-ai | `hear/deployments/gateway.py` — Ray ingress and submission transports |
| A04 | hear-ai | `hear/services/jobs/submission.py` — allowed jobs, durable submission and credential/replay policy |
| A05 | hear-ai | `hear/orchestrator.py` — job routing, event snapshots, failure/restart behavior, pipeline and lineage |

Platform references used to check the design: Protocol Buffers proto3 language guide (field-number compatibility, unknown fields and presence); ARQ documentation (pessimistic execution and replay); Redis persistence and Pub/Sub documentation; Docker resource constraints and Compose deploy specification; gRPC Python AsyncIO documentation; Ray resource scheduling documentation. Proposed values and target behaviors in this plan are engineering recommendations, not claims that these libraries automatically provide the complete application guarantee.
