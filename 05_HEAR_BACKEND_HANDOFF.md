# Hear backend required changes — 20 September 2026

No backend files were edited. Paths below refer to the backend repository, whose actual current bodies and migration heads must be inspected before implementation. The master authority remains `00_IMPLEMENTATION_ORDER.md` and `02_HEAR_BACKEND_FILE_BY_FILE.md`.

## 1. Compatibility changes to consume now

### Credential lifetime and HTTP/gRPC errors

Files: `services/ai/storage.py`, `services/ai/client.py`, `core/worker/handlers/ai.py`, both settings modules.

The AI still enforces its configured legacy admission reserve. Its default is 86400 seconds. Issue real scoped access whose remaining provider lifetime exceeds that reserve plus permitted pre-start wait, network delay and clock-skew margin. Do not set the issuer lifetime equal to the admission reserve, reduce validation to bypass the reserve, or relabel an unchanged key with a later JSON timestamp.

Insufficient remaining lifetime now maps to HTTP 422 with `detail="storage_credentials_expiring"`, or gRPC FAILED_PRECONDITION with that same detail. Treat only this typed condition as queued/waiting for a real credential refresh. Keep malformed requests, wrong owner, changed destination and unrelated 422 failures permanent. Preserve job identity and semantics during credential-only replay. A rejected admission is not proof of a durable new AI job; the backend retains its own saved request.

### Subscription recovery

File: `grpc_client/client.py`, then existing journal/SSE owners.

Each subscriber now has its own bounded live queue. A slow subscriber can receive `stream_reset` with `error="subscriber_overflow_reconnect_required"`, followed by a current snapshot and stream termination. On this event, reconcile with GetResult and reconnect the same authenticated job/run. Do not create a new inference job.

`job_snapshot` carries current nonterminal status; persisted terminal snapshots retain the existing job_completed/job_failed/job_cancelled names. Apply monotonic state rules: a stale snapshot cannot regress completed, cancelled or awaiting-approval state. This v1 mechanism provides state recovery, not a durable replay of every intermediate event. Complete journal replay still belongs in backend EventJournal.

### Schema synchronization

Copy `hear/proto/pipeline.proto` into the backend's authoritative corresponding proto path and regenerate Python, type stubs and gRPC bindings using the agreed toolchain. Do not copy only generated Python or manually patch descriptors.

`PipelineEvent` retains field numbers 10, 11 and 12 but makes progress_pct, elapsed_seconds and estimated_remaining optional. Use HasField to distinguish absent values from measured zero; do not write `value or old_value` for progress.

`HealthReply` adds:

| Number | Field | Meaning |
|---|---|---|
| 7 | control_ready | Gateway/control path readiness |
| 8 | service_epoch | Instance identity of the health reader, not an execution claim |
| 9 | capabilities | Struct keyed by model deployment name, with state and running replica count |
| 10 | models_loaded | Deployments observed ready |
| 11 | gpu_metrics_available | False when worker GPU measurements are not available |
| 12 | optional error | Sanitized probe/control failure code |

Existing GPU fields are not authoritative when gpu_metrics_available is false. A CPU gateway does not report its own CUDA visibility as model health. The current capability reader reports ready, cold, warming, unavailable and disabled; it does not yet measure busy capacity. `/ready` follows control readiness. An optional-model failure degrades capability health without claiming the entire control plane is unavailable.

This reader is not startup fault isolation: enabled models are still bound into the legacy application graph. A failed enabled constructor can still prevent application deployment. Independent application/execution profiles remain part of the unfinished migration.

### Controls and quality

Preserve same_speaker=false through both submission transports; omitted retains the existing true default. Preview assessment failure is now `passed=false`, `status="unavailable"`, `error="quality_assessment_failed"`. Never turn missing/unavailable quality into passed=true. AI confirmation now rejects expired and nonpassing previews.

Legacy AI confirmation still synthesizes and uses AI persistence. Do not call that implementation the new exact-candidate approval flow. Replace its callers through the migration below before removing it.

Resolver and training RPCs were removed under the earlier explicit user instruction. Remove their AI client calls. Keep backend resolver/product functionality. Training restoration requires an explicit scope decision; do not silently recreate AI training endpoints.

## 2. Backend ownership implementation, in dependency order

| Files | Required implementation and exit proof |
|---|---|
| `models/processing.py`, actual migration directory | Extend ProcessingJob; add ProcessingJobAttempt with unique job/fence and current execution claim. Reuse existing result counters and AudioTrack.audio_revision. Pin semantic request/source/policy/model identities without storing renewable secrets in result metadata. Prove migration/backfill/rollback and concurrent claims. |
| `services/ai/repository.py` | Transaction-scoped queries/CAS for admission, dispatch, execution claim, renewal, cancellation, progress, result receipt/application and expired work. Never hold a shared session during network inference. Prove old fence/runner/source cannot write. |
| `models/stream_event.py`, `services/events/journal.py`, `transport.py` | Extend existing producer-event dedupe and pending delivery; append event and state in one transaction. Preserve global cursor semantics and paginate replay. Redact nested grants and signed URLs. Prove committed results survive Redis/SSE failure. |
| `services/ai/scheduler.py` | Refactor existing scheduler to AIJobDispatcher. Persist attempt/fence plus dispatch intent atomically. Count inference separately from received/approval-ready state. Preserve per-user/family fairness and bounded queries. Reconcile lost acknowledgements before new attempts. |
| `services/ai/__init__.py`, `service.py`, `submission_policy.py`, `constants.py` | Real injected AIJobService; no empty inheritance or global transport singleton. Keep publish-only policy and ProcessingJob authority. Define state/error sets once. Saved creator requests must not wait for GPU completion. |
| `services/ai/client.py`, `grpc_client/client.py`, worker handlers/enqueue | Inject process-owned pooled clients; close streams/tasks explicitly. Keep v1 while it drains. Add bounded server-streaming ExecuteAttempt only once the jointly reviewed protocol is implemented on AI. Preserve acceptance uncertainty and explicit false/zero/absent values. |
| `services/ai/storage.py`, `b2_validator.py` | Issue actual owner-scoped source-read/output grants just in time. Prefix output by owner/job/attempt/execution. Refresh only the same current identity/scope. Keep root provider keys backend-only. Test against real provider expiry and wrong-prefix access. |
| `api/v1/internal/ai_execution.py` | Thin authenticated claim, heartbeat, grants/refresh, events and result endpoints under the exact attempt URL. Delegate to repository/grant/result owners. Enforce identity/fence/source and bounded payload/timeouts. Verify manifests outside long transactions, then lock/recheck before commit. |
| `services/ai/job_result_processor.py`, `handlers.py`, `callbacks.py` | One receipt/application path for stream, control report and manifest reconciliation. Preserve received and application-only retry. No GPU resubmission for notification/application failure. Keep coercion pure and current-result/source checks authoritative. |
| `services/ai/media.py`, `cleanup.py` | Durable immutable candidates with digest, source revision, attempt/fence, validation and fixed expiry. Approve the exact candidate without synthesis. Import old preview/tombstone obligations before AI deletion. Cleanup rechecks exact object ownership and all live references. |
| All six `services/audio_source/` files | Reuse revision-checked AudioSourceMutationService. Commit source/approval/journal together; durable derivative intents follow commit. Double confirm increments revision once; stale candidates cannot overwrite current source. |
| `services/ai/tagging.py`, `track_state.py`, `audio_joiner.py` | Idempotent accepted metadata, publication-preserving projections and only genuinely required bounded media assembly. Keep pipeline, reconstruction, Magic Clean and publish identities separate. |
| `services/ai/notifications.py`, `sse_publisher.py` | Deliver committed event IDs with recipient/channel idempotency receipts. No domain mutations or expiry resets while notifying. Provider failure must not change source/job success or rerun inference. |
| `services/ai/reconciler.py` | Paginate due/expired/ambiguous attempts, inspect the known manifest before replacement, and schedule application/delivery/cleanup separately. Do not stop at the first 500 jobs or use process-local subscriptions as durable inventory. |
| System-health/probe/policy/repository and system-incidents services | Real bounded AI probe, capability-aware hysteresis, epoch tracking, one deduplicated outage/recovery incident. Cold/disabled/busy are not identical to service down. Probe even when no jobs run. |
| Creator/job/preview routes, API schemas, app/worker lifespan and configuration | Compose injected services; preserve public shapes and permission checks; expose latest processing state separately from published track state. Add protocol/lease/grant limits without duplicate endpoint aliases or session singletons. |
| Backend resolver, speed renderer, waveform, catalogue and publication services | Preserve existing owners. Remove AI resolver/speed dependencies after verified migration. Waveform completion must not trigger speeds and publication must not wait for waveform generation. |

## 3. Joint cutover contract — not yet shipped

The current AI code does not implement ExecuteAttempt, JobExecutor, BackendClient or result-manifest ingestion. Do not enable the new backend route against it merely because the additive health protocol is available.

Before cutover, jointly implement strict protocol_version, backend/job/track/user identity, attempt_id/fence, execution_id/worker_epoch, semantic_request_hash, source_asset/revision/digest, timeline/reference identities, policy/model/taxonomy revisions and lease/deadline fields. Backend execution claim must commit before AI downloads, inference or output writes. Identical replay must not start another writer. Different live runner must conflict.

The executor must upload immutable candidate objects and checksums before result.json, omit credentials/local paths/signed URLs, and report that known manifest idempotently. Backend locks/rechecks current attempt and source before accepting/applying. Cancel, old fence, supersession and stale source win over late progress/results. A lost stream after output must recover the manifest rather than rerun inference.

Pin each job to exactly one recovery owner. Keep v1 SQL/readers until every old job, candidate, lineage record and cleanup obligation has an explicit migrated disposition. Only then remove AI ORM/queue/recovery/preview/catalogue code. Do not run backend retries and AI recovery for the same new-protocol job.

## 4. Acceptance and release checklist

Run T01–T44 from the master document across both repositories. In particular, require real concurrent dispatcher/claim tests; delayed old-fence events; zero/absent progress; expired grants; stream loss after manifest; backend loss mid-inference; double approval and stale source; published-track failure; notification interruption; pagination over 500 jobs; and kill/restart tests at download, model, encode, upload, apply and notify.

Use real scoped B2 credentials and provisioned GPU model artifacts for release tests. Record memory, cold/warm latency, audio quality, queue fairness, and all skipped checks. Roll back routing for new jobs without undoing durable fences or losing pending candidates. Unit tests alone do not authorize production rollout.
