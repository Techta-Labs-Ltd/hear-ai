# Hear: implementation order and shared contract

Version: 2.0 — 20 September 2026  
Status: implementation instructions; no repository or production changes applied by this document.  
AI baseline: `Techta-Labs-Ltd/hear-ai@a89ee0ff9b231351034d612b7c4dfa6f8e42b091`  
Backend baseline: `Techta-Labs-Ltd/hear-backend@60387821fab069d4ea80cc0910ec09779b221cde`

## 1. Use these files, not the earlier architectural sketches

Implement [the Hear-AI file map](01_HEAR_AI_FILE_BY_FILE.md) and [the backend file map](02_HEAR_BACKEND_FILE_BY_FILE.md) in the order below. This document is the authority for cross-service identity, transaction ownership, migration and acceptance. The two file maps specify changes inside each repository. The earlier audit remains evidence, not an alternative implementation design.

The maps cover every identified Hear-AI package and every identified file in its `services` subpackages, plus all 18 files in the backend `services/ai` directory and related integration boundaries. Detailed source was read for the critical paths; other entries are explicitly directed refactors against an inspected directory inventory, not claims that every implementation line has been audited. Inspect the current body before editing an inventory-only entry. New paths and methods are marked as new or target names. Reconcile any commits after these baselines; do not overwrite unrelated work.

## 2. Fixed decisions

| Decision | Required implementation |
|---|---|
| Business job ownership | Keep the existing backend `ProcessingJob` authoritative and reuse existing `StreamEvent`/`EventJournal` for durable events. Do not create a second business-job store in Hear-AI. |
| Attempt ownership | Backend creates the attempt, increments its fence, grants execution, renews its lease and decides retries. |
| AI execution | Ray Serve owns model replicas and bounded execution. An inference worker executes one authorised attempt, not its own durable business retry loop. |
| Code organisation | Reuse existing service files. Add one common `JobExecutor` and one extracted `PipelineService`. Refactor Magic Clean and reconstruction in their existing `service.py` files. Do not add a second set of workflow wrappers. |
| Model access | Inject an async `RayModelClient` or a narrow collaborator through constructors. Resolve deployment names only at composition boundaries. |
| Audio transport | Acquire an immutable source once per execution workspace. Send bounded windows or short reference clips to model actors, not whole recordings. |
| Results | Upload attempt-specific immutable candidates, then a result manifest. Backend accepts and applies the current result idempotently. |
| Approval | Approve the exact candidate that was previewed. Confirmation must not run synthesis again. |
| Publication | Preserve published state during background processing and failure. Use the existing canonical audio-source mutation service and existing publish policy. |
| Notifications | Commit durable events before delivery. Reuse backend SSE, WebSocket, AI notification, health and incident services. |
| Removed AI responsibilities | Remove AI SQL jobs, preview persistence, catalogue mutation, business retry ownership, resolver and speed generation after their caller/data gates pass. |
| Retained backend responsibilities | Keep backend PostgreSQL, relevant Redis uses, resolver, speed rendering, waveform rendering, source ownership and publication controls. |

## 3. Code cleanliness rules

Write direct implementation, not explanatory comment blocks. Do not add commented-out implementations, migration stories inside functions, TODO/FIXME placeholders, empty exception handlers, debug prints, unexplained catch-all fallbacks or duplicate `v2` service trees. Keep design explanations here and in tests. Preserve required copyright notices, generated-file headers, shebangs and narrowly justified tooling directives.

Use constructor injection for services that own clients, model handles, configuration, policies or external resources. Keep pure transformations as functions. Do not create a class merely to rename one function. Do not use a global service locator or a generic dependency bag accessible throughout the domain. Do not share a mutable ORM session between concurrent jobs.

The composition root may construct concrete collaborators. Domain constructors only assign dependencies and validate local configuration. They must not run migrations, fetch networks, start unowned loops, inspect every model or download weights. Model-replica startup may load its already provisioned model. Manage tasks, channels and executors with explicit start/close lifecycle, not `__del__` as the main correctness mechanism.

Refactor one behaviour boundary at a time. Preserve public contract and golden audio tests while moving code. Delete the old implementation when the new owner is active; a temporary adapter must have a named removal gate. Do not simultaneously alter DSP thresholds, dependency versions and service ownership in the same change.

## 4. One execution path

Backend user API → existing AIJobService → backend durable job/dispatch intent → existing scheduler refactored into AIJobDispatcher → bounded ExecuteAttempt stream → Ray ExecutionDeployment → JobExecutor → selected existing service → injected Ray model deployments → immutable candidate and manifest → backend JobResultProcessor → approval/source service → committed events and notification delivery.

The creator HTTP request remains asynchronous and returns a durable job identifier. The backend execution worker, not the creator request, keeps the server-streaming `ExecuteAttempt` call outstanding. The Ray execution call remains outstanding for actual work. Do not return an acceptance response and start an unbounded, detached task per recording.

Use a server-streaming execution contract; do not require a newer bidirectional-streaming feature just to perform this migration. Validate the selected API against the Ray and gRPC versions actually locked in the image. Keep v1 `/process`, `SubmitJob`, `Subscribe` and `GetResult` compatibility only for explicitly identified legacy jobs until they drain. V2 durable results live in the backend; an AI process-local registry is never their authority.

## 5. Identity and replay contract

| Field | Authority and rule |
|---|---|
| `protocol_version` | Backend pins a job to legacy or new execution. Never let both protocols recover the same job. |
| `backend_id`, `job_id`, `track_id`, `user_id` | Backend-authorised identity. Authenticate the backend before using any supplied identity. |
| `semantic_request_hash` | Hash immutable source identity/revision, job type, controls, timeline/edit data and destination identity. Exclude renewable secrets, expiry and progress. |
| `attempt_id`, `fence` | Backend creates a new attempt and strictly increasing fence atomically. Do not reuse an expired execution as a fresh attempt. |
| `execution_id`, `worker_epoch` | The actual runner acquires a backend execution claim before side effects. A duplicate RPC must not start a second writer for the same attempt. |
| `source_asset_id`, `source_revision`, source digest | Identify the authorised input, not whichever URL the track happens to have when a retry begins. |
| `speaker_reference_asset`, `timeline_basis` | Reconstruction explicitly separates reference voice from splice source and the coordinate system used by edits. |
| `model_revision`, `policy_revision`, `taxonomy_revision` | Pin the processing inputs needed for reproducibility. A retry does not silently switch semantic versions. |
| `lease_expires_at`, `deadline_at` | Backend decides lease/deadline. Worker uses conservative local monotonic elapsed-time checks and stops before unauthorised work. |
| `event_id`, worker event ordinal | Identify producer reports. Backend uses the existing persisted StreamEvent ID as the authoritative replay cursor after validation; gaps between global IDs across different streams are normal. |

Acquire the execution claim with a compare-and-set transaction over current attempt/fence, nonterminal job, valid lease and unclaimed runner. Return the existing claim state for an identical replay. A different live runner receives a conflict/defer response, not permission to infer again. After lost ownership, only the backend may create a replacement attempt.

Validate backend, job, attempt, fence, execution and source identity on progress, lease renewals, grant refresh, result reports and cancellation acknowledgements. Once a newer fence exists, an old worker cannot update the current job. Cancellation is durable in the backend before signalling a running request. A queued or progress message cannot resurrect a completed, cancelled or approval-ready result.

## 6. Control operations and result commit

Add one authenticated internal backend router at the proposed `src/app/api/v1/internal/ai_execution.py`. Route all operations to the same AI job/repository owners; do not implement separate business logic in endpoints.

| Operation | Required behaviour |
|---|---|
| Acquire execution | Atomically claim the current attempt and return its execution identity, lease, approved source references and grants. |
| Heartbeat/renew | Verify current ownership; update liveness and return continue/cancel/drain plus any renewed lease. Report progress independently of high-frequency heartbeat. |
| Refresh storage grant | Renew actual permitted access for the same attempt and destinations. Never change audio semantics or extend scope. |
| Record progress/result | Validate producer event identity, deduplicate, persist state/event in one transaction and acknowledge only after commit. |

This control API replaces pod-local recovery ownership. It is not an additional scheduler or an invitation to deliver unsafely authenticated arbitrary callbacks. Use bounded transport retries for control delivery only. The returned stream terminal message and the control report both converge on the same idempotent backend result ingestion method.

Write output beneath the authorised owner prefix, followed by job, attempt and execution identity. Never allow two executions to share a mutable scratch key. Write candidate audio and ancillary results first, verify them, and write `result.json` last. Store object identities, hashes, durations, validation, model/policy revisions and typed payload in that manifest. Exclude credentials, temporary signed URLs and local filesystem paths.

The backend checks the known manifest key before replacing an ambiguous attempt. Accept a late complete result only when that attempt is still current, not cancelled, not superseded and policy permits acceptance after lease expiry. Reconciliation must lock/compare the same current-attempt row used by the dispatcher. A newer attempt wins once its fence is committed. Do not claim physical exactly-once execution; require idempotent, current-result application.

## 7. State mapping

Reuse current backend status values, including `dispatching` and `awaiting_approval`; do not replace them with incompatible names everywhere.

| Durable state | Meaning | Capacity rule |
|---|---|---|
| `queued` | Saved, awaiting eligibility/capacity/credentials/healthy capability | No GPU budget consumed. Store a separate blocked reason. |
| `dispatching` | Short-lived durable backend claim | Count dispatch budget; timeout is reconciled, not blindly reset. |
| `submitted` | Sent or acknowledged, not yet proven started | Preserve acceptance uncertainty and reconcile the same attempt. |
| `processing` | Current worker actually started | Count active execution/model-family budget. |
| `received` | Durable result received, application pending | Release inference capacity; use separate result-application budget. |
| `awaiting_approval` | Durable candidate is ready for a user decision | No GPU or dispatch slot held. Do not extend approval expiry on every notification. |
| `completed`, `failed`, `cancelled` | Final processing/business outcome under existing contract | Terminal writes are monotonic. |

Keep job processing status separate from `AudioTrack.status`. Respect `pipeline.generate_on_publish_only`. Do not run pipeline unexpectedly when that policy defers it until publication. Keep waveform and speed work behind their existing canonical-revision policy; waveform completion must not trigger speed generation. A publish request must not wait for waveform work.

## 8. Downtime rules

| Failure | Required response |
|---|---|
| AI unreachable before definite acceptance | Persist waiting state, reconcile same attempt and retry delivery with bounded backoff. Do not classify as bad audio. |
| Model cold or capacity full | Defer that family, do not consume inference failure attempts or report the whole service down. |
| Missing weights or failed model constructor | Mark only that capability unavailable, open one deduplicated incident and keep unrelated jobs usable. |
| Worker dies during processing | Expire its lease, inspect its manifest, retry only when no acceptable result exists. |
| Backend unavailable | Stop new execution claims. Existing work follows its lease; stop safely before expiry and retain only authorised attempt artifacts. Never buffer unlimited events. |
| Control report/stream lost after output | Recover the existing manifest; do not rerun the model merely to repeat notification. |
| Result application or notification provider fails | Retry backend application/delivery only, never GPU inference. |
| Source changes before result approval | Mark result stale; do not overwrite the newer source. |

Expose process liveness, control readiness, model capability states and attempt liveness separately. Probe from the backend; an abruptly dead pod cannot send a shutdown notification. Store worker epoch and capability changes through existing health/incident services. Use hysteresis and notification dedupe; do not email once per waiting job.

## 9. Ordered work packages

Do not mark a package complete merely because a test file exists. Record actual command, environment and result. Test names below are required behaviours, not claims of existing or passing tests.

| Package | Do this | Required tests and exit gate |
|---|---|---|
| P00 Baseline | Pin both source snapshots, lockfile, image, GPU and model artifacts. Capture representative public requests/results and baseline audio. Inventory imported symbols before moving them. | T01 import/contract inventory; T02 baseline job-type parity. No unrelated upgrades. |
| P01 Immediate correctness | Repair TTL contract on both sides; fix live event overflow/fan-out; make health truthful; honour `same_speaker=false`; stop quality errors being labelled passed. | T03–T07. Keep legacy DB/recovery operational. |
| P02 Backend ownership | Extend ProcessingJob and add current attempts/journal using existing storage. Refactor scheduler, result processor and worker adapters. Add current-attempt/source CAS and short transactions. | T08–T13. Two dispatchers cannot own the same attempt; zero/backward events cannot corrupt final state. |
| P03 Protocol and transport | Add ExecuteAttempt and internal control endpoints. Generate both client/server stubs in build. Add strict versioned identity, actual grants and manifest ingestion/reconciliation. | T14–T19. Duplicate/ambiguous delivery is reconciled; old and new jobs retain separate authorities. |
| P04 AI dependency cleanup | Add common executor, extract PipelineService, refactor existing Magic Clean/reconstruction services; inject clients and immutable policy/taxonomy. Remove hidden global access from migrated paths. | T20–T24. Services construct without network/DB; standalone jobs work without prior pipeline. |
| P05 Bounded audio/model boundaries | Replace whole-file bytes with disk-backed decoding and bounded windows. Move model work to window methods, keep responsive lifecycle/cancellation and measured GPU concurrency. | T25–T29. Long-file memory/disk bounded, correct timestamps and no leaked workspaces/native jobs. |
| P06 Preview/source ownership | Move AI preview records to existing backend job metadata/approval. Approve exact artifact with revision check; route all canonical changes through audio_source service. | T30–T33. Double approval idempotent; no resynthesis; published tracks survive failure. |
| P07 Health and delivery | Implement real capability probe, outbox-backed job/incident notifications, journal replay, startup reconciliation and planned drain. | T34–T37. Recovery after both services restart and after Redis/SSE/email interruption. |
| P08 Remove misplaced persistence/features | Migrate previews, lineage, catalogue/keyword/training data and cleanup ownership; move resolver/speed callers; drain legacy jobs. Remove AI SQL/Redis/application retry files only after dependency scan. | T38–T40. No remaining live AI SQL consumers; training and discovery still work; Alexa/resolver/speed behaviour preserved. |
| P09 Production release | Test real GPU/audio/storage stack, mixed-family fairness, full pod loss and rolling replacement. Canary new attempts; preserve rollback readers and source revisions. | T41–T44. Record measured performance and every skipped test. No production-ready claim on HTTP 200 alone. |

Some work overlaps, but deletion never precedes its ownership replacement. P04 can run with compatibility adapters until P05 changes model transport. Each adapter must be removed or explicitly retained for v1 only before P08 closes.

## 9a. Exact immediate patch set before architectural deletion

Apply these fixes to the existing path first, with tests; do not wait for the full migration and do not bypass validation.

| Existing location | Direct edit |
|---|---|
| AI `services/jobs/submission.py`: `_magic_clean_storage_ttl_is_safe`, `_validate_magic_clean_storage_ttl` | Compare actual remaining validity against the declared admission reserve. Serialize insufficient validity as the stable `storage_credentials_expiring` condition. Catch/map it before generic permanent ValueError handling. Keep wrong-owner, malformed and changed-destination requests rejected. |
| Backend `services/ai/storage.py`: `AIStorageContext.build`, and both settings modules | Require actual issued lifetime to exceed admission reserve plus maximum permitted pre-start wait, transport/skew margin. Do not issue exactly the minimum required remaining TTL. Fail contradictory configured policies before admission; generate grants just in time in the new protocol. |
| Backend `core/worker/handlers/ai.py`: `_submit_to_ai`, HTTP error classification | Treat the specific credential-validity condition as a durable queued/grant-refresh state. Keep other invalid 422 requests permanent; do not make every validation failure retryable. Do not mark accepted-but-queued or terminal replay as processing. |
| AI `orchestrator.py`: `_push_event`, `subscribe` | Give each subscription its own bounded queue while v1 drains. Register before taking its current-state snapshot, send to every subscriber, make overflow explicitly trigger snapshot/reconnect, and retain terminal DB recovery. Unregister in each subscriber's finally path; a dropped put cannot remove the only terminal notification route. |
| AI `orchestrator.py`: `_process_edit_transcript` | Read the normalized stored `same_speaker` value and forward it instead of hardcoding true. Preserve omitted-as-true compatibility, explicit false and both transports. |
| AI `services/reconstruction/service.py`: quality exception and confirm path | Remove exception-as-passed quality. Move approval to exact candidate application through backend; do not acknowledge confirmation then resynthesize. Keep existing clients under the versioned adapter until candidate migration is ready. |
| AI `services/transport/operations.py`: `health`; backend `system_health/probes/runpod.py` | Replace unconditional healthy/unknown results with actual bounded control/capability checks. Do not use the CPU gateway's CUDA visibility as the model worker's health. |

## 10. Mandatory acceptance catalogue

| ID | Required assertion |
|---|---|
| T01 | Import settings/contracts/gateway on a CPU test process without accessing model directories, opening a database or creating hardcoded logs. |
| T02 | Existing job types and typed result variants remain compatible; `rebuild` is not accidentally changed into synthesis. |
| T03 | Default issuer/validator pair accepts normal transit delay; true expiry, wrong owner and destination changes still reject. |
| T04 | Two subscribers each receive ordered terminal state; overflow/reconnect recovers journal data. |
| T05 | CPU gateway liveness does not report worker GPUs; cold, failed, busy and ready models produce distinct capability status. |
| T06 | Explicit `same_speaker=false` survives both REST/gRPC and direct/edit-transcript paths. |
| T07 | Quality-assessor exceptions never return `passed=true`; intentional removed stems and valid silence do not fail unrelated retention gates. |
| T08 | Two backend dispatchers claim each current attempt only once under load. |
| T09 | Duplicate ExecuteAttempt cannot acquire a second execution/writer for the same attempt. |
| T10 | Late result/progress/cancellation from an old fence cannot overwrite the current attempt or canonical source. |
| T11 | Progress zero and optional absent values are distinguished; delayed retry cannot resurrect terminal/approval-ready state. |
| T12 | Received and awaiting-approval jobs do not occupy GPU admission capacity. |
| T13 | Crash between DB claim and queue enqueue is repaired from durable backend state without duplicate execution. |
| T14 | Lost submission acknowledgement reconciles same identity before a new attempt is created. |
| T15 | Result manifest uploaded before disconnect is recovered without another model invocation. |
| T16 | Partial upload or incomplete manifest cannot become a visible candidate. |
| T17 | Grant refresh returns real scope/expiry and cannot widen source/destination or alter controls. |
| T18 | Backend outage stops new claims and running work at safe lease boundaries; no unlimited RAM event buffer. |
| T19 | Legacy and new protocol coexist with exactly one authoritative recovery owner per job. |
| T20 | Constructor injection tests replace each external collaborator without monkeypatching global singletons. |
| T21 | Magic Clean executes for an authorised source with no AI track/transcript history and with LLM/Fish unavailable. |
| T22 | Direct reconstruction works without Magic Clean/tagging/discovery; edit-transcript requests ASR only when required. |
| T23 | Pipeline retains transcription, moderation/flagging, permitted tagging/categorisation/discovery and delivery semantics. |
| T24 | Policy/taxonomy state is backend-scoped and immutable for a running job; updates do not leak across tenants. |
| T25 | One-minute, one-hour and longest-supported recordings have bounded host/GPU/object-store memory and disk usage. |
| T26 | Chunk boundaries preserve word offsets, channels, silence maps and continuity without duplicate/missing speech. |
| T27 | Native inference/encoder cancellation finishes or terminates before workspace cleanup or slot release. |
| T28 | OOM is classified as capacity/model failure, not silence; poisoned model replica is replaced through supported Ray lifecycle. |
| T29 | Mixed long pipeline, short reconstruction and Magic Clean traffic is bounded and fair under real model residency limits. |
| T30 | Candidate remains available across browser, backend and AI restarts; approval reuses its exact verified content. |
| T31 | Double confirm changes canonical revision once; stale source/expired candidate cannot apply. |
| T32 | Cumulative reconstruction edits use explicit timeline basis and preserve untouched spans within declared codec tolerance. |
| T33 | Published track remains published on any AI failure; only latest job status shows the failure. |
| T34 | Failed probes open one incident; recovery resolves it once; cold/busy states do not trigger outage spam. |
| T35 | Notification/SSE failure after DB commit is retried from durable journal/outbox; no result rollback or re-inference. |
| T36 | More than one page of active jobs is reconciled after restart; no first-500-only recovery. |
| T37 | Planned drain rejects new execution, bounds existing work and preserves results after replacement. |
| T38 | Migrated worker startup and dependency scan contain no live AI SQL/Redis job/preview/training persistence requirement. |
| T39 | Backend resolver and existing speed URLs remain functional after AI-side removal; 1x source reuse and waveform policy remain intact. |
| T40 | Every migrated legacy pending result/preview/cleanup record has an explicit owner and retention/rollback disposition. |
| T41 | Kill worker during download, inference, encoding, upload and report; correct result or retry follows without stale deletion. |
| T42 | Restart backend after result receipt before apply and after apply before notify; visible mutation and notification are idempotent. |
| T43 | Test locked package/model artifact stack with a real GPU and actual scoped B2 storage; record peak memory, cold/warm time, queue lag and audio quality. |
| T44 | Roll back routing for new jobs without restoring stale attempt fences, losing pending candidates or breaking old result readers. |

## 11. Definition of done

Each file-map entry must be closed with changed path/symbol, migrated caller list, actual test evidence and deletion status. Run format/lint/type checks, source unit tests, both-side contract tests, real model/storage integration tests and failure drills. Keep the measured results separate from proposed targets.

Do not call the migration complete while an old global client, hidden retry queue, AI database model, runtime installer, unbounded full-recording transfer or alternative canonical-update path is still used by the migrated protocol. Do not delete meaningful functionality to make the import scan pass. A release with skipped GPU tests remains unverified for production, not silently green.

## Source basis

The original audit records source observations. This implementation order adds design decisions rather than claiming new production measurements. Principal baselines: [AI orchestrator](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/orchestrator.py), [AI source tree](https://github.com/Techta-Labs-Ltd/hear-ai/tree/a89ee0ff9b231351034d612b7c4dfa6f8e42b091), [backend AI services](https://github.com/Techta-Labs-Ltd/hear-backend/tree/60387821fab069d4ea80cc0910ec09779b221cde/src/app/services/ai), [backend job model](https://github.com/Techta-Labs-Ltd/hear-backend/blob/60387821fab069d4ea80cc0910ec09779b221cde/src/app/models/processing.py), [Ray Serve gRPC documentation](https://docs.ray.io/en/latest/serve/advanced-guides/grpc-guide.html). Validate documentation APIs against the committed lockfile, not the latest documentation version alone.
