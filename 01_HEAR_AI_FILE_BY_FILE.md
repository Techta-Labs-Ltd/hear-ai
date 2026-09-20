# Hear-AI: direct file-by-file implementation plan

Version 2.0 — 20 September 2026  
Baseline: `a89ee0ff9b231351034d612b7c4dfa6f8e42b091`  
Read `00_IMPLEMENTATION_ORDER.md` first. P00–P09 are implementation packages; T01–T44 are mandatory acceptance behaviours defined there.

## A00. Scope, naming and implementation discipline

These are instructions to change the implementation, not a second audit. Existing paths are taken from the inspected repository tree. Specific current methods named below were visible in inspected source unless marked as target names. Per-file directions for remaining DSP, training, scripts and support modules are prescribed responsibilities; do not interpret them as a claim that every internal algorithm has been tested.

Do not introduce `application/`, `domain/`, `infrastructure/`, `use_cases/`, an abstract repository for every file, or three duplicate workflow frameworks just to satisfy this plan. Reuse the current `core`, `models`, `deployments`, `services`, `proto`, `training`, `tests` and `scripts` packages. Keep the service class names unless their responsibility genuinely changes.

The main new runtime files are `services/jobs/executor.py`, `services/jobs/pipeline.py`, `deployments/execution.py`, `core/backend_client.py` and `core/health.py`. Move the shared noise processor to `core/noise.py`; this is a relocation, not another implementation. New release configuration such as `serve.yaml`, a controlled image build and focused tests is allowed. Do not create both these files and the alternative `application/workflows.py` layout from the earlier sketch.

Refactor concrete consumers in the same package that changes a constructor. Do not leave no-argument compatibility constructors that quietly recreate globals. Keep temporary transport compatibility only where an identified v1 client still needs it. No explanatory source-comment blocks, commented-out code, placeholder methods or debug prints. Preserve generated/legal/tool-required directives. Put behavioural rationale in these plans and tests.

## A01. Root entry point — `main.py`

**Action: simplify startup; keep one production entry point. Packages P01, P04, P09.**

Retain `main`, `run`, `configure_process` and `validate_runtime` where they remain useful. Pass the same validated `Settings` instance throughout startup. Move the import of model/execution graph code out of module scope so `--validate-only` and the CPU gateway do not import the orchestrator, Fish Speech or every GPU model before configuration checks can run.

Change `validate_runtime` into explicit validation of the selected deployment profile and enabled capabilities. The gateway profile validates transport/auth/protocol configuration, not every model directory. Each model profile validates its own installed modules, native libraries, manifest and required files before it advertises readiness. Keep actual MossFormer `last_best_checkpoint` validation: that fix already exists. Keep Demucs manifest/member checks and Fish codec/weight checks for their enabled profiles. Do not replace file validation with directory existence.

Set offline library behaviour and logging before importing model packages. Remove hardcoded log-directory assumptions. Configure structured stdout logging once. Keep any file sink optional, configured, writable and created only in bootstrap. Do not set global warning filters inside service imports.

Use deployment configuration to start independent enabled graphs. Distinguish connecting to an externally managed Ray cluster from creating a local development runtime. Do not shut down a shared Serve instance simply because this driver attached to it. Own-and-close rules must match what this process actually created. Keep Ray administrative interfaces private.

**Delete here after P08:** AI database validation, runtime DDL dependencies, legacy orchestrator recovery launch, resolver/speed startup wiring and required modules that no longer have callers. Never download models, regenerate stubs, install packages or run schema migrations on application startup.

**Done when:** T01, T05, T19, T37 and T38 pass; an unavailable optional model does not prevent the gateway and unrelated capabilities from starting.

## A02. Configuration — `hear/config.py`, `.env.example`

**Action: retain one typed configuration module. Packages P01–P04, P08.**

Construct settings once at bootstrap. Pass relevant immutable settings/policy objects to services and deployment constructors. Remove the module-level `settings = Settings()` dependency from migrated domain code. Do not scatter `os.getenv` calls across processing methods.

Validate each range and relationship: positive bounded concurrency, nonnegative overlap smaller than half a window, finite timeouts/TTL, supported codecs, maximum source size/duration, workspace quota, maximum model-window bytes and enabled model artifact paths. Fail on conflicting configuration rather than silently selecting a fallback.

Make `QWEN_LLM_ENABLED` and Fish/capability enablement actually control graph construction and readiness. Preserve separate policies for pipeline, Magic Clean and reconstruction; do not add a separate flag per helper. Keep model residency, CPU execution concurrency, backend lease and storage validity conceptually distinct.

Repair the legacy storage TTL pair with the backend in P01. Do not set an invented new expiry on unchanged provider credentials. For new execution, consume the actual attempt grant and its validity; remove the universal Magic-Clean-only 24-hour waiting reserve after the backend queue and refresh path are active.

Remove `DATABASE_URL`, `DB_*`, `ORCHESTRATOR_*` business retry/queue options and storage-context-at-rest encryption options from the final AI execution profile only after their consumers migrate. Remove AI `RESOLVER_*`, `PIPELINE_SPEED_MULTIPLIERS` and obsolete settings after caller removal. Keep delivery bitrate/sample-rate/chunking and reconstruction timing settings. Do not confuse reconstruction tempo correction with Alexa speed generation.

Add the minimal missing settings for backend control endpoint/auth, selected protocol/profile, attempt deadlines, capability/resource budgets, window/input/disk bounds, TLS and controlled log level. Document units and required relationships in the runbook rather than duplicating settings in multiple environment files. Exclude secrets from logs and settings dumps.

**Done when:** parameterised settings tests reject contradictory values; default backend/AI credential contract passes T03 and enabled-capability tests pass T05.

## A03. Build and packaging — `pyproject.toml`, `uv.lock`, `patches/whisperx-asr-qwen.patch`, `venv`, `.gitignore`

**Action: make the tested runtime reproducible without runtime installation. Packages P00, P08, P09.**

Keep `uv.lock` authoritative and regenerate it deliberately during a dependency change, never on pod startup. Record the actual locked Ray/gRPC/PyTorch/Transformers versions; a permissive version range is not the deployed version. Pin VCS-based WhisperX to a reviewed immutable revision when updating the lock/source declaration. Verify the existing patch and resolved upstream revision together; either bake the patch once during build or use a fork revision containing it, never both blindly.

Separate the CPU gateway/execution dependencies from optional GPU/training dependencies where needed for independent imports. Make Fish Speech and required model-engine packages explicit provisioned build inputs; do not rely on an accidental mutable `/workspace` install. Build FFmpeg/ffprobe and required codec/filter support into the image. Put provisioning into a controlled image build or operator command, not model constructors.

Remove SQLAlchemy/psycopg2 and AI-only encryption/Redis dependencies only after import and functional ownership gates pass. Do not remove a package merely because its most obvious caller moved: check training, loaders, previews and cleanup too. Keep a dependency actually required by a remaining model/runtime library. Remove resolver-only packages only after their remaining NLP/model uses are checked.

Remove the tracked machine-specific `venv` symlink from the repository. Do not recursively delete the operator's live environment. Ignore generated caches, local audio, secrets, model weights and machine environments. Commit generated protocol code according to the existing client packaging contract, not transient model outputs.

Add or update the production image build and CI configuration as release files. Verify `--validate-only`, enabled-profile imports and locked dependencies inside the built image. Keep model downloads as an explicit provisioning step with revision/digest verification.

**Done when:** a clean build has no undeclared runtime package installation and T01, T38 and T43 pass.

## A04. Composition — `hear/deployments/app.py`

**Action: make this the only deployment graph composition boundary. Packages P04, P05.**

Replace the all-or-nothing `build_application()` construction with explicit gateway, execution-profile and model-profile builders in this file. These target builder functions may be named `build_gateway`, `build_execution` and `build_model`; do not create a new factory file for every deployment. Import model-specific modules only inside the relevant builder.

Bind the stable gateway separately from optional model families so one failed optional constructor does not block all entry points. Use independently deployable model applications where needed for fault isolation. Record each application/deployment name once in `serve.yaml` or equivalent deployment configuration. Resolve cross-application handles at composition/replica-startup boundaries and inject them; do not allow services to look them up by string globally.

Construct one `RayModelClient` per execution replica from the relevant handles. Construct one common `JobExecutor` with a map of supported job types to service execution methods. Build `PipelineService`, `MagicCleanAudioEnhancer` or the refactored `RegenerationService` only for the execution profile that needs them. Shared pure services may be reused in that replica; mutable per-job state must not be shared.

Use three independently bounded execution profiles: pipeline/text operations, Magic Clean, reconstruction/editing. These can use the same `ExecutionDeployment` class with different configuration; do not copy its implementation three times. They do not imply three physical GPUs. Do not introduce another durable queue inside the graph.

**Delete:** `set_model_client`, implicit singleton registration, unconditional model binds and passing the old orchestrator everywhere after legacy drain.

**Done when:** injected fake handles construct each profile independently, T20–T24 pass, and turning off Fish/LLM does not break Magic Clean imports or processing.

## A05. Gateway — `hear/deployments/gateway.py`

**Action: retain HTTP/gRPC transport, authentication and delegation only. Packages P01, P03, P04.**

Inject the execution handles, health reader, backend control client and authentication registry. Remove `init_db()`, loader `.load()` calls, `orchestrator.recover_jobs.remote()` and global model-client registration from its constructor. Gateway startup must not run taxonomy mutation, model inference or business recovery.

Keep `/` as identity, expose lightweight liveness, and make `/ready` reflect gateway/control readiness rather than loading every model. Return a bounded capability summary through health without pretending local gateway CUDA describes workers. Set request/response limits and deadlines, redact secrets and authenticate before dispatch.

Keep existing `/process`, `/discovery`, `SubmitJob`, `Subscribe` and `GetResult` only through a version-aware compatibility adapter for legacy jobs. Map requests once into typed internal envelopes. Do not duplicate defaults between REST and protobuf conversion. Preserve `same_speaker` presence, stem zeros, optional fields and idempotency conflicts.

Add the new `ExecuteAttempt` method. Forward a bounded server-streaming request to the chosen execution profile and keep it outstanding. Propagate cancellation to the downstream Serve response, but rely on backend lease/cancel state for durability. Translate capacity exhaustion, unavailable capability, invalid identity and invalid audio into distinct protocol errors. Do not hide unexpected exceptions as successful empty results.

Do not expose preview approval, catalogue mutation, training persistence or canonical track writes as gateway-local operations in the target. Redirect identified compatibility callers to backend owners; remove those legacy RPC implementations after their callers migrate.

**Done when:** T01, T03–T06, T14 and T19 pass; gateway contains no ORM, native audio processing or business retry loop.

## A06. Common Ray executor — new `hear/deployments/execution.py`

**Action: add one thin Ray wrapper, not a scheduler. Packages P03–P05.**

Target class: `ExecutionDeployment`. Constructor dependencies: immutable execution-profile configuration, injected model handles, backend control client factory, workspace/storage configuration. Construct the profile's services and `JobExecutor` in this replica's composition boundary.

Keep a Serve request active for the full attempt. Set finite `max_ongoing_requests` and supported finite queue/backpressure limits based on the locked Ray version. Use a small model-call look-ahead budget. Reject/defer overload to the backend; do not accumulate an unbounded list of task objects or audio windows.

Use CPU resources here. Any CUDA-dependent computation still performed by the old orchestrator/synthesizer must move into its model deployment before removing its GPU allocation. Keep local preprocessing explicitly CPU-bound with a bounded executor/subprocess budget.

Expose minimal capability/lifecycle state: accepting, draining, current instance epoch and bounded in-flight count. Do not store durable jobs/results here or rely on request count as the backend's persistent ledger. Register running child tasks only for cancellation/lifecycle; close and await them on drain.

**Done when:** T09, T18, T25, T27, T29 and T37 pass with multiple execution replicas and no hidden task-per-job backlog.

## A07. Common attempt lifecycle — new `hear/services/jobs/executor.py`

**Action: extract lifecycle once from `hear/orchestrator.py`. Packages P02–P05.**

Target class: `JobExecutor`. Constructor dependencies: backend control client, workspace factory, per-attempt storage factory, job-type-to-service execution mapping, immutable execution policy and clock/metrics collaborators. Do not pass a mutable database session or global container. `execute(request)` is the primary target method.

Acquire the current backend execution claim before source download, model invocation or artifact writes. Validate protocol, semantic identity, current fence, source revision and deadline. Open one execution-scoped workspace. Create structured progress events and a bounded lease-renewal task. Call exactly one selected service; do not implement DSP or branch-specific lineage inside the executor.

The selected service returns typed local artifact descriptors plus business result data, not uploaded URLs or a `result.__dict__` containing local paths. The executor uploads returned artifacts under the exact authorised execution prefix, verifies them, writes the immutable manifest last and reports its reference. Pure text jobs still produce a recoverable manifest. Never upload the entire source again merely to create a result.

Keep transport retry bounded and separate from model execution. A failed result report retries that report or leaves a known manifest for reconciliation; it does not call the service again. On model/network/audio error, return a typed failure with retryability evidence; only the backend decides the next attempt. Do not increment a business attempt counter in this process.

On cancellation/lease loss, cancel downstream responses and signal native work, wait for stopped work, then release workspace/model-call resources. Persist or report cleanup ownership through the backend where needed. Do not erase a completed candidate because the final acknowledgement was ambiguous. Do not suppress cleanup failures without recording them.

**Extract from orchestrator:** lifecycle portions of `_process`, completion/reporting and common stage timing. **Do not copy:** SQL sessions, `_recovery_loop`, `_fair_scheduler`, `_scheduled_runs`, shared subscriber queue or per-workflow processing branches.

**Done when:** T08–T19, T27 and T41–T42 pass, and the executor imports no AI ORM model or concrete DSP processor.

## A08. Old orchestration and jobs — `hear/orchestrator.py`, `services/jobs/submission.py`, `services/jobs/scheduler.py`

**Action: extract deliberately, then delete the obsolete owners. Packages P01–P08.**

| Current symbol/responsibility | Exact target |
|---|---|
| `_process_pipeline`, `_process_discovery`, `_run_discovery`, pipeline compression orchestration and text-only result branching | New `services/jobs/pipeline.py` / `PipelineService`. Reuse categorization, moderation, transcription and discovery services. |
| `_process_magic_clean` | Existing `services/magic_clean/service.py` execution method; common upload/manifest in JobExecutor. |
| `_process_reconstruct`, `_process_edit_transcript` | Existing `services/reconstruction/service.py` / refactored RegenerationService. |
| `resolve_reconstruction_reference_url`, `_resolve_reconstruction_reference_url`, source lineage job queries | Backend source/AI service ownership; pass explicit source/reference/timeline in new request. Retain pure comparison logic only in backend migration tests where needed. |
| `_magic_clean_lineage_jobs`, `_magic_clean_jobs_outside_scope`, hash-alias history scan, reuse candidate selection | Backend indexed source/lineage/reuse policy. Remove global AI history scans. |
| `_record_magic_clean_cleanup_tombstone`, `_cleanup_magic_clean_artifact` and DB tombstone recovery | Backend `services/ai/cleanup.py`; executor only reports owned uncommitted artifact references. |
| `_push_event`, `subscribe` and terminal replay | Backend event journal/replay. Keep corrected v1 fan-out only during drain. |
| `_set_stage`, `_push_stage_result` | AttemptContext event reporting with definitions in `models/stages.py`; backend validates/persists. |
| `_schedule_job`, `_dispatch_loop`, `_run_scheduled`, `_pending_job`, fairness and job slots | Backend `services/ai/scheduler.py`; finite AI admission belongs only to ExecutionDeployment. |
| `recover_jobs`, `_recovery_loop`, retry decisions | Backend reconciler and current attempt records. |
| `get_stats` business queue data | Backend indexed counts. AI reports only its local capability/capacity telemetry. |
| Hardcoded reconstruction logging handlers | Root logging configuration; delete import-time handlers. |

In `submission.py`, preserve request normalization and semantic conflict behaviour until the new contract is active. Move reusable pure normalization into `models/schemas.py`; put durable idempotency in backend AIJobService. Separate renewable grant data from semantic fingerprint. Remove SQL insertion and queue dispatch from the final AI submission path. Retain v1 adapter only for known legacy requests, then delete it.

In `scheduler.py`, port useful pure round-robin test cases into the existing backend scheduling tests. Do not keep AI `FairJobScheduler` alongside backend job ownership. Delete it after no v1 jobs are owned by AI. Do not leave a scheduler with renamed variables pretending it is just an executor.

**Done when:** all mapped methods have exactly one live owner, all callers migrated, T19/T38/T40 pass, and old orchestrator is removed rather than reduced to another global router.

## A09. Audio acquisition and conversion — `core/audio_utils.py`, `core/downloader.py`

**Action: consolidate external audio I/O while retaining reusable conversion functions. Packages P04, P05.**

Add `AudioIO` in the existing `audio_utils.py`; inject a pooled HTTP client, workspace policy, FFmpeg runner/configuration and allowed-source policy. Move `download_audio` acquisition into this class. Keep small deterministic codec/bitrate/probe transformations as functions where they do not own state. Do not introduce another broad `utils_v2.py`.

Accept authorised immutable object references from the backend. Validate scheme/host/redirect targets, DNS/IP destination policy and input byte/duration limits before streaming. Do not allow arbitrary private-network URL fetching. Stream to an owned workspace file while hashing; use actual probed content type/codec instead of trusting `.wav` in a filename. Verify source revision/digest and response size before processing.

Provide explicit consumer representations: ASR mono at its required sample rate, stereo-preserving Magic Clean representation, and reconstruction source/reference representations with timeline and channel metadata. Convert only when required. Decode long inputs into bounded file-backed windows and reuse their prepared representation within the same attempt. Do not call `read_bytes`, `response.content`, whole-file `torchaudio.load` or equivalent on unbounded recordings.

Track FFmpeg subprocesses, timeouts and cancellation under the workspace lifecycle. Check exit status and stderr without logging private URLs. Preserve existing MP3 delivery helpers where required, and stream output encoding. Use a tested large-file representation when standard WAV limits are exceeded. Do not blindly force all workflows through one lossy conversion.

Keep `downloader.py` as a temporary import adapter only until every caller moves. Then delete it. Pass a local path only within its owning execution process/filesystem. Across model actors pass bounded data/object references with explicit sample rate/shape, never assume `/workspace/audio/...` exists on another node.

**Done when:** T25–T27 pass; acquisition/conversion counts show no duplicate full-source download within one new-protocol attempt.

## A10. Workspace and blocking work — `core/hear_temp.py`, `core/blocking.py`, `core/gpu.py`, `deployments/audio_cleanup.py`, `tools/clean_temp.py`

**Action: one local resource owner, no distributed lock illusion. Packages P04, P05, P08.**

Refactor `hear_temp.py` into a workspace manager using backend/job/attempt/execution identity, node identity, disk quotas and a runtime registry. Keep file creation, registration and deletion here. Remove SQL `db` parameters from migrated callers. Never delete active files solely because they are old; the owning process lock/runtime marker and lease-aware cleanup policy must distinguish active, stopped and abandoned work. Validate deletion paths against the configured root, including symlinks.

Retain cancellation-safe patterns in `blocking.py`: signal native work, await termination, then propagate cancellation. Use a bounded executor owned by the replica; do not create a daemon thread for each model response. Remove duplicate one-line wrappers from `services/magic_clean/blocking.py` after imports move here. Do not spread `asyncio.to_thread` calls without a concurrency budget.

Delete `core/gpu.py` process-global `cuda_inference_lock` after all consumers use actual model-actor concurrency. A process-local lock cannot coordinate different Ray actors. Keep serial access at a non-thread-safe model's owning deployment; do not remove its protection just because the global lock is deleted.

Change `audio_cleanup.py` to node-local abandoned-workspace sweeping only. Run one correctly placed sweeper per node, or perform bounded sweeps in each node's workspace manager; one arbitrarily scheduled replica cannot clean every node's local disk. Remove remote B2 tombstone SQL reconciliation here; backend owns remote artifact cleanup. `tools/clean_temp.py` must call the same workspace policy, support safe dry-run/selection and never recursively erase active/model/cache directories.

**Done when:** T27, T38 and T41 pass, including cancellation during encoding/upload and two executions on different nodes.

## A11. Object storage — `core/storage.py`

**Action: retain a narrow scoped storage adapter. Packages P01, P03–P05.**

Keep or refactor `B2Storage` with explicit constructor dependencies: a provider client/factory and immutable attempt grant/destination. Remove choosing credentials from arbitrary globals. Do not share mutable grant state across users. The backend issues actual restricted grants; AI neither invents expiry nor receives a provider master key.

Retain safe key joining, backend/bucket/prefix checks, expiry enforcement, streamed upload, checksum verification and sanitised errors. Supply a public artifact descriptor method rather than calling a private `_public_url` from orchestration. Have JobExecutor publish all final candidates/manifests through this adapter; DSP/model services must not upload independently.

Use bounded multipart settings appropriate to source limits; abort incomplete multipart sessions on owned cancellation where supported. Refresh only the current execution grant through BackendClient. Do not start inference while a known-invalid destination grant makes completion impossible. Store actual stable object keys in manifests, not expiring signed URLs.

Remove `storage_for_job`, `encrypt_storage_context`, `decrypt_storage_context` and DB-specific coupling after legacy records move to backend ownership. Keep at-rest secret handling only where still required by a retained compatibility path until drain; do not remove it prematurely. Remote cleanup must use exact version/key ownership, not a job-wide prefix that another attempt may occupy.

**Done when:** T03, T15–T18 and T41 pass against the actual storage adapter.

## A12. Auth, policy and context — remaining `core` files

**Action: remove mutable runtime globals and SQL from read-only processing inputs. Packages P03, P04, P08.**

| File | Direct implementation instruction | Acceptance |
|---|---|---|
| `backend_registry.py` | Keep authentication and allowed backend/storage mapping. Parse once at composition, inject a read-only registry, validate actual worker/gateway credentials and rotation. Do not store per-job mutable credentials in registry state. | Wrong backend/tenant cannot acquire or read another execution; T03/T10. |
| `processing_context.py` | Keep existing transcript normalization/TrackData behaviour. Add target runtime AttemptContext for identity, deadline, workspace, cancellation and progress sink. Keep it per execution; exclude ORM instances and hidden service lookups. | Concurrent jobs do not share mutable state; T20. |
| `content_context.py` | Preserve pure contextual label rules and their tests. Remove imported global I/O if any; do not introduce one class per heuristic. | Existing editorial/assistive-tech/wildlife regressions remain unchanged; T23. |
| `category_loader.py` | Keep read-only parsing/lookup. Replace SQL-backed loading and in-request add/ensure persistence with a versioned taxonomy snapshot supplied by backend. Emit label proposals in results, not writes. | Same request revision gives same catalog; T24/T38. |
| `discovery_taxonomy.py` | Keep discovery-specific parsing, inject immutable revision. Reuse shared fetch/cache infrastructure, avoid a second updater/retry loop. | Missing/replaced taxonomy has explicit state and no cross-tenant bleed. |
| `keyword_loader.py` | Make keyword sets immutable per backend/policy revision. Remove request-time `sync`/`sync_platform_keywords` mutation from moderation/categorization. Backend owns reviewed updates. | Two backend policies remain independent; T24. |
| `platform_settings.py` | Replace per-service hidden global reads with a typed policy snapshot passed into the job/service. Keep one compatibility adapter until all callers migrate, then remove the adapter if no longer needed. | Publish-only and blocked-keyword policy remains intact; T23/T24. |
| `discovery_sort.py` | Move user-facing catalogue sorting/pagination to backend query ownership. Retain a pure ranking helper only if generated-result ranking uses it; otherwise delete after its callers migrate. | Stable ordering/tenant filtering and no AI catalogue DB dependency. |
| `db_gate.py` | Keep only for legacy drain. Move any reusable transaction behaviour to existing backend repositories; do not copy a generic retry loop around unsafe non-idempotent operations. Delete from AI at P08. | T38. |
| `__init__.py` | Keep import-safe exports only; do not instantiate clients, loaders, settings or models. | T01. |

New `core/backend_client.py`: implement one injected pooled async client for execution claim, lease, grant refresh and idempotent reports defined in the master contract. Keep TLS/auth, deadlines, sanitisation and bounded transport retry here. Expose typed methods, not arbitrary authenticated HTTP forwarding. No business-job scheduler, SQL or model retry policy belongs in this file.

New `core/health.py`: aggregate cached model-replica lifecycle/capability snapshots, instance epoch and control readiness. Do not import CUDA in the gateway or call inference to answer a probe. Treat scaled-to-zero as cold; missing artifacts as unavailable; saturated capacity as busy. Apply stale snapshot handling. Backend owns incident notification, not this reader.

## A13. Schemas and stages — `models/schemas.py`, `models/stages.py`, `models/discovery.py`, `models/database.py`

**Action: retain typed values; remove worker persistence only after migration. Packages P02–P04, P08.**

In `schemas.py`, keep existing REST DTOs and their versioned compatibility. Add target typed execution request, source reference, timeline/edit specification, grant metadata, local artifact descriptor, execution result and progress types. Validate once at transport entry; do not propagate untyped `dict` through every layer. Treat zero/false/omitted distinctly. Require all stem percentages together or use one documented default. Reject nonfinite/out-of-bounds timestamps and invalid edit overlap before inference.

Keep secret-bearing grant types separate from serialisable result/telemetry types. Make local artifact descriptors internal; require explicit conversion to public artifact manifests so local paths and secrets never leak. Avoid creating another duplicate `contracts.py` with equivalent models.

In `stages.py`, keep one authoritative stage definition per job type. Make emitted stages match execution: audio tag, direct reconstruction, edit-transcript downloading/transcribing/diffing/reconstructing, Magic Clean chunk progress and final validation/upload. Emit stage starts when work actually begins, not two immediate labels before a monolithic remote call. Distinguish percentage from measurement; do not invent time remaining as a correctness signal. Keep backend labels compatible.

In `discovery.py`, retain value objects and normalization for generated metadata. Remove catalogue SQL coupling if present, not discovery generation. Use typed optional missing metrics instead of implying a default zero is a measurement.

In `database.py`, inventory every model and caller, including AiJob, AiTrackJob, RegenerationPreview, CategoryTrainingExample, loaders, cleanup and temp tracking. Migrate required records and external callers first. Delete this module only after T38/T40; do not replace it with SQLite/Redis or a local JSON job database. `models/__init__.py` must not construct engines or eagerly import removed models.

**Done when:** T02, T06, T10–T11, T19, T24 and T38 pass.

## A14. Native model access — `services/model_client.py`, `services/llm.py`

**Action: one asynchronous model adapter; no service locator or sync thread bridge. Packages P04, P05.**

For `RayModelClient`, replace `_client`, `set_model_client` and `get_model_client` with constructor injection. Keep an explicit typed set of required/optional handles per execution profile. An absent optional handle produces a typed capability-unavailable result; it must not trigger hidden local model loading.

Migrate `transcribe_sync`, `moderate_sync`, `nli_sync`, `sentiment_sync` and `llm_generate_sync` callers to awaited methods. Then delete `_resolve_sync`, its daemon-thread/future wait path and obsolete sync methods. Preserve the `hypothesis_template` parameter in the new async NLI method. Keep speech reference ID, language and seed forwarding intact. Parse/validate model results at this boundary, not with repeated JSON conversions through the service stack.

Use bounded `AudioWindow` / short-clip inputs for transcription and enhancement. Cap generated speech response sizes using text/segment limits; long output becomes bounded segments, not an unlimited bytes object. Preserve response cancellation/deadline propagation. Never pass storage credentials to model actors that only need samples/text.

In `llm.py`, retain `LLMService` as the owner of prompts, generation options and schema validation; constructor-inject model client and immutable LLM policy. Remove `get_llm_service` singleton access. Make inference methods async, respect enablement, and use explicit output parsing/error types. Keep intended moderation/categorization/discovery fallbacks visible in the calling service, not broad catches silently returning success.

Keep the currently working engine during structural extraction. Implement a vLLM-backed deployment only as a separate tested engine change after the exact model/quantisation and installed Ray/vLLM compatibility are verified. Do not force ASR, Fish Speech or every model through vLLM. No second pod-local retry engine is allowed. Record engine/model revision in results.

**Done when:** T20/T23 and grep/import checks show no live global model-service accessor or event-loop-blocking sync response resolution.

## A15. Pipeline extraction — new `services/jobs/pipeline.py`

**Action: extract one cohesive service, not one class per stage. Packages P04, P05.**

Target class: `PipelineService`. Constructor dependencies: AudioIO, TranscriptionService, ModerationService, CategorizationService, discovery service and immutable processing policy. Do not instantiate these inside `execute`.

Move current pipeline/discovery orchestration from the orchestrator. Keep a clear job-type dispatch in this class for pipeline, transcription, categorization, audio_tag, discovery and rebuild. Reuse the same stage functions rather than copying pipeline bodies for each type.

| Job type | Preserve and implement |
|---|---|
| `pipeline` | Acquire source, transcribe, moderate/flag, run permitted categorization/tagging/discovery and required delivery conversion. Do not make Magic Clean/reconstruction hidden mandatory stages. |
| `transcription` | Transcribe and return its established typed result; do not run tagging, discovery or synthesis. |
| `audio_tag` | Transcribe the bounded short utterance; return at most two cleaned suggestions and source identity. Do not run full moderation/discovery unless existing product policy explicitly requires it. |
| `categorization` | Use supplied transcript when valid, otherwise required transcription; preserve current moderation/discovery contract rather than silently reducing output. |
| `discovery` | Produce discovery metadata using the existing source/transcript semantics, not a catalogue listing. |
| `rebuild` | Use corrected text and existing text/pipeline semantics. Do not synthesize merely because the name resembles reconstruct. |

Return typed result and any local delivery artifacts for JobExecutor to publish. Keep blocked/flagged behaviour and empty/silent-content semantics explicit. Do not classify an arbitrary model exception as no speech. Pass policy/taxonomy revision through all stages. Avoid repeat transcript decoding or redundant full-text model calls where cached same-attempt stage data is valid.

No SQL track rows, backend metadata fetches, storage credential selection, global model lookups, event journal writes or business retries belong here. Original publication state is not a pipeline concern.

**Done when:** T02/T20/T23/T24 and golden job-type result tests pass against both current API adapters.

## A16. Transcription service — `services/transcription/service.py`, `services/transcription/chunks.py`

**Action: retain result processing, replace whole-recording transport and hidden clients. Packages P04, P05.**

Change `TranscriptionService.__init__` to require model client, AudioIO/window reader and immutable transcription policy. Remove the service-wide lock that serializes unrelated jobs after model safety is enforced at the model deployment. A model's own concurrency remains bounded. `transcribe` accepts an authorised source/prepared file description and attempt context, not an unbounded file bytes argument.

Replace `run_in_executor(client.transcribe_sync, ...)` with direct awaited model-window calls. Keep at most the configured small number of windows outstanding. Pass required sample rate, offset, language and bounded batch policy. Do not create a parallel ASR implementation for reconstruction; inject this same service.

Retain `_process_result`, `_credible_segments` and text normalization as cohesive result-processing logic. Move confidence thresholds out of globals into injected policy. Keep short-utterance and hallucination filtering tests, including legitimate short “thank you” speech. Distinguish no-speech detection from a model failure. Validate timestamps and handle absent word boundaries deliberately; do not throw away valid segment-level transcription or invent measured confidence when the model did not supply it.

Use actual probed audio duration separately from the last spoken segment end. Preserve existing public fields through the compatibility serializer; add explicit presence/measurement status where needed rather than silently changing the meaning of a zero.

In `chunks.py`, retain `adaptive_batch_size`, offset shifting and final merge behaviour. Replace the source of `iter_audio_chunks` so production callers do not need an entire decoded waveform. Keep short-array helpers for bounded unit tests only. Clamp windows to source duration, preserve global word times and deduplicate overlap according to a tested rule. Do not run the full file through a second concatenation just to compute final transcript text.

**Done when:** T06/T22/T25/T26/T28 pass; no long-recording `Path.read_bytes` → transcribe path remains.

## A17. Transcription model — `deployments/transcription.py`

**Action: model-only bounded inference. Packages P04, P05.**

Constructor-load the local ASR/aligner artifacts for this deployment only. Inject model configuration, batch/window limits and device policy. Validate exact artifact content and tokenizer/generation settings against the locked model before reporting ready. Keep the implemented CUDA-health failure signal and supported replica replacement.

Replace `transcribe(audio_bytes, batch_size)` as the new-protocol entry with a target `transcribe_window(window, options)` method accepting bounded samples/metadata. Remove whole-file tempfile writing and `whisperx.load_audio` for new calls. Return a typed/dict bounded window transcript with offsets applied in one agreed place; avoid a JSON string inside another model response when no external wire compatibility requires it.

Run synchronous native inference on one bounded, replica-owned execution lane so health/cancel handling remains responsive. Do not allow concurrent calls into a non-thread-safe model simply because inference moved to a thread. Limit torch/CPU threads to the assigned budget. Synchronise actual native completion before releasing its slot.

Narrow the broad `(IndexError, ValueError)` “no output” catch. Only an explicit library no-speech result becomes an empty transcript; programming errors and malformed model output must remain failures. On CUDA context corruption mark unhealthy; let the deployment lifecycle replace the replica. Backend decides whether to schedule another attempt.

Remove the global `torch.nn.functional.pad` monkeypatch. Place any necessary conversion into the exact bounded model adapter or pinned upstream patch, with a regression fixture proving why it is needed. Do not globally change third-party behaviour for unrelated processors. Remove redundant per-window full GC/cache flushes unless measurements demonstrate a need; deleting references and model lifecycle ownership are not interchangeable with emptying allocator cache.

**Done when:** T01/T25–T28 and actual ASR/aligner fixtures pass with the locked stack.

## A18. Magic Clean execution service — `services/magic_clean/service.py`

**Action: reuse MagicCleanAudioEnhancer as the cohesive CPU execution service. Packages P01, P04, P05.**

Constructor dependencies: AudioIO, async model client, delivered-audio validator/metrics and immutable Magic Clean policy. Do not construct MossFormer, Demucs, dynamics, noise and unrelated models here. Their native model ownership moves to the Magic Clean deployment. Remove service `load`, `_loaded` and `_gpu_lock` after new model-window calls are active; model readiness/concurrency belongs to its actual actor.

Add the target `execute(request, context)` entry to acquire/verify the authorised source once, normalize controls, perform streamed cleaning, validate the local delivery and return typed local artifacts/results. Keep the render helper `enhance` only where it makes reuse clear, not as a second orchestration path. Do not add a wrapper `MagicCleanWorkflow` that merely calls this service.

Remove direct B2 upload, uploaded-key deletion, storage-context decryption and job SQL from the service. JobExecutor publishes final artifacts/manifest once. Return local data only inside the same owning execution process; the public serializer must never expose the deleted `local_path` that the current result object can carry.

Keep file/PCM hash verification where meaningful, but compute each required digest once in the workspace pipeline and reuse it. The backend supplies explicit immutable lineage/source identity. Remove AI history searches and duplicate source acquisition from the orchestrator/deployment path. Never substitute a different audio version because a URL string matches old output.

Use an async bounded streaming path over the source file. The model client enhances only bounded windows; the service writes their retained cores to disk, performs global silence/mastering policy once and validates the delivered encoded file. Progress is based on processed windows/stages, not fake separating/mixing events after the whole job returns.

**Fix the check that blocks legitimate jobs:** correct the issuer/validator TTL contract before this extraction. Preserve real expiry/ownership constraints. Classify insufficient grant validity as waiting for credentials, not corrupted audio. For audio checks distinguish hard integrity failures from calibrated quality warnings; do not suppress all validation to make a request succeed.

**Done when:** T03/T07/T21/T25–T27 pass; new Magic Clean has no dependency on an existing AiTrackJob, LLM, Fish Speech or prior transcript.

## A19. Magic Clean model wrapper — `deployments/magic_clean.py`

**Action: own native processors and window inference, not files/jobs/storage. Packages P04, P05.**

Change the constructor to receive immutable model/profile configuration and construct the native pipeline from explicit processors. Load only pre-provisioned local MossFormer/Demucs artifacts here and report actual readiness. Do not instantiate the CPU `MagicCleanAudioEnhancer` service inside this actor once A18 is migrated.

Add target `enhance_window(window, controls, profile_revision)` with bounded shape/sample-rate validation and cancellation-safe native execution. The GPU pipeline returns enhanced samples of the expected channels/sample count plus small diagnostics. Convert transfer buffers to the agreed CPU/Ray representation before returning; do not retain every finished tensor.

Delete URL downloading, B2Storage construction, complete-file upload, job-temp cleanup and full-asset hashing from this model deployment. Those belong to the execution process. Preserve `max_ongoing_requests=1` initially for non-thread-safe processors and measure before raising it. Keep logical GPU/resource configuration tied to measured peak use, not an assumed VRAM fraction.

**Done when:** T21/T25/T27–T29 pass and the model actor accepts no backend storage secrets or long recording URLs.

## A20. Magic Clean pipeline — `services/magic_clean/pipeline.py`

**Action: retain the actual DSP pipeline and tighten its interface. Packages P04, P05.**

Keep `MagicCleanPipeline` and `MagicCleanProfile`. Replace `Any` collaborators with concrete processor types or narrow typed protocols actually used by the pipeline. Construct them in the model composition boundary, not during arbitrary processing calls. Keep model `load` startup separate from inference; `_ensure_stem_loaded` should assert readiness or complete controlled startup, not fetch weights on a request.

Keep `process` as a bounded window operation with `finalise=False` for the production streaming path. Do not normalize loudness or remove global silence independently per window. Keep short-array `process_chunked` only for explicitly size-limited local testing/tools, or remove it after callers switch to disk streaming. Its pieces-plus-concatenation approach must not remain a long-recording production fallback.

Retain stem mixing and level semantics. Test `speech=0`, `music=0`, `background=0`, all defaults and partial-field rejection. The current residual `background = input - vocals - music` is not proof of a separately identified environmental-noise stem; preserve/document its actual semantics and do not rename a slider without testing the product behaviour.

Inspect `_protect_source_activity` with dedicated fixtures. It restores source in detected collapsed frames, not necessarily the whole track. Record the proportion of frames/samples restored and the trigger reason. Test quiet voiced speech, genuine noise reduction, separator failure and intentionally muted stems. Correct false-positive restoration using a versioned, tested guard policy and smooth transition handling; do not simply delete protection or set dry mix to zero everywhere. Never reintroduce muted music/background through full-mixture restoration.

Keep disabled spectral suppression and fixed-tone shaping disabled unless a separate audio-quality change proves them safe. Preserve zero-strength identity. Keep finite-sample/channel/length validation as hard guards. Move full-file mastering/silence orchestration to the streaming/service boundary while retaining reusable bounded DSP methods.

**Done when:** T07/T21/T26 plus before/after speech-retention and noise-only audio fixtures pass. Record an explicit DSP-policy change separately from structural extraction.

## A21. Magic Clean streaming — `services/magic_clean/streaming.py`

**Action: one production long-file implementation. Packages P04, P05.**

Refactor `clean_file_streaming` to use an injected async window enhancer instead of requiring local GPU processors. Decode with bounded buffers, include overlap/context margins, write only agreed retained cores, and bound look-ahead. Do not queue every window before awaiting results. Apply offsets and overlap retention once; add continuity fixtures at the first, internal and final boundaries.

Use disk-backed intermediate/reference files in the current execution workspace. Maintain one global source/output silence decision and one delivery mastering/encoding pass according to existing policy. Preserve retained-timeline information when silence is cut. Hash/validate while streaming or by bounded file passes; do not concatenate all chunks into a giant tensor for finalisation.

Progress and cancellation hooks take AttemptContext collaborators, not raw global callbacks or a database. If a remote window fails, do not silently substitute raw input and label enhancement successful. Report the precise failed window/stage, preserve safe partial outputs for owned cleanup and leave retry policy to backend.

Keep final encoded-output validation after encoding. Validation of the intermediate waveform is not evidence the MP3 delivery is correct. Publish no final artifact until those hard checks complete.

**Done when:** T25–T27 and encoder/last-window failure fixtures pass with bounded memory.

## A22. Magic Clean processing files — every module

**Action: keep useful signal-processing units, remove orchestration and duplicate I/O. Packages P04, P05.**

These per-file directions preserve existing algorithms until their own fixtures authorise a change. Do not assume a helper is unused just because it is absent from one happy path.

| Existing file | Required changes inside the file | Constructor / test rule |
|---|---|---|
| `processing/audio_io.py` | Move duplicate source/file conversion into core AudioIO. Retain only genuinely Magic-Clean-specific bounded tensor/file-format conversion; delete the module if no unique behaviour remains. | No client/credential selection. Test channel/sample-rate parity. |
| `processing/helpers.py` | Keep pure small tensor/math helpers that have multiple real callers. Merge single-use helpers into their owning processor and update imports; no new helper classes. | Boundary/empty/nonfinite fixtures; no side effects. |
| `processing/mossformer.py` | Keep MossFormer2Enhancer as native model adapter. Inject model path, explicit device/dtype and inference policy. Validate checkpoint before ready, run bounded windows, fail typed on invalid output. | Model-replica owns load/unload; no HTTP, SQL, download or retry scheduler. Test quiet speech, stereo shape and corrupted weights. |
| `processing/stems.py` | Keep StemSeparator with injected device/model configuration. Load Demucs at replica startup, cap native segment/batch use and preserve named-stem/channel semantics. | Do not reload model per window. Test controls, residual sum and output length. |
| `processing/noise.py` | Move the reusable NoiseReducer implementation to `core/noise.py`; update Magic Clean and reconstruction imports together and remove the original duplicate. Preserve exact zero-strength identity and optional suppression policy. | Inject any mutable model/device dependency; retain pure signal functions as functions. No two noise implementations. |
| `processing/speech.py` | Keep SpeechProcessor for bounded speech-specific filters only. Do not hide ASR or source acquisition here. Preserve explicit opt-in tone/de-essing behaviour. | Parameters from immutable profile. Test sibilance, quiet speech and identity when disabled. |
| `processing/dynamics.py` | Keep DynamicsProcessor for reusable compression/limiting primitives. Make device and parameters explicit; do not auto-select CUDA in a CPU execution process. Ensure global loudness/mastering is applied once by the streaming owner. | Test peaks, transients, silence and no chunk-boundary pumping. |
| `processing/silence.py` | Keep SilenceProcessor, disk-backed silence decision and retained interval map. Inject threshold/timing policy and optional detector adapter. Preserve source/output-union speech protection and do not trim per-window independently. | Test quiet speech, leading/trailing silence, short gaps, cut_silence=false and cross-window pauses. |
| `processing/quality.py` | Keep QualityMetrics as measurement logic. Return measured/unavailable/error status explicitly; compute bounded or streaming statistics for long files. | Missing metric is not zero-pass. Test silence, clipping and metric exceptions. |
| `processing/validation.py` | Keep hard guards for decode success, finite samples, legitimate duration/channel rules, integrity and clipping. Separate heuristic quality warnings from hard failure. Use requested stem/silence semantics when applying audibility/retention checks. Stream long comparisons and cap reference buffers. | Tests must cover valid intentional silence, quiet speech, channel loss, malformed output and encoded delivery. |
| `processing/__init__.py` | Keep exports only. | Import must not load models. |

## A23. Remaining Magic Clean files — `models.py`, `lineage.py`, `cleanup.py`, `blocking.py`, `__init__.py`

**Action: keep values, move ownership, delete redundant wrappers. Packages P01–P08.**

In `models.py`, preserve `StemLevels`, defaults and `ContentMode`; make validation explicit and immutable. Refactor EnhancementResult into an internal typed local result without pretending a deleted path is a usable public artifact. Public URL/key population happens once after JobExecutor upload.

In `lineage.py`, separate pure digest/normalization utilities from database/history-based source resolution. Move digest streaming into core audio/storage helpers when shared. Move durable lineage selection, alias/reuse authorization and engine-version reuse policy to the backend AI/source services. Require explicit source/parent identities in new execution requests. Remove the AI-wide all-history/hash-ownership search rather than wrapping it in a new class. Keep authorised duplicate assets distinct from unauthorised cross-tenant references.

In `cleanup.py`, move remote candidate/tombstone decisions to existing backend `services/ai/cleanup.py`. Preserve reconciliation of old unresolved tombstones during migration. AI local workspace cleanup remains in core/hear_temp; do not retain both remote-cleanup authorities. Delete this module at P08 after ownership records/callers migrate.

In `blocking.py`, redirect callers to core/blocking and delete the one-line duplicate after the same release's imports/tests pass. `__init__.py` remains import-safe only.

**Done when:** T03/T07/T10/T21/T38/T40 pass; no lineage/cleanup data is discarded merely to remove SQL imports.

## A24. Reconstruction execution — `services/reconstruction/service.py`

**Action: repurpose the existing RegenerationService; no parallel workflow class. Packages P01, P04–P06.**

Constructor dependencies: AudioIO, TranscriptionService, SpeechSynthesizer, RegenerationQualityAssessor and immutable reconstruction policy. Remove constructing its own quality assessor. Add target `execute(request, context)` as the service entry for direct reconstruct/edit_transcript and an explicitly supported deletion operation. Keep shared internal rendering helpers, not independent implementations per transport.

Move direct/edit-transcript orchestration from the old orchestrator. Direct reconstruct uses provided timed changes with a valid timeline. Edit-transcript obtains a source-revision-matching transcript/alignment or runs independent transcription, computes a validated diff, then calls the same synthesizer. Honour `same_speaker=false`; do not force true. Define an explicit no-change result for a no-op edit rather than inventing a synthesis failure.

Use the submitted splice-source revision and a separately authorised speaker-reference source. Do not silently replace current splice audio with the immutable root. Require explicit coordinate mapping/cumulative original-coordinate edits when duration changed; never infer timestamps from a matching URL. Validate overlapping/out-of-range changes before model work.

Remove RegenerationPreview/SessionLocal persistence, encryption helpers, `_commit`, `_download_to_temp` and duplicate download/quality paths. Assess local rendered output before upload, rather than downloading the candidate again. Replace the `quality_metrics={passed: True, error: ...}` exception branch with explicit failed/unavailable assessment; a hard validation failure cannot be a passing preview.

Move durable `create_preview`, `confirm_preview`, `rollback_preview`, `get_preview`, preview expiry and approval ownership to backend AI/source services. A compatibility route must delegate to that owner; it must not keep a second AI preview table. Replace the debug-only `_broadcast_event` with actual AttemptContext progress reporting for execution stages. Backend alone publishes durable user notifications.

`confirm_preview` must cease downloading and resynthesizing the original. Produce the complete candidate once during execution, or assemble from exact already-approved segments without generating new speech. Backend confirmation applies that exact verified candidate with source-revision CAS. Do not mark confirmed before the canonical transaction succeeds.

**Done when:** T06/T07/T22/T30–T33 pass, including approval with Fish Speech offline.

## A25. Speech synthesis — `services/reconstruction/synthesizer.py`

**Action: keep synthesis/splicing logic, remove hidden infrastructure. Packages P04–P06.**

Change SpeechSynthesizer constructor to require async model client, transcription/alignment collaborator, TTSPostProcessor, shared NoiseReducer only if needed, AudioIO and immutable synthesis policy. Delete `_transcriber_instance`, `_get_transcriber`, `get_model_client`, module FileHandlers and no-argument construction. Remove pseudo `load()` flags that merely read configuration or print an obsolete HTTP status; model readiness comes from the injected capability.

Keep `reconstruct_segments`, `generate_preview` and `remove_segment` where they provide different required render outputs, but funnel replacement generation through one implementation. Change them to return typed local render artifacts/segment timing. Remove storage uploads, storage-key ownership, global temp selection and direct B2 dependencies from synthesis; JobExecutor publishes once.

Retain source-speaker reference selection, bounded reference duration, pitch-preserving pace correction, internal pauses, safe tempo limits and deterministic seed behaviour. Do not alter these algorithms during dependency extraction. Make source revision, edits and intended deterministic controls explicit in tests; do not accidentally change voice/pace on identical retry. LLM text preparation must not rewrite user content; reconstruction must still work with LLM unavailable using its supported deterministic preparation path.

Read only required source/reference intervals for synthesis. Cap generated text and segment duration/token budget. For full rebuilt tracks, stream unchanged spans and replacement segments into a file-backed output; do not load/concatenate the entire recording in memory. Preserve sample rate/channels and record the edit time map plus actual output duration. Specify codec tolerance for unchanged spans rather than claiming bit-identical MP3 after a necessary re-encode.

Keep model calls awaited; do not hold a process-global GPU lock while waiting for a remote actor. Run expensive CPU postprocessing in a bounded local executor. Respect cancellation at reference extraction, each synthesis request and splice/encode stage.

**Done when:** T22/T25–T28/T30–T32 pass with repeated and cumulative edits.

## A26. Reconstruction support — every remaining file

**Action: retain cohesive pure processing and move durable ownership out. Packages P04–P06.**

| File | Exact direction | Acceptance |
|---|---|---|
| `diff.py` | Keep compute_edit_segments, edit_segments_to_changes and punctuation/mishearing helpers as deterministic text/timestamp transformations. Use explicit timeline/source revision; no service lookups or I/O. Define deletion/no-op/overlap semantics once in schema and tests. | Word alignment, punctuation-only edits, deletion, insertion and changed-duration cumulative fixtures. |
| `audio_buffer.py` | Keep a bounded audio/segment buffer type only if it has distinct value. Add shape/sample-rate validation and explicit maximum size. Do not use it as a full-recording accumulator; merge into an existing result type if it is only a duplicate container. | T25 and bounded reference/segment tests. |
| `tts_post_processor.py` | Keep TTSPostProcessor with explicit policy/device. Consolidate the common synthesis postprocessing sequence here; no downloads, uploads, credentials, DB or model-service globals. Preserve pacing/pitch/loudness and pause regressions. | Quiet/short speech, long replacement, leading silence, source mismatch and no unexpected clipping. |
| `voice_profile.py` | Make reference/profile extraction explicit and scoped to authorised source revision. Bound cache size/lifetime; key by backend/owner/source/model/policy, not bare track ID. Remove hidden global storage selection. Let backend own durable profile metadata where persistence is required. | No cross-tenant reuse; revision change invalidates cached profile; no unsupported whole-file loads. |
| `quality.py` | Constructor-inject metric providers/policy. Keep RegenerationQualityAssessor purely assessing supplied bounded/local render data. Distinguish assessed failure from metric unavailable; never promote exception to pass. | T07 and candidate-quality fixtures. |
| `dnsmos.py` | Keep a model/metric adapter only. Load its native model under an explicit declared model-worker owner or a separate controlled metric runtime, not a hidden CUDA allocation in the CPU gateway. Use bounded windows, explicit local artifacts and unavailable status. | Missing metric weights do not falsify quality; no uncontrolled runtime download. |
| `__init__.py` | Export types only; no model load, logger FileHandler or global instance. | T01/T20. |

## A27. Fish Speech — `deployments/fish_speech.py`

**Action: retain the model owner; make inference bounded and honest. Packages P04, P05.**

Inject Fish model paths, device/dtype/quantisation and text/reference limits. Load configured codec and weights once in the owning replica. Reconcile the configured codec path with the path actually opened; do not expose a setting ignored by implementation. Keep loading offline and governed by enabled capability.

Run the synchronous inference engine via a bounded replica-owned worker so gRPC/Ray lifecycle stays responsive. Keep serial native access initially. Validate reference clip length/bytes, input text and supported language options. Keep deterministic seed/reference propagation; never silently ignore a public control without a contract decision.

Check engine result status and require valid nonempty finite audio for successful generated speech. If no final audio arrives, return a typed failure rather than encoding an empty array as success. Cap returned audio size and split lengthy requests through synthesis service segment limits.

Implement explicit lifecycle teardown for the engine, queue/worker thread and decoder; destructor remains only a fallback. Mark fatal CUDA/engine faults unhealthy and let Serve replace the replica. Do not add a business retry loop or reload on every call.

**Done when:** T22/T27–T29/T43 pass with a missing model, empty engine result, cancellation and actual voice-reference tests.

## A28. Small models and LLM — `deployments/language_models.py`

**Action: retain two model roles, not uncontrolled inference in async methods. Packages P04, P05.**

For SmallModelsDeployment, constructor-load only enabled toxic/sentiment/NLI and reviewed trained-classifier artifacts. Give each declared model explicit device/memory policy. Expose bounded text-window inference and correct label mapping; do not evaluate only the first arbitrary 512 characters of an unbounded transcript and present it as full-content coverage.

For LLMDeployment, honour enablement before binding/loading. Inject model path, token/context limits, engine configuration and concurrency. Keep generation in its actual model runtime, not the caller event loop. Implement real supported batched generation with attention masks/output mapping and bounded total tokens, or expose honest sequential processing; do not name a serial loop a throughput optimisation.

Limit request queues and native concurrency. Enforce input/context/token budgets before GPU work. Handle malformed output at the typed service boundary and fatal model failures at replica health. Do not use global `device_map=auto` assumptions to exceed assigned resources. Correct missing/unused imports/type names during linting without asserting they all cause a runtime exception.

Any vLLM switch is confined to LLMDeployment and its locked dependencies/configuration; keep the service contract unchanged, verify the selected quantisation/model on hardware and retain one retry owner. Do not couple the success of Magic Clean or direct reconstruction to that switch.

**Done when:** T20/T23/T28/T29/T43 pass with late-transcript content, mixed token lengths and bounded batching.

## A29. Categorization — `services/categorization/service.py`

**Action: add an explicit constructor; remove global policy mutation and SQL. Packages P04, P08.**

Constructor dependencies: async model client, optional LLMService, immutable taxonomy/policy snapshot and trained-classifier adapter/handle. Move global `category_loader`, discovery taxonomy and `get_llm_service`/`get_model_client` access to those collaborators. Remove module-level logging/warning configuration.

Keep `categorize` as one cohesive processing entry. Retain useful keyword, shortlist, merge, normalisation, blocked-keyword and editorial-rule helpers inside the file. Do not split every `_apply_*` method into a separate class/file. Use async model calls directly; only substantial pure CPU work goes to the bounded executor.

Replace `category_loader.add_tag`, `ensure_labels` persistence and `_log_auto_example` SQL with typed label/training proposals returned to backend. No request changes a shared taxonomy used by another user. Resolve platform settings once per job revision, not with hidden per-stage network/global reads.

Process long text using token-aware bounded windows/aggregation and an explicit candidate budget. Preserve full-context intent, stable ranking, maximum tag/category counts and relevant test cases. Reuse scores from the same model/input revision rather than running equivalent NLI calls repeatedly. Keep per-track results separate before multi-track aggregation; preserve identity when merging.

**Done when:** T20/T23/T24/T38 pass; categorization is reusable without opening an AI database or mutating a singleton.

## A30. Discovery — `services/categorization/discovery.py`

**Action: keep metadata generation, remove catalogue ownership. Packages P04, P08.**

Retain the existing discovery generation service and result-building helpers. Add constructor-injected model/LLM client, taxonomy reader and immutable discovery policy. Remove `get_discovery_service` singleton access from orchestrator/callers when the service is injected.

Keep transcript-to-discovery metadata, descriptions and search phrase generation. Bound prompts/context, validate result shape, preserve content identity and taxonomy revision. Return proposed metadata; do not save catalogue rows, reindex the backend or query all AI jobs here. Remove duplicated tag/category inference when a valid same-request result is available.

Move list/latest/trending queries to the backend's indexed catalogue. Do not delete the generation service simply because catalogue reads move out. A missing optional LLM must use an explicitly tested fallback or unavailable capability, not fabricated descriptions.

**Done when:** T23/T24/T38 and discovery output fixtures pass with no hidden DB/global service access.

## A31. Moderation — `services/moderation/service.py`

**Action: inject dependencies and preserve safety policy while making evaluation complete. Packages P04, P05, P08.**

Constructor dependencies: model client, optional LLMService, immutable backend-specific keyword/moderation policy and trained harm model adapter. Remove `harm_keyword_loader` request mutation and `get_*` service locators. Route trained classifier inference through a declared model owner; do not load another GPU model inside a supposedly CPU-only helper.

Keep severity/intent/keyword rules and their thresholds unchanged during extraction. Separate keyword evaluation, local model evaluation and fallback decision within this cohesive service. Do not convert every branch into a class. Process all required transcript windows; preserve source offsets/evidence for flagged content. Do not miss harmful content merely because it appears beyond the initial model text slice.

Replace automatic `_learn_phrases` persistence with a backend-owned reviewed proposal/event. Do not silently teach one backend's policy from another backend's audio. Keep auto-learning opt-in semantics and version updates durable.

Model unavailable, malformed prediction and timeout are evaluation errors or explicit partial assessment, never implicitly “safe”. Empty/non-speech handling remains an explicit product policy. Do not alter publication policy or bypass flagging as a performance shortcut.

**Done when:** T23/T24/T28/T38 plus late-transcript flagging, quoted/context-sensitive and model-unavailable fixtures pass.

## A32. Transport services and protobuf — `services/transport/grpc.py`, `services/transport/operations.py`, `proto/`

**Action: keep one translation/auth layer and remove misplaced business operations. Packages P03, P04, P06, P08.**

In `grpc.py`, constructor-inject execution handles/adapter, health reader and registry rather than constructing `Operations()` and reading SQL. Keep job-to-payload mapping and ownership validation during v1 drain. Add typed ExecuteAttempt mapping with explicit presence rules and public/internal field separation. Consolidate error translation once; do not silently drop identity fields or convert all failures to empty structs.

In `operations.py`, move preview CRUD/approval/expiry to backend AI/source services; move catalogue listing to backend catalogue; move training example ingestion and policy mutation to backend-owned operations. Keep direct moderate/categorize only as stateless delegates to injected services and a supported policy context. Replace unconditional healthy output with core/health. Remove `_training_tasks`, SQL reads/writes and per-gateway RegenerationService ownership after caller migration. Delete Operations if no distinct active responsibility remains; do not retain a pass-through class only to preserve the filename.

Extend `proto/pipeline.proto` additively for new execution envelopes/stream results and explicit optional measurement fields. Preserve field numbers and current oneofs during compatibility; reserve removed field names/numbers rather than reuse them. Regenerate `pipeline_pb2.py`, `pipeline_pb2.pyi` and `pipeline_pb2_grpc.py` together with the same build toolchain and verify schema equivalence against backend generated descriptors. Do not edit generated Python manually or regenerate it at runtime.

Remove `resolver.proto`, `resolver_pb2.py` and `resolver_pb2_grpc.py` only after backend resolver callers migrate. Keep legacy job RPCs until records/callers drain; V2 GetResult is backend-owned rather than an AI database query. `services/transport/__init__.py` and `proto/__init__.py` must remain import-safe.

**Done when:** T02/T06/T14/T19/T30/T38/T39 pass and no transport layer owns a second service implementation.

## A33. Training and learned inference — every `training` file

**Action: retain supported training, relocate its data authority and scheduling. Packages P04, P08.**

| File | Required direction |
|---|---|
| `categorizer_infer.py` | Keep trained-classifier loading/prediction behind an injected, revision-pinned model owner. Remove global invalidation coupling with SQL-mutating requests. Load once per actual model replica; cache within a bounded lifecycle. |
| `harm_infer.py` | Apply the same scoped model ownership. Expose typed prediction/unavailable status; no hidden GPU allocation inside moderation's CPU process. |
| `categorizer_train.py` | Accept a versioned dataset artifact, target and output prefix; remove SessionLocal/AI SQL reading. Run through the existing controlled Ray training mechanism under backend-owned low-priority job/approval ownership, not one background task in every gateway. Publish candidate model/metrics/manifest; backend approves/pins a revision. |
| `migrate_catalog_to_db.py` | Replace its AI-DB destination with an explicit one-time export/import migration to backend-owned catalogue data. Keep migration history outside production imports; remove obsolete script after retained data is reconciled. |
| `seed_from_catalog.py` | Make seeding an operator/backend dataset-building step with explicit inputs and idempotency. Do not import it during API startup or create training examples on every category request. |
| `seed_harm_examples.py` | Preserve reviewed examples as controlled data inputs with version/owner. Move backend persistence ownership, keep idempotent seeding and prevent uncontrolled live-policy updates. |
| `__init__.py` | No training start, model load, engine or global cache construction on import. |

Preserve the feature before deleting SQL packages: dataset creation, training request, execution, artifact publication, approval and inference loading must all have named owners. Limit training capacity so it cannot starve interactive audio jobs. Do not change model policy automatically on every new example.

**Done when:** T23/T24/T38/T40 plus an explicit seed → train → review → pin → infer integration test pass.

## A34. Resolver and speed removal — `hear/resolver/`, `deployments/resolver.py`, resolver proto, speed references

**Action: remove these responsibilities from AI only. Package P08.**

Remove AI resolver deployment/entry-point registration, resolver package, its dedicated tests/configuration/proto generation and runtime dependencies after proving backend resolver callers are active. Remove stale project description/docs referring to a combined AI/resolver server. Do not delete backend resolver classes or public Alexa functionality.

Remove AI speed-generation task code and its imports/dispatch/settings/proto fields through the compatibility gate. `scripts/generate_alexa_playback.py` is an identified script to inspect: classify pre-recorded playback prompt generation separately from generating per-track speed layers. Move supported one-time prompt generation to controlled tooling or retain it there; do not blindly delete required Alexa prompt assets because of its name.

Reserve removed protocol fields. Preserve required source delivery encoding and reconstruction pitch/tempo adjustment. Move per-track speed rendering to the established backend/CPU renderer, keeping 1x source reuse, current URLs and canonical-source revision invalidation. Do not regenerate speeds for transient preview artifacts or trigger speed work from waveform completion.

**Done when:** T39/T40 pass and no AI runtime resolver/speed caller remains. Delete only identified obsolete code, not every function containing the word speed.

## A35. Scripts, cleanup tools and operational documentation

**Action: one repeatable operating path. Packages P00, P08, P09.**

| File/section | Direct action |
|---|---|
| `scripts/download_models_ray.py` | Keep explicit operator/build provisioning. Pin exact artifacts/digests, fail partial snapshots and align with profile manifests. Never call it from main, constructors or inference. |
| `scripts/runpod-workspace-env.sh` | Keep deterministic persistent cache/workspace configuration; do not install/update packages implicitly. Align paths with Settings and actual mounts. |
| `scripts/postgres-env.sh`, `scripts/sync-postgres-password.sh` | Retain only for legacy drain/operator migration. Remove from final AI runtime instructions and repository after AI SQL ownership is gone; do not alter backend database secrets. |
| `scripts/live_test.py`, `scripts/live_regeneration_local_test.py`, `scripts/smoke_test.py` | Consolidate repeated transport setup, point to supported versioned endpoints, require explicit nonproduction/test data and actual expected assertions. No embedded service/storage credentials or automatic production destructive actions. |
| `scripts/generate_alexa_playback.py` | Apply the explicit prompt-versus-speed classification in A34, then preserve/move only the supported tooling responsibility. |
| `tools/clean_temp.py` | Use the same workspace policy as A10, not independent recursive deletion rules. |
| `README.md` | Replace stale orchestration/default counts with actual final responsibilities and startup commands. Document legacy drain separately, not contradictory parallel instructions. |
| `docs/BACKEND_INTEGRATION.md`, `docs/PER_BACKEND_JOB_INTEGRATION.md`, `docs/GRPC.md` | Document one shared versioned contract: ownership, identity, lease, error classes, manifests, replay and exact preview approval. Remove contradictions about callbacks/health or which side stores jobs. |
| `docs/AUDIO_JOBS_BACKEND_RUNBOOK.md` | Preserve historical incident as dated evidence; mark existing checkpoint validation fixed. Add tested TTL, missing-model, grant refresh, downtime and result-reconciliation runbooks. |
| `docs/MAGIC_CLEAN_PROFESSIONAL_GRADE_PLAN.md`, `docs/SYSTEM_IMPROVEMENT_REPORT.md` | Mark old architectural proposals superseded where they conflict. Preserve useful DSP evidence and acceptance fixtures, not two active implementation authorities. |
| `hear/__init__.py`, all service/package `__init__.py`, deployment `.gitignore` | Export symbols or ignore true generated files only. Remove eager construction, compatibility re-exports with no callers and wildcard import side effects. |

## A36. Tests and deletion gate

**Action: extend real regression tests, do not replace them with source-string checks. Packages P00–P09.**

Add focused test modules for constructor injection/import safety, protocol/identity, executor lifecycle, bounded audio I/O, per-family model adapters, Magic Clean validation/controls, reconstruction timeline/approval contract, pipeline policy, health/drain and removed dependency boundaries. Reuse existing fixture/test modules where their scope matches; the suggested topics do not require one file per assertion.

Keep current golden audio and public result fixtures before changing DSP. Use fakes for external model/storage/control boundaries in unit tests, then run real Ray multi-replica, GPU, FFmpeg and scoped-storage integration tests. A fake-model success is not model validation. Test longest-supported recordings and mixed traffic rather than only short clips.

Before removing AI persistence, list every remaining reference to AiJob, AiTrackJob, RegenerationPreview, CategoryTrainingExample, SessionLocal, init_db, commit_with_retry, storage_for_job and encrypted storage contexts. Resolve every live owner; no broad exception that hides the missing DB. Also scan global get/set model/LLM/discovery clients, unbounded file-byte loads, resolver/speed registrations, hardcoded FileHandlers and detached task registries.

Require the implementation agent to report actual changed paths, removed paths, constructors/callers migrated, package gates passed, exact tests run and blockers. Do not report “everything working” because imports or mocked tests pass. Complete T01–T44 across both repositories.

## Baseline source references

Directions above are proposed changes. Key current-code observations are supported by these pinned sources:

- [Entrypoint](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/main.py), [config](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/config.py), [graph](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/deployments/app.py), [gateway](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/deployments/gateway.py).
- [Orchestrator](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/orchestrator.py), [submission](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/services/jobs/submission.py), [model client](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/services/model_client.py).
- [Magic Clean service](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/services/magic_clean/service.py), [pipeline](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/services/magic_clean/pipeline.py), [model deployment](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/deployments/magic_clean.py).
- [Reconstruction service including current confirm/assessment paths](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/services/reconstruction/service.py), [synthesizer](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/services/reconstruction/synthesizer.py).
- [Transcription service](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/services/transcription/service.py), [model deployment](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/deployments/transcription.py), [Fish Speech](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/deployments/fish_speech.py), [language models](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/deployments/language_models.py).
- [Categorization](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/services/categorization/service.py), [moderation](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/services/moderation/service.py), [transport operations](https://github.com/Techta-Labs-Ltd/hear-ai/blob/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/services/transport/operations.py).
- Inventory basis: [services](https://github.com/Techta-Labs-Ltd/hear-ai/tree/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/services), [training](https://github.com/Techta-Labs-Ltd/hear-ai/tree/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/hear/training), [scripts](https://github.com/Techta-Labs-Ltd/hear-ai/tree/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/scripts), [patches](https://github.com/Techta-Labs-Ltd/hear-ai/tree/a89ee0ff9b231351034d612b7c4dfa6f8e42b091/patches).
