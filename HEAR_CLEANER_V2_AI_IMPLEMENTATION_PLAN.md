# HEAR Cleaner v2 — Hear-AI implementation plan

**Owner:** `hear-ai`  
**Prepared:** 21 September 2026  
**Status:** Standalone implementation specification; not deployed or benchmark-certified.  
**Basis:** Existing combined implementation plan and implementation pack, separated by team.

Read this document as the task plan for its named repository. Shared interfaces are included here so implementation does not depend on the combined document. Coordinate changes to shared definitions across all three repositories. Do not overwrite unrelated work; reconcile the reviewed baseline with current approved implementation branches.

## Contents

1. [Ownership and fixed decisions](#section-1)
2. [Current runtime findings and invariants to preserve](#section-2)
3. [Profile semantics and effective processing settings](#section-3)
4. [Backend-to-AI execution, manifests and recovery](#section-4)
5. [Internal versioning and public integration boundary](#section-5)
6. [Existing-file migration map](#section-6)
7. [Target classes, engine adapters and long-form execution](#section-7)
8. [Old-model and legacy-runtime removal manifest](#section-8)
9. [A40 memory, concurrency and host resource plan](#section-9)
10. [Mastering, validation and audio-quality gates](#section-10)
11. [Execution security and storage handoff](#section-11)
12. [Metrics, readiness and operations](#section-12)
13. [Tests and AI acceptance criteria](#section-13)
14. [Ordered delivery, dependencies and coding-agent instruction](#section-14)
15. [AI-owned and shared backlog](#section-15)
16. [Source baseline and references](#section-16)

---

<a id="section-1"></a>

## 1. Ownership and fixed decisions

### This document owns

Replace the old cleaner runtime with DeepFilterNet3, an audio-only SAM Audio Small adapter and conservative CPU noise-profile processing. Implement the typed executor, model loading, bounded long-form processing, cancellable subprocesses, quality checks, lossless/delivery artifacts and A40 certification. Remove old cleaner models and their executable wiring after the replacement is accepted.

### This document does not own

Do not build profile-selection UI, decide publication state, approve candidates, create a second durable business-job database or own backend retry/cleanup policy. AI reports attempt results; the backend decides whether a result is current and applicable. Do not delete models or loading patches required by unrelated AI features.

### Required handoffs

| From/to | Contract |
|---|---|
| Backend → AI | Authenticated attempt identity/fence, pinned input, resolved CleanPlan, scoped grants and deadline. |
| AI → Backend | Capability/readiness manifest, compact attempt progress and immutable terminal manifest reference. |
| AI + Operations → Backend | Measured A40 limit evidence, approved runtime/checkpoint/precision/long-form digests and supported input limits. |
| Backend → Frontend | Public job/review projection and media access. AI has no direct browser integration. |


### Final cleaner lineup

| Profile | Engine | User intent |
|---|---|---|
| Natural | DeepFilterNet3 | Clean speech while prioritising the original voice. |
| Voice Focus | SAM Audio Small, audio-only adapter | Isolate speech from distracting sound and music. |
| Music & Atmosphere | Conservative CPU noise-profile DSP | Reduce steady hiss/hum while retaining intended music and atmosphere. |

Do not add MossFormer, ClearVoice, Demucs, SAM Base/Large or another enhancement-model fallback to the completed v2 cleaner. Preserve models required by transcription, alignment, moderation, discovery and reconstruction. CPU analysis support is not another cleaning engine.

### Ownership boundary

Frontend handles settings, comparison and user decisions. Backend owns the profile catalogue, permissions, immutable source/revision identity, durable jobs, attempts, result ingestion, candidate approval and cleanup. Hear-AI executes an authenticated, fully resolved attempt and produces immutable artifacts. Only backend approval can change the track's active audio.

The browser never calls Hear-AI directly and never receives worker or object-store credentials. The backend remains the durable job authority. Neither a UI store nor a GPU worker is a second business-job database.

### Non-negotiable integration rules

Sample previews and full candidates are separate jobs. Samples always have `can_apply=false`. Processing, rejection, cancellation and failure must not replace or unpublish current audio. A full candidate is applied only against its captured expected active revision. Shared contracts are versioned, and no component may silently choose a different profile or default.

The aggregate cleaner device-memory release budget is **12,000,000,000 bytes** on the A40, with normal operation targeted below 10,000,000,000 bytes. This is a certification requirement, not an existing measurement. Frontend sees availability; backend dispatches certified capabilities; AI/operations implement and measure the budget.

---

<a id="section-2"></a>

## 2. Current runtime findings and invariants to preserve

### Hear-AI still has old-model and business-persistence wiring

The cleaner contains MossFormer/Demucs adapters and a stem-mix pipeline. `pyproject.toml` installs `clearvoice`, `demucs`, SQLAlchemy and a PostgreSQL driver. Model requirements also exist in settings, startup and provisioning. These must be removed or migrated at all their call sites, not just from one class. [R7–R12]

`NoiseReducer` is also used by reconstruction. Preserve the required shared functionality or move it with tests before deleting the old module. Do not remove it by filename alone. [R13]

### Existing audio-integrity work should survive

Keep the useful guarantees already implemented: bounded file processing, immutable source hashes, timeline/shape validation, finite-sample checks, delivered-file validation, cancellation propagation and lineage protection. Port their tests to the new engine contracts. Retiring model-specific tests does not justify dropping these invariants. [R10]

---

<a id="section-3"></a>

## 3. Profile semantics and effective processing settings

### Three profiles with explicit semantics

| ID | Display label | Copy | Engine | Default |
|---|---|---|---|---|
| `natural` | Natural | Reduce background noise while keeping your voice sounding natural. | DeepFilterNet3 | Standard; loudness adjustment on; pauses unchanged |
| `voice_focus` | Voice Focus | Isolate speech and remove distracting background sound, including music. | SAM Audio Small | Fixed, validated speech-isolation preset; loudness adjustment on; pauses unchanged |
| `music_atmosphere` | Music & Atmosphere | Reduce steady hiss and hum while keeping music and background atmosphere. | CPU noise-profile DSP | Light; loudness adjustment off; no pause shortening |

These are content-preservation choices, not quality tiers. Do not label Voice Focus as universally better, premium or studio quality. Backend capability status determines availability.

### Controls

| Control | Natural | Voice Focus | Music & Atmosphere |
|---|---|---|---|
| Noise reduction | Light / Standard / Strong | No arbitrary strength slider in v2 | Light / Standard |
| Initial proposed numeric settings | attenuation limit 12 / 18 / 24 dB | one fixed operation | reduction 3 / 6 dB |
| Match comparison loudness | On | On | On |
| Adjust output loudness | On by default | On by default | Off by default |
| Shorten long pauses | Off; only available after edit-map tests pass | Off; only available after edit-map tests pass | Unsupported |
| Noise-only reference section | Not required | Not required | User-selected or explicitly validated reference |
| Channel policy | Preserve source | Explicit mono output policy | Preserve source |

Numeric presets are **initial engineering defaults to calibrate before release**. They are not measured suppression promises. Once approved, freeze them in a versioned backend profile catalogue. Changes require a new profile version/digest.

DeepFilterNet's existing `atten_lim_db` parameter supports an attenuation-limit control. Its zero value does not implement an identity bypass; use an explicit bypass operation when needed. The function also resets recurrent state, which matters for long-form processing. [E2]

Do not expose reverb removal, de-clipping, voice replacement, arbitrary target-speaker selection or free-text SAM prompts in this release. They are separate capabilities and are not implied by selecting a cleaner profile.

### Voice Focus channel policy

Use a tested backend-controlled speech-target prompt. Version its wording and embeddings. Default intent is all wanted speech, not only the loudest speaker. This is a release requirement to test, not something a prompt guarantees.

The initial SAM adapter returns mono speech. Mono source audio is supported directly. For stereo input, require acknowledgement of mono output and preflight the channel layout. Correlated dual mono may be processed according to a pinned channel policy. Anti-phase audio, spatial mixes and split-speaker channels must not be blindly averaged. Reject unsupported layouts or require a separate, explicit channel-edit operation. Do not discard one channel silently. [E1]

Natural and Music & Atmosphere must preserve channel count and test stereo image, polarity and per-channel speech retention. Independent channel processing is not automatically stereo-safe.

### Music/atmosphere noise references

The selected noise-only interval is referenced against the **actual input revision** being processed. Validate bounds and warn when there is likely speech or musical content. A negative speech-activity result is not proof of noise-only audio. Do not treat the first 0.5 seconds or the quietest segment as noise unconditionally.

For the first release, require explicit confirmation of a chosen noise sample when automatic analysis cannot confidently support it. A validated `afftdn` configuration can capture a noise profile and control reduction and smoothing; linked noise-floor handling does not itself guarantee identical channel processing. [E3]

Keep reduction conservative; do not advertise arbitrary overlapping-event removal for this profile.

### Scope of the first release

Full cleaned candidates replace a whole recording revision. The selection handles initially choose a **sample-preview interval**, not permission to replace a full track with a short clip. Partial-range destructive application is out of scope for this cutover; it needs separate splice, channel-layout and edit-map tests.

---

<a id="section-4"></a>

## 4. Backend-to-AI execution, manifests and recovery

### Transport stays outside the cleaner

Use the existing authenticated gRPC transport for the current Pod path. Add a provider adapter behind a shared `AiExecutionClient` boundary. Do not put gRPC calls inside a model adapter or require browser connectivity for a job to survive.

For future serverless execution, `RunPodExecutionClient` submits the same attempt envelope to a queue-based endpoint. Its handler calls `CleanExecutor.execute()` and returns the same manifest reference. RunPod documents handler-based execution and warns that higher concurrency needs memory testing; keep cleaner concurrency at one. [E6]

### Attempt envelope

Required internal fields:

```text
contract_version, backend_id/tenant_scope
job_id, attempt_id, fence, provider
purpose: sample_preview | full_candidate
input: revision/media/object-version/checksum/sample metadata
expected_active_audio_revision
plan: profile/version, engine/runtime/checkpoint digest, effective options,
      channel policy, sample/long-form policy, deterministic seed policy
artifact_prefix, manifest destination, scoped storage grants
execution deadline, heartbeat/lease policy
correlation_id
```

Credentials are passed only through an authorised worker channel. Persist their references separately from semantic job identity. The backend signs or otherwise authenticates execution tickets; workers cannot invent tenant scope, destinations or a newer attempt fence.

### Acceptance and durability

The backend commits a job and dispatch intent before returning 202. A background dispatcher leases work and submits an attempt. Ray acknowledgement means an executor accepted work, not that a business job became durable in Ray.

If the transport fails after possible acceptance, reconcile the same attempt using its provider ID and expected manifest location. Do not immediately create another job with new IDs. Repeated execution remains possible under failure: enforce idempotent effects and stale-attempt fencing, not a false exactly-once-inference promise.

### Result bundle

Upload artifacts under an immutable attempt-specific prefix. Then upload `manifest.json` **last**, containing checksums, sizes, source identity, plan/runtime identity, warnings, metrics and artifact references. This final marker denotes a complete bundle. Multiple object uploads are not an atomic transaction; the manifest-last convention makes partial uploads distinguishable.

Recommended artifact roles:

- `cleaned_master`: lossless FLAC of the mastered candidate.
- `delivery_audio`: MP3 for normal playback.
- `comparison_source`: short authorised preview source or its signed media reference.
- `edit_map`: only when timing changes; otherwise explicit identity mapping metadata.
- `validation_report`: integrity/perceptual-warning evidence, no invented certainty.
- `manifest`: bundle identity and provenance.

Do not retain full target/residual debug audio for every user by default. It increases storage and can contain speech believed removed by the user. Keep short-lived diagnostic artifacts only under an explicit internal policy.

### Result ingestion and CPU protection

AI sends compact progress and one terminal manifest reference. Do not stream huge PCM arrays, waveforms, word arrays or complete JSON results on every progress event.

On a terminal event, authenticate and perform bounded schema checks, persist an inbox record and acknowledge. A bounded worker verifies the manifest/artifact metadata and registers the candidate. It does not need to run in the gRPC receive coroutine.

Reuse existing `EventJournal`/stream infrastructure. Limit routine progress to at most one update per second per active job, coalescing unchanged values. Terminal and state-transition events must be durable and never dropped. Release values are budgets to measure, not justification for losing business events.

Frontend event projection includes job ID, active attempt ID, state version, purpose, stage, progress and candidate/review status. It never includes object-store credentials or raw model stack traces.

### Failure and recovery rules

| Failure | Required result |
|---|---|
| Browser closes | Job continues; reopening hydrates backend state. |
| Redis queue data is lost | Backend dispatcher/reconciler redelivers durable eligible jobs. |
| AI Pod crashes before manifest | Backend retries a fenced attempt against the immutable source. |
| Crash after upload but before notification | Reconciler finds/verifies the complete expected manifest. |
| Duplicate/stale result arrives | Inbox deduplicates; inactive fence cannot create/apply an active candidate. |
| Cancellation races completion | Database state wins; a cancelled/inactive attempt cannot become applicable. |
| Storage grant expires | Refresh through the backend at authorised boundaries, or fail cleanly; never substitute a different source. |
| Source changes | Existing recording remains active; candidate becomes stale. |
| Validation fails | Candidate cannot be applied; original remains untouched. |
| GPU budget exceeded | Fail/recycle worker, report typed error; never raise the cap or switch models silently. |
| Node disk budget exceeded | Reject before decoding where possible; terminate bounded writers and clean attempt-local files. |

### No PostgreSQL in Hear-AI

Move cleaner job ownership, lineage lookups, retry decisions and cleanup tombstone ownership into the backend. Hear-AI keeps only ephemeral execution state and model caches.

The current repository still contains AI-side PostgreSQL dependencies. First move all remaining shared job-runtime consumers behind backend-issued attempts, preserving other job types. Then remove AI SQLAlchemy/driver dependencies and database bootstrap from the new runtime. Do not delete the database module while unrelated workflows still import it. The cleaner-specific image can exclude it earlier through dependency isolation, but the shared project cutover is complete only when remaining consumers are migrated. [R7, R8]

---

<a id="section-5"></a>

## 5. Internal versioning and public integration boundary

These are proposed v2 contracts from the implementation pack, not live service responses. UUIDs, hashes and revision numbers are illustrative. These shared definitions must remain identical across the frontend, backend and AI documents.

### Internal execution and manifest versioning

The internal attempt contract contains secrets/grants and must not be embedded wholesale in public ProcessingJobResponse. Public projection is allowlisted, not a direct serialisation of callback_payload/result_metadata.

Use one canonical protobuf definition and regenerate AI/backend stubs together. Add messages/fields using previously unused field numbers. Do not renumber old fields or manually edit generated Python/TypeScript. Mark deprecated legacy fields, then reserve their numbers/names when removed from the canonical writer contract. Capability negotiation must reject an unsupported v2 execution instead of silently dropping unknown profile options and applying old defaults.

Internal manifest verification must establish source, output object identity/checksums, accepted attempt, plan/runtime version, scope and deadline. A manifest hash authenticates integrity relative to the trusted expected hash; by itself it does not prove authorisation or perceptual quality.

### What the worker reports, and what it must not report

Workers report job/attempt/fence, purpose, processing stage and terminal artifact identity through the authenticated backend transport. Backend owns state_version, review decisions and the public SSE projection. The worker must never emit an authoritative `clean.applied` event or choose a new track revision.

Execution states: `queued`, `dispatched`, `processing`, `validating`, `succeeded`, `failed`, `cancelled`.

Review states: `not_applicable`, `pending`, `applied`, `rejected`, `stale`, `expired`.

A sample with successful execution has review=`not_applicable`, can_apply=false. A valid full candidate has review=`pending` and may be applied. A job can have completed inference while waiting for human review; it should not keep a GPU lease or an execution timeout active.

Maintain a legacy `status` projection for existing clients during transition: full pending candidates map to `awaiting_approval`; approved jobs map to `completed`; samples have a typed purpose and are excluded from the old pending-review list. Do not make `awaiting_approval` a provider execution state.

---

<a id="section-6"></a>

## 6. Existing-file migration map

### Existing AI files and runtime surfaces

| Existing location | Replacement action |
|---|---|
| `hear/services/magic_clean/service.py` | Replace self-constructed old model chain with dependency-injected CleanExecutor. No durable job ownership. |
| `hear/services/magic_clean/pipeline.py` | Retire old stem-routing/residual-mixing branches; leave one authoritative executor. |
| `hear/services/magic_clean/models.py` | Move v2 plans/results/errors to contracts; remove old ContentMode/StemLevels defaults from execution. |
| `hear/services/magic_clean/streaming.py` | Retain bounded processing invariants; use selected engine sessions and cancellable subprocesses. |
| `hear/services/magic_clean/processing/mossformer.py` | Delete after replacement/invariant tests pass. |
| `hear/services/magic_clean/processing/stems.py` | Delete Demucs cleaner adapter. |
| `hear/services/magic_clean/processing/quality.py` | Replace heuristic quality-score contract with measured availability and explicit gate outcomes. |
| `hear/services/magic_clean/processing/validation.py` | Retain and adapt PCM/codec integrity checks and their tests. |
| `hear/services/magic_clean/processing/silence.py` | Move optional pause editing into edit-map-producing implementation; disabled until certified. |
| `hear/services/magic_clean/lineage.py` | Move business lineage resolution to backend; reuse pure hashing/verification without DB coupling. |
| `hear/services/magic_clean/cleanup.py` | Move durable tombstone ownership to backend; retain only attempt-local cleanup responsibilities in AI. |
| `hear/deployments/magic_clean.py` | Thin Pod/Ray adapter calling the shared executor. |
| `hear/deployments/app.py`, `hear/orchestrator.py` | Rewire cleaner construction/attempt execution without damaging retained workflows. |
| `hear/services/jobs/submission.py` | Migrate cleaner admission/default/business persistence to backend-issued attempts. |
| `hear/proto/pipeline.proto` and generated modules | Add v2 messages and regenerate canonical stubs with backend; never edit generated code manually. |
| `hear/config.py`, `main.py`, `.env.example` | Replace old model settings/startup requirements with capability-specific pinned configuration. |
| `hear/tools/model_provisioning.py` | Stage only approved cleaner assets and verify checksums; remove obsolete provisioning. |
| `pyproject.toml` and package lock | Isolate/pin cleaner dependencies; remove old packages only after reverse-consumer checks. |
| `hear/core/noise.py` | Preserve or migrate reconstruction consumer before any shared-file deletion. |
| Existing cleaner/runtime tests | Port invariant coverage; remove assumptions tied to the obsolete model chain. |

The package trees later in this document show proposed new target locations. Existing-file actions are carried forward from the reviewed plan, not a fresh repository audit. Rebase against current implementation branches before editing. [R7–R13]

---

<a id="section-7"></a>

## 7. Target classes, engine adapters and long-form execution

Keep the cleaner in its existing service package. The proposed structure is intentionally smaller than a general-purpose audio framework.

```text
hear/services/magic_clean/
  contracts.py                 typed plans/results/errors, no transport
  service.py                   CleanExecutor
  inspection.py                source and channel/resource inspection
  artifacts.py                 immutable bundle writing
  quality.py                   integrity and wanted-content gates
  mastering.py                 one encode/master path
  streaming.py                 bounded processing coordination
  pause_editing.py              optional edit map and pause renderer
  engines/
    base.py                    CleanEngine protocol, EngineSession
    deepfilter.py              DeepFilterNet3 only
    sam_audio.py               SAM Small adapter only
    noise_profile.py           CPU DSP only
  processing/
    audio_io.py                validated decode/read/resample helpers
    validation.py              retained low-level PCM/codec invariants

hear/runtime/cleaner/           NEW target runtime package
  factory.py                   constructor wiring
  config.py                    budgets and pinned runtime descriptors
  model_registry.py            only allowed cleaner engines
  resource_guard.py            device/host/scratch budgets
  subprocesses.py              cancellable bounded FFmpeg processes
  longform_sam.py              SAM-specific long-form inference

hear/deployments/magic_clean.py EXISTING Pod/Ray adapter, becomes thin
hear/entrypoints/cleaner_serverless.py  NEW future provider adapter
```

Do not create duplicate audio I/O or subprocess utilities when an existing general utility already satisfies the contract; move/reuse it with callers and tests. Replace old processing modules incrementally, leaving only the new authoritative route at cutover.

### Constructor responsibilities

| Class | Injected dependencies | Owns |
|---|---|---|
| `CleanExecutor` | inspector, engine registry, streaming executor, quality gate, pause editor, masterer, artifact writer | One complete attempt, not durable business scheduling |
| `EngineRegistry` | immutable runtime/model descriptors, loaders, device/resource guard | Load only approved engines; report capabilities |
| `DeepFilterEngine` | pinned model, resampler/state factory | Natural inference contract |
| `SamSpeechEngine` | pinned audio-only model, fixed prompt features, long-form policy | Speech target extraction contract |
| `NoiseProfileEngine` | validated filter builder and subprocess runner | Conservative CPU noise-profile operation |
| `AudioQualityGate` | signal validators and optional existing CPU speech-activity analyser | Integrity and risk classification |
| `AudioMasteringService` | subprocess runner, output policy | One final mastering/encoding owner |
| `ArtifactWriter` | scoped storage interface and checksums | Artifacts then manifest, never track approval |
| `ExecutionContext` | attempt/fence, cancellation token, deadline, budget, progress sink, workspace | Attempt-local runtime state |

Use typed `Protocol` interfaces where interchangeable implementations actually exist. Avoid `Any`, service-locator globals and constructor-driven network/database side effects. Dataclasses are sufficient for immutable value objects; do not turn every helper into a class.

`CleanExecutor.execute(plan, context) -> CleanResultManifest` is shared by Pod and Serverless adapters. It receives a fully resolved plan and never asks a database which profile to choose.

### Processing sequence

`validate ticket → verify source → preflight resources → bounded decode/inspect → selected engine → align/validate PCM → optional pause edit → master once → encode → validate exact encoded artifact → upload artifacts → upload manifest → report result`.

A failure at any stage must not be converted into a success containing untouched audio labelled as cleaned. An explicit `no_change` outcome may return the original with a clear reason; it is not an exception fallback.

### DeepFilterNet3 details

- Pin `DeepFilterNet3` explicitly; do not rely on a package's changing default model.
- Preserve model-rate requirements and account for analysis/synthesis delay. Decode at float precision, resample through a single approved path and validate the source/output time relationship.
- Keep recurrent/filter/STFT state per job and channel layout. The convenience `enhance()` function resets state, so implement validated contextual blocks or continuous state handling, not independent tiny calls. [E2]
- Bounded block sizes and CPU/disk output assembly. Never move an entire multi-hour waveform to CUDA.
- Explicit attenuation limit values, no additional default postfilter, no MossFormer or Demucs stage.
- Test zero/bypass, short clips, tails, channel asymmetry and already-clean speech.

### SAM Small audio-only adapter

Use `facebook/sam-audio-small` only. Pin checkpoint and source/runtime revisions. SAM includes required audio codec and conditioning components plus optional capabilities; reducing memory requires inspecting their loading behaviour, not just setting an inference boolean. [R14, E1]

Implementation requirements:

1. Construct an audio-only runtime with required separator, audio codec and text-conditioning behaviour intact.
2. Do not instantiate optional text/visual rankers or the automatic span predictor for this route.
3. Avoid loading unused vision weights to GPU; retain equivalent no-video dimensions/conditioning. Do not globally call `.cuda()` on a full multimodal bundle and then delete modules after the peak allocation occurred.
4. Encode approved fixed prompts on CPU or in a separately bounded stage and cache embeddings by prompt/model/precision digest. Preserve required T5 conditioning; no zero-vector substitute.
5. Stage weights from CPU; use a validated precision policy. Keep numerically sensitive components in FP32 where required. Do not assume whole-model BF16 or quantisation is quality-neutral.
6. Use `reranking_candidates=1`, no automatic span prediction, batch=1.
7. Loader key exceptions must be narrowly allowlisted for intentionally excluded optional modules. Never use blanket `strict=False` to conceal missing core weights.
8. Verify that the audio-only adapter matches the corresponding full runtime's text-only outputs within a defined tolerance on fixed-noise fixtures before accepting memory optimisations.
9. Target and residual are model outputs, not guaranteed exact algebraic complements. Do not create three independent sliders or assume they sum exactly to the source. [R15]
10. Fail typed on no target, hallucinated target, invalid samples or suspicious speech loss. No automatic extra denoiser.

A VAD already required by the project may remain as CPU analysis support. It is not a third cleaner engine. Do not load Whisper, Qwen, SAM Judge or another large model into the cleaner process for every quality check.

### Long-form SAM execution

Initial engineering candidate: 10-second model windows with 2-second overlap, batch=1. These are not published or certified memory/quality numbers. Release only the window/precision configuration that passes A40 and boundary tests.

Do not independently generate random overlapping windows and concatenate them. Implement a SAM-aware long-form method. When using multi-diffusion-style execution, keep the shared latent state on CPU/memory-mapped storage, process bounded overlapping windows at each solver evaluation, accumulate and normalise predictions, and update the global state only after the whole step. Do not update one window in place and let the next see a different solver time.

Pin solver settings and window-grid origin. Bound audio-feature encoding and target decoding as well as the separator. Avoid a hidden full-file codec allocation defeating the window budget.

The inspected public `separate()` method performs generation on its input batch and does not expose a complete drop-in multi-hour executor. Building and validating the long-form wrapper is a release task, not already-completed upstream integration. [R15]

Checkpointing is optional for the first reliable release: a crash may rerun the attempt from immutable input. Only resume when source/plan/runtime checksums match and the checkpoint includes all needed latent/solver/RNG/window state. A half-written waveform is not sufficient to resume a generative process correctly.

### Pause editing

Keep disabled by default. Initial proposed rule: shorten only clearly non-speech gaps longer than 1.5 seconds toward 0.6 seconds, with protected source/output speech boundaries, conservative padding and short non-speech joins. These thresholds require listening calibration.

Produce a sample-index-based edit map. Normal segments have exact affine source/output mappings. Crossfades identify both source regions and gain envelopes. Never claim a single offset covers a recording with several removed gaps.

No pause cutting when wanted-content evidence is uncertain or when profile=music_atmosphere. Do not discard breaths, unvoiced consonants or dramatic pauses based solely on energy. No automatic removal on an entirely silent/noise-only input.

---

<a id="section-8"></a>

## 8. Old-model and legacy-runtime removal manifest

### Delete from the completed v2 cleaner

| Item | Required removal |
|---|---|
| `hear/services/magic_clean/processing/mossformer.py` | Delete after v2 engine contracts and invariant tests replace callers. |
| `hear/services/magic_clean/processing/stems.py` | Delete old Demucs cleaner adapter. |
| Old `pipeline.py` | Delete or replace entirely with the new executor; no dormant old branch. |
| `ContentMode`, `StemLevels`, `DEFAULT_STEM_LEVELS` in old cleaner models | Remove from v2. Preserve a separate temporary old-result reader only where necessary. |
| Old fixed EQ/de-esser/compressor chain | Remove from cleaner execution and retire unreferenced modules; do not keep a second mastering implementation. |
| `clearvoice` and `demucs` dependencies | Remove from new cleaner dependency set and root project when reverse-call audit confirms no retained consumer. |
| MossFormer/Demucs settings | Remove `MOSSFORMER_MODEL_PATH`, `DEMUCS_MODEL`, `DEMUCS_MODEL_PATH` from config/startup/environment/docs. |
| `hear/tools/model_provisioning.py` entries | Remove obsolete checkpoint provisioning; add checksum-verified DeepFilterNet3/SAM Small assets. |
| `/models/mossformer2-se-48k`, `/models/demucs` in deployment | Remove from new image/active volume after old attempts drain and reference verification. Never wipe the entire models volume. |
| Legacy submit mapping | Remove new-job support after consumer cutover; reject old bodies explicitly. |
| Heuristic `quality_score` as release proof | Remove from new UI and v2 quality contract. Legacy readers may display historic values as legacy, not recompute them. |
| AI business PostgreSQL dependency | Remove after job-runtime migration for all shared consumers. Backend PostgreSQL stays. |

### Do not add

No SAM Base/Large, Resemble Enhance, second general-purpose denoiser, runtime model-choice menu, always-loaded SAM Judge/CLAP/ImageBind, automatic chain of denoisers, or hidden ClearVoice/Demucs compatibility engine.

### Preserve

Transcription/alignment models and patches; moderation/categorisation/discovery models; Fish Speech or other currently required reconstruction engines; existing CPU analysis required for safety; audio codecs/resampling; B2 integration; source hashing and lineage rules; reliable job/transport contracts for non-cleaner operations.

The current dependency file contains Whisper/Qwen-related components and a shared NumPy constraint attributed to the old runtime. Removing ClearVoice is **not** permission to upgrade NumPy/Torch/Transformers across the entire application without regression tests. Separate a cleaner dependency group/image and pin a compatible runtime. [R7]

### Removal acceptance checks

Scan source, startup, deployment templates, provisioning, environment files, package locks and documentation. After deletion, no v2 executable path may import or provision ClearVoice, MossFormer or Demucs. Archived historical documentation and old-result schema names may remain explicitly labelled.

Build the new image with the old model directories absent and external model downloads disabled; Natural and Music & Atmosphere must start independently of SAM access, while Voice Focus advertises unavailable until its pinned assets are present and certified. No missing cleaner model should crash transcription/reconstruction workers.

Do not count a comment saying “removed” as removal. Check the built image dependency inventory and actual runtime imports.

---

<a id="section-9"></a>

## 9. A40 memory, concurrency and host resource plan

### Budget definition

Use decimal bytes to honour the stated cap without accidentally treating 12 GiB as 12 GB:

```text
cleaner_gpu_limit_bytes       = 12_000_000_000
cleaner_gpu_operating_target  = 10_000_000_000
initial_torch_allocator_cap  =  9_000_000_000
cleaner_gpu_concurrency      = 1
sam_batch_size               = 1
sam_reranking_candidates     = 1
```

12,000,000,000 bytes is about 11.18 GiB. Report raw bytes plus the displayed unit in telemetry. Count all cleaner processes, loaded models, CUDA contexts, reserved allocations and non-Torch workspaces. Do not double-count Torch allocated+reserved; allocated is included in reserved. Use a whole-process device measurement separately from framework measurements.

Ray GPU fractions are scheduling units, not hard VRAM partitions. PyTorch's per-process fraction limits its caching allocator rather than every possible GPU allocation. Monitoring alone can observe an overshoot only after it happens. Do not call any of these a proved host-level hard limit. [E4, E5]

Configure an allocator cap before CUDA model allocation, a supported platform-level memory control when available, and certified bounded workloads with headroom. If the hosting environment cannot enforce a hard physical partition, state that limitation operationally and require test-certified admission beneath the cap. Do not promise a universal hard partition on an unverified Pod host.

### Deployment shape

One cleaner inference actor/process per A40 initially. One active GPU job at a time across Natural and Voice Focus, including previews. Music & Atmosphere executes on a bounded CPU lane and does not allocate CUDA memory. CPU analysis/mastering/encoding use a small separate worker pool with explicit thread limits.

Reuse the same Pod where appropriate; this plan does not require buying another GPU. A cleaner-only image/process boundary isolates dependencies and startup without changing backend ownership.

Maintain an aggregate device admission budget for other AI services. A local cleaner semaphore prevents cleaner-cleaner contention, not contention from transcription/TTS actors. Do not run two models simply because their Ray fractions add to less than one when their measured peak memory does not fit the device.

During deployment, drain or stop the old cleaner before loading a second copy on the same card. Blue/green cleaner overlap must also count toward the total 12 GB cleaner budget. Rollback swaps images, not loads both engines concurrently.

### Loading and OOM policy

Lazy-load only the selected engine. A small DeepFilter model may remain cached alongside SAM only when the combined measured budget permits; otherwise offload/release it before loading SAM. Startup checks must include load-time peaks, not just inference peaks.

On a budget/OOM fault, mark the attempt failed with a typed code, terminate or recycle the affected worker when needed, and release its lease. One infrastructure retry is allowed only when an actual transient resource/leak issue is resolved. Never retry indefinitely, enlarge the budget, silently reduce quality, quantise automatically, or substitute another profile.

If SAM Small with validated optimisations does not pass the cap, Voice Focus remains unavailable. Optimise and recertify that adapter; do not quietly deploy a >12 GB configuration. This is the budget gate, not a claim of current benchmark success.

### Host resources and file size

Bound downloads by authorised size, enforce byte and wall-clock limits during decode, restrict supported mono/stereo layouts and validate actual decoded length. A small compressed file can expand substantially.

Compute scratch reservation from `sample_rate * channels * seconds * 4` for each floating-point PCM copy, plus outputs, model/runtime overhead, validation and safety margin. Account for all concurrent CPU jobs. Prefer streamed scans and reuse decoded PCM where provenance is preserved instead of repeatedly decoding whole files.

Publish the certified maximum duration/bytes in backend capabilities. Include at least hour-long soak tests and the intended maximum duration; do not adopt a marketing duration limit until its scratch, memory and execution budget has passed. Queue delay and credential lifetime are separate from audio duration.

---

<a id="section-10"></a>

## 10. Mastering, validation and audio-quality gates

### One authoritative master

Initial output policy:

- Float PCM processing; retain a lossless 24-bit FLAC candidate master after mastering, recording bit depth and any dither policy.
- 128 kbps mono MP3 for mono voice; 192 kbps stereo MP3 for stereo/mixed material, pending the listening gate.
- With speech loudness adjustment enabled: target -19 LUFS mono / -16 LUFS stereo, bounded +6 dB gain initially.
- Music & Atmosphere loudness adjustment off by default; preserve musical dynamics.
- Limit and validate delivered true peak at or below -1 dBTP; allow bounded codec-correction re-encoding from the lossless intermediate, never from a previous MP3.
- Record source rate, internal model rate, delivery rate and exact timebase. Upsampling a low-bandwidth input is not evidence of recovered detail.

These are product targets to validate. Report achieved values when safe gain/peak constraints prevent the nominal loudness. Do not compress heavily merely to display a target number. Digital silence has unavailable loudness; use null+reason rather than a fake measured -99 value in v2.

### Hard integrity gates

Source checksum; authorised source revision; complete decode; finite values; expected format/layout; exact PCM timebase without pause edits; declared channel conversion; no missing tails/duplicated chunks; valid codec decode; delivered duration within the codec's justified allowance; clipping/peak policy; coherent artifact checksum/manifest; no result from a stale attempt.

Hard-invalid results are never applicable. Human approval is not a bypass for malformed or corrupted artifacts.

### Wanted-content risk gates

Use source/output speech/activity evidence, per-channel checks, aligned energy/spectral comparisons and boundary discontinuity detection to flag suspicious loss. Calibrate on real recordings. ASR agreement, VAD activity and no-reference quality metrics are evidence, not a proof of all-word preservation.

For Natural, disappearing wanted speech is unacceptable. For Voice Focus, intended background suppression is expected; compare protection to target-speech evidence, not to all source energy. For Music & Atmosphere, measure image/transient changes and inspect removed content for unwanted musical loss.

Use outcomes `passed`, `review_required`, `rejected`, `not_applicable`, with supporting metrics and warning intervals. A reviewed candidate can be applied only if hard integrity passes and no blocking policy violation remains.

### Evaluation outside the cleaner runtime

Build a versioned, permissioned test corpus with clean references where available, noisy real-world recordings, silence/noise-only inputs, music/voice mixtures, UK accents, quiet/older speakers and multi-speaker recordings. Do not treat the largest suppression amount as the best quality.

Run objective/reference tests where applicable, plus blinded loudness-matched listening. Evaluate full hard cases, not only handpicked previews. Keep heavy evaluation models outside the production cleaner process and memory budget; do not introduce another denoiser as a fallback.

---

<a id="section-11"></a>

## 11. Execution security and storage handoff

Authorise every job, sample, candidate, apply/reject and media-signing request. Check group/organisation ownership at both submission and review. Untrusted user text never becomes a model path, shell command, ffmpeg expression or storage prefix.

Resolve media from backend records. Internal downloads validate scheme/host/redirect destinations and reject metadata/internal-address access unless an explicitly authorised storage route needs it. Use argument arrays, not shell interpolation. Restrict codecs, timeouts, subprocess output capture and upload size.

Persist immutable source/media identifiers and checksums; issue short-lived read/write grants at dispatch. Refresh grants through backend authorisation for long jobs. The old “exactly 24 hours remaining” admission pattern is not the new contract: queue and execution timing should not make an otherwise authorised job depend on a one-day frontend credential.

Store candidates under attempt-specific keys; keep incomplete uploads isolated. Cleanup is owned by the backend and uses service-owned authorisation, not expired job grants. Mark-and-sweep/delete only unreferenced objects. Preserve active media, approved history, cleaner roots and pending-review objects.

Initial candidate retention proposal: sample previews 24 hours, unreviewed full candidates 7 days; backend capabilities expose the actual expiry. Existing application retention requirements take precedence. Accepted outputs follow track history policy. Rejecting a candidate does not delete a source/root artifact still referenced by another revision.

Do not log signed query strings, storage secrets, private audio paths or transcript content by default. Use correlation IDs and redacted diagnostic metadata.

---

<a id="section-12"></a>

## 12. Metrics, readiness and operations

Record queue time separately from model loading, decoding, inference, mastering, validation and upload. Include job/attempt/fence, profile and model digests, input duration/channels, outcome and warning counts.

Track total GPU process memory, Torch allocated/reserved peaks, CPU/thread utilisation, host RSS, scratch occupancy, processing-time/audio-time ratio, queue depth, OOMs, cancelled attempts, model-load failures, stale results, apply conflicts, duplicate events, validation rejects, reviewed speech-loss warnings and original-restore frequency.

Readiness must be per capability: Natural may be ready while SAM is unavailable. Publish profile health to the backend; keep frontend “temporarily unavailable” accurate. Do not fail all AI health merely because an optional cleaner profile cannot load.

Use the existing operational-notification flow with severity: HIGH when cleaning cannot execute safely; MEDIUM for degraded/unavailable optional profile; LOW for verified recovery. Group incidents rather than emailing per failed chunk.

---

<a id="section-13"></a>

## 13. Tests and AI acceptance criteria

### Engine and audio tests

- Zero input, near-silence, noise-only, tiny clip, malformed input, NaN/Inf, extreme gain and corrupted decode.
- 8/16/22.05/24/32/44.1/48/96 kHz input coverage where certified; mono/stereo; unsupported multichannel rejection.
- Quiet words, breaths, unvoiced consonants, accents, simultaneous speakers, similar background voice, singing and applause.
- Fan/hiss/hum, keyboard, dogs, traffic, wind, handling noise, clipping and dropouts with honest limitations.
- Clean input transparency, source loss flags, SAM no-target and hallucination safeguards.
- Word at every window boundary, arbitrary final partial window, no duplicated or lost samples, no unexplained beginning/end fades.
- Anti-phase/dual-mono/split-speaker cases; no hidden channel discard.
- Noise sample contains speech/music; reference selection out of bounds; no automatic first-frame capture.
- Loudness unavailable on silence/short clips, post-codec true peak, no repeated lossy clean.
- Silence edit map, crossfades, caption mapping and no accidental trim when disabled.

### A40 and deployment tests

Run on the actual supported A40 image with real weights: cold load, hot load, maximum supported windows, target codec decode, repeated short jobs, longest supported recording, model switching, cancellation, failed allocation and worker recycling. Include other permitted AI workloads and deployment replacement; measure aggregate cleaner memory.

Release only if all certified workloads remain under the 12 GB cap with the intended enforcement/admission strategy. Record measured peak bytes, runtime versions, image digest, checkpoints, precision policy and fixtures. The spec itself is not that evidence.

### Regression tests outside cleaning

Transcription and timestamps; Qwen/Whisper loading patches; reconstruction/voice synthesis; moderation; categorisation; discovery; initial upload and publish; waveform/speed generation; deletion/history; backend SSE clients. Delete no shared dependency until its retained consumers pass.

### AI acceptance checklist

- [ ] Only DeepFilterNet3, SAM Audio Small and CPU noise-profile processing are cleaner engines.
- [ ] Required codec/text conditioning remains intact; optional SAM omissions are narrowly validated.
- [ ] Core state/output checks fail typed; no silent old-model or cross-profile fallback.
- [ ] Long-form codec, separator and decoding all remain bounded.
- [ ] Delay, partial tails, channels and timeline checks pass for each certified input layout/rate.
- [ ] Exact encoded outputs and complete manifest bundles are validated.
- [ ] Cancellation/deadline stops child work safely and preserves attempt identity.
- [ ] Source/plan/runtime hashes and fences are reported without owning backend approval state.
- [ ] Actual A40 evidence covers cold load, inference, model switching, failures and deployment overlap below the aggregate 12 GB budget.
- [ ] Old model imports, packages, provisioning and startup requirements are absent from the built v2 image.
- [ ] Shared reconstruction/transcription/loading-patch regressions pass after removals.
- [ ] No claims of professional quality or memory compliance are based solely on mocks or checkpoint file sizes.

---

<a id="section-14"></a>

## 14. Ordered delivery, dependencies and coding-agent instruction

### Ordered AI delivery

1. Agree the authenticated attempt/result contract with backend. Create a cleaner-only dependency boundary and fake engine.
2. Build resource guards, safe decoding, cancellable subprocesses and attempt-local workspaces.
3. Implement DeepFilterNet3, SAM Small audio-only and CPU noise-profile adapters independently.
4. Implement SAM-aware bounded long-form processing, including codec and output decoding, and certify boundary behaviour.
5. Consolidate wanted-content/integrity checks, optional edit maps, one mastering path and manifest-last artifacts.
6. Integrate the thin Pod/Ray adapter, regenerated protobuf and backend grant/result reconciliation.
7. Run real-model listening, A40 memory, long-duration and coexistence tests; advertise only certified capabilities.
8. Coordinate migration of shared AI-side durable job consumers before deleting root-project DB dependencies.
9. Drain old cleaner attempts and physically remove obsolete adapters/packages/provisioners/settings/unused weights.
10. Complete canary and rollback evidence. Add the Serverless adapter later using the same executor and separate provider certification.

### Dependencies delivered by other owners

Backend C01/C03/C06 provide schemas, source/attempt identities and resolved profile plans. C08/C17/C18 provide durable dispatch, scoped grants, artifact ingestion and reconciliation. Backend-led C19 completes shared persistence ownership migration. All-client C24 precedes stopping supported legacy writers. QA C26 must inspect audio, not only numeric metrics.

### Coding-agent handoff

Build one typed executor and keep transports thin. Maintain a deletion ledger naming every old symbol, its consumers and replacement tests. Never delete unrelated ASR/TTS/moderation/discovery models or loading patches. Report actual model/image digests, tests executed, measured memory, listening evidence and unrun gates. If SAM cannot meet the certified cap, report it unavailable instead of deploying above budget or silently substituting a model.

---

<a id="section-15"></a>

## 15. AI-owned and shared backlog

Task IDs and dependencies are retained from the original 30-task backlog. Shared tasks appear in more than one team document but remain one coordinated work item, not duplicated implementations.

| ID | Owner/repository | Work | Depends on | Acceptance evidence |
|---|---|---|---|---|
| C00 | All | Pin implementation commits, inventory imports/API consumers/current jobs, capture baseline fixtures. | None | Baseline ledger with exact refs and protected workflows. |
| C01 | Backend + AI | Freeze profile/purpose/source/attempt/result contract; one canonical protobuf/OpenAPI source. | C00 | Valid/invalid fixtures and generated-type checks in both services. |
| C09 | AI | Create cleaner-only dependency/runtime boundary, factory, contracts and fake engine. | C01 | No backend database import in new cleaner executor; non-cleaner workers unaffected. |
| C10 | AI | Implement `ResourceGuard`, cancellable FFmpeg runner, bounded source read/decoding. | C09 | Cancellation/timeout/disk-limit tests; process group stopped; no unbounded stderr capture. |
| C11 | AI | Implement pinned DeepFilterNet3 adapter with per-job state and delay-correct processing. | C09,C10 | Short/tail/stereo/state tests, real-model smoke and bypass identity. |
| C12 | AI | Implement SAM Small audio-only loader without optional rankers/span model or GPU vision loading. | C09,C10 | Required-weight validation, fixed-input parity check, measured load peaks. |
| C13 | AI | Implement bounded SAM long-form codec/separator/decode route. | C12 | Boundary tests and long-form/peak-memory evidence; no naive independent random joins. |
| C14 | AI | Implement conservative CPU noise-profile adapter and noise-reference validation. | C09,C10 | Speech/music-contaminated reference handling, channel and transient checks. |
| C15 | AI | Consolidate mastering, lossless artifact, delivery encoding and exact-file validation. | C10 | Loudness/codec/duration/finiteness tests with no second master path. |
| C16 | AI | Wanted-content risk checks and optional pause editor/edit map. | C11–C15 | Injected speech-loss failures rejected/flagged; map handles joins and caption positions. |
| C17 | AI + backend | Manifest-last immutable artifacts, grant refresh and result reconciliation. | C08,C15 | Partial upload ignored; crash after manifest recovered; source/artifact hashes checked. |
| C18 | AI + backend | Thin Pod/Ray adapter and regenerated protobuf transport. | C01,C08,C17 | Full end-to-end Pod processing, result registration and cancellation. |
| C19 | Backend | Migrate remaining AI business-job DB ownership to backend across retained job types. | C03,C08,C18 | Transcription/reconstruction/pipeline recovery tests without AI business PostgreSQL. |
| C25 | AI + operations | A40 memory, longest-input, coexistence, model-switch and deployment-drain certification. | C11–C18 | Recorded aggregate memory below 12,000,000,000 bytes on approved workloads. |
| C26 | QA | Blinded matched-loudness evaluation; edge-case and already-clean corpus. | C11–C18 | Reviewed per-profile quality results; hard failures not hidden by average score. |
| C27 | All | Stop old admissions/drain attempts; remove old adapters/packages/config/assets and AI DB dependencies. | C19,C24–C26 | Built-image/runtime inventory proves old cleaner absent; regressions pass. |
| C28 | All | Canary/rollback/restore-original drills, runbooks, archive old plans as superseded. | C27 | Release checklist signed with actual evidence and current image/model digests. |
| C29 | AI + backend | Future serverless adapter using shared executor and provider-pinned attempts. | C18,C25,C28 | Separate cold-start/deadline/retry/concurrency/grant tests on actual Serverless. |

---

<a id="section-16"></a>

## 16. Source baseline and references

Source: `HEAR_CLEANER_V2_FULL_IMPLEMENTATION_PLAN.md` and its accompanying implementation pack, prepared 21 September 2026. This document separates their existing instructions by owner; it is not a new repository audit or benchmark.

| Repository | Reviewed branch | Reviewed commit |
|---|---|---|
| `hear-frontend` | `dev` | `f730521a9871320f2ecc85c900624ee9437d2ae7` |
| `hear-backend` | `main` | `9897ac8b6be9d9f5a11467e290b687a6632537f9` |
| `hear-ai` | `main` | `f50e8dc0e9bc0894f104cc18f34a65853cd98228` |

Reference keys below retain the original plan's labels. External URLs are retained source references, not a fresh verification or a dependency version pin.

- **E1:** Meta SAM Audio repository/model card: `https://github.com/facebookresearch/sam-audio`; `https://huggingface.co/facebook/sam-audio-small`.
- **E2:** DeepFilterNet enhancement implementation: `https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/df/enhance.py`.
- **E3:** FFmpeg noise reduction and encoding filters: `https://ffmpeg.org/ffmpeg-filters.html#afftdn`.
- **E4:** Ray accelerator/fractional-GPU documentation: `https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html`.
- **E5:** PyTorch allocator-limit documentation: `https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.set_per_process_memory_fraction.html`.
- **E6:** RunPod Serverless handler contract: `https://docs.runpod.io/serverless/workers/handler-functions`.
- **R7:** hear-ai at `f50e8dc0e9bc0894f104cc18f34a65853cd98228`, `pyproject.toml`.
- **R8:** same AI commit, `hear/services/jobs/submission.py`; `hear/services/jobs/workflows.py`.
- **R9:** same AI commit, `hear/services/magic_clean/models.py`.
- **R10:** same AI commit, `hear/services/magic_clean/service.py`, `pipeline.py`, `streaming.py`, `processing/validation.py`.
- **R11:** same AI commit, `hear/config.py`, `main.py`, `.env.example`, `hear/tools/model_provisioning.py` model references.
- **R12:** same AI commit, cleaner adapter files `processing/mossformer.py`, `processing/stems.py` and their test references.
- **R13:** same AI commit, `NoiseReducer` references in `hear/core/noise.py`, `hear/services/magic_clean/service.py`, `hear/services/reconstruction/synthesizer.py`.
- **R14:** Meta SAM Audio `sam_audio/model/model.py`, constructor and config; inspected blob `2ac5ddc7d317c1d5f472d3b24209e4a6ccc8eb51`.
- **R15:** same Meta model implementation, `separate()`, codec target/residual decoding and result handling.

Record actually tested image, package and checkpoint revisions in the release manifest. Proposed files, settings and wire shapes are implementation decisions, not assertions that they are already deployed.
