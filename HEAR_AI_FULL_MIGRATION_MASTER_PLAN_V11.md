# HEAR AI — Full Pod + Serverless Migration Master Plan V11

**Audit baseline**
- Hear AI: `Techta-Labs-Ltd/hear-ai`
- Original planning baseline: `d4965c4b1324df7617e71adb688a46d94b4179d0` (**334** tracked files)
- Latest runtime source: `ee11f2afb2e5057deaeee23c502f9a1cde96d730` (`ee11f2a Remove obsolete Magic Clean runtime code`), with **276** tracked files
- Hear Backend integration baseline: current backend branch audited during this planning pass
- Migration register refreshed: 2026-09-26

**Pod transport update — 2026-09-27:** The Pod uses a protected HTTP/SSE ingress backed by a Pod-local durable RabbitMQ role queue. The backend submits the versioned attempt envelope; the Pod queues it, a local worker consumes it when GPU capacity is available, and the API streams queued and canonical execution events. Serverless uses native RunPod dispatch and emits the same canonical events without RabbitMQ. The backend remains the durable source for job state and client-facing reconnectable SSE.

**Magic Clean update — 2026-09-27:** Magic Clean has two production profiles: `natural` for DeepFilterNet3 denoising, including recording hiss and steady unwanted noise, and `sam_audio` for text-prompted separation with official SAM Audio Base. SAM Audio ambient mode generates two candidates and uses the official CLAP text ranker; event mode uses the official PE Audio Frame span predictor. The SAM prompt names one target sound. `action=remove` publishes the residual and `action=isolate` publishes only the target. Isolate attempts that do not contain an audible target fail with `target_not_detected` and publish no candidate artifact. The retired `voice_focus` and `music_atmosphere` profiles are rejected at ingress.

This is the migration blueprint for preserving the current product while removing the old AI-side control plane. It is deliberately stricter than a refactor outline: it defines the final runtime, inference engine policy, model residency, build/image strategy, Pod and Serverless transports, SSE semantics, B2 behavior, migration phases, deletion gates, tests, and an exhaustive file register.

---

## 1. Non-negotiable final state

The final platform has exactly four **logical AI jobs**:

1. `pipeline`
2. `transcription`
3. `reconstruction`
4. `magic_clean`

There may be multiple **execution profiles** for one logical job when that prevents unrelated models from being loaded. Magic Clean routes to `natural` or `sam_audio` while the durable job type remains `magic_clean`.

The final Hear AI worker:
- does not own application PostgreSQL;
- does not own application Redis;
- does not create business retries;
- does not own browser SSE;
- does not run gRPC;
- does not run Ray Serve;
- does not download all models at startup;
- does not load models unrelated to its worker capability;
- does not apply dependency patches on every production cold start;
- does not create or mutate durable preview/job state.

Hear Backend is the durable control plane.

---

## 2. Target architecture

```text
                                Creator frontend
                                      |
                                      v
                             HEAR BACKEND FASTAPI
               +----------------------+----------------------+
               | PostgreSQL                                  |
               | Redis progress + SSE                        |
               | AI attempts/outbox/result inbox             |
               +----------------------+----------------------+
                                      |
                         ExecutionRouter / policy
                          /                       \
                         /                         \
                        v                           v
                 POD PROVIDER                RUNPOD SERVERLESS
              HTTP/SSE -> local RabbitMQ      native /run
                        |                           |
                        v                           v
                 capability worker          streaming handler
                        \                           /
                         \                         /
                          +---- shared JobExecutor-+
                                      |
             +------------------------+-----------------------+
             |                        |                       |
       PipelineWorkflow      TranscriptionWorkflow    ReconstructionWorkflow
                                                             |
                                                       MagicCleanWorkflow
                                      |
                               InferenceManager
                                      |
                    role/profile-specific local engines
                                      |
                         bounded audio + Backblaze B2
```

The backend normalizes both execution providers into the same canonical `ExecutionEvent` and result manifest.

---

## 3. Why FastAPI remains

FastAPI remains in two places.

### Hear Backend

It is the stable application/control API:
- create/query/cancel jobs;
- internal attempt claim;
- Pod event ingestion;
- provider webhook ingestion;
- result acceptance;
- SSE.

### Hear AI Pod

It provides operational control and a streaming attempt endpoint:
- `/healthz`
- `/readyz`
- `/capabilities`
- `/drain`
- `POST /v1/attempts/stream`

The backend sends a versioned `AttemptEnvelope`; the Pod publishes it to a durable local RabbitMQ role queue. A local worker claims it through the backend, executes it when GPU capacity is available, and the Pod API returns queued and canonical events as SSE. The backend persists events and owns reconnectable user-facing SSE. Serverless uses RunPod's native dispatch and emits the same canonical event schema.

### RunPod Serverless

Queue-based Serverless uses the native RunPod handler. Do not start a second HTTP server just to call the same Python code.

---

## 4. Provider behavior

### 4.1 Pod

```text
backend creates durable job and attempt
  -> backend POSTs AttemptEnvelope to Pod /v1/attempts/stream
  -> Pod claims attempt through backend
  -> JobExecutor
  -> canonical ExecutionEvent values stream back as SSE
  -> backend persists events and final outcome
  -> backend serves reconnectable client SSE
```

The HTTP request is the Pod transport. The backend remains the job database and owns retries and reconciliation after a dropped stream.

### 4.2 RunPod Serverless

```text
backend transaction
  -> job + attempt + provider dispatch record
  -> RunPod /run
  -> save provider_job_id
  -> streaming handler calls same JobExecutor
  -> handler yields canonical ExecutionEvent
  -> backend consumes RunPod /stream
  -> same AIEventService -> Redis -> SSE
  -> webhook/status/B2 manifest -> AIOutcomeService
```

Use the platform operations directly:
- `/run` for normal long jobs;
- `/stream` for live incremental results;
- `/status` for reconciliation;
- `/cancel` for cancellation;
- `/health` for provider health;
- webhook as a completion wake-up;
- `/runsync` only for short canaries/diagnostics.

The handler must be a streaming handler and yield bounded JSON events. Set aggregate streaming only if the total emitted event set remains intentionally small.

---

## 5. Canonical job contracts

Every attempt has:

```text
schema_version
job_id
run_id
attempt_id
job_type
operation
track_id
owner/tenant scope
source_revision
source URL/object identity
source hashes where known
options
artifact prefix
reporting/provider context
deadline
```

The worker never trusts a user-supplied arbitrary callback URL or arbitrary object-store prefix.

### Pipeline

Stages:
1. prepare
2. transcribe
3. transcript correction where current behavior requires it
4. moderate/flag
5. categorize/tag
6. discovery
7. canonical delivery encoding
8. manifest/outcome

`audio_tag`, `categorization`, and `discovery` stop being durable job identities. Their behavior is preserved as pipeline capabilities/stages or deliberately migrated call surfaces.

Pipeline workers receive a versioned backend catalog containing categories, tags, keyword rules, moderation keywords, and discovery taxonomy paths. The runtime validates the catalog version and constructs immutable per-worker configuration; it does not load these values from application database globals or silently fall back to stale process state.

### Transcription

Stages:
1. source prepare
2. bounded windows
3. ASR
4. forced alignment / word timestamps
5. offset merge
6. result manifest

### Reconstruction

One durable job with operations:
- `replace_segments`
- `edit_transcript`
- `rebuild`
- `remove_segments`
- preview compute where needed

If reconstruction needs transcript/timestamps and they are absent, the backend should run/require a canonical transcription prerequisite instead of forcing the reconstruction worker to load Qwen ASR alongside Fish Speech.

### Magic Clean

One durable job, multiple execution profiles:
- `natural`
- `sam_audio`
- no stem-separation profile; Demucs and the legacy speech/music/background mixing controls were removed by latest source commit `ee11f2a`

This keeps the product's one Magic Clean job while loading only the engine needed for the selected request.

---

## 6. Inference-engine decision

There is **no universal inference server** in the initial target.

Use constructor-owned local engines behind `InferenceManager`.

### Qwen ASR — production migration engine

**Initial migration:** keep the existing patched WhisperX/Qwen3-ASR + Qwen3 ForcedAligner behavior.

Reason: current transcript/alignment/punctuation/word timestamp behavior is a product contract and is already patched/tested.

**Optimization candidate after migration:** vLLM Qwen3-ASR + vLLM Qwen3 ForcedAligner.

vLLM now supports Qwen3-ASR and the forced-aligner model. It is a serious target, but it becomes production only after golden parity for:
- transcript;
- language;
- word segmentation;
- timestamps;
- punctuation;
- silent/no-content behavior;
- long-audio offsets;
- VRAM;
- cold/warm startup;
- real-time factor.

Do not delete the patch in this migration.

### Pipeline text LLM

Current `QWEN_LLM_ENABLED` is disabled by default. Therefore:
- do not download or load Qwen2.5-7B in normal pipeline workers when disabled;
- if enabled, use vLLM as the preferred engine after output tests;
- create a larger `pipeline-llm` execution image/profile if necessary, without creating a fifth logical job.

### Small models

Toxicity, sentiment, and NLI remain current Transformers implementations for migration safety.

After parity:
- benchmark CPU ONNX Runtime;
- use it if it frees GPU memory without hurting throughput/SLO.

### Fish Speech

Keep the native Fish Speech inference engine.
- Pin source revision.
- Build/install it in the image.
- Load once per reconstruction worker.
- Never clone `main` during worker startup.
- Fail readiness on missing CUDA for a GPU role instead of silently switching to CPU.

### Magic Clean

Use the cleaner-v2 profile registry as the target model-loading policy, but do not declare profile parity until the current product goldens pass.

Target engines:
- Natural -> DeepFilterNet3 only
- SAM Audio -> official SAM Audio Base only

Do not eagerly load unrelated cleaner engines in one process.

After profile certification, delete legacy cleaner model paths that have no remaining product caller.

---

## 7. Model loading: exact worker capabilities

### 7.1 Transcription worker

Required:
- Qwen3-ASR-1.7B
- Qwen3-ForcedAligner-0.6B
- VAD assets actually required by the patched WhisperX path

Not loaded:
- Qwen 7B text LLM
- Fish Speech
- DNSMOS
- MossFormer
- SAM
- DeepFilter cleaner
- classifier models unless this worker also has pipeline capability

### 7.2 Pipeline worker

Required:
- transcription set
- toxic-BERT
- sentiment
- NLI

Optional:
- Qwen2.5-7B only when pipeline LLM feature is enabled

Not loaded:
- Fish Speech
- reconstruction quality stack
- Magic Clean engines

A persistent Pod may advertise capabilities `[pipeline, transcription]`, because pipeline already owns the transcription model set. This is a good configuration for the single Pod you currently have.

### 7.3 Reconstruction worker

Required:
- Fish Speech S2-Pro
- Fish codec
- DNSMOS/quality model

Not loaded:
- Qwen ASR by default
- pipeline classifiers
- Magic Clean models

If transcript/timestamps are missing, use a transcription prerequisite.

### 7.4 Magic Clean Natural

Required:
- DeepFilterNet3 runtime/assets;
- an externally pinned certification-file SHA-256 and an absolute evidence path whose bytes match the certified digest;
- a certification with a measured peak device-memory value;
- device-wide free-memory admission before loading, with a 2 GB safety reserve.

### 7.5 Magic Clean SAM Audio

Required:
- Meta SAM Audio source at commit `bb4c6999d2677c7402360e426afc01ddfad6dce0` through its supported `SAMAudio` and `SAMAudioProcessor` APIs;
- `facebook/sam-audio-base` at revision `81f64008f9f957c2b57a45923fb5299c66e9d186`;
- `google-t5/t5-base` at revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1` with pinned file hashes;
- `lukewys/laion_clap` at revision `b3708341862f581175dba5c356a4ebf74a9b6651` for official text reranking and `facebook/pe-a-frame-large` at revision `40187271298f84e2966d4518c88dde698540c9ad` for official span prediction, with pinned file hashes;
- a required lowercase text prompt naming the target sound, `remove` mapped to the residual output, `isolate` mapped to the target output, `prompt_mode=ambient|event`, and a deterministic seed;
- ambient mode with two CLAP-ranked candidates, event mode with PE span prediction, a 75-second context window with 5-second overlap for longer inputs, one mono GPU profile worker, prompt text bound to its SHA-256, and rejection when the residual collapses active source audio for two continuous seconds;
- mono input runs directly; stereo requires explicit acknowledgement and measured channel correlation of at least `0.98`, then a bounded average-to-mono conversion; reject all other stereo layouts and bind the threshold/conversion policy into runtime identity;
- a deployment certificate with `license_review_approved=true`, approved prompt/profile certification, and current audio quality evidence.
- an externally pinned certification-file SHA-256 and an absolute evidence path whose bytes match the certified digest;
- a certification with a measured peak device-memory value and device-wide free-memory admission before loading, with a 2 GB safety reserve.

The model manifest records the SAM License as `review_required`. Hugging Face access to the SAM checkpoint is gated; provision it before deployment and keep the runtime profile disabled until license review is recorded.

---

## 8. One Pod today, Serverless beside it

The architecture supports your current reality: one warm Pod plus Serverless.

Recommended initial routing:

```text
pipeline       -> existing warm Pod
transcription  -> existing warm Pod
reconstruction -> Serverless
magic_clean    -> Serverless profile endpoint
```

The Pod image is `pipeline` capable, so it loads ASR + pipeline classifiers, not reconstruction or cleaner models.

If the Pod is saturated, backend policy may spill a pipeline/transcription attempt to the matching Serverless endpoint.

Later you can add dedicated Pods without changing job contracts.

Do not create a "load every model" Pod as the permanent design.

---

## 9. Serverless endpoint topology

Logical job types stay four, but endpoints can be execution-profile-specific:

```text
hear-pipeline
hear-transcription
hear-reconstruction
hear-magic-clean-natural
hear-magic-clean-voice-focus
hear-magic-clean-music-atmosphere
```

This is intentional: endpoint specialization is how we avoid cold-starting models that the job does not use.

### Cached-model policy

RunPod currently permits one cached Hugging Face model per endpoint.

Choose the largest/highest cold-start model for that slot:
- transcription -> Qwen3-ASR
- pipeline -> Qwen3-ASR
- reconstruction -> Fish Speech repo if compatible/useful
- Magic Clean profile -> its primary HF model when applicable

Secondary assets use:
- a pinned network volume;
- or the role image if small/stable enough.

Do not put all model repos in every Docker image.

---

## 10. Automatic patch lifecycle

Current assets are retained:

```text
patches/manifest.json
patches/whisperx-asr-qwen.patch
hear/tools/dependency_patches.py
```

The current manifest already pins:
- WhisperX VCS revision;
- original source SHA256;
- patched source SHA256;
- patch SHA256.

### Production build

```text
uv sync --project deploy/runtime --frozen --no-default-groups --group <workload> --group <provider>
python -m hear.tools.dependency_patches
python -m hear.tools.dependency_patches --check
pytest patch/ASR contract tests
build immutable image
```

Apply and verify the dependency patch only for pipeline and transcription images.

### Runtime

Runtime performs `--check` only.

It must never mutate site-packages on normal Pod/Serverless boot.

If the patched digest is wrong:
- fitness/readiness fails;
- the worker accepts zero jobs.

Only transcription/pipeline role images need the WhisperX patch dependency.

---

## 11. Model manifest and provisioning

Replace the current global `MODEL_MANIFEST` with a versioned manifest whose entries contain:

```text
logical_name
hf_repo / source
exact revision/commit
roles/profiles
local logical path
required files
optional hashes
license/provenance metadata
engine adapter
```

The current provisioner downloads every listed model and does not pin each Hugging Face model revision. That is removed.

### Pod model preparation

- persistent model volume;
- `provision_models --role pipeline`;
- download missing exact revisions only;
- verify files/hashes;
- worker startup reads only local files;
- offline mode during normal execution.

### Serverless

- primary model uses RunPod cached model when useful;
- secondary assets on versioned network volume or role image;
- fitness check verifies exact expected snapshot/files;
- no model download inside the per-job handler.

---

## 12. Docker build

Create one Dockerfile with shared layers and role targets.

Target images:

```text
hear-ai-pod-pipeline
hear-ai-pod-transcription
hear-ai-pod-reconstruction
hear-ai-pod-magic-clean-natural
hear-ai-pod-magic-clean-voice-focus
hear-ai-pod-magic-clean-music-atmosphere

hear-ai-serverless-pipeline
hear-ai-serverless-transcription
hear-ai-serverless-reconstruction
hear-ai-serverless-magic-clean-natural
hear-ai-serverless-magic-clean-voice-focus
hear-ai-serverless-magic-clean-music-atmosphere
```

Build from one source tree. Registry layer deduplication keeps common dependency/code layers shared.

### Dependency extras

Keep the runtime dependency contract and lock together in `deploy/runtime/pyproject.toml` and `deploy/runtime/uv.lock`. The repository root `pyproject.toml` contains shared pytest, Ruff, and mypy configuration only. Base project dependencies provide the shared runtime core. Role/provider groups are:

```text
pod
serverless
transcription
pipeline
pipeline-llm
reconstruction
magic-clean-natural
magic-clean-voice-focus
magic-clean-music-atmosphere
dev
```

Do not add a Demucs dependency group; the latest cleaner source explicitly removes that runtime and its model provisioning.

Do not keep Ray, grpcio, SQLAlchemy and psycopg in every final AI worker after their migration gates.

### Do not install in production AI image

- PostgreSQL server/client solely for the old AI DB
- Supervisor for Ray topology
- benchmark evidence
- tests
- documentation
- old protobuf generators
- model downloader used only by deployment preparation

Use `.dockerignore`.

---

## 13. Pod process design

One Pod worker process has:
- FastAPI/Uvicorn API for operational routes and SSE attempt submission;
- a local RabbitMQ connection, durable role queue publisher, and queue consumer;
- one process-scoped backend HTTP client;
- one `JobExecutor`;
- one `InferenceManager`;
- one worker capability/profile;
- bounded native/background executors.

`POST /v1/attempts/stream` accepts the same `AttemptEnvelope` used by Serverless, queues it durably in local RabbitMQ, and streams a queued event followed by canonical `ExecutionEvent` values. RabbitMQ's AMQP listener binds to loopback on the Pod. The backend authenticates the request, persists the streamed events, and owns durable status and browser SSE.

Heavy synchronous model/audio work runs outside the event loop through the owned bounded executor/native path so `/healthz`, `/drain`, lease handling and cancellation stay responsive.

Initial GPU-heavy concurrency: **1** per worker.

The RabbitMQ role queue buffers work while the single GPU worker is busy. The broker is local to the Pod and is not shared with Serverless.

---

## 14. Serverless process design

Module import / worker startup:
1. load config;
2. resolve role/profile;
3. verify patch if applicable;
4. resolve cached/network-volume model paths;
5. construct `InferenceManager`;
6. warm engines;
7. pass RunPod fitness checks.

Handler:

```text
async streaming handler
  -> parse AttemptEnvelope
  -> claim Hear attempt
  -> async for ExecutionEvent in JobExecutor.stream(...)
       -> RunPod progress_update(compact summary)
       -> yield canonical event
  -> final event contains compact manifest
```

`return_aggregate_stream=True` is permitted only while events remain bounded.

Large audio lives in B2, not request JSON.

---

## 15. SSE and progress

### Pod path

```text
JobExecutor
 -> Pod HTTP SSE response
 -> backend PodProgressBridge
 -> AIEventService
 -> AIProgressStore
 -> Redis
 -> SSEPublisher
 -> browser
```

### Serverless path

```text
JobExecutor yields
 -> RunPod /stream
 -> backend RunPodProgressBridge
 -> AIEventService
 -> same AIProgressStore
 -> Redis
 -> same SSEPublisher
 -> browser
```

The frontend has one event contract.

### Backend SSE fixes

Current backend must change to:
1. subscribe Redis;
2. replay durable journal events after `Last-Event-ID`;
3. send current authorized progress snapshot;
4. stream buffered/live Redis;
5. emit keepalive comments.

Do not emit the cached latest event before older journal events.

Do not hold request-scoped SQLAlchemy sessions for the entire SSE connection.

### Progress state

Use attempt-scoped Redis:

```text
ai:progress:{job_id}:{attempt_id}
ai:progress-current:{job_id}
ai:track-current-job:{track_id}
```

Use an atomic Lua/transaction update to reject stale attempt/source/sequence data.

Numerical progress is replaceable. Final result is durable.

---

## 16. Direct Pod HTTP reliability

The backend calls the Pod API only after committing the job and attempt. It bounds active Pod streams and treats the backend claim as the execution fence. A dropped stream stops the worker heartbeat; lease expiration and backend reconciliation decide whether to retry. The Pod keeps GPU concurrency bounded and rejects excess requests with `429 Retry-After`.

The Pod endpoint requires a service bearer key and validates the worker capability before execution. It does not retain a second in-memory job database; reconnectable status and event replay come from the backend's durable event journal and Redis progress state.

---

## 17. Backend dispatch outbox

The backend commits the job and attempt before calling the Pod API or RunPod. A durable dispatch record and reconciliation handle timeouts where a provider accepted a request but the backend did not record the response. Pod HTTP and RunPod dispatch use the same attempt identity and claim fence.

---

## 18. Backblaze B2

Keep B2 as durable artifact storage.

### Fix current double-read behavior

Current `B2Storage.upload_file()` can upload an artifact and then GET the entire object again to recompute SHA256.

Remove that from the normal success path.

Target:
1. hash while local artifact is created or in one local pass;
2. upload with metadata;
3. HEAD object;
4. verify length/metadata/version;
5. persist compact manifest;
6. optional independent audit verification runs separately.

Use boto3 `TransferConfig`/multipart behavior for large files.

Every artifact prefix includes backend/tenant/job/attempt identity and cannot escape the scoped prefix.

---

## 19. Transcription performance

Keep the good existing file-backed `TranscriptionService.transcribe_file()` design and make it the primary long-audio path.

Refactor the old Ray deployment into `QwenAsrEngine`.

Requirements:
- model loaded once;
- forced aligner loaded once;
- no per-job model initialization;
- no full long-audio decode before chunking;
- bounded disk reads;
- tested resampling at window boundaries;
- exact timestamp offset merge;
- no unbounded transcript/event payloads;
- no `torch.cuda.empty_cache()` on every chunk unless profiling proves it helps.

Keep the `<=16 MB` byte-based path only for small reference audio where it is genuinely appropriate.

---

## 20. Reconstruction performance

Current reconstruction logic must preserve output behavior but improve memory.

### Immediate fixes
- instantiate DNSMOS ONNX session once per worker;
- inject it into post-processing;
- avoid re-loading model/session per score;
- reuse Fish Speech engine;
- use existing transcript/timestamps as inputs where possible.

### Timeline renderer

Replace whole-source tensor cloning/repeated splice with:
- disk-backed source;
- ordered original coordinate changes;
- copy unchanged ranges;
- insert generated audio;
- crossfade;
- write output incrementally.

Implement in Python first with SoundFile/FFmpeg, then use C++/pybind11 only if profiling shows the assembly remains material.

---

## 21. Magic Clean consolidation

The repo currently contains both legacy Magic Clean processing and a large cleaner-v2 runtime/engine tree.

Do not ship two permanent cleaners.

Migration:
1. freeze current UI/input/output goldens;
2. certify `natural` and `sam_audio` on real representative audio;
3. remove the retired stem-level/Demucs, fixed Voice Focus, and noise-profile paths;
4. map each profile to one engine worker;
5. move common validation/mastering/artifact code into canonical `hear/services/magic_clean`;
6. remove cleaner-v2 gRPC/wire layer;
7. remove duplicated legacy engine implementations after no caller remains;
8. keep certification tests;
9. move benchmark JSON out of production image.

No profile should cause every cleaner model to load.

---

## 22. Health/readiness

### Pod `/readyz`

Ready only when:
- patch is correct for ASR roles;
- required model revisions/files exist;
- GPU type/VRAM meets role policy;
- eagerly loaded engines are healthy; lazily loaded Magic Clean profiles have verified certificates, assets, exact runtime dependencies, and sufficient GPU headroom before admission;
- scratch capacity available;
- backend claim endpoint configured;
- Pod API bearer key configured;
- worker is not at its active-job limit;
- worker is not draining.

Magic Clean readiness is intentionally non-allocating: the selected profile validates pinned assets and dependencies, and GPU profiles sample live device memory. Model allocation occurs after backend claim when the first authorized attempt loads that profile. Readiness does not claim model warm-up or first-inference success; those remain measured deployment gates.

### RunPod fitness checks

Register:
- GPU/VRAM;
- required model files;
- patch digest;
- FFmpeg/ffprobe;
- scratch;
- engine warm-up;
- required environment.

If check fails, worker must receive no job.

Backend `/health` combines:
- DB;
- Redis realtime/cache;
- Pod worker registrations;
- RunPod `/health`;
- Pod stream admission/rejections;
- current attempt liveness;
- result reconciliation backlog.

Scale-to-zero Serverless with no backlog is not an outage.

---

## 23. Current code that must disappear

Final AI runtime must have no:
- `DatabaseRuntime`;
- AI SQL job models;
- AI fair scheduler;
- AI job recovery loop;
- `PipelineGrpcService`;
- pipeline protobuf;
- Ray `DeploymentHandle`;
- Ray Serve app/gateway;
- `ModelClientRegistry`;
- `serve.start/run`;
- local PostgreSQL scripts;
- Ray download/start scripts;
- gRPC-only docs/tests;
- durable aliases for old job names.

Deletion happens only after the replacement behavior is tested and current callers are moved.

---

## 24. Backend migration required

Create/extend cohesive backend owners:

```text
AIJobService
AIJobRepository
AIExecutionAttempt
AIDispatchOutbox
ExecutionRouter
PodHttpProvider
RunPodServerlessProvider
PodProgressBridge
RunPodProgressBridge
AIEventService
AIProgressStore
AIOutcomeService
AIReconciliationService
```

Do not create a microservice/class for every SQL statement.

### Serverless progress bridge

A bounded async service owns RunPod `/stream` connections.

If active stream count reaches provider/configured limits:
- jobs still execute;
- fall back to `/status` progress polling for overflow;
- completion still comes from webhook/status/B2 manifest.

On backend restart, rebuild bridges for active provider jobs from PostgreSQL.

---

## 25. Provider selection

Backend configuration is per logical job/profile.

Initial proposal for the one existing Pod:

```text
pipeline: pod_preferred
transcription: pod_preferred
reconstruction: serverless
magic_clean.natural: serverless
magic_clean.sam_audio: serverless
```

Policy may spill Pod work to Serverless after:
- Pod capacity full;
- Pod stream admission rejects or Pod draining/degraded;
- Pod draining/degraded.

Provider selection is stored on the attempt so it never changes mid-attempt.

---

## 26. CI/CD

### Pull request CI

Run:
- Ruff/mypy;
- architecture import rules;
- four-job contract tests;
- patch manager unit tests;
- ASR patch golden tests;
- transcription window tests;
- reconstruction tests;
- Magic Clean certified profile tests;
- Docker target build smoke tests;
- no-network runtime verification;
- legacy-import denylist after each deletion gate.

### Image release

Use immutable tags:

```text
hear-ai-pipeline:<git-sha>
hear-ai-transcription:<git-sha>
...
```

Record:
- Git SHA;
- patch SHA;
- model manifest SHA;
- engine revision;
- Docker digest.

Never deploy `latest` as the only production identifier.

---

## 27. Migration phases

### M0 — Freeze current behavior
Capture real golden fixtures/results for all four jobs, old aliases, previews, cancellation, long audio and failures.

### M1 — Four-job contracts
Introduce new contracts and backend compatibility mapping. No execution change.

### M2 — Backend attempts/outbox/outcome/progress
Create the durable control plane before moving workers.

### M3 — Fix SSE
Correct replay ordering, close DB before stream, add attempt-scoped progress store and keepalives.

### M4 — Build/image/model system
Role extras, Docker targets, automatic build patch, model manifest, offline startup.

### M5 — Extract JobExecutor/workflows
Old Ray/gRPC transport calls the new workflow code first.

### M6 — Local inference engines
Extract Qwen, classifiers, Fish Speech and Magic Clean model ownership from Ray wrappers.

### M7 — Pod-local RabbitMQ and HTTP/SSE transcription and pipeline
Run RabbitMQ on the Pod, verify durable role-queue admission and worker consumption, then measure SSE behavior and failure recovery. Serverless stays on native RunPod dispatch while emitting the same canonical events.

### M8 — Move AI durable state to backend
Previews/lineage/cleanup state migrated; remove AI DB.

### M9 — Reconstruction worker
Cut Fish Speech/reconstruction to the new executor.

### M10 — Magic Clean profile workers
Certify and cut profiles one at a time.

### M11 — RunPod Serverless
Deploy role/profile endpoints; enable native `/stream`, `/status`, `/cancel`, webhook and provider health.

### M12 — Remove gRPC
Only when all callers and administrative operations are replaced.

### M13 — Remove Ray Serve
Only when all engines are local and parity is proven.

### M14 — Optimize inference
Benchmark vLLM ASR/forced aligner, vLLM LLM, ONNX small models, native timeline renderer.

### M15 — Final deletion
Delete old scripts/docs/tests/dependencies/aliases and prevent reintroduction with CI.

---

## 28. Rollout order and rollback

Cut over one logical path at a time:
1. standalone transcription;
2. pipeline;
3. reconstruction;
4. each Magic Clean profile.

Each cutover has:
- feature flag/provider routing;
- canary percentage;
- golden comparison;
- error/latency dashboards;
- rollback to old runtime while old code still exists.

Delete old runtime only after a soak period and verified no-caller telemetry.

---

## 29. Required production tests

### Transport/control
- DB commit before Pod HTTP dispatch crash
- Pod HTTP timeout after backend claim
- Pod SSE disconnect and lease expiry
- claim race
- worker crash after claim
- backend restart
- RunPod stream disconnect
- RunPod webhook loss
- RunPod status/result expiry
- provider retry/redelivery
- cancel race

### SSE
- reconnect after missed progress
- durable replay + snapshot ordering
- stale attempt event
- out-of-order sequence
- Redis restart
- slow client
- long connection without SQL session

### Models
- patch missing/wrong revision
- model file missing
- no network startup
- cold and warm loads
- wrong GPU/VRAM
- OOM
- engine unhealthy restart

### Audio
- long mono/stereo
- silence
- speech/music mixtures
- profile goldens
- reconstruction exact intervals
- B2 upload interruption
- local scratch exhaustion
- source revision changes

---

## 30. Performance measurements

Per role/profile record:
- container image size;
- cold boot to fitness start;
- model resolution time;
- model load time;
- warm-up time;
- ready time;
- queue delay;
- time to first progress;
- real-time factor;
- CPU-seconds/audio-minute;
- peak host RAM;
- peak VRAM;
- scratch peak;
- B2 download/upload time;
- result acceptance latency.

Optimize from measurements, not assumptions.

---

## 31. Definition of done

The project is finished only when:
- there are exactly four logical job types;
- the existing Pod runs a capability image without unrelated models;
- Serverless endpoints load only the selected role/profile models;
- current Qwen patch is automatically applied during build and verified on boot;
- no normal job start downloads models;
- RunPod `/stream` drives Serverless live progress;
- Pod progress uses backend event ingestion;
- both normalize to the same SSE contract;
- Redis loss cannot lose final state;
- Pod HTTP/API loss cannot erase a committed job;
- B2 holds durable artifacts/manifests;
- stale/duplicate attempts cannot apply twice;
- AI workers have no application DB;
- gRPC/Ray Serve/old orchestrator are gone;
- cleaner duplicate runtimes are consolidated;
- all golden audio/output tests pass on Pod and Serverless.

---

## 32. New files to add

```text
Dockerfile
.dockerignore
models/manifest.yaml

hear/entrypoints/pod.py
hear/entrypoints/serverless.py
hear/bootstrap.py

hear/contracts/jobs.py
hear/contracts/events.py
hear/contracts/outcomes.py
hear/contracts/errors.py

hear/security/grants.py

hear/api/app.py
hear/api/routers/health.py
hear/api/routers/control.py

hear/runtime/attempt_stream.py
hear/api/routers/jobs.py

hear/execution/context.py
hear/execution/executor.py
hear/execution/reporter.py
hear/execution/native.py
hear/execution/resource_lease.py

hear/workflows/pipeline.py
hear/workflows/transcription.py
hear/workflows/reconstruction.py
hear/workflows/magic_clean.py

hear/inference/manager.py
hear/inference/qwen_asr.py
hear/inference/text_generation.py
hear/inference/small_models.py
hear/inference/fish_speech.py
hear/inference/magic_clean_registry.py
hear/inference/magic_clean_factory.py
hear/runtime/cleaner/path_cleanup.py

hear/audio/io.py
hear/audio/workspace.py
hear/audio/chunking.py
hear/audio/magic_clean_streaming.py

hear/storage/b2.py
hear/health/service.py
hear/compat/dependency_patches.py

scripts/provision_models.py
scripts/validate_image.py
tools/benchmarks/
```

The exact number of modules can be reduced where two small owners are naturally cohesive. The rule is clear ownership, not file proliferation.

---

## 33. Exhaustive current-file migration register

The companion register is [HEAR_AI_FULL_MIGRATION_MASTER_PLAN_V11_REGISTER.csv](HEAR_AI_FULL_MIGRATION_MASTER_PLAN_V11_REGISTER.csv). It has 376 unique paths spanning the 334-file planning baseline, all 276 paths tracked by latest source `ee11f2a`, and four V11 runtime/official SAM additions. The `latest_source_status` column records whether each path remains tracked in that source commit, was removed by it, or was added only in V11. Each entry has a disposition, target, and action.

### File-by-file register

| Current file | Disposition | Target | Action |
|---|---|---|---|
| `.env.example` | **REWRITE** | `.env.example` | Remove retired runtime settings; document role, provider, Pod API key, model roots, profiles, health settings, and the catalog service key. |
| `.gitignore` | **KEEP_UPDATE** | `.gitignore` | Keep; add local model manifests, build outputs, benchmark outputs and container-local scratch exclusions. |
| `deploy/cleaner/deepfilter3.ini` | **DEV_ONLY** | `tools/cleaner or docs/evidence` | Keep only if needed for certified cleaner build/benchmarks; exclude production image. |
| `deploy/cleaner/package-files.json` | **DELETE_AFTER_MERGE** | `deploy/runtime/pyproject.toml + uv.lock` | Merge cleaner dependencies into role-specific runtime groups; remove duplicate package/lock. |
| `deploy/cleaner/pyproject.toml` | **DELETE_AFTER_MERGE** | `deploy/runtime/pyproject.toml + uv.lock` | Merge cleaner dependencies into role-specific runtime groups; remove duplicate package/lock. |
| `deploy/cleaner/sam-small-optional-keys.json` | **REMOVE_PER_LATEST_SOURCE** | `tools/cleaner or docs/evidence` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `deploy/cleaner/uv.lock` | **DELETE_AFTER_MERGE** | `deploy/runtime/pyproject.toml + uv.lock` | Merge cleaner dependencies into role-specific runtime groups; remove duplicate package/lock. |
| `deploy/supervisord.conf` | **DELETE** | `container runtime / orchestrator` | New image runs one clear role; no embedded Ray head + app Supervisor topology. |
| `hear/__init__.py` | **KEEP_MINIMAL** | `hear/__init__.py` | Package marker; remove exports of legacy runtime. |
| `hear/config.py` | **REWRITE** | `hear/config.py` | Typed core settings plus role-specific validation; no DATABASE_URL/RAY/GRPC settings in final AI runtime. |
| `hear/orchestrator.py` | **DELETE_AFTER_EXTRACTION** | `hear/workflows/* + hear/execution/executor.py` | Extract behavior into four workflows and delete the stateful mega-orchestrator. |
| `hear/runtime/__init__.py` | **KEEP_MINIMAL** | `hear/runtime/__init__.py` | Package marker; remove exports of legacy runtime. |
| `hear/services/__init__.py` | **KEEP_MINIMAL** | `hear/services/__init__.py` | Package marker; remove exports of legacy runtime. |
| `hear/services/model_client.py` | **DELETE_AFTER_ENGINE_CUTOVER** | `hear/inference/manager.py` | Replace Ray handles/global registry with constructor-injected local engines. |
| `main.py` | **REPLACE** | `hear/entrypoints/pod.py + hear/entrypoints/serverless.py` | Remove all-in-one Ray/DB/gRPC startup; use explicit Pod and Serverless entrypoints. |
| `patches/manifest.json` | **KEEP_HARDEN** | `patches/manifest.json` | Keep exact WhisperX revision and before/after/patch digests; add build metadata and CI enforcement. |
| `patches/whisperx-asr-qwen.patch` | **KEEP_HARDEN** | `patches/whisperx-asr-qwen.patch` | Mandatory build-time patch until a separately approved vLLM ASR parity migration. |
| `pyproject.toml` | **UPDATE** | `pyproject.toml + deploy/runtime/pyproject.toml` | Keep root test/lint/type configuration only; define role-specific production dependency groups in the runtime project. |
| `uv.lock` | **DELETE** | `deploy/runtime/uv.lock` | Remove the duplicate root lock; CI and images use the single role-specific runtime lock with frozen/locked installs. |
| `.github/workflows/architecture.yml` | **REWRITE** | `.github/workflows/ci.yml` | Replace architecture-only CI with lint, typing, architecture, runtime contract/workflow tests, and builds for all 14 supported image targets. |
| `HEAR_AI_GRPC_PUBLISH_RELIABILITY_PLAN.md` | **DELETE_AFTER_CUTOVER** | `-` | Legacy gRPC architecture documentation; keep in Git history, remove from active docs after HTTP/RunPod cutover. |
| `HEAR_CLEANER_V2_AI_IMPLEMENTATION_PLAN.md` | **ARCHIVE_AFTER_MERGE** | `docs/archive/` | Cleaner-v2 implementation plan becomes historical after certified engines are merged into canonical Magic Clean. |
| `README.md` | **UPDATE** | `README.md` | Rewrite for four jobs, Pod-local RabbitMQ and HTTP/SSE, RunPod native stream/status/cancel, role images, backend SSE and operational runbook. |
| `deploy/cleaner/README.md` | **UPDATE** | `deploy/cleaner/README.md` | Replace retired gRPC and experiment guidance with the canonical V11 profile targets, admission requirements, and certification status. |
| `docs/AUDIO_JOBS_BACKEND_RUNBOOK.md` | **UPDATE** | `docs/AUDIO_JOBS_BACKEND_RUNBOOK.md` | Archive the legacy guide; document current role images, startup, health, scratch handling, and release gates. |
| `docs/BACKEND_INTEGRATION.md` | **UPDATE** | `docs/BACKEND_INTEGRATION.md` | Archive the legacy guide; document current attempt contracts, reporting endpoints, provider paths, and backend-owned durability requirements. |
| `docs/CLASS_OWNERSHIP_AND_PATCHES.md` | **UPDATE** | `docs/CLASS_OWNERSHIP_AND_PATCHES.md` | Archive the legacy guide; document V11 module ownership, dependency groups, and the Qwen/WhisperX patch. |
| `docs/CLEANER_V2_DELIVERY_LEDGER.md` | **UPDATE** | `docs/CLEANER_V2_DELIVERY_LEDGER.md` | Archive the legacy ledger; state each profile's runtime status and remaining certification evidence. |
| `docs/ENVIRONMENTS.md` | **UPDATE** | `docs/ENVIRONMENTS.md` | Archive the legacy guide; document current runtime variables, profile settings, patches, and local setup. |
| `docs/GRPC.md` | **DELETE_AFTER_CUTOVER** | `-` | Legacy gRPC architecture documentation; keep in Git history, remove from active docs after HTTP/RunPod cutover. |
| `docs/MAGIC_CLEAN_PROFESSIONAL_GRADE_PLAN.md` | **UPDATE** | `docs/MAGIC_CLEAN_PROFESSIONAL_GRADE_PLAN.md` | Archive the legacy plan; define profile-specific certification and production release evidence. |
| `docs/PER_BACKEND_JOB_INTEGRATION.md` | **UPDATE** | `docs/PER_BACKEND_JOB_INTEGRATION.md` | Archive the legacy guide; map each canonical job payload to its runtime path and expected result. |
| `docs/SYSTEM_IMPROVEMENT_REPORT.md` | **UPDATE** | `docs/SYSTEM_IMPROVEMENT_REPORT.md` | Archive the legacy report; summarize the V11 runtime, implemented changes, and remaining integration/release gates. |
| `hear/proto/CLEANER_V2.md` | **NOT PRESENT** | `docs/BACKEND_INTEGRATION.md` | This file is absent in the migration checkout; its current attempt and transport contract is documented in the backend integration guide. |
| `deploy/cleaner/evidence/cleaner-wheel-df3-isolation-2026-09-21.json` | **ARCHIVE** | `CI/GitHub release evidence; excluded by .dockerignore` | Preserve useful benchmark evidence outside the production image; remove from hot deployment package. |
| `deploy/cleaner/evidence/cleaner-wheel-isolation-2026-09-21.json` | **ARCHIVE** | `CI/GitHub release evidence; excluded by .dockerignore` | Preserve useful benchmark evidence outside the production image; remove from hot deployment package. |
| `deploy/cleaner/evidence/cleaner-worker-factory-2026-09-21.json` | **ARCHIVE** | `CI/GitHub release evidence; excluded by .dockerignore` | Preserve useful benchmark evidence outside the production image; remove from hot deployment package. |
| `deploy/cleaner/evidence/df3-a40-synthetic-hour-2026-09-21.json` | **ARCHIVE** | `CI/GitHub release evidence; excluded by .dockerignore` | Preserve useful benchmark evidence outside the production image; remove from hot deployment package. |
| `deploy/cleaner/evidence/df3-a40-synthetic-smoke-2026-09-21.json` | **ARCHIVE** | `CI/GitHub release evidence; excluded by .dockerignore` | Preserve useful benchmark evidence outside the production image; remove from hot deployment package. |
| `deploy/cleaner/evidence/df3-cpu-synthetic-hour-2026-09-21.json` | **ARCHIVE** | `CI/GitHub release evidence; excluded by .dockerignore` | Preserve useful benchmark evidence outside the production image; remove from hot deployment package. |
| `deploy/cleaner/evidence/noise-profile-cpu-synthetic-hour-2026-09-21.json` | **ARCHIVE** | `CI/GitHub release evidence; excluded by .dockerignore` | Preserve useful benchmark evidence outside the production image; remove from hot deployment package. |
| `deploy/cleaner/evidence/sam-audio-only-core-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-codec-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-codec-graph-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-codec-roundtrip-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-conditioned-solver-16step-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-conditioned-solver-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-convolution-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-cpu-rounding-diagnostic-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-cublas-retention-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-cuda-window-crossing-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-factory-real-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-feature-activation-disk-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-feature-disk-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-feature-recurrent-disk-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-file-pipeline-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-file-pipeline-real-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-installed-cuda-repeat-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-installed-cuda-short-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-installed-raw-conditioning-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-installed-v2-inference-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-isolated-wheel-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-joint-decoder-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-meta-codec-real-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-meta-core-real-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-mixed-cpu-backend-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-noise-reproducibility-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-noise-resource-failures-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-pcm-adapter-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-plan-resampling-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-prompt-cache-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-recurrent-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-separation-pipeline-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-small-download-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-strict-checkpoint-real-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-t5-offline-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-watermark-decoder-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `deploy/cleaner/evidence/sam-watermark-faults-cpu-2026-09-21.json` | **REMOVE_PER_LATEST_SOURCE** |Removed by latest source commit `ee11f2a`|This experiment records the retired custom SAM implementation; official SAM Audio certification and parity evidence must be generated separately.|
| `hear/core/__init__.py` | **AUDIT** | `hear/core/__init__.py` | Review callers during migration and place under the owning four-job/service module. |
| `hear/core/backend_registry.py` | **REPLACE** | `hear/security/grants.py` | Replace static backend registry/service-key ownership with scoped signed attempt grants issued by backend. |
| `hear/core/blocking.py` | **REWRITE** | `hear/execution/native.py` | Keep bounded native execution helper; delete DB-specific executor and make cancellation/resource ownership explicit. |
| `hear/core/category_loader.py` | **REWRITE** | `hear/services/pipeline/configuration.py` | Load immutable/versioned pipeline configuration supplied by backend/build assets; remove AI database ownership. |
| `hear/core/db_gate.py` | **DELETE** | `-` | AI workers no longer access application PostgreSQL. |
| `hear/core/discovery_taxonomy.py` | **REWRITE** | `hear/services/pipeline/configuration.py` | Load immutable/versioned pipeline configuration supplied by backend/build assets; remove AI database ownership. |
| `hear/core/downloader.py` | **MOVE_REWRITE** | `hear/audio/io.py` | Keep bounded download/decode behavior; use shared httpx client, attempt workspace and source identity checks. |
| `hear/core/gpu.py` | **DELETE** | `-` | Remove the unused global CUDA lock; no repository caller depends on it. |
| `hear/core/health.py` | **REWRITE** | `hear/health/service.py` | Remove Ray Serve health; expose role model readiness, patch/model manifest, GPU/scratch and active capacity. |
| `hear/core/hear_temp.py` | **DELETE** | `hear/audio/workspace.py` | Remove database-coupled global temp API; reconstruction and other workflows use attempt-scoped workspaces with deterministic cleanup. |
| `hear/core/keyword_loader.py` | **REWRITE** | `hear/services/pipeline/configuration.py` | Load immutable/versioned pipeline configuration supplied by backend/build assets; remove AI database ownership. |
| `hear/core/noise.py` | **KEEP_REFACTOR** | `hear/core/noise.py` | Retain the reconstruction synthesis noise reducer; it remains imported by the V11 synthesizer. |
| `hear/core/storage.py` | **MOVE_REWRITE** | `hear/storage/b2.py` | Keep scoped B2 prefix checks; shared client/TransferConfig, local hash + HEAD verification, no full read-back after every upload. |
| `hear/deployments/.gitignore` | **DELETE_AFTER_CUTOVER** | `-` | Ray Serve deployment/control layer is removed after local engine/runtime cutover. |
| `hear/deployments/__init__.py` | **DELETE_AFTER_CUTOVER** | `-` | Ray Serve deployment/control layer is removed after local engine/runtime cutover. |
| `hear/deployments/app.py` | **DELETE_AFTER_CUTOVER** | `-` | Ray Serve deployment/control layer is removed after local engine/runtime cutover. |
| `hear/deployments/audio_cleanup.py` | **DELETE_AFTER_CUTOVER** | `-` | Ray Serve deployment/control layer is removed after local engine/runtime cutover. |
| `hear/deployments/fish_speech.py` | **EXTRACT_THEN_DELETE** | `hear/inference/fish_speech.py` | Move model construction/inference logic into local role engines; remove @serve deployment wrapper. |
| `hear/deployments/gateway.py` | **DELETE_AFTER_CUTOVER** | `-` | Ray Serve deployment/control layer is removed after local engine/runtime cutover. |
| `hear/deployments/language_models.py` | **EXTRACT_THEN_DELETE** | `hear/inference/text_generation.py + hear/inference/small_models.py` | Move model construction/inference logic into local role engines; remove @serve deployment wrapper. |
| `hear/deployments/magic_clean.py` | **EXTRACT_THEN_DELETE** | `hear/inference/magic_clean.py` | Move model construction/inference logic into local role engines; remove @serve deployment wrapper. |
| `hear/deployments/transcription.py` | **EXTRACT_THEN_DELETE** | `hear/inference/qwen_asr.py` | Move model construction/inference logic into local role engines; remove @serve deployment wrapper. |
| `hear/models/__init__.py` | **DELETE_OR_MINIMIZE** | `hear/contracts/__init__.py` | Package marker only after model split. |
| `hear/models/database.py` | **DELETE_AFTER_STATE_MIGRATION** | `hear-backend models/migrations` | Move jobs, previews, lineage and durable state to backend; remove SQLAlchemy DB runtime from AI. |
| `hear/models/discovery.py` | **KEEP_REFACTOR** | `hear/models/discovery.py` | Keep the pure discovery serialization/domain types used by categorization and its callers. |
| `hear/models/schemas.py` | **MOVE_REWRITE** | `hear/contracts/jobs.py + events.py + outcomes.py` | Strict four-job/attempt contracts; remove transport-specific and legacy job aliases. |
| `hear/models/stages.py` | **DELETE_AFTER_STATE_MIGRATION** | `hear-backend job stage catalog` | The legacy catalog covers job types outside the canonical four; backend owns durable stage labels while V11 events carry stage identifiers. |
| `hear/proto/__init__.py` | **DELETE_IF_EMPTY** | `-` | Proto package disappears when no gRPC contracts remain. |
| `hear/proto/cleaner_v2.proto` | **DELETE_AFTER_CLEANER_CONSOLIDATION** | `hear/contracts/magic_clean.py` | Keep only until useful cleaner-v2 contract semantics are represented by Pydantic/internal contracts. |
| `hear/proto/cleaner_v2_pb2.py` | **DELETE_AFTER_CLEANER_CONSOLIDATION** | `hear/contracts/magic_clean.py` | Keep only until useful cleaner-v2 contract semantics are represented by Pydantic/internal contracts. |
| `hear/proto/cleaner_v2_pb2.pyi` | **DELETE_AFTER_CLEANER_CONSOLIDATION** | `hear/contracts/magic_clean.py` | Keep only until useful cleaner-v2 contract semantics are represented by Pydantic/internal contracts. |
| `hear/proto/cleaner_v2_pb2_grpc.py` | **DELETE_AFTER_CLEANER_CONSOLIDATION** | `hear/contracts/magic_clean.py` | Keep only until useful cleaner-v2 contract semantics are represented by Pydantic/internal contracts. |
| `hear/proto/pipeline.proto` | **DELETE_AFTER_GRPC_CUTOVER** | `-` | Pipeline gRPC contract/generated code replaced by HTTP/JSON contracts and RunPod native operations. |
| `hear/proto/pipeline_pb2.py` | **DELETE_AFTER_GRPC_CUTOVER** | `-` | Pipeline gRPC contract/generated code replaced by HTTP/JSON contracts and RunPod native operations. |
| `hear/proto/pipeline_pb2.pyi` | **DELETE_AFTER_GRPC_CUTOVER** | `-` | Pipeline gRPC contract/generated code replaced by HTTP/JSON contracts and RunPod native operations. |
| `hear/proto/pipeline_pb2_grpc.py` | **DELETE_AFTER_GRPC_CUTOVER** | `-` | Pipeline gRPC contract/generated code replaced by HTTP/JSON contracts and RunPod native operations. |
| `hear/runtime/cleaner/__init__.py` | **MOVE_KEEP** | `hear/inference/magic_clean_components/` | Retain SAM/DeepFilter implementation only if certification tests pass; exclude from unrelated role images. |
| `hear/runtime/cleaner/deepfilter_loader.py` | **MOVE_KEEP** | `hear/inference/magic_clean_components/` | Retain SAM/DeepFilter implementation only if certification tests pass; exclude from unrelated role images. |
| `hear/runtime/cleaner/executor.py` | **MERGE** | `hear/workflows/magic_clean.py` | Merge execution semantics into canonical MagicCleanWorkflow/engine session. |
| `hear/runtime/cleaner/factory.py` | **MOVE_REWRITE** | `hear/inference/magic_clean_factory.py` | Build profile-specific engine workers without model downloads or transport startup. |
| `hear/runtime/cleaner/grpc_ingress.py` | **DELETE** | `-` | Transport-specific cleaner gRPC ingress is obsolete. |
| `hear/runtime/cleaner/inspection_worker.py` | **MOVE_KEEP** | `hear/services/magic_clean/runtime/` | Retain bounded/safety utilities; remove transport/Ray assumptions. |
| `hear/runtime/cleaner/longform_sam.py` | **REMOVE_PER_LATEST_SOURCE** | `hear/inference/magic_clean_components/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `hear/runtime/cleaner/mapped_residency.py` | **REMOVE_PER_LATEST_SOURCE** | `hear/services/magic_clean/runtime/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `hear/runtime/cleaner/metrics.py` | **MOVE_KEEP** | `hear/services/magic_clean/runtime/` | Retain bounded/safety utilities; remove transport/Ray assumptions. |
| `hear/runtime/cleaner/model_registry.py` | **MOVE_REWRITE** | `hear/inference/magic_clean_registry.py` | Keep certified profile/runtime identity; load only selected profile engine. |
| `hear/runtime/cleaner/noise_reference.py` | **MOVE_KEEP** | `hear/services/magic_clean/runtime/` | Retain bounded/safety utilities; remove transport/Ray assumptions. |
| `hear/runtime/cleaner/resampling.py` | **MOVE_KEEP** | `hear/services/magic_clean/runtime/` | Retain bounded/safety utilities; remove transport/Ray assumptions. |
| `hear/runtime/cleaner/resource_guard.py` | **MOVE_KEEP** | `hear/services/magic_clean/runtime/` | Retain bounded/safety utilities; remove transport/Ray assumptions. |
| `hear/runtime/cleaner/s3_verification.py` | **MOVE_KEEP** | `hear/inference/magic_clean_components/` | Retain SAM/DeepFilter implementation only if certification tests pass; exclude from unrelated role images. |
| `hear/runtime/cleaner/sam_checkpoint.py` | **REMOVE_PER_LATEST_SOURCE** | `hear/inference/magic_clean_components/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `hear/runtime/cleaner/sam_codec_graph.py` | **REMOVE_PER_LATEST_SOURCE** | `hear/inference/magic_clean_components/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `hear/runtime/cleaner/sam_codec_loader.py` | **REMOVE_PER_LATEST_SOURCE** | `hear/inference/magic_clean_components/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `hear/runtime/cleaner/sam_conditioning.py` | **REMOVE_PER_LATEST_SOURCE** | `hear/inference/magic_clean_components/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `hear/runtime/cleaner/sam_convolution.py` | **REMOVE_PER_LATEST_SOURCE** | `hear/inference/magic_clean_components/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `hear/runtime/cleaner/sam_core_loader.py` | **REMOVE_PER_LATEST_SOURCE** | `hear/inference/magic_clean_components/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `hear/runtime/cleaner/sam_features.py` | **REMOVE_PER_LATEST_SOURCE** | `hear/inference/magic_clean_components/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `hear/runtime/cleaner/sam_loader.py` | **MOVE_KEEP** | `hear/inference/magic_clean_components/` | Retain SAM/DeepFilter implementation only if certification tests pass; exclude from unrelated role images. |
| `hear/runtime/cleaner/sam_noise.py` | **REMOVE_PER_LATEST_SOURCE** | `hear/inference/magic_clean_components/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `hear/runtime/cleaner/sam_pcm.py` | **REMOVE_PER_LATEST_SOURCE** | `hear/inference/magic_clean_components/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `hear/runtime/cleaner/sam_pipeline.py` | **REMOVE_PER_LATEST_SOURCE** | `hear/inference/magic_clean_components/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `hear/runtime/cleaner/sam_prompt_cache.py` | **REMOVE_PER_LATEST_SOURCE** | `hear/inference/magic_clean_components/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `hear/runtime/cleaner/sam_recurrent.py` | **REMOVE_PER_LATEST_SOURCE** | `hear/inference/magic_clean_components/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `hear/runtime/cleaner/speech_activity.py` | **MOVE_KEEP** | `hear/services/magic_clean/runtime/` | Retain bounded/safety utilities; remove transport/Ray assumptions. |
| `hear/runtime/cleaner/speech_risk.py` | **MOVE_KEEP** | `hear/services/magic_clean/runtime/` | Retain bounded/safety utilities; remove transport/Ray assumptions. |
| `hear/runtime/cleaner/subprocesses.py` | **MOVE_KEEP** | `hear/services/magic_clean/runtime/` | Retain bounded/safety utilities; remove transport/Ray assumptions. |
| `hear/runtime/cleaner/wire.py` | **REWRITE_OR_DELETE** | `hear/contracts/magic_clean.py` | Remove wire/gRPC coupling; preserve only canonical plan/result conversions. |
| `hear/runtime/cleaner/worker_lease.py` | **MOVE_REWRITE** | `hear/execution/resource_lease.py` | Retain local exclusive-resource protection; backend owns distributed attempt lease. |
| `hear/services/categorization/__init__.py` | **KEEP_REFACTOR** | `hear/services/categorization/__init__.py` | Pipeline domain service; inject engines/configuration, no global model registry or AI DB. |
| `hear/services/categorization/discovery.py` | **KEEP_REFACTOR** | `hear/services/categorization/discovery.py` | Pipeline domain service; inject engines/configuration, no global model registry or AI DB. |
| `hear/services/categorization/service.py` | **KEEP_REFACTOR** | `hear/services/categorization/service.py` | Pipeline domain service; inject engines/configuration, no global model registry or AI DB. |
| `hear/services/llm.py` | **REWRITE** | `hear/inference/text_generation.py + pipeline service` | Remove ModelClientRegistry; default no model load when feature disabled; vLLM engine when enabled. |
| `hear/services/moderation/__init__.py` | **KEEP_REFACTOR** | `hear/services/moderation/__init__.py` | Pipeline domain service; inject engines/configuration, no global model registry or AI DB. |
| `hear/services/moderation/service.py` | **KEEP_REFACTOR** | `hear/services/moderation/service.py` | Pipeline domain service; inject engines/configuration, no global model registry or AI DB. |
| `hear/services/jobs/__init__.py` | **DELETE_IF_EMPTY** | `-` | Old job-control package removed after cutover. |
| `hear/services/jobs/credentials.py` | **DELETE_AFTER_BACKEND_CUTOVER** | `hear-backend AI control plane` | Durable scheduling/submission/credential refresh belongs to backend, not GPU worker. |
| `hear/services/jobs/scheduler.py` | **DELETE_AFTER_BACKEND_CUTOVER** | `hear-backend AI control plane` | Durable scheduling/submission/credential refresh belongs to backend, not GPU worker. |
| `hear/services/jobs/submission.py` | **DELETE_AFTER_BACKEND_CUTOVER** | `hear-backend AI control plane` | Durable scheduling/submission/credential refresh belongs to backend, not GPU worker. |
| `hear/services/jobs/workflows.py` | **REWRITE** | `hear/contracts/jobs.py + hear/execution/router.py` | Exactly four canonical jobs; legacy aliases only at backend compatibility boundary. |
| `hear/services/magic_clean/__init__.py` | **KEEP_REFACTOR** | `hear/services/magic_clean/__init__.py` | Retain canonical Magic Clean behavior; remove duplicated transport/state ownership. |
| `hear/services/magic_clean/artifacts.py` | **KEEP_REFACTOR** | `hear/services/magic_clean/artifacts.py` | Keep canonical Magic Clean domain/safety logic; align to one job type and profile-specific engines. |
| `hear/services/magic_clean/cleanup.py` | **REMOVE_PER_LATEST_SOURCE** | Removed by latest source commit `ee11f2a` | Latest source no longer contains this module; scoped attempt cleanup is implemented in `hear/runtime/cleaner/path_cleanup.py`. |
| `hear/services/magic_clean/contracts.py` | **KEEP_REFACTOR** | `hear/services/magic_clean/contracts.py` | Keep canonical Magic Clean domain/safety logic; align to one job type and profile-specific engines. |
| `hear/services/magic_clean/engines/__init__.py` | **KEEP_MOVE** | `hear/inference/magic_clean/` | Profile-specific engine implementation; load only when the selected profile worker starts. |
| `hear/services/magic_clean/engines/base.py` | **KEEP_MOVE** | `hear/inference/magic_clean/` | Profile-specific engine implementation; load only when the selected profile worker starts. |
| `hear/services/magic_clean/engines/deepfilter.py` | **KEEP_MOVE** | `hear/inference/magic_clean/` | Profile-specific engine implementation; load only when the selected profile worker starts. |
| `hear/services/magic_clean/engines/noise_profile.py` | **KEEP_MOVE** | `hear/inference/magic_clean/` | Profile-specific engine implementation; load only when the selected profile worker starts. |
| `hear/services/magic_clean/engines/sam_audio.py` | **KEEP_MOVE** | `hear/inference/magic_clean/` | Profile-specific engine implementation; load only when the selected profile worker starts. |
| `hear/services/magic_clean/inspection.py` | **KEEP_REFACTOR** | `hear/services/magic_clean/inspection.py` | Keep canonical Magic Clean domain/safety logic; align to one job type and profile-specific engines. |
| `hear/services/magic_clean/lineage.py` | **REMOVE_PER_LATEST_SOURCE** | Removed by latest source commit `ee11f2a` | Durable lineage belongs to the backend; runtime provenance is represented by execution events and artifacts. |
| `hear/services/magic_clean/mastering.py` | **KEEP_REFACTOR** | `hear/services/magic_clean/mastering.py` | Keep canonical Magic Clean domain/safety logic; align to one job type and profile-specific engines. |
| `hear/services/magic_clean/models.py` | **REMOVE_PER_LATEST_SOURCE** | Removed by latest source commit `ee11f2a` | Latest source no longer contains this module; current Magic Clean contracts live in `hear/services/magic_clean/contracts.py`. |
| `hear/services/magic_clean/pipeline.py` | **CONSOLIDATE** | `hear/workflows/magic_clean.py + hear/inference/magic_clean/` | Keep only behavior still needed after cleaner-v2 profile certification; remove duplicate legacy engine pipeline. |
| `hear/services/magic_clean/processing/__init__.py` | **REMOVE_PER_LATEST_SOURCE** | Removed by latest source commit `ee11f2a` | These legacy MossFormer/Demucs processing components are not part of the supported three-profile cleaner; retain only in the pre-V11 archive when needed for audit. |
| `hear/services/magic_clean/processing/audio_io.py` | **REMOVE_PER_LATEST_SOURCE** | Removed by latest source commit `ee11f2a` | These legacy MossFormer/Demucs processing components are not part of the supported three-profile cleaner; retain only in the pre-V11 archive when needed for audit. |
| `hear/services/magic_clean/processing/dynamics.py` | **REMOVE_PER_LATEST_SOURCE** | Removed by latest source commit `ee11f2a` | These legacy MossFormer/Demucs processing components are not part of the supported three-profile cleaner; retain only in the pre-V11 archive when needed for audit. |
| `hear/services/magic_clean/processing/mossformer.py` | **REMOVE_PER_LATEST_SOURCE** | Removed by latest source commit `ee11f2a` | These legacy MossFormer/Demucs processing components are not part of the supported three-profile cleaner; retain only in the pre-V11 archive when needed for audit. |
| `hear/services/magic_clean/processing/quality.py` | **KEEP_REFACTOR** | `hear/services/magic_clean/processing/quality.py` | Keep canonical Magic Clean domain/safety logic; align to one job type and profile-specific engines. |
| `hear/services/magic_clean/processing/silence.py` | **REMOVE_PER_LATEST_SOURCE** | Removed by latest source commit `ee11f2a` | These legacy MossFormer/Demucs processing components are not part of the supported three-profile cleaner; retain only in the pre-V11 archive when needed for audit. |
| `hear/services/magic_clean/processing/speech.py` | **REMOVE_PER_LATEST_SOURCE** | Removed by latest source commit `ee11f2a` | These legacy MossFormer/Demucs processing components are not part of the supported three-profile cleaner; retain only in the pre-V11 archive when needed for audit. |
| `hear/services/magic_clean/processing/stems.py` | **REMOVE_PER_LATEST_SOURCE** | Removed by latest source commit `ee11f2a` | These legacy MossFormer/Demucs processing components are not part of the supported three-profile cleaner; retain only in the pre-V11 archive when needed for audit. |
| `hear/services/magic_clean/processing/validation.py` | **REMOVE_PER_LATEST_SOURCE** | Removed by latest source commit `ee11f2a` | These legacy MossFormer/Demucs processing components are not part of the supported three-profile cleaner; retain only in the pre-V11 archive when needed for audit. |
| `hear/services/magic_clean/quality.py` | **KEEP_REFACTOR** | `hear/services/magic_clean/quality.py` | Keep canonical Magic Clean domain/safety logic; align to one job type and profile-specific engines. |
| `hear/services/magic_clean/service.py` | **REWRITE** | `hear/workflows/magic_clean.py` | Remove global eager MossFormer/Demucs load and broad GPU lock; delegate to selected engine/profile. |
| `hear/services/magic_clean/streaming.py` | **REMOVE_PER_LATEST_SOURCE** | Removed by latest source commit `ee11f2a` | Latest source no longer contains this module; retain only the bounded attempt workspace and profile engine implementations. |
| `hear/services/reconstruction/__init__.py` | **KEEP_REFACTOR** | `hear/services/reconstruction/__init__.py` | Retain reconstruction domain behavior under canonical reconstruction job. |
| `hear/services/reconstruction/audio_buffer.py` | **KEEP_REFACTOR** | `hear/services/reconstruction/audio_buffer.py` | Retain reconstruction domain behavior under canonical reconstruction job. |
| `hear/services/reconstruction/dnsmos.py` | **KEEP_OPTIMIZE** | `hear/services/reconstruction/dnsmos.py` | Load ONNX/DNSMOS session once per reconstruction worker and inject it. |
| `hear/services/reconstruction/quality.py` | **KEEP_REFACTOR** | `hear/services/reconstruction/quality.py` | Retain reconstruction domain behavior under canonical reconstruction job. |
| `hear/services/reconstruction/service.py` | **REWRITE** | `hear/workflows/reconstruction.py` | Move preview persistence/confirmation to backend; keep compute service pure. |
| `hear/services/reconstruction/synthesizer.py` | **REFACTOR** | `hear/services/reconstruction/synthesizer.py` | Preserve voice behavior; replace whole-file cloning/splice assembly with bounded streaming timeline renderer. |
| `hear/services/reconstruction/tts_post_processor.py` | **KEEP_OPTIMIZE** | `hear/services/reconstruction/tts_post_processor.py` | Inject shared DNSMOS scorer; preserve pitch/loudness/quality gates. |
| `hear/services/reconstruction/voice_profile.py` | **DELETE_AFTER_STATE_MIGRATION** | `hear-backend voice profile storage` | This persistent per-user store has no runtime caller; profile persistence belongs with backend user state. |
| `hear/services/transcription/__init__.py` | **KEEP_REFACTOR** | `hear/services/transcription/__init__.py` | Keep bounded file-window transcription/result policy; inject QwenAsrEngine instead of RayModelClient. |
| `hear/services/transcription/service.py` | **KEEP_REFACTOR** | `hear/services/transcription/service.py` | Keep bounded file-window transcription/result policy; inject QwenAsrEngine instead of RayModelClient. |
| `hear/services/transport/__init__.py` | **DELETE_IF_EMPTY** | `-` | Old transport package removed. |
| `hear/services/transport/grpc.py` | **DELETE_AFTER_GRPC_CUTOVER** | `-` | Obsolete transport. |
| `hear/services/transport/operations.py` | **SPLIT** | `hear/workflows/* + hear-backend APIs` | Compute operations move to workflows; durable/admin/preview operations move backend. |
| `hear/tools/__init__.py` | **KEEP_MINIMAL** | `hear/tools/__init__.py` | Package marker/tooling only. |
| `hear/tools/check_architecture.py` | **REWRITE** | `hear/tools/check_architecture.py` | Enforce new dependency direction and ban Ray/gRPC/AI-DB imports after deletion gates. |
| `hear/tools/clean_temp.py` | **REWRITE** | `hear/audio/workspace.py / ops tool` | Attempt-scoped cleanup only; no global unsafe cleanup. |
| `hear/tools/dependency_patches.py` | **KEEP_HARDEN** | `hear/compat/dependency_patches.py` | Build-time apply + runtime verify only; immutable production dependencies. |
| `hear/tools/model_provisioning.py` | **REWRITE** | `hear/models/manifest.py + scripts/provision_models.py` | Pin exact model revisions; provision per role outside hot startup; remove Ray task. |
| `hear/utils/__init__.py` | **KEEP_REFACTOR** | `hear/utils/__init__.py` | Retain pure reusable logic; remove dead callers after workflow extraction. |
| `hear/utils/audio.py` | **MOVE_REFACTOR** | `hear/audio/` | Consolidate shared bounded audio I/O/DSP. |
| `hear/utils/audio_dsp.py` | **MOVE_REFACTOR** | `hear/audio/` | Consolidate shared bounded audio I/O/DSP. |
| `hear/utils/content_context.py` | **KEEP_REFACTOR** | `hear/utils/content_context.py` | Retain pure reusable logic; remove dead callers after workflow extraction. |
| `hear/utils/discovery_sort.py` | **KEEP_REFACTOR** | `hear/utils/discovery_sort.py` | Retain pure reusable logic; remove dead callers after workflow extraction. |
| `hear/utils/processing_context.py` | **KEEP_REFACTOR** | `hear/utils/processing_context.py` | Retain pure reusable logic; remove dead callers after workflow extraction. |
| `hear/utils/timing.py` | **KEEP_REFACTOR** | `hear/utils/timing.py` | Retain pure reusable logic; remove dead callers after workflow extraction. |
| `hear/utils/transcript_diff.py` | **KEEP_REFACTOR** | `hear/utils/transcript_diff.py` | Retain pure reusable logic; remove dead callers after workflow extraction. |
| `hear/utils/transcription_chunks.py` | **KEEP_HARDEN** | `hear/utils/transcription_chunks.py` | Golden-test chunk offsets/boundaries and reuse in Qwen engine. |
| `scripts/benchmark_cleaner_deepfilter.py` | **DEV_ONLY** | `tools/benchmarks/` | Keep benchmarking only; exclude image. |
| `scripts/benchmark_cleaner_noise_profile.py` | **DEV_ONLY** | `tools/benchmarks/` | Keep benchmarking only; exclude image. |
| `scripts/benchmark_cleaner_sam_core.py` | **REMOVE_PER_LATEST_SOURCE** | `tools/benchmarks/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `scripts/benchmark_cleaner_sam_features.py` | **REMOVE_PER_LATEST_SOURCE** | `tools/benchmarks/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `scripts/benchmark_cleaner_sam_installed.py` | **REMOVE_PER_LATEST_SOURCE** | `tools/benchmarks/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `scripts/benchmark_cleaner_solver.py` | **REMOVE_PER_LATEST_SOURCE** | `tools/benchmarks/` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `scripts/bootstrap-pod.sh` | **REPLACE** | `Dockerfile + deployment scripts` | No apt/git/model/database setup at runtime; immutable image + mounted model assets. |
| `scripts/build_cleaner_wheel.py` | **DELETE_AFTER_MERGE** | `-` | Separate cleaner wheel disappears when cleaner code/dependencies are consolidated. |
| `scripts/download_models_ray.py` | **DELETE_AFTER_CUTOVER** | `-` | Obsolete AI Postgres/Ray/gRPC tooling. |
| `scripts/generate_alexa_playback.py` | **KEEP_AUDIT** | `scripts/generate_alexa_playback.py` | Offline utility; verify no obsolete AI speed-layer responsibility before retaining. |
| `scripts/generate_cleaner_proto.py` | **DELETE_AFTER_CUTOVER** | `-` | Obsolete AI Postgres/Ray/gRPC tooling. |
| `scripts/generate_service_key.py` | **MOVE_BACKEND_OR_DEV** | `hear-backend ops` | AI no longer owns durable backend registry; keep only if needed during migration. |
| `scripts/live_regeneration_local_test.py` | **REWRITE_DEV** | `tools/smoke/` | Use new four-job Pod HTTP/SSE and RunPod provider contracts. |
| `scripts/live_test.py` | **REWRITE_DEV** | `tools/smoke/` | Use new four-job Pod HTTP/SSE and RunPod provider contracts. |
| `scripts/load-env.sh` | **DEV_ONLY** | `scripts/load-env.sh` | Keep local tooling only; production receives environment directly. |
| `scripts/postgres-env.sh` | **DELETE_AFTER_CUTOVER** | `-` | Obsolete AI Postgres/Ray/gRPC tooling. |
| `scripts/runpod-workspace-env.sh` | **DELETE** | `-` | No active caller needs the old workspace path helper; images receive paths from runtime settings and the old tests wrote into host-level directories. |
| `scripts/setup_runtime.py` | **REWRITE** | `scripts/setup_runtime.py` | Select a role/provider dependency group from the runtime project and apply/check the WhisperX patch only for pipeline and transcription. |
| `scripts/smoke_test.py` | **REWRITE_DEV** | `tools/smoke/` | Use new four-job Pod HTTP/SSE and RunPod provider contracts. |
| `scripts/start-hear-ray-server.sh` | **DELETE_AFTER_CUTOVER** | `-` | Obsolete AI Postgres/Ray/gRPC tooling. |
| `scripts/start-postgres.sh` | **DELETE_AFTER_CUTOVER** | `-` | Obsolete AI Postgres/Ray/gRPC tooling. |
| `scripts/start-with-env.sh` | **DEV_ONLY** | `scripts/start-with-env.sh` | Keep local tooling only; production receives environment directly. |
| `scripts/sync-postgres-password.sh` | **DELETE_AFTER_CUTOVER** | `-` | Obsolete AI Postgres/Ray/gRPC tooling. |
| `tests/__init__.py` | **AUDIT_ADAPT** | `tests/__init__.py` | Update imports/fixtures for four-job architecture; delete only when no preserved behavior remains. |
| `tests/generate_transcript.py` | **AUDIT_ADAPT** | `tests/generate_transcript.py` | Update imports/fixtures for four-job architecture; delete only when no preserved behavior remains. |
| `tests/integration/test_categorizer_wildlife.py` | **KEEP_ADAPT** | `tests/integration/test_categorizer_wildlife.py` | Run against local engines and optional GPU CI; preserve algorithm outputs. |
| `tests/integration/test_synthesizer.py` | **KEEP_ADAPT** | `tests/integration/test_synthesizer.py` | Run against local engines and optional GPU CI; preserve algorithm outputs. |
| `tests/integration/test_transcribe_url.py` | **KEEP_ADAPT** | `tests/integration/test_transcribe_url.py` | Run against local engines and optional GPU CI; preserve algorithm outputs. |
| `tests/integration/test_tts_post_processor.py` | **KEEP_ADAPT** | `tests/integration/test_tts_post_processor.py` | Run against local engines and optional GPU CI; preserve algorithm outputs. |
| `tests/test_architecture.py` | **AUDIT_ADAPT** | `tests/test_architecture.py` | Update imports/fixtures for four-job architecture; delete only when no preserved behavior remains. |
| `tests/test_audio_delivery.py` | **KEEP_ADAPT** | `tests/test_audio_delivery.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_audio_tag_contract.py` | **REPLACE** | `tests/test_pipeline_tagging_contract.py` | Audio tagging is no longer a canonical job type; preserve stage behavior. |
| `tests/test_backend_storage.py` | **AUDIT_ADAPT** | `tests/test_backend_storage.py` | Update imports/fixtures for four-job architecture; delete only when no preserved behavior remains. |
| `tests/test_bootstrap_pod.py` | **REWRITE** | `Docker/model-manifest/runtime tests` | Validate immutable image, role model manifest, patch check and provider environment. |
| `tests/test_category_loader.py` | **AUDIT_ADAPT** | `tests/test_category_loader.py` | Update imports/fixtures for four-job architecture; delete only when no preserved behavior remains. |
| `tests/test_cleaner_v2_artifacts.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_artifacts.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_capabilities.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_capabilities.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_contracts.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_contracts.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_deepfilter.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_deepfilter.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_deepfilter_loader.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_deepfilter_loader.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_deepfilter_real.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_deepfilter_real.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_executor.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_executor.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_factory.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_factory.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_grpc_ingress.py` | **DELETE_REPLACE** | `tests/test_http_runpod_contracts.py` | Replace legacy transport tests with FastAPI/RunPod stream contract tests. |
| `tests/test_cleaner_v2_inspection.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_inspection.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_mastering.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_mastering.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_metrics.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_metrics.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_noise_profile.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_noise_profile.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_noise_reference.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_noise_reference.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_packaging.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_packaging.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_quality.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_quality.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_resampling.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_resampling.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_runtime.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_runtime.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_s3_verification.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_s3_verification.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_sam_checkpoint.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_checkpoint.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_codec_graph.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_codec_graph.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_codec_loader.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_codec_loader.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_conditioning.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_conditioning.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_convolution.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_convolution.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_core_loader.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_core_loader.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_engine.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_engine.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_features.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_features.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_file_pipeline.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_file_pipeline.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_loader.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_sam_loader.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_sam_noise.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_noise.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_pcm.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_pcm.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_pipeline.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_pipeline.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_plan.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_plan.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_probe.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_probe.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_prompt_cache.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_prompt_cache.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_recurrent.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_recurrent.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_sam_solver.py` | **REMOVE_PER_LATEST_SOURCE** | `tests/test_cleaner_v2_sam_solver.py` | Latest source commit `ee11f2a` removes this experimental custom SAM component; the official SAM Audio adapter is the supported voice-focus path. |
| `tests/test_cleaner_v2_source_staging.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_source_staging.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_speech_activity.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_speech_activity.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_speech_risk.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_speech_risk.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_wire.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_wire.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_cleaner_v2_worker_lease.py` | **KEEP_ADAPT** | `tests/test_cleaner_v2_worker_lease.py` | Keep certification/safety tests; rename away from v2 once canonical Magic Clean owns them; remove transport-specific expectations. |
| `tests/test_content_context.py` | **KEEP_ADAPT** | `tests/test_content_context.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_database_initialization.py` | **DELETE_REPLACE** | `hear-backend tests` | AI database ownership removed; migrate durable-state tests to backend. |
| `tests/test_db_gate.py` | **DELETE_REPLACE** | `hear-backend tests` | AI database ownership removed; migrate durable-state tests to backend. |
| `tests/test_dependency_patches.py` | **KEEP_EXPAND** | `tests/test_dependency_patches.py` | Add build-time apply/runtime verify and failure-on-unpatched-image cases. |
| `tests/test_diff_engine.py` | **KEEP_ADAPT** | `tests/test_diff_engine.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_discovery.py` | **KEEP_ADAPT** | `tests/test_discovery.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_discovery_sort.py` | **KEEP_ADAPT** | `tests/test_discovery_sort.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_environment_credentials.py` | **AUDIT_ADAPT** | `tests/test_environment_credentials.py` | Update imports/fixtures for four-job architecture; delete only when no preserved behavior remains. |
| `tests/test_fair_scheduler.py` | **DELETE_REPLACE** | `backend scheduler/outbox/provider tests` | Old AI scheduler/subscription tests replaced by backend authority and provider adapters. |
| `tests/test_fastapi_ingress.py` | **REWRITE** | `test_fastapi_ingress.py` | Test Pod health/control routers and backend internal event/outcome routes; no direct AI durable submission. |
| `tests/test_grpc_contracts.py` | **DELETE_REPLACE** | `tests/test_http_runpod_contracts.py` | Replace legacy transport tests with FastAPI/RunPod stream contract tests. |
| `tests/test_hear_temp.py` | **DELETE** | `tests/test_audio_workspace.py` | Replace database-coupled temp tests with attempt-workspace isolation and cleanup coverage. |
| `tests/test_immediate_correctness.py` | **AUDIT_ADAPT** | `tests/test_immediate_correctness.py` | Update imports/fixtures for four-job architecture; delete only when no preserved behavior remains. |
| `tests/test_independent_lanes.py` | **REWRITE** | `test_independent_lanes.py` | Test four logical job types plus Magic Clean execution profiles/provider pools. |
| `tests/test_job_error_reporting.py` | **AUDIT_ADAPT** | `tests/test_job_error_reporting.py` | Update imports/fixtures for four-job architecture; delete only when no preserved behavior remains. |
| `tests/test_job_submission.py` | **DELETE_REPLACE** | `backend scheduler/outbox/provider tests` | Old AI scheduler/subscription tests replaced by backend authority and provider adapters. |
| `tests/test_job_subscriptions.py` | **DELETE_REPLACE** | `backend scheduler/outbox/provider tests` | Old AI scheduler/subscription tests replaced by backend authority and provider adapters. |
| `tests/test_magic_clean_cleanup.py` | **KEEP_ADAPT** | `tests/test_magic_clean_cleanup.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_magic_clean_dsp_safety.py` | **KEEP_ADAPT** | `tests/test_magic_clean_dsp_safety.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_magic_clean_lineage.py` | **KEEP_ADAPT** | `tests/test_magic_clean_lineage.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_magic_clean_model_path.py` | **KEEP_ADAPT** | `tests/test_magic_clean_model_path.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_magic_clean_orchestrator_guards.py` | **REWRITE** | `tests/test_magic_clean_workflow_guards.py` | Orchestrator removed; preserve safety invariants in workflow. |
| `tests/test_magic_clean_pipeline.py` | **KEEP_ADAPT** | `tests/test_magic_clean_pipeline.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_magic_clean_service_safety.py` | **KEEP_ADAPT** | `tests/test_magic_clean_service_safety.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_magic_clean_stages.py` | **KEEP_ADAPT** | `tests/test_magic_clean_stages.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_magic_clean_streaming.py` | **KEEP_ADAPT** | `tests/test_magic_clean_streaming.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_magic_clean_validation.py` | **KEEP_ADAPT** | `tests/test_magic_clean_validation.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_model_client_async.py` | **DELETE_REPLACE** | `tests/test_inference_manager.py` | Ray model client removed. |
| `tests/test_model_provisioning.py` | **REWRITE** | `Docker/model-manifest/runtime tests` | Validate immutable image, role model manifest, patch check and provider environment. |
| `tests/test_moderation_context.py` | **KEEP_ADAPT** | `tests/test_moderation_context.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_native_worker.py` | **REWRITE** | `tests/test_native_execution.py` | Test bounded local engine execution and cancellation. |
| `tests/test_pipeline_stage_reports.py` | **AUDIT_ADAPT** | `tests/test_pipeline_stage_reports.py` | Update imports/fixtures for four-job architecture; delete only when no preserved behavior remains. |
| `tests/test_reconstruction_lineage.py` | **KEEP_ADAPT** | `tests/test_reconstruction_lineage.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_reconstruction_synthesizer.py` | **KEEP_ADAPT** | `tests/test_reconstruction_synthesizer.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_runpod_workspace_env.py` | **DELETE** | `-` | Remove tests for an unused environment helper that created host-level model directories; current role configuration is validated by runtime bootstrap. |
| `tests/test_runtime_config.py` | **REWRITE** | `Docker/model-manifest/runtime tests` | Validate immutable image, role model manifest, patch check and provider environment. |
| `tests/test_runtime_setup.py` | **REWRITE** | `Docker/model-manifest/runtime tests` | Validate immutable image, role model manifest, patch check and provider environment. |
| `tests/test_service_health.py` | **REWRITE** | `test_service_health.py` | Test role readiness/model manifest/GPU/scratch instead of Ray Serve replicas. |
| `tests/test_supervisor_config.py` | **DELETE_REPLACE** | `tests/test_docker_images.py` | Supervisor/Ray topology removed. |
| `tests/test_transcription_chunks.py` | **KEEP_ADAPT** | `tests/test_transcription_chunks.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_transcription_no_speech.py` | **KEEP_ADAPT** | `tests/test_transcription_no_speech.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_transcription_windows.py` | **KEEP_ADAPT** | `tests/test_transcription_windows.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_tts_pitch_matching.py` | **KEEP_ADAPT** | `tests/test_tts_pitch_matching.py` | Preserve behavior golden tests while replacing architecture-specific fixtures. |
| `tests/test_typed_grpc.py` | **DELETE_REPLACE** | `tests/test_http_runpod_contracts.py` | Replace legacy transport tests with FastAPI/RunPod stream contract tests. |
| `tests/test_voice_profile.py` | **DELETE_AFTER_STATE_MIGRATION** | `hear-backend profile storage tests` | Source test covers an unused persistent per-user store that is not part of the V11 worker runtime. |
| `tests/test_workflow_registry.py` | **AUDIT_ADAPT** | `tests/test_workflow_registry.py` | Update imports/fixtures for four-job architecture; delete only when no preserved behavior remains. |
| `tests/transcript_dump.json` | **AUDIT_ADAPT** | `tests/transcript_dump.json` | Update imports/fixtures for four-job architecture; delete only when no preserved behavior remains. |
