# HEAR AI Production Cutover

**Status:** Production cutover runbook  
**Requested target:** RunPod Serverless queue endpoints; use section 34 for the current setup  
**Canonical Pod runtime:** Bazel-built `runpod-stack` image  
**Serverless runtime:** Same source and contracts, separate role images and startup policy (section 31)  
**Canonical source target:** `release/hear-ai-production-v11`

**Current Serverless status, 2026-10-03:** Both role images were built, published
and checked through Bazel → Docker in the `hear-ai` repository. Both RunPod
queue endpoints now exist: pipeline `8ewxonopk5ex1p`, cleaner `e2ysmfujllh9ur`.
Each template's immutable image and all 37 live environment settings were read
back and verified. Backend Serverless dispatch from PR 88 is merged and
successfully deployed; full CI passed (1,995 tests, lint and type checks).
The replacement RunPod API key is valid. Worker startup still requires a GHCR
pull credential: the images are private and this RunPod account has no registry
authentication configured. No real Serverless jobs have run. Production runtime
dispatch remains disabled. See section 35 for created endpoint URLs. Sections
32 and 33 record earlier Pod work and do not apply to Serverless.

**Review:** 2026-10-03 — Bazel Docker build/publication, live backend transport
and a real CPU cleaning job passed. GPU Pipeline and Serverless acceptance remain pending.

**Latest execution check:** 2026-10-03 — env setup, Docker publication and a
real CPU cleaning job passed; GPU cutover remains pending.
The full `runpod-stack` image has now been built and published through Bazel → Docker on the
production backend's Docker host; image ID
`sha256:ee06c90f6668bb0ff041a4bbd70375f4d47e64bc25f90e78ae8e9e74d3dfa5f4`.
The external, mode-0600 `/root/hear-ai-config/production.env` is fully populated
with the live backend service key, ingress token and ownership policy on both
hosts. Authenticated protocol and catalogue routes have been fixed, deployed
and accepted. A real CPU cleaning canary has passed against the Docker image, including live
B2 readback and duplicate fencing.
GPU acceptance remains pending, and `HEAR_AI_RUNTIME_V1` remains disabled.
See section 32 for evidence.

---

## 1. Final architecture

Bazel is already the build entrypoint for HEAR AI.

The intended **Pod** production flow is:

```text
HEAR AI source
    ↓
Bazel
    ↓
deterministic Docker build context
    ↓
Docker Buildx target: runpod-stack
    ↓
build stage provisions and verifies pinned models
    ↓
final image contains runtime + models
    ↓
RunPod starts container
    ↓
production env is loaded externally
    ↓
API / RabbitMQ workers start with models COLD
    ↓
job arrives
    ↓
required model lazy-loads from the image into GPU
    ↓
job completes
    ↓
model remains warm for its idle TTL
    ↓
no more jobs
    ↓
model is evicted from GPU
```

Bazel is therefore responsible for producing the reproducible build input and invoking the image build.

**Models must never be downloaded into `/workspace`.**

The model files belong inside the production image at:

```text
/models
```

On a Pod, HEAR loads those image-contained models into GPU when required. On
Serverless, core models preload before accepting jobs and remain resident for the
worker lifetime. Optional Sound Cleanup/AudioSep models still load on demand.
Packaging eliminates runtime downloads; it does not eliminate image pulls,
library initialization, disk reads, or transfer of weights into GPU memory.

---

# 2. Canonical branch

Create one production branch containing the complete validated HEAR AI lineage.

```text
release/hear-ai-production-v11
```

Required known lineage:

```text
5ac3c46  perf: restore ASR batching and remove redundant normalization measurements
63dc0c0  feat: finalize RunPod concurrency cleanup and Bazel release setup
d3ec9a0  feat: lazily load GPU engines and evict them after idle
2c45abc  fix: add reconstruction build compiler
ba2f015  build: make RunPod image self-provisioning
```

The final production branch must contain the equivalent of all of these changes.

Do not deploy from:

```text
perf/*
fix/*
old cleaner branches
old sound-cleanup worktrees
temporary deployment branches
```

After the release has passed production acceptance, merge:

```text
release/hear-ai-production-v11
    ↓
main
```

Then remove obsolete branches only after checking that none contain unique commits.

---

# 3. Clean the RunPod filesystem

The RunPod host should not contain loose HEAR model payloads or old test/runtime directories.

## Keep

```text
/workspace/hear-ai-v11/
```

This is the canonical HEAR AI source checkout.

Keep production configuration outside the image:

```text
/root/hear-ai-config/production.env
/root/hear-ai-config/runtime.env
```

Optionally retain one final Bazel build artifact:

```text
/root/hear-ai-v11/builds/<final-release-artifact>
```

`/models` may exist as an empty host directory, but production must not depend on host model files.

## Remove

Remove stale HEAR files such as:

```text
/root/hear-ai-v11/deployment-audit-*
/root/hear-ai-v11/fish-nf4-setup
/root/hear-ai-v11/loadtest-*
/root/hear-ai-v11/performance-*
/root/hear-ai-v11/revised-limits-*
/root/hear-ai-v11/root-migration-*
/root/hear-ai-v11/simulation-*
/root/hear-ai-v11/sound-cleanup-build
/root/hear-ai-v11/*.log
/root/hear-ai-v11/admission.lock*
```

Remove old model payloads and caches:

```text
/models/*
/root/hear-ai-v11/models/*
old Hugging Face caches
old Torch caches
old model exports
```

Remove old workspace trees:

```text
/workspace/hear-ai
/workspace/.cache
old HEAR AI fixtures/downloads outside the canonical repo
```

Do **not** delete:

```text
/workspace/hear-ai-v11/.git
/root/hear-ai-config
```

---

# 4. Stop every old HEAR AI process first

Before cleanup or deployment:

```bash
tmux kill-session -t hear-ai-runtime 2>/dev/null || true

pkill -TERM -f 'hear.entrypoints' || true
pkill -TERM -f 'scripts.simulation_backend' || true
```

Verify:

```bash
ps -eo pid,args | grep -E '[h]ear.entrypoints|[s]imulation_backend|[r]un_pod_stack'
```

Expected:

```text
no HEAR AI runtime processes
```

Verify GPU:

```bash
nvidia-smi
```

Expected before the new container starts:

```text
GPU model memory ≈ 0 MiB
```

---

# 5. Bazel is the production build entrypoint

The final build remains Bazel-driven.

Required Bazel targets:

```bash
bazel build //:image_context //:runpod_image
bazel run //:runpod_image -- --dry-run
bazel build //:runpod_container
```

The Bazel context must contain:

```text
Dockerfile
hear/**
scripts required by production
patches/**
deploy/runtime/**
deploy/cleaner/**
```

The context must NOT contain:

```text
.env files
credentials
TLS keys
user recordings
MP3/WAV/FLAC test audio
host model directories
simulation-only scripts
large caches
```

---

# 6. The Docker build must provision the models

The `runpod-stack` image is self-provisioning.

The build stage must download and verify every required model from pinned sources.

The build must not copy models from the host.

## Pipeline models

Provision from `hear/model_manifest.json`:

```text
Qwen/Qwen3-ASR-1.7B
Qwen/Qwen3-ForcedAligner-0.6B
unitary/toxic-bert
cardiffnlp/twitter-roberta-base-sentiment-latest
cross-encoder/nli-distilroberta-base
```

Pipeline provisioning:

```bash
python -m hear.tools.model_provisioning \
  --role pipeline \
  --model-root /models
```

The pinned revisions in `hear/model_manifest.json` are authoritative.

---

# 7. WhisperX + Qwen transcription stack

The production transcription architecture is:

```text
WhisperX framework
    ↓
local Silero VAD
    ↓
Qwen3-ASR 1.7B
    ↓
Qwen3 Forced Aligner 0.6B
    ↓
word-level timestamps
```

WhisperX is patched by:

```text
patches/whisperx-asr-qwen.patch
```

The build must run:

```bash
python -m hear.tools.dependency_patches
python -m hear.tools.dependency_patches --check
```

The patch is required for:

```text
Qwen integration
forced-aligner support
batch limits
timestamp handling
device handling
VAD integration
language mapping
```

The runtime must fail if the expected WhisperX source revision does not match the patch manifest.

---

# 8. Magic Clean model provisioning

Provision DeepFilterNet3 during the Docker build:

```bash
python scripts/provision_magic_clean_models.py \
  --model-root /models \
  --engine deepfilter
```

Final image path:

```text
/models/magic-clean/DeepFilterNet3
```

DeepFilterNet remains the primary cleaner.

Do not restore:

```text
SAM-Audio
MossFormer
obsolete Magic Clean engines
```

---

# 9. Sound Cleanup assets

The production image also provisions:

```text
/models/sound-cleanup-v1-runtime
/models/sound-cleanup-specialist/runtime
```

These correspond to:

```text
PANNs event analysis
Silero speech protection
AudioSep selected overlap repair
```

Provisioning must validate the pinned manifest hashes.

Expected runtime environment:

```dotenv
HEAR_SOUND_CLEANUP_BUNDLE=/models/sound-cleanup-v1-runtime
HEAR_SOUND_CLEANUP_SEPARATOR_BUNDLE=/models/sound-cleanup-specialist/runtime
```

The build verifies pinned source checkpoints/revisions, checks export equivalence,
and writes the two manifest SHA256 values to `/models/sound-cleanup-release.env`.
The Pod and Serverless launchers parse this image-contained metadata using the
safe dotenv loader. Record these values with the image digest. Runtime verifies
the manifests and their asset hashes against this release metadata.
Do not reuse an older image's export hashes: serialized TorchScript exports can
produce different bytes when rebuilt.

Sound Cleanup stays opt-in.

Automatic bark/event thresholds are not yet considered calibrated enough to silently alter recordings.

Selected overlap repair remains approval-required.

---

# 10. Fish Speech NF4

Fish S2 Pro NF4 is provisioned during image build into:

```text
/models/fish-speech/s2-pro-nf4-runtime
```

Provisioning command:

```bash
python scripts/provision_fish_nf4.py \
  --model-root /models
```

Fish source is installed in:

```text
/opt/fish-speech
```

Fish reconstruction means:

```text
text-to-speech transcript editing
```

It does **not** mean FFmpeg-only splicing.

## Production licensing gate

The current manifest still contains:

```text
fish-speech-s2-pro
license_status = permission_required
```

Do not fake this approval.

The default image omits Fish weights. After legitimate permission, setting
`HEAR_FISH_LICENSE_APPROVED=true` for the Bazel image builder forwards the Docker
build argument to provision them. This flag does not change the license manifest:
production reconstruction remains blocked until the deployment policy/manifest
also records legitimate permission.

Pipeline and Magic Clean must not be blocked by the Fish licensing status.

---

# 11. Final Docker image layout

Expected final image:

```text
/app/hear
/app/scripts
/app/patches

/opt/hear-ai-v11/venvs/pipeline
/opt/hear-ai-v11/venvs/reconstruction
/opt/hear-ai-v11/venvs/magic_clean_natural

/opt/hear-ai-v11/shared

/opt/fish-speech

/models/
    qwen3-asr-1.7b/
    qwen3-forced-aligner/
    toxic-bert/
    twitter-roberta-sentiment/
    nli-distilroberta/
    magic-clean/DeepFilterNet3/
    sound-cleanup-v1-runtime/
    sound-cleanup-specialist/runtime/
    sound-cleanup-release.env
    fish-speech/s2-pro-nf4-runtime/
```

The Fish directories exist only in an approved build.

No model should be loaded from `/workspace`.

---

# 12. Fix Docker dependency deduplication

The previous image failed because the dedupe script used:

```python
Path.rename()
```

across Docker overlay filesystems.

That produced:

```text
OSError: [Errno 18] Invalid cross-device link
```

The corrected implementation must use:

```python
shutil.copytree(...)
```

followed by removal of the duplicate and creation of the shared symlink.

The production build must prove that this step completes.

---

# 13. Runtime environment files

The image must never contain production secrets.

Preferred runtime env:

```text
/root/hear-ai-config/production.env
```

Fallback:

```text
/root/hear-ai-config/runtime.env
```

`run_pod_stack.sh` should prefer:

```text
/root/hear-ai-config/production.env
/root/hear-ai-config/runtime.env
```

The production container should fail closed if a required env file is absent.

This file requirement applies to the Pod stack. Serverless reads deployment
settings directly from the endpoint environment and has no host env-file dependency.

Environment variables should be parsed using the existing safe dotenv loader.

Do not `source` arbitrary credential files as shell code.

---

# 14. Production runtime mode

Production must use:

```dotenv
HEAR_RUNTIME_MODE=production
```

The simulation backend must not run.

Do not run:

```text
scripts.simulation_backend
run_simulation_stack.sh
simulation-local backend identity
simulation S3 endpoint
```

---

# 15. Production worker layout

Pod target:

```text
Pipeline             1 worker process
Magic Clean          4 worker processes
Fish Reconstruction  2 worker processes when licensed
Gateway               1
RabbitMQ               local runtime broker
```

Reconstruction is omitted by default while its licensing gate is closed. A
Serverless worker runs one role, one SDK handler, and no local RabbitMQ or gateway.

No standalone Transcription GPU worker is required.

Transcription jobs route through the Pipeline Qwen/WhisperX stack.

That avoids a duplicate Qwen model.

---

# 16. Concurrency limits

Pod production settings (not a cluster-wide Serverless limit):

The block below includes the intended licensed reconstruction capacity. Until
that gate opens, the image defaults to Pipeline and Magic Clean only and omits
the reconstruction keys. Enabling licensed reconstruction requires explicitly
adding its role and capacities to the deployment environment.

```dotenv
HEAR_HOST_MAX_CONCURRENT_JOBS=10

HEAR_POD_ROLE_LIMITS={"pipeline":7,"magic_clean_natural":4,"reconstruction":2}

HEAR_POD_PROCESS_LIMITS={"pipeline":7,"magic_clean_natural":1,"reconstruction":1}

HEAR_WORKER_REPLICAS={"pipeline":1,"magic_clean_natural":4,"reconstruction":2}
```

The host-wide 10-job limit always wins for the Pod stack. It is a local admission
limit; separate Serverless workers need endpoint limits and backend admission.

The per-type limits are:

```text
Pipeline        7
Magic Clean     4
Reconstruction  2
```

---

# 17. Lazy GPU loading

Pod models start cold. The Serverless startup profile in section 31 preloads core
models and disables HEAR idle eviction while the worker remains alive.

The API and RabbitMQ consumers stay alive.

Production TTLs:

```dotenv
HEAR_GPU_IDLE_EVICTION_ENABLED=true

HEAR_PIPELINE_IDLE_TTL_SECONDS=600
HEAR_MAGIC_CLEAN_IDLE_TTL_SECONDS=300
HEAR_RECONSTRUCTION_IDLE_TTL_SECONDS=1200
HEAR_AUDIOSEP_IDLE_TTL_SECONDS=90
```

Flow:

```text
worker starts
    ↓
model cold
    ↓
job received
    ↓
single-flight model load
    ↓
job runs
    ↓
last borrower finishes
    ↓
idle timer
    ↓
new job?
    ├─ yes → keep warm
    └─ no → evict model from GPU
```

Fish eviction terminates its supervised inference child process.

This is not GPU-to-RAM offloading.

---

# 18. Bazel production build

After the production branch is complete:

```bash
cd /workspace/hear-ai-v11

git status --short

bazel clean

bazel build //:image_context //:runpod_image

bazel run //:runpod_image -- --dry-run
```

Then build the real OCI image from the Bazel context on a Buildx-capable host:

```bash
bazel run //:runpod_image -- \
  --target runpod-stack \
  --tag <registry>/hear-ai-runtime:<release-sha> \
  --push
```

Record:

```text
Git commit SHA
image tag
image digest
Bazel context SHA256
build date
Sound Cleanup exported manifest SHA256 values
```

`HEAR_IMAGE_REVISION` defaults to the Bazel context SHA256. Record the Git SHA
separately. The source context is deterministic; fetching pinned sources and
exporting models does not guarantee a byte-identical OCI image across rebuilds.

Deploy by image digest when possible.

For a local build and controlled container test on a Docker-capable GPU host:

```bash
bazel run //:runpod_image -- --target runpod-stack --tag hear-ai:production
bazel run //:runpod_container -- \
  --image hear-ai:production \
  --env-file /root/hear-ai-config/production.env
```

The container launcher requests GPUs, mounts only the external configuration
file read-only, and uses models packaged in the image. It rejects a configuration
that selects simulation mode. It runs in the foreground and publishes
`127.0.0.1:8000` for local acceptance checks. Use `--publish 0.0.0.0:8000:8000`
when the deployment's backend routing requires it. With a remote Docker context,
the env-file path must exist on the Docker host.

The `scripts/build_production_image.sh` convenience wrapper invokes these Bazel
build targets using Docker Buildx. The canonical image builder has no Kaniko
fallback.

---

# 19. Image acceptance tests

Before deployment verify:

```text
Pipeline environment imports
Reconstruction environment imports
Magic Clean environment imports
WhisperX patch check
Qwen model manifest
Fish model hashes
DeepFilter hashes
Sound Cleanup bundle hash
AudioSep bundle hash
no secrets in layers
no workspace model dependency
```

Run:

```bash
ruff check hear tests scripts

python -m hear.tools.check_architecture

python -m mypy

pytest tests -q
```

The previous finalized runtime passed:

```text
668 tests
12 skipped
mypy: 0 errors in 139 source files
Ruff: passed
architecture checks: passed
```

Run these again on the final canonical production branch.

---

# 20. Backend must be deployed before AI traffic is switched

The current backend source contains the new runtime code.

Required backend routes:

```text
GET  /api/v1/internal/ai/runtime/protocol
GET  /api/v1/internal/ai/runtime/catalog

POST /api/v1/internal/ai/attempts/{attempt_id}/claim
POST /api/v1/internal/ai/attempts/{attempt_id}/heartbeat
POST /api/v1/internal/ai/attempts/{attempt_id}/events
POST /api/v1/internal/ai/attempts/{attempt_id}/outcome
```

The observed live backend containers were still running an older image and returned `404` for the protocol route.

Therefore:

```text
backend source ready != backend production deployment ready
```

Redeploy the backend before enabling the v1 runtime.

---

# 21. Backend feature flag

Deploy the backend first with:

```dotenv
HEAR_AI_RUNTIME_V1=false
```

Verify routes and authentication.

Then deploy HEAR AI.

Run controlled production canaries.

Only after canaries pass:

```dotenv
HEAR_AI_RUNTIME_V1=true
```

---

# 22. Backend → HEAR AI job flow

Final production flow:

```text
hear-backend
    ↓
create ProcessingJob
    ↓
build AttemptEnvelope
    ↓
verify source hash/revision
    ↓
issue attempt grant
    ↓
POST HEAR AI /v1/attempts
    ↓
RabbitMQ
    ↓
worker claim
    ↓
heartbeat
    ↓
progress events
    ↓
model processing
    ↓
Backblaze upload
    ↓
outcome callback
    ↓
backend validates artifacts
    ↓
result processing / approval
```

The backend owns:

```text
job identity
source revision
storage scope
callback grant
result persistence
approval state
```

HEAR AI owns:

```text
model inference
artifact generation
progress events
```

---

# 23. SSE flow

The backend runtime callback path is already designed to feed the existing SSE infrastructure.

Flow:

```text
AI worker
    ↓
POST /internal/ai/attempts/{attempt_id}/events
    ↓
RuntimeAttemptService
    ↓
persist job.stage + job.progress_percentage
    ↓
database commit
    ↓
SSEPublisher.track_progress()
    ↓
Redis latest-event cache + pub/sub
    ↓
SSE service
    ↓
frontend
```

Track endpoint:

```text
GET /api/v1/sse/tracks/{track_id}/events
```

The stream supports:

```text
live Redis pub/sub
latest event replay
database fallback
Last-Event-ID
```

The frontend should not poll the AI processing endpoint.

---

# 24. SSE verification

Run backend tests:

```bash
pytest \
  tests/test_ai_runtime_v1.py \
  tests/test_ai_sse_lifecycle.py \
  tests/test_sse_startup.py \
  -q
```

Production canary must verify:

```text
job submitted
    ↓
SSE: submitted/enqueued
    ↓
SSE: processing
    ↓
stage changes visible
    ↓
progress percentage visible
    ↓
reconnect
    ↓
latest event replayed
    ↓
job completes
    ↓
SSE completion / awaiting approval event
```

SSE must keep working independently of the normal web API workers.

---

# 25. Production canaries

Run these before turning the v1 flag on globally.

## Pipeline

Prove:

```text
backend
→ /v1/attempts
→ WhisperX + Qwen ASR
→ Qwen Forced Aligner
→ categorization/moderation
→ artifact upload
→ outcome
→ SSE completion
```

## Transcription

Prove:

```text
job_type=transcription
→ Pipeline queue
→ same Qwen/WhisperX stack
→ no dedicated transcription GPU model
→ transcript manifest
→ word timestamps
→ backend result
→ SSE
```

## Magic Clean

Prove:

```text
DeepFilterNet3
→ full cleaned FLAC
→ MP3
→ validation JSON
→ backend artifact validation
→ approval required
→ SSE
```

## Sound Cleanup

Prove selected repair:

```text
AudioSep loads only when required
selected region changes
outside selected region unchanged
speech protection passes
approval required
```

## Background cleanup

Prove:

```text
stationary cleanup executes
no high-confidence speech-loss frames
result marked for listening review
```

Do not claim certified hum removal until perceptual acceptance is complete.

## Fish

Run only after legitimate production licensing approval.

---

# 26. GPU eviction production check

Start the image with no jobs.

Expected:

```text
API ready
RabbitMQ consumers ready
models cold
GPU close to zero
```

Submit one job.

Expected:

```text
required model loads
job completes
model stays warm
```

Wait past the configured TTL.

Expected:

```text
model weights evicted
API still ready
RabbitMQ still connected
worker process still alive
```

A small CUDA-context footprint after first use is acceptable.

---

# 27. Production cutover sequence

Execute in this order.

1. Stop every existing HEAR AI process.
2. Clean stale host directories and loose model payloads.
3. Consolidate source into `release/hear-ai-production-v11`.
4. Ensure `ba2f015` self-provisioning image changes are in the branch.
5. Run complete code tests.
6. Run Bazel build.
7. Build and push the `runpod-stack` OCI image.
8. Record image digest.
9. Deploy updated backend with `HEAR_AI_RUNTIME_V1=false`.
10. Verify runtime protocol/catalog routes.
11. Deploy HEAR AI production container using the new image.
12. Mount/load `/root/hear-ai-config/production.env`.
13. Verify `/healthz`.
14. Verify `/readyz`.
15. Verify `/capabilities`.
16. Verify `/metrics`.
17. Run Pipeline production canary.
18. Run transcription production canary.
19. Run Magic Clean production canary.
20. Verify SSE for every canary.
21. Verify GPU idle eviction.
22. Enable `HEAR_AI_RUNTIME_V1=true`.
23. Monitor jobs, queues, GPU, callbacks and SSE.
24. Merge release branch into `main`.
25. Remove obsolete branches and remaining temporary build artifacts.

---

# 28. Rollback

If a production canary fails:

```dotenv
HEAR_AI_RUNTIME_V1=false
```

This stops new v1 job submissions.

Keep:

```text
previous backend image digest
previous HEAR AI image digest
new release image digest
```

Do not overwrite source audio.

AI artifacts are attempt-scoped, so rollback does not require restoring originals.

Do not silently remap legacy reconstruction jobs.

---

# 29. Definition of done

The migration is complete only when all of these are true.

- [ ] Only the canonical HEAR AI source repository remains on RunPod.
- [ ] No loose HEAR AI model files remain on the host.
- [ ] No stale simulation/load-test/audit directories remain.
- [ ] Production env is outside Git and Docker image.
- [ ] `release/hear-ai-production-v11` contains all required production commits.
- [ ] Bazel context builds successfully.
- [ ] Buildx produces the complete `runpod-stack` image.
- [ ] Pipeline models are provisioned inside the image.
- [ ] DeepFilterNet3 is provisioned inside the image.
- [ ] Sound Cleanup assets are provisioned inside the image.
- [ ] Fish NF4 assets are provisioned only in an approved image; otherwise Fish remains disabled.
- [ ] WhisperX patch is applied and verified.
- [ ] Full HEAR AI tests pass.
- [ ] Updated backend image is deployed.
- [ ] Runtime protocol route returns 200 instead of 404.
- [ ] Production HEAR AI API is running.
- [ ] Pipeline canary completes.
- [ ] Transcription canary completes.
- [ ] Magic Clean canary completes.
- [ ] SSE displays real progress without polling.
- [ ] GPU models lazy-load on demand.
- [ ] GPU models evict after idle TTL.
- [ ] Fish is enabled only after legitimate production permission.
- [ ] `HEAR_AI_RUNTIME_V1=true` is enabled only after all prior gates.
- [ ] Release branch is merged into `main`.
- [ ] Obsolete branches/worktrees are removed.

---

# 30. Final target

This is the Pod cutover target. Serverless migration uses the same source release
with the role images and additional acceptance gates in section 31.

```text
ONE source branch
ONE Bazel build
ONE production runpod-stack image

models packaged into image
secrets external
Pod: models cold on startup, lazy-load, evict after idle
Serverless: core models preload, stay warm until worker termination

backend owns jobs
HEAR AI owns inference
Backblaze owns artifacts
SSE owns live UI progress

no loose models
no stale worktrees
no simulation runtime
no frontend polling
```

This is the production state HEAR AI should end in.

---

# 31. RunPod Serverless compatibility and latency gates

## Images and entrypoints

Build role-specific images from the same Bazel context. Do not launch
`runpod-stack` as a queue-based Serverless worker: its command starts a gateway
and RabbitMQ consumers instead of the RunPod SDK handler.

| Docker target | Jobs | Image-contained assets |
| --- | --- | --- |
| `pipeline-serverless` | Pipeline and transcription | Qwen ASR, forced aligner, three classifiers, packaged Silero VAD |
| `transcription-serverless` | Transcription only; optional separate endpoint | Qwen ASR, forced aligner, packaged Silero VAD |
| `magic-clean-natural-serverless` | Magic Clean and requested Sound Cleanup | DeepFilterNet3, verified PANNs/Silero/AudioSep exports |
| `reconstruction-serverless` | Reconstruction after licensing approval | Fish NF4 and codec in an approved build |
| `pipeline-llm-serverless` | Optional Pipeline with Qwen LLM | Pipeline assets plus pinned Qwen LLM |

Use `pipeline-serverless` for transcription so both job types reuse the same Qwen
instance. Keep the optional LLM image out of the default release; it adds image
size and VRAM demand. The cleaner provisioning environment and Rust compiler
remain in build stages. Final workers contain one role environment and its
models, without the all-role stack or local broker.

Model provisioning includes each manifest's declared weight files and small
tokenizer/configuration assets. Alternate weight formats and unused binary
exports are excluded to reduce image transfer and storage overhead.

```bash
bazel run //:runpod_image -- \
  --target pipeline-serverless \
  --tag <registry>/hear-ai-pipeline:<release-sha> \
  --push

bazel run //:runpod_image -- \
  --target magic-clean-natural-serverless \
  --tag <registry>/hear-ai-cleaner:<release-sha> \
  --push
```

Each command starts `scripts/run_serverless.sh`, loads any build-generated Sound
Cleanup hash metadata, then executes `python -m hear.entrypoints.serverless`.
Dependencies and models are installed during build. Offline Hugging Face and
Transformers settings remain enabled in the final image. Supply endpoint secrets
as environment variables, including `HEAR_BACKEND_INTERNAL_URL`,
`HEAR_BACKEND_SERVICE_KEY` for Pipeline, and `HEAR_ENGINE_REVISION`.
`HEAR_WORKER_ID` derives from the RunPod worker identity unless explicitly supplied;
generation is unique per process. Do not share a fixed identity among replicas.

Use writable local `/audio` scratch with enough storage for the configured job
budget; Magic Clean defaults to at least 8 GiB free. Models remain under `/models`.
No model provisioning, dependency installation, export, or runtime setup script
may execute when the worker starts or handles a job.

## Startup and model lifetime

Serverless images default to:

```dotenv
HEAR_RUNTIME_MODE=production
HEAR_SERVERLESS_PRELOAD_MODELS=true
HEAR_GPU_IDLE_EVICTION_ENABLED=false
HEAR_SERVERLESS_MAX_CONCURRENT_JOBS=1
WHISPER_BATCH_SIZE=8
WHISPER_LONG_AUDIO_BATCH_SIZE=8
WHISPER_CHUNK_SECONDS=240
OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=2
```

Core Qwen/aligner/classifier, DeepFilter, or licensed Fish models preload before
SDK polling begins. Startup emits a duration for each preloaded resource. Readiness
must pass before and after preload; a failed preload exits after closing resources.
Preloading with HEAR idle eviction enabled is rejected before model allocation.
This avoids unloading a resident model while RunPod keeps an active worker ready.
Optional Sound Cleanup and AudioSep stay on demand and reuse their loaded state.

For deliberate scale-to-zero operation, setting preload to `false` allows lazy
loading. Its first request pays model initialization; do not describe this profile
as having zero model-load latency.

The RunPod scheduler receives the same concurrency limit as HEAR's admission
semaphore. Start with one job per worker. Increase Pipeline concurrency only after
measuring throughput, tail latency and peak VRAM on the target GPU. Qwen's native
inference lane serializes model calls while downloads/uploads can overlap.
DeepFilter admits one inference session per worker; Fish requires concurrency one.
The Pod's host lock and 10-job limit do not coordinate separate Serverless workers.
Preserving a global limit requires backend admission across endpoints.

## Endpoint configuration

Use a queue-based endpoint and submit canonical `AttemptEnvelope` payloads under
RunPod's `input` field through `/run`. The backend needs a RunPod submission adapter;
the Pod's `POST /v1/attempts` transport does not automatically switch to Serverless.
Verify that adapter before migrating traffic. The existing handler retains backend
claim, heartbeat, progress, artifact and outcome callbacks, so SSE continues through
the backend. RunPod progress/results are supplementary and do not replace persistence.

For latency-sensitive traffic, keep at least one active worker **with core models
preloaded**. Active workers incur idle charges. Enable FlashBoot and verify the
actual setting after deployment. For flex workers, choose an idle timeout matching
traffic gaps; 300 seconds is an initial measurement candidate, not a guarantee.
The provider's shutdown timer is independent of HEAR's model TTLs. Configure
execution timeout and job TTL to cover long recordings, queueing, initialization,
uploads and callbacks. Align these with each attempt deadline and storage expiry.
Select CUDA 12.8-compatible drivers and benchmark any GPU fallback before enabling it.
[RunPod endpoint settings](https://docs.runpod.io/serverless/endpoints/endpoint-configurations)
and [optimization guidance](https://docs.runpod.io/serverless/development/optimization)
describe these controls; [concurrent handlers](https://docs.runpod.io/serverless/workers/concurrent-handler)
document scheduler concurrency.

## Required evidence before claiming Serverless readiness

- [ ] Build each selected image from the Bazel context and record its digest, size and model/export hashes.
- [ ] Start it on the target GPU with endpoint secrets and no host model/env mounts.
- [ ] Confirm models load from the image while runtime model-download network access is unavailable.
- [ ] Confirm preload finishes before SDK polling and warm jobs do not reload core weights.
- [ ] Confirm repeated cleaner readiness checks reuse verified hashes and changed assets fail readiness.
- [ ] Exercise canonical Pipeline, transcription and Magic Clean jobs through the actual RunPod SDK and endpoint.
- [ ] Verify backend claims, heartbeats, progress, outcomes, artifacts and SSE, including retries and duplicate delivery.
- [ ] Measure fresh-worker image initialization, model preload, first-job time, warm-job p50/p95, real-time factor, and peak VRAM separately.
- [ ] Repeat after provider idle shutdown, with FlashBoot revival, and with a new uncached worker.
- [ ] Test supported maximum recording duration, cancellation, timeout, scratch cleanup and concurrent jobs.
- [ ] Record explicit latency/throughput budgets and pass them on the deployed GPU before switching backend traffic.

Image-contained weights remove runtime download overhead. Preloading plus active
workers avoids core weight loading on requests served by those warm workers.
Fresh workers and scale-out still have startup costs; FlashBoot is an optimization,
not proof of a latency guarantee. Local unit tests do not establish GPU performance
or confirm the external backend transport is deployed.

---

# 32. Execution evidence — 2026-10-03

| Check | Result |
| --- | --- |
| Full AI test suite | 687 passed, 12 skipped, 3 dependency/serialization warnings in 168.80 seconds; subsequent focused configuration and image checks also passed |
| Static checks | Ruff, mypy (140 source files), architecture and shell syntax passed |
| Bazel production input | `//:image_context`, `//:runpod_image`, `//:runpod_container` passed; context SHA256 `7ef75975d2ad2fbcfc7f8e8df1021f91566d8ee54b1900609175e66a5b53d2e6`, 173 runtime files; archive contents and file hashes verified; credentials, audio and host weights excluded |
| Full Docker build | Actual Bazel → Docker build on backend Docker host; candidate image ID `sha256:ee06c90f6668bb0ff041a4bbd70375f4d47e64bc25f90e78ae8e9e74d3dfa5f4`; Docker layer export and unpack succeeded; the subsequent `docker save` exceeded the SSH 55-minute limit; [hear-ai publication workflow 37125681683](https://github.com/Techta-Labs-Ltd/hear-ai/actions/runs/37125681683) passed, with verified unchanged runtime files and configuration except repository labels |
| External production environment | 46 populated settings, matched to live backend credentials and storage ownership policy; mode 0600, parent 0700; configured on this Pod and Docker builder host |
| Authenticated production preflight | HTTP 200; matching protocol; no configuration blockers, missing configured models or license blockers; configuration check does not establish image/GPU inference |
| Live backend protocol/catalogue | [PR 87](https://github.com/Techta-Labs-Ltd/hear-backend/pull/87) merged and deployed at `dac6f7e73c2aaa35bad59ff824a2a91a6a6e89d5`; authenticated protocol and catalogue pass; catalogue version 1340458090, 2,587 categories, 21,218 tags, 5 keyword rules and 3 harm keywords |
| Backend validation | 1,942 tests passed, 26 skipped; lint and type checks passed; production deploy workflow 37118466944 succeeded |
| Real Docker cleaning canary | Passed on the CPU Docker builder, using live backend and B2; job `d87b3a35-5773-41a8-a0fe-b489f9ec35c3`, attempt `396b0991-b49e-4ddb-bf20-034ab50a7f9d`; HTTP 202, 5 progress events, outcome persisted, result applied, completion SSE delivered, approval preview saved, original preserved, B2 digest verified, duplicate delivery fenced; no GPU acceptance claimed |
| Current RunPod access | Injected API key rejected by REST/GraphQL; saved account authenticates but has no Pods or team memberships and cannot access `0as9lqk138vfwz`; backend deployment and AI repository/production Actions environments have no RunPod key |
| Dispatch cutover | `HEAR_AI_RUNTIME_V1=false`; traffic has not switched |
| GPU acceptance / Serverless | Not yet verified for this Docker image; required before claiming GPU or Serverless readiness |

The current Pod cannot perform nested Docker builds because the required bind
mount is denied. The real image was therefore built on the existing authorized
Docker host through the backend's production SSH workflow. No host model directory
was supplied to the build or container. The Docker launcher defaults to all GPUs;
`--gpus none` supports an explicit CPU diagnostic canary and `--detach` supports
supervised startup.

The additive backend protocol/catalogue fix uses the existing internal service-key
protection and live database catalogue. It does not enable runtime dispatch.
The tested source archive and publishing workflow are stored in the `hear-ai`
repository on `ops/hear-ai-runpod-access-20261003`. This remains an isolated
ops branch rather than the canonical AI release branch.

Private local configuration evidence is in
`/root/hear-ai-build/production-live-config-accepted.json`. On the Docker builder,
image artifacts and private canary logs are under `/var/lib/hear/ai-cutover`.
The image build workflow is
[37117480599](https://github.com/Techta-Labs-Ltd/hear-backend/actions/runs/37117480599).
Credentials remain outside the image and repository.

The successful real canary workflow is
[37120370492](https://github.com/Techta-Labs-Ltd/hear-backend/actions/runs/37120370492).
The local receipt is `/root/hear-ai-build/real-cleaner-canary-receipt.json`.
Its 11-second DeepFilter output is 265,581 bytes with SHA256
`c5a306e222efaba0936ea29584654f2a0d33e7913f43e07bd2aad4ba7bc96658`.
The worker attempt completed. The backend job correctly remains
`awaiting_approval`, with its preview applied at `2026-10-03T11:32:30.862508Z`.
The CPU canary container was stopped after the check.

The published candidate is:

```text
ghcr.io/techta-labs-ltd/hear-ai:cutover-7ef75975d2ad
ghcr.io/techta-labs-ltd/hear-ai@sha256:0fd7977c9e3b5f9225daf290206ef06ddaa75bf123fb6ce02781747dc6e1c3f5
```

The package is linked to `Techta-Labs-Ltd/hear-ai` and was published by the
[hear-ai publishing workflow](https://github.com/Techta-Labs-Ltd/hear-ai/actions/runs/37125681683).
Registry read access using that repository's own token, the package repository
association, and OCI source labels were verified. The runtime filesystem and
configuration match the tested candidate; only repository labels changed.
The published image configuration is `sha256:6ca153ff7446fbc2a7777e8803be5ef2d573f8d5ed0cd5039071317cfc993ed0`.
The original canary configuration was
`sha256:559feefe3008f198b773cfc20c54375b5285e468fa1265cbbf7d97f1602e4818`.
The original Bazel-built Docker archive was transferred with its full SHA256
verified, then published using `bazel run //:publish_existing_runpod_image`.

On a Docker-capable GPU host with registry pull access and the protected env file:

```bash
bazel run //:runpod_container -- \
  --image ghcr.io/techta-labs-ltd/hear-ai@sha256:0fd7977c9e3b5f9225daf290206ef06ddaa75bf123fb6ce02781747dc6e1c3f5 \
  --env-file /root/hear-ai-config/production.env \
  --name hear-ai-production --publish 0.0.0.0:8000:8000 --detach
```

GPU deployment of this candidate has not occurred. The provided RunPod key also
returns Unauthorized using the documented GraphQL `api_key` query parameter.
The saved account does not own the current Pod. Owning-account access remains
necessary to deploy and validate the image on GPU before enabling dispatch.

The Docker archive completed after the original SSH timeout and was independently
verified against the tested image configuration and Bazel context. All expected
layer files are present, runtime mode is production, and production credentials
are absent from image configuration. Its size is 14871298048 bytes.

Archive verification workflow: [37121187717](https://github.com/Techta-Labs-Ltd/hear-backend/actions/runs/37121187717).
Archive SHA256: `9d590fe8a6c1a4093583f618632907fc594a8b9d8234815507bd4043ef5aa0f5`.

The obsolete `hear-ai-production` package and backend AI publishing files were
removed after successful `hear-ai` publication. The temporary authenticated
archive transfer service was stopped and removed.
Cleanup verification: [37127018275](https://github.com/Techta-Labs-Ltd/hear-backend/actions/runs/37127018275).

## 33. Pod-only setup and host-model offload, 2026-10-03

Use the already published Bazel-built image:

```text
ghcr.io/techta-labs-ltd/hear-ai@sha256:0fd7977c9e3b5f9225daf290206ef06ddaa75bf123fb6ce02781747dc6e1c3f5
```

Create or edit a RunPod custom template in the account owning the Pod. Configure
its GHCR registry authentication with a GitHub username and a classic token with
`read:packages` access to the private `hear-ai` package. Attach the existing
persistent volume at `/workspace`; use a 100 GB container disk and expose HTTP
port `8000`.

The existing production environment has been encrypted at
`/workspace/hear-ai-deploy/production.env.enc`. Its decryption key is stored only
in `/root/hear-ai-config/config-decryption.key`, mode `0600`. Add the value from
that key file to a RunPod secret, then provide it to the container as the
`HEAR_CONFIG_KEY` environment variable. Copy it into the secret before replacing
the current container, since `/root` is ephemeral. Keep the key outside Git.

Set the container startup command to:

```bash
bash /workspace/hear-ai-deploy/start-production.sh
```

The startup script verifies the encrypted file checksum, decrypts the configured
production environment, verifies the plaintext checksum, writes it under
`/root/hear-ai-config/production.env` with mode `0600`, and starts the image's
Pod stack. The encryption/decryption roundtrip and startup shell syntax passed.
The persistent volume does not honor restrictive POSIX modes; its temporary
plaintext configuration copy was removed. Only the encrypted environment is
stored there.

After deployment, the gateway base URL is:

```text
https://<POD_ID>-8000.proxy.runpod.net
```

Check `/healthz`, `/readyz`, and `/capabilities`. Require `/readyz` HTTP 200 with
production mode and ready pipeline/cleaning lanes. Set the backend `HEAR_HTTP_URL`
to that gateway base URL and ensure `HEAR_AI_INGRESS_TOKEN` matches the worker's
`HEAR_POD_API_KEY`. Keep `HEAR_AI_RUNTIME_V1=false` until the deployed Docker image
passes real GPU Pipeline and cleaning job acceptance. A live GPU deployment has
not occurred in this workspace; nested Docker is unavailable in the current Pod.

On a separate Docker-capable GPU host, the existing Bazel launcher remains valid:

```bash
bazel run //:runpod_container -- \
  --image ghcr.io/techta-labs-ltd/hear-ai@sha256:0fd7977c9e3b5f9225daf290206ef06ddaa75bf123fb6ce02781747dc6e1c3f5 \
  --env-file /root/hear-ai-config/production.env \
  --publish 0.0.0.0:8000:8000 --detach
```

All 173 host-model files were copied and SHA256-verified at
`/workspace/hear-ai-models`. Root disk space recovered: 12,826,763,264 bytes.
`/models` is now a compatibility symlink to that archive. Pipeline and
transcription model manifests validated with no missing assets after the move.
The archive is a backup: the production runtime rejects model storage under
`/workspace` and uses the models packaged at `/models` inside its Docker image.
That storage guard remains in place. No models are currently loaded on the host
GPU: 0 MiB memory used and no compute processes.

Offload receipt: `/workspace/hear-ai-models-offload-20261003.json`.

Reference documentation:
- [RunPod custom templates](https://docs.runpod.io/pods/templates/create-custom-template)
- [RunPod exposed ports](https://docs.runpod.io/pods/configuration/expose-ports)
- [GHCR authentication](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)

## 34. Current RunPod Serverless setup, 2026-10-03

Deploy two **queue-based Serverless endpoints** from these published images.
The pipeline image handles pipeline and transcription jobs; the cleaner image
handles natural cleaning jobs. Image build, publication and Docker startup
structure checks passed through Bazel in
[hear-ai workflow 37130037490](https://github.com/Techta-Labs-Ltd/hear-ai/actions/runs/37130037490).
The RunPod SDK loads, required model files are packaged, and the default command
is `bash /app/scripts/run_serverless.sh`. These checks ran without a GPU; they
are not real Serverless inference acceptance.

| Endpoint | Published Docker image |
| --- | --- |
| Pipeline + transcription | `ghcr.io/techta-labs-ltd/hear-ai:pipeline-serverless-20261003` |
| Natural cleaner | `ghcr.io/techta-labs-ltd/hear-ai:cleaner-serverless-20261003` |

Use immutable image references when creating templates:

```text
Pipeline:
ghcr.io/techta-labs-ltd/hear-ai@sha256:7e1a31a220cac843a47153435c502d96d2297a3f49bc783cd4fe594ef61be43c

Cleaner:
ghcr.io/techta-labs-ltd/hear-ai@sha256:2eeefddf57476729f816681be90dc5f03674b13af33ed0ced44cb7a381443802
```

Both images used Bazel context SHA256
`7cbb82c9e43da883189ec4a5c6f63355dffc49079e345ec8c8a7622bb7226001`.
Publication receipts are in `docs/verification/production-cutover-20261003.json`.
Images and image publishing remain owned by `Techta-Labs-Ltd/hear-ai`.

In RunPod, select **Serverless → New Endpoint → Import from Docker Registry**.
Create one endpoint for each image. Select queue mode, one A40 GPU per worker,
a 50 GB container disk, active workers 0, max workers 1, idle timeout 5 seconds,
and FlashBoot enabled. Start with an endpoint execution timeout of 7,200 seconds;
the backend submission policy additionally limits each job to its attempt
deadline. Leave Docker entrypoint/start command overrides empty so the image's
Serverless command runs. No exposed HTTP port or Pod proxy URL is required.
Configure RunPod registry authentication for the private GHCR package using a
GitHub username and a classic token with `read:packages` access.

The role environments are already populated with the live backend URL, service
credential and ownership policy, outside Git, mode `0600`:

- Pipeline: `/root/hear-ai-config/serverless-pipeline.env` and
  `/root/hear-ai-config/serverless-pipeline.json`.
- Cleaner: `/root/hear-ai-config/serverless-cleaner.env` and
  `/root/hear-ai-config/serverless-cleaner.json`.

Use the JSON values as template/endpoint environment variables, or put sensitive
values into RunPod secrets and reference them in the environment. Private REST
API deployment plans, including the immutable images and actual worker envs,
are `/root/hear-ai-config/serverless-pipeline-deployment.json` and
`/root/hear-ai-config/serverless-cleaner-deployment.json`. These plans were applied using the accepted replacement RunPod key.
The returned template and endpoint IDs are in section 35. The private-registry
authentication ID still needs to be attached to each template. Preserve these private files before replacing the
current container because `/root` is ephemeral.

Common worker settings are:

```text
HEAR_RUNTIME_MODE=production
HEAR_MODEL_ROOT=/models
HEAR_TEMP_DIR=/audio
HEAR_SERVERLESS_PRELOAD_MODELS=true
HEAR_GPU_IDLE_EVICTION_ENABLED=false
HEAR_SERVERLESS_MAX_CONCURRENT_JOBS=1
```

The pipeline role is `HEAR_WORKER_ROLE=pipeline`; the cleaner role is
`HEAR_WORKER_ROLE=magic_clean_natural`. Serverless preloads its image models when
a worker starts. RunPod terminates idle workers when active workers is zero,
which releases all their GPU models. Keep HEAR's separate idle eviction disabled
with Serverless preloading; the entrypoint rejects enabling both.
The host root model offload is already complete (section 33).

After RunPod creates each endpoint, its URL is:

```text
https://api.runpod.ai/v2/<ENDPOINT_ID>
POST /run
GET /status/<RUNPOD_JOB_ID>
GET /health
```

The `/run` body must contain the backend's signed attempt under `input`.
The provider API key authenticates requests to RunPod; the HEAR service key
still authenticates worker callbacks to the backend. They serve separate roles.

The backend previously submitted only to Pod `/v1/attempts`. The required
Serverless transport is now in
[draft backend PR 88](https://github.com/Techta-Labs-Ltd/hear-backend/pull/88).
It preserves the canonical signed envelope, uses the remaining attempt lifetime
for RunPod execution timeout and queue TTL, validates acceptance, and records
the provider job and endpoint IDs. Transcription uses the pipeline endpoint.
Full CI passed: 1,995 tests passed, 26 skipped; lint and type checking passed.
The adapter was merged and deployed successfully by workflow 37131507076.
The isolated canary configuration is prepared with the actual endpoint IDs.
Production runtime dispatch stays disabled pending acceptance. Configure:

```text
HEAR_AI_TRANSPORT=serverless
HEAR_RUNPOD_API_KEY=<valid RunPod provider key>
HEAR_RUNPOD_ENDPOINTS_JSON={"pipeline":"<PIPELINE_ENDPOINT_ID>","magic_clean_natural":"<CLEANER_ENDPOINT_ID>"}
```

Keep `HEAR_AI_RUNTIME_V1=false` until a real pipeline, transcription and cleaner
attempt has passed through the deployed Serverless workers with backend claim,
heartbeat, progress, artifact readback and terminal outcome verified. Confirm
workers return to zero afterward. The earlier real CPU cleaner canary validates
the backend contract; it does not validate RunPod Serverless GPU execution.

The first RunPod key was rejected. Its replacement was accepted and both
endpoints were created (section 35). Current blocker: the private GHCR images
require a registry pull credential, which is absent from the RunPod account.
Production is not yet ready to receive jobs through Serverless.

References: [RunPod worker deployment](https://docs.runpod.io/serverless/workers/deploy),
[endpoint settings](https://docs.runpod.io/serverless/endpoints/endpoint-configurations),
[queue requests](https://docs.runpod.io/serverless/endpoints/send-requests),
and [endpoint creation API](https://docs.runpod.io/api-reference/endpoints/POST/endpoints).


## 35. Created Serverless endpoints, 2026-10-03

The replacement RunPod API key was accepted by the management API at
16:55 Africa/Lagos. Both queue endpoints were created in that account and their
saved templates were independently read back. Each image digest and all 37
worker environment values match its protected deployment plan.

| Role | Endpoint name | Endpoint ID | Template ID |
| --- | --- | --- | --- |
| Pipeline + transcription | `hear-ai-pipeline-20261003` | `8ewxonopk5ex1p` | `rdumefwltc` |
| Natural cleaner | `hear-ai-cleaner-20261003` | `e2ysmfujllh9ur` | `l1swqghx1x` |

Actual job submission URLs:

```text
https://api.runpod.ai/v2/8ewxonopk5ex1p/run
https://api.runpod.ai/v2/e2ysmfujllh9ur/run
```

Both endpoint `/health` routes accepted authenticated requests. Active workers
is 0; maximum workers is 1 per endpoint; idle timeout is 5 seconds; GPU is A40
with one GPU per worker; container disk is 50 GB. Queue acceptance of a real
signed HEAR job and GPU inference have not been verified.

Backend PR 88 was merged and deployed successfully by
[workflow 37131507076](https://github.com/Techta-Labs-Ltd/hear-backend/actions/runs/37131507076).
The real backend canary driver and protected provider configuration are prepared
on the production backend. Live import of the Serverless adapter and disabled
production runtime dispatch were verified by
[workflow 37136562408](https://github.com/Techta-Labs-Ltd/hear-backend/actions/runs/37136562408).
The canary driver has not executed because the RunPod image pull credential is
still missing. Provider credentials are encrypted to a
private key held only on the backend host; no plaintext key is committed.
The canary changes settings only in its own driver process. General production
runtime dispatch remains disabled pending real acceptance.

The remaining setup requirement is registry authentication. Anonymous GHCR
pull was rejected, and the RunPod registry-auth list is empty. Add a GHCR
credential at RunPod Credentials → Container Registry Auth, using a GitHub
username and a classic token with `read:packages` access to the private
`Techta-Labs-Ltd/hear-ai` package. Attach its ID to both templates, then validate
real cleaning, pipeline and transcription jobs and worker scale-to-zero.

Private endpoint receipt: `/root/hear-ai-config/serverless-endpoints.json`.
Public verification receipt: `docs/verification/production-cutover-20261003.json`.
