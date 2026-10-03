# Hear AI

Python 3.12 audio workers for Pipeline, Transcription, Fish reconstruction, and
Magic Clean. Pod and RunPod Serverless entrypoints share contracts, execution,
workflows, inference engines, and Backblaze B2 artifact storage.

The backend owns durable jobs, dispatch, retries, progress, and approval. Workers
claim an attempt, process its authorized source, upload scoped artifacts, and
report events and the outcome to the owning backend.

| Job | Worker role | Processing |
| --- | --- | --- |
| `pipeline` | `pipeline` | Transcription, moderation, categorization, discovery, and audio export |
| `transcription` | `pipeline` or `transcription` | Qwen ASR with aligned timestamps |
| `reconstruction` | `reconstruction` | Fish Speech S2 Pro (bf16) edits and timeline assembly |
| `magic_clean` | `magic_clean_natural` | DeepFilterNet3 with four cleaning profiles |

See [audio jobs](docs/AUDIO_JOBS.md) for cleaning, optional event repair, and Fish
editing; [model notices](docs/SOUND_CLEANUP_MODEL_NOTICES.md) preserve asset provenance.

## Install and provision

Use Python 3.12, `uv`, and the role groups locked in
[deploy/runtime/pyproject.toml](deploy/runtime/pyproject.toml) and
[uv.lock](deploy/runtime/uv.lock). FFmpeg is required. Pods also need RabbitMQ
with [rabbitmq.conf](deploy/runtime/rabbitmq.conf) and its loopback AMQP listener.

From the checkout root, install each selected role:

```bash
python3.12 scripts/setup_runtime.py --role pipeline --provider pod
python3.12 scripts/setup_runtime.py --role magic_clean_natural --provider pod
```

Role environments live under `/opt/hear-ai-v11/venvs/<role>`; Serverless setup uses
`<role>-serverless`. Add `--feature qwen_llm` only for the optional pipeline LLM.
The setup applies the pinned Qwen/WhisperX patch from [patches](patches/manifest.json);
readiness verifies it again. Missing or changed dependency patches prevent readiness.

Provision verified model files before starting a worker:

```bash
/opt/hear-ai-v11/venvs/pipeline/bin/python -m hear.tools.model_provisioning \
  --role pipeline --model-root /models
/opt/hear-ai-v11/venvs/magic_clean_natural/bin/python \
  scripts/provision_magic_clean_models.py --model-root /models --engine deepfilter
```

Pass `--feature qwen_llm` when provisioning models for that feature. Model roots
must be outside this source checkout and off RunPod network volumes
(`/workspace`, `/runpod-volume`), including through symlinks; the runtime refuses
to start otherwise, because those FUSE mounts are slow and shared. Weights live on
the container root disk at `/models`: baked into Serverless and Pod images, or
provisioned once on a development Pod. A symlinked model root is canonicalized at
startup since engines open pinned assets with `O_NOFOLLOW`. Keep scratch and
download caches outside the repo. Workers load models offline; model identities and approval
status are in [model_manifest.json](hear/model_manifest.json).

Serverless targets provision their role assets during image build. Fish requires
its pinned upstream source, the official bf16 weights, and licensing approval before
production deployment; see the [Fish instructions](docs/AUDIO_JOBS.md#reconstruction).

## Configure and start

Start with [.env.example](.env.example). Supply real backend credentials, ownership
policy, revisions, and storage/transport settings through a secret store or an
external environment file. The Pod launcher accepts `HEAR_ENV_FILE`; otherwise it
looks for `/root/hear-ai-config/production.env`, then
`/root/hear-ai-config/runtime.env`, legacy `/root/hear-ai-v11/runtime.env`, and `.env`.
Production fails if its selected environment file is absent.

| Setting | Purpose |
| --- | --- |
| `HEAR_WORKER_ROLE`, `HEAR_POD_STACK_ROLES` | Role and explicit Pod consumer lanes |
| `HEAR_WORKER_ID`, `HEAR_WORKER_GENERATION` | Worker identity and lease fencing; generated if omitted |
| `HEAR_IMAGE_REVISION`, `HEAR_ENGINE_REVISION` | Immutable software/model identity |
| `HEAR_MODEL_ROOT`, `HEAR_TEMP_DIR` | External models and writable attempt scratch |
| `HEAR_MODEL_PATHS_JSON`, `HEAR_MAGIC_CLEAN_MODEL_DIR` | Per-model weight directory overrides keyed by manifest logical name; cleaner checkpoint directory |
| `HEAR_MIN_FREE_SCRATCH_BYTES` | Readiness free-space threshold |
| `HEAR_BACKEND_INTERNAL_URL` | Backend callback base including its API prefix |
| `HEAR_BACKEND_SERVICE_KEY` | Pipeline catalogue and backend protocol credential |
| `HEAR_PIPELINE_CATALOG_BACKEND_ID` | Backend that owns this pipeline catalogue |
| `HEAR_BACKEND_POLICY_JSON` / `HEAR_BACKEND_REGISTRY_JSON` | Source, callback, token, and storage authorization |
| `HEAR_POD_API_KEY`, `HEAR_RABBITMQ_URL` | Pod ingress bearer credential and local broker |
| `HEAR_POD_MAX_CONCURRENT_JOBS`, `HEAR_SERVERLESS_MAX_CONCURRENT_JOBS` | Per-worker admission limits |
| `HEAR_HOST_MAX_CONCURRENT_JOBS`, `HEAR_HOST_JOB_LOCK_PATH` | Shared Pod admission ceiling and lock |
| `HEAR_API_MAX_BODY_BYTES` | Bounded attempt request size |
| `HEAR_MAGIC_CLEAN_MODEL_DEVICE` | Cleaner device, normally `cuda:0` |
| `AUDIO_DOWNLOAD_MAX_BYTES`, `AUDIO_DOWNLOAD_READ_TIMEOUT_SECONDS`, `AUDIO_DECODE_TIMEOUT_SECONDS` | Download/decode resource limits |
| `FISH_SPEECH_HOME`, `FISH_SPEECH_MODEL_ROOT` | Pinned upstream Fish source checkout and optional separate model root |

Configure the backend registry with distinct ingress-token hashes, correct
callback/source origins, buckets, endpoints, public URLs, and attempt prefixes.
Pipeline catalogue ownership is one backend per worker; use separate pipeline
deployments for different backend taxonomies. Tokens, storage keys, and reporting
grants belong in backend/worker secrets, never frontend requests or logs.

Cleaner resource limits are `MAGIC_CLEAN_SCRATCH_BYTES`, `MAGIC_CLEAN_MAX_INPUT_BYTES`,
and `MAGIC_CLEAN_MAX_FRAMES`. Every model directory is resolved through
[model_manifest.json](hear/model_manifest.json) under `HEAR_MODEL_ROOT`; set
`HEAR_MODEL_PATHS_JSON` to point a logical model name at a different directory, for
example custom or fine-tuned weights with the same file layout. Overrides are
validated at startup and readiness checks the overridden directory.

Start the Pod gateway and consumers:

```bash
HEAR_ENV_FILE=/path/outside/repo/production.env bash scripts/run_pod_stack.sh
```

Pipeline also accepts transcription jobs; a dedicated transcription consumer is
optional. The Pod exposes `/healthz`, `/readyz`, `/capabilities`, `/metrics`, and
`/drain`. Serverless uses RunPod dispatch without RabbitMQ. Start admission at one
job and measure combined GPU/RAM use before increasing workers or limits.

Models load on first use and evict after idle time while workers stay ready.
`HEAR_GPU_IDLE_EVICTION_ENABLED` enables this policy. Pipeline, cleaner, Fish, and
AudioSep idle TTLs default to 600, 300, 1200, and 90 seconds; override the respective
`HEAR_PIPELINE_IDLE_TTL_SECONDS`, `HEAR_MAGIC_CLEAN_IDLE_TTL_SECONDS`,
`HEAR_RECONSTRUCTION_IDLE_TTL_SECONDS`, and `HEAR_AUDIOSEP_IDLE_TTL_SECONDS` settings.
Fish eviction closes its supervised child process. Plan for CPU staging during
cold load and residual CUDA context memory after model eviction.

## Jobs and callbacks

The backend creates a durable job and a versioned `AttemptEnvelope` binding source
revision/hash, exact options, backend identity, deadline, reporting grant, and
scoped storage. Submit to `POST /v1/attempts` with the Pod bearer token; 202 means
confirmed RabbitMQ publication. `POST /v1/attempts/stream` is an optional SSE preview.
For Serverless, submit the same envelope in RunPod's `input` field.

From the backend, use `hear.dispatch` instead of hand-written HTTP calls. It is
importable with only `httpx` and `pydantic` installed:

```python
from hear.dispatch import DispatcherFactory

dispatcher = DispatcherFactory(os.environ).build()  # HEAR_AI_TRANSPORT=pod|serverless
receipt = await dispatcher.submit(envelope)         # same DispatchReceipt for both
```

Pod needs `HEAR_POD_BASE_URL` and `HEAR_POD_API_KEY`; Serverless needs
`HEAR_RUNPOD_API_KEY` and `HEAR_RUNPOD_ENDPOINTS_JSON` mapping worker roles to
endpoint IDs, for example `{"pipeline": "...", "magic_clean_natural": "..."}`.
Both transports run the identical workflow and report the identical
`ExecutionEvent`/`ExecutionOutcome` payloads to the backend callbacks; the
transport response is only an acceptance receipt or an optional preview.

Workers claim before execution, heartbeat, upload artifact manifests, and report
canonical events/outcomes independently of the submitting connection. Callback
paths use the deployment-owned backend base, disable redirects, and path-escape
attempt IDs. Registry deployments select the attempt's registered backend.
Every callback includes `X-AI-Attempt-Grant`, `X-AI-Worker-ID`, and
`X-AI-Worker-Generation`.

| POST path suffix | Payload |
| --- | --- |
| `/internal/ai/attempts/{attempt_id}/claim` | Worker identity and execution-scope hash |
| `/internal/ai/attempts/{attempt_id}/heartbeat` | Worker identity, generation, and sequence |
| `/internal/ai/attempts/{attempt_id}/events` | `ExecutionEvent` |
| `/internal/ai/attempts/{attempt_id}/outcome` | `ExecutionOutcome` |

Only an `execute` claim permits processing. Other decisions are `already_completed`,
`cancelled`, `stale`, `not_current`, and `lease_unavailable`. The backend must fence
leases, apply outcomes idempotently, deduplicate events by ID, reconcile interrupted
attempts, and persist client-facing progress. It validates scoped artifact hashes
before applying or approving a candidate. Keep the original audio until approval.
Executable contracts are in `hear/contracts/` and `hear/execution/reporter.py`.

Check deployment configuration without processing a job:

```bash
/opt/hear-ai-v11/venvs/pipeline/bin/python -m scripts.check_job_runtime \
  --env-file /path/outside/repo/production.env
```

Release validation needs a new backend-issued attempt, accepted lease/outcome,
independent B2 read/hash validation, approval, cancellation/retry checks, and
full-recording listening. For Pods, also test broker saturation/recovery. Preserve
existing messages until reconciled during migration. Readiness or local tests
do not establish live backend/B2 integration.

## Build and publish

```bash
docker build --target pipeline-serverless -t hear-ai:pipeline-serverless .
docker build --target magic-clean-natural-serverless -t hear-ai:cleaner-serverless .
docker build --target transcription-pod -t hear-ai:transcription-pod .
```

Bazel produces a deterministic source context with a SHA-256 manifest. Buildx
requires a Docker-capable builder:

```bash
bazel build //:image_context
bazel run //:runpod_image -- --target pipeline-serverless --tag YOUR_REGISTRY/hear-ai:REVISION --push
bazel run //:check_serverless_image -- YOUR_REGISTRY/hear-ai:REVISION
```

Use `--target runpod-stack` for the combined Pod image, then launch it with
`scripts/run_production_container.sh`, an immutable image, and an external env file.
The stack isolates Fish/cleaner environments and shares byte-identical dependency
trees. The Serverless publication workflow builds the current checkout on manual
invocation. Building/publishing an image does not deploy an endpoint.

## Test

CI runs lint, typing, architecture, tests, and supported Docker builds using the
locked development groups plus pinned CPU inference/audio packages:

```bash
uv sync --project deploy/runtime --locked --group pod --group serverless --group dev
uv run --project deploy/runtime ruff check hear tests scripts
uv run --project deploy/runtime mypy
uv run --project deploy/runtime python -m hear.tools.check_architecture
uv run --project deploy/runtime python -m pytest tests tests/integration -q
```

For real-model API testing, install and provision the selected roles, including
Fish for reconstruction. Configure a fresh simulation root outside the checkout:

```bash
/opt/hear-ai-v11/venvs/pipeline/bin/python -m scripts.setup_simulation \
  --root /tmp/hear-simulation --audio-file /path/to/recording.mp3
HEAR_SIMULATION_ROOT=/tmp/hear-simulation bash scripts/run_simulation_stack.sh
```

From another shell:

```bash
HEAR_SIMULATION_ROOT=/tmp/hear-simulation \
  /opt/hear-ai-v11/venvs/pipeline/bin/python -m scripts.test_simulated_jobs
```

The gateway/consumers use production workflows and real models; only the backend,
catalogue, and S3 service are simulated. They bind to loopback, use a generated
local CA, verify claims and uploaded hashes, and reject production identities.
The API is on port 8000; test backend/storage use HTTPS port 18081. Setup refuses
to overwrite existing config. `HEAR_CANARY_JOB_TYPES` selects a subset of the four
job types; `HEAR_CANARY_COPIES` selects one to ten copies. Logs, certificates,
artifacts, and reports stay under the simulation root. This does not validate
production integration, commercial model permission, or maximum hardware capacity.

Runtime code is under `hear/`, tests under `tests/`, required operational helpers
under `scripts/`, and dependency/configuration assets under `deploy/` and `patches/`.
Generated audio, caches, credentials, and model weights stay out of source control.
