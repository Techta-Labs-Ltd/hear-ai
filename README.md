# Hear AI

Hear AI is one Python project containing the audio intelligence pipeline,
and model deployments. Ray Serve owns model lifecycle,
scheduling, the FastAPI ingress, and the built-in gRPC proxy.

The staged refactor is tracked in [implementation status](04_IMPLEMENTATION_STATUS.md).
Required backend changes are in [the backend handoff](05_HEAR_BACKEND_HANDOFF.md).
The active runtime is still the legacy protocol; backend-owned execution is not enabled.

## Runtime architecture

```text
HTTP client -> Ray Serve HTTP proxy :8000 -> FastAPI ingress --+
                                                               |
gRPC client -> Ray Serve gRPC proxy :50051 --------------------+
                                                               v
                                                Gateway (application=hear)
    |-- Orchestrator
    |-- Whisper + Qwen aligner
    |-- Qwen LLM
    |-- Toxicity, sentiment, and NLI models
    |-- DeepFilterNet and MossFormer2
    `-- Fish Speech
```

FastAPI runs inside the Ray Serve ingress; there is no separate Uvicorn
process, gRPC server, model sidecar, or runtime installer. Internal calls use
injected Ray Serve deployment handles.
Required Python packages, native libraries, and PostgreSQL must be available
before the process starts. The server applies its verified dependency patch and
uses Ray to provision missing model artifacts before it creates Serve deployments.

## Package management

The project uses `uv` and commits `uv.lock`; there is no `requirements.txt` or
hand-managed virtual environment workflow. Resolve dependencies during a
controlled development/build step:

```bash
python scripts/setup_runtime.py
```

Setup installs the committed lockfile and automatically applies verified dependency patches.
See [class ownership and patch automation](docs/CLASS_OWNERSHIP_AND_PATCHES.md).
Production images should run `python scripts/setup_runtime.py --no-dev` while being built.
When the image already provides the locked packages in its system Python, run
uv in no-project mode. This does not create a project environment or install
anything during startup:

```bash
uv run --no-project python main.py
```

For KubeRay, use a Ray image that already contains `uv` and set
`RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook` on
every Ray pod. Keep the project directory as the working directory so Ray and
`uv` discover the same lockfile. The dependency environment must be present
before the Serve application starts.

## RunPod persistent workspace

RunPod treats `/root` as ephemeral. Before installing dependencies or starting
the server, source the workspace environment helper:

```bash
cd /workspace/hear-ai
source scripts/runpod-workspace-env.sh
```

The helper only creates repository-local directories and exports cache paths.
When the server starts, Ray downloads missing model artifacts into `/models`
and startup applies the verified dependency patch.
`scripts/download_models_ray.py` remains available for manual pre-warming.
Keep `.env` model paths aligned with `.env.example`.

After a CUDA or base-image maintenance event, rebuild the project environment
from the committed lockfile instead of reusing a virtual environment created
against the previous image:

```bash
cd /workspace/hear-ai
source scripts/runpod-workspace-env.sh
mv .venv .venv.pre-cuda13  # recoverable backup, if an old environment exists
python scripts/setup_runtime.py
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run --no-project python main.py --validate-only
```

Do not run the removal command during preparation; it is a post-maintenance
operator step. The locked CUDA 12.8 PyTorch build should be tested against the
new NVIDIA driver before changing dependency versions.

## Start

Copy `.env.example` to `.env` (or the deployment secret store) and set every
required model path. The application reads `/workspace/hear-ai/.env` through
`pydantic-settings`; Supervisor also exports that file before starting Ray so
shell tools and Python see the same values. Explicit Supervisor overrides such
as `RAY_ADDRESS=auto` take precedence over the file.

`AI_SERVICE_URL` and `AI_SERVICE_SECRET` are backend-client names and are not
AI-server settings. On this service, authentication is configured with the
`BACKEND_REGISTRY_JSON` SHA-256 digest; the backend keeps the matching
plaintext `HEAR_SERVICE_KEY`. Registrations are environment-scoped; see
[development and production credentials](docs/ENVIRONMENTS.md).

The server stores only service-key SHA-256 digests. Generate a new backend key
and install its digest with:

```bash
python scripts/generate_service_key.py --backend-id backend-a --env-file .env --write
```

Save the printed `HEAR_SERVICE_KEY` in the backend's secret store. The backend
must send that same plaintext key as `X-Service-Key`/`x-api-key`; never commit
the plaintext key or put it in `BACKEND_REGISTRY_JSON`.

Validate the immutable runtime without starting Ray:

```bash
uv run --no-project python main.py --validate-only
```

Start the complete application:

```bash
uv run --no-project python main.py
```

`main.py` connects to `RAY_ADDRESS` or creates a local Ray runtime, starts the
Ray Serve HTTP and gRPC proxies, registers the generated protobuf servicers,
and deploys the single `hear` application. Before Serve starts, a Ray task
downloads any missing artifacts into the filesystem-root `/models` directory
and applies required dependency patches. Model paths must stay under that one
directory; do not use a separate workspace model cache.

### Supervisor deployment

Use [deploy/supervisord.conf](deploy/supervisord.conf) as the process-manager
configuration. It starts a Ray head first, waits for it in
`scripts/start-hear-ray-server.sh`, then starts `main.py` with
`RAY_ADDRESS=auto`. Supervisor restarts either process if it exits; first
startup provisions the `/models` cache from the Ray cluster before Serve
accepts traffic.

On a new root-capable pod, one command installs the OS/Python/Fish Speech
dependencies, creates local PostgreSQL credentials, and prepares the root-level
runtime directories:

```bash
cd /workspace/hear-ai
sudo scripts/bootstrap-pod.sh
supervisord -c deploy/supervisord.conf
```

Use `scripts/bootstrap-pod.sh --start` to start Supervisor immediately after
setup. The first server start downloads models into `/models`; it does not
store weights under `/workspace`.

When deploying code or changing `MAGIC_CLEAN_ENGINE_REVISION`, first drain
queued/running jobs, then restart both Ray processes. Restarting only the
application process can retain existing Serve actors with stale imports or
settings, causing engine-revision mismatches:

```bash
supervisorctl -c deploy/supervisord.conf stop hear-ray-server
supervisorctl -c deploy/supervisord.conf restart ray-head
supervisorctl -c deploy/supervisord.conf start hear-ray-server
```

Wait for `/health` to report `healthy` and `control_ready: true` before
submitting new jobs.

## Availability and concurrency

The production defaults run two stateless gateway replicas. Ray Serve
load-balances requests across them and performs rolling replacement. A small
Ray deployment sweeps abandoned audio from `HEAR_TEMP_DIR` at the configured
interval, while normal job completion and failure paths clean their own files.

The orchestrator is intentionally a single stateful replica because it owns
live `Subscribe` streams. It admits at most
`ORCHESTRATOR_MAX_CONCURRENT_JOBS` jobs (three by default); additional work is
reported as queued and starts when a slot becomes available. Durable job state
remains in PostgreSQL, while coordination and request routing use Ray rather
than Redis.

## FastAPI

Ray Serve hosts these system endpoints on `HTTP_PORT` (default `8000`):

- `GET /`: service identity
- `GET /health`: aggregate pipeline health
- `GET /ready`: pipeline readiness status
- `POST /process`: idempotent submission for every asynchronous job type

`POST /process` requires `X-Service-Key` for the submitted `backend_id`.
Every request must include that registered backend identity and a job-scoped
`storage` object containing an allowed B2 endpoint/bucket, temporary credentials,
a user/job folder prefix, public base URL, and expiry. Missing or mismatched
backend/storage context is rejected. The request `job_id` is its idempotency key:
an identical resend returns the original `run_id` and current status, while a
different semantic payload for the same key returns HTTP `409`. A queued Magic
Clean job may accept an authenticated credential-only refresh with a later
expiry; deliberate reruns must still use a new `job_id`.

OpenAPI documentation is exposed at `/docs` and `/openapi.json` only when
`ENABLE_DOCS=true`. Typed application operations remain on gRPC.

## gRPC

Contracts and checked-in client stubs live in `hear/proto`. Every call must
include:

- `application: hear` for Ray Serve application routing
- `x-api-key: <registered backend service key>` for backend-bound authentication

The Pipeline service covers progress streaming, results, cancellation, queue
status, moderation, categorization, reconstruction, discovery, administration,
and aggregate health. Hear submits over REST, then consumes `Subscribe` and
`GetResult` over gRPC. Terminal results are persisted and replayed after a
stream reconnect. Jobs and results are isolated by the backend identity resolved
from `x-api-key`; one backend cannot read or cancel another backend's jobs.
Artifact results contain `backend_id`, `bucket_name`, `b2_key`, and a URL joined
from the submitted public base URL, but never storage credentials.

For reconstruction requests sent through `SubmitJob` or `CreatePreview`, an
omitted `same_speaker` field defaults to `true`. Send the optional field
explicitly as `false` to opt out of matching the source speaker.
Reconstruction measures the first-to-last aligned source speech span and applies
a bounded, pitch-preserving tempo correction to match the generated delivery rate
while retaining natural internal pauses. A longer or shorter replacement can
therefore change the segment and rebuilt-track duration; clients must use the
returned duration instead of assuming the old interval length.

Ray Serve provides the gRPC proxy. Do not start `grpc.aio.server`, install
packages, generate stubs, or download models in `main.py`.

For asynchronous `SubmitJob` same-interval retries, submitting the exact
rebuilt URL from an earlier completed job with the same `backend_id`,
`track_id`, and change intervals keeps that current audio for splicing but
resolves the original source for voice, pitch, and pace reference. Isolated
segment URLs and changed intervals fail closed because their timestamps cannot
be mapped safely to the original speaker. URL aliases, re-uploads, and changed
track IDs are deliberately not guessed across security boundaries; those clients
should retain the immutable original and submit the cumulative change set in
original-track coordinates.

To regenerate stubs during a controlled build/development step:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. \
  hear/proto/pipeline.proto
```

The generator version must be compatible with the protobuf runtime baked into
the deployment image.

## Project layout

- `main.py`: only production entry point
- `hear/config.py`: unified settings
- `hear/deployments/`: Ray models, audio cleanup, orchestrator, and FastAPI/gRPC gateway graph
- `hear/proto/`: Pipeline protobuf contracts/stubs
- `hear/services/`: application and audio-processing services
- `tests/`: unit, contract, and integration tests

Outbound HTTP/S3 integrations to the Hear backend, taxonomy CDN, and object
storage remain supported. Job result callbacks are not used; Hear consumes
results through gRPC. System probes and job submission use FastAPI, while typed
result and operation traffic uses gRPC.
