# Hear AI

Hear AI is one Python project containing the audio intelligence pipeline,
and model deployments. Ray Serve owns model lifecycle,
scheduling, the FastAPI ingress, and the built-in gRPC proxy.

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
Required Python packages, native libraries, PostgreSQL, and model artifacts
must be provisioned before the process starts.

## Package management

The project uses `uv` and commits `uv.lock`; there is no `requirements.txt` or
hand-managed virtual environment workflow. Resolve dependencies during a
controlled development/build step:

```bash
uv lock
uv sync --frozen
```

Production images should run `uv sync --frozen --no-dev` while being built.
When the image already provides the locked packages in its system Python, run
uv in no-project mode. This does not create a project environment or install
anything during startup:

```bash
uv run --no-project python main.py
```

For KubeRay, use a Ray image that already contains `uv` and set
`RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook` on
every Ray pod. Keep the project directory as the working directory so Ray and
`uv` discover the same lockfile. The dependency environment and local model
artifacts must be present before the Serve application starts.

## Start

Copy `.env.example` to the deployment secret store and set every required
model path. Validate the immutable runtime without starting Ray:

```bash
uv run --no-project python main.py --validate-only
```

Start the complete application:

```bash
uv run --no-project python main.py
```

`main.py` connects to `RAY_ADDRESS` or creates a local Ray runtime, starts the
Ray Serve HTTP and gRPC proxies, registers the generated protobuf servicers,
and deploys the single `hear` application.

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
backend/storage context is rejected. The request `job_id` is its idempotency key: an identical resend returns the original
`run_id` and current status, while a different payload for the same key returns
HTTP `409`. Deliberate reruns must use a new `job_id`.

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

Ray Serve provides the gRPC proxy. Do not start `grpc.aio.server`, install
packages, generate stubs, or download models in `main.py`.

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
- `hear/proto/`: Pipeline and Resolver protobuf contracts/stubs
- `hear/services/`: application and audio-processing services
- `hear/resolver/`: resolver domain implementation
- `tests/`: unit, contract, and integration tests

Outbound HTTP/S3 integrations to the Hear backend, taxonomy CDN, and object
storage remain supported. Job result callbacks are not used; Hear consumes
results through gRPC. System probes and job submission use FastAPI, while typed
result and operation traffic uses gRPC.
