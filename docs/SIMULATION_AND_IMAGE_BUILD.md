# Live API simulation and RunPod image build

The updated API is tested with real DeepFilterNet, Qwen ASR/alignment, pipeline
classification/moderation and Fish S2 Pro NF4. Only the backend, catalogue and S3
storage are simulated. The previous API and FFmpeg-only reconstruction processes
are stopped. Recordings and verified model files are not removed.

## Test service

API: port 8000, `/docs`, `/capabilities`, `POST /v1/attempts`.
Mode is explicitly `simulation`; this is not a connection to the production app.
Fake backend/storage bind only to HTTPS loopback port 18081. Clients trust a
local test CA rather than disabling TLS verification. No production credentials
are loaded by the test service. Its minimal S3 emulator supports these small,
single-part test artifacts; it is not a full Backblaze/AWS implementation.

The owning fake backend registers a scoped attempt, verifies claims and worker
identity, records progress, receives the canonical outcome, and reads uploaded
files back to verify SHA-256 and length. A duplicate completed job is not rerun.

Run the canary from the source checkout:

```bash
HEAR_CANARY_COPIES=2 /opt/hear-ai-v11/venvs/test/bin/python -m scripts.test_simulated_jobs
```

This submits eight jobs, two of each type. The configured limits are one active
job per role and two active jobs across the Pod. Both limits use process-shared
locks; RabbitMQ retains work waiting for capacity. These are tested settings,
not a claim that two is the hardware's maximum throughput.

Model locations remain on the root filesystem: `/models` and
`/root/hear-ai-v11/models`. Model loading is offline during job execution.
Simulation grants no commercial licence; external backend identities, sources
and buckets are rejected in this mode. Production mode retains its normal checks.

## Bazel / container build

```bash
bazel build //:image_context
bazel run //:runpod_image -- --dry-run
bazel run //:runpod_image -- --tag YOUR_REGISTRY/hear-ai:VERSION --push
```

Bazel produces a deterministic source-only context with a SHA-256 file manifest.
The executable target then drives Docker Buildx. It requires a Docker-capable
builder host; the inference Pod itself does not need Docker. The context and
build-target dry run can be validated on the Pod without a Docker daemon.
The final image build/push is a separate step, not implied by a successful dry run.

`runpod-stack` packages all four roles. Pipeline and transcription share an
installed Python environment, not an unsafe concurrent model object. Fish and
cleaning keep their own dependency environments. Byte-identical Torch/NVIDIA
package trees are shared in the final image; differing versions are never merged.
Application source is copied after dependency installation for layer-cache reuse.

Use `--target reconstruction-serverless`, `transcription-serverless`,
`pipeline-serverless` or `magic-clean-natural-serverless` for role-specific images.
These retain the same job contracts and workflows. No credentials, recordings or
model weights are included in the source context. Provision verified model files
into the documented root paths separately, before starting consumers. Do not
re-download models per job or copy `/workspace` into the image wholesale.
