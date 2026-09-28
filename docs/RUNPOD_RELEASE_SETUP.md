# RunPod release setup

This release runs three worker roles: Pipeline, Fish reconstruction, and Magic Clean.
Standalone transcription jobs route through Pipeline, so the Qwen ASR/aligner is loaded once.
The verified whole-Pod admission ceiling is 10 workflows: Pipeline 7, Magic Clean 4,
and Fish reconstruction 2, subject to the shared global ceiling.

Models are external runtime assets and MUST NOT be downloaded into /workspace.
General models and cleanup assets use /models. Fish NF4 uses
/root/hear-ai-v11/models. Audio scratch may use /workspace/hear-ai-v11/.runtime-audio;
scratch is not a model location and is excluded from Git/Bazel image contexts.

Required cleanup assets:
- /models/magic-clean/DeepFilterNet3
- /models/sound-cleanup-v1-runtime
- /models/sound-cleanup-specialist/runtime
The image pins the verified cleanup manifest hashes. A mismatch prevents the affected
worker from starting instead of silently disabling validation.

Fish uses the verified S2 Pro NF4 runtime under /root/hear-ai-v11/models, with source
code baked into /opt/fish-speech in the image. Two Fish replicas are used to provide
two concurrent reconstruction jobs; Fish does not intentionally offload model layers
to CPU RAM during inference.
## Memory behaviour

Qwen and Fish are explicitly CUDA-resident; this release does not use CPU model
offloading. Host RAM is Python/native state, audio buffers, file cache, and loader
state. Fish NF4 can temporarily stage about 18.5 GiB RSS while cold-loading.

GPU engines now use lazy loading and idle eviction. A cold worker keeps its RabbitMQ
consumer and HTTP readiness while holding no model weights in VRAM. Production idle
TTLs are 600 s for Pipeline/Qwen, 300 s for DeepFilterNet, 1200 s for Fish, and
90 s for AudioSep.

On the real A40 proof run, cold startup was 3 MiB VRAM. Pipeline + Magic Clean + Fish
jobs peaked at 10,165 MiB, then fell to 798 MiB after the short proof TTLs while the
API and consumers remained ready. A second Pipeline job reloaded to 9,394 MiB and
returned to 798 MiB after idle eviction. After Sound Cleanup, four long-lived cleaner
processes retained about 1,848 MiB of CUDA-context/runtime overhead after model
eviction. See GPU_IDLE_LIFECYCLE.md for the exact lifecycle evidence.

## Audio cleanup acceptance

Selected bark-over-speech repair completed through the API using the pinned AudioSep
specialist. The repaired region passed the no-new-tone check with zero detected
speech-loss frames and zero removed-speech frames; only the selected region changed.
The result remains approval-required.

Stationary background cleanup also completed with zero high-confidence speech-loss
frames. It remains listening-review required and does not claim certified hum removal.

Automatic animal detection is NOT calibrated yet. The controlled bark scored 0.2291
while the matching no-bark speech control scored 0.0004, but the current detector
threshold did not select the bark. Keep automatic event cleanup disabled by default
until a larger positive/negative calibration set establishes kind-specific thresholds.
## Bazel build

Bazel 8.4.2 produces a deterministic credential-free Docker context:

    bazel build //:image_context //:runpod_image
    bazel run //:runpod_image -- --dry-run

Verified context:
- /root/hear-ai-v11/builds/hear-runtime-context-122b18a6e7ee.tar
- SHA-256: 122b18a6e7eeddc801b6d4b7560a89008eb154d7b2e04dee8891225b79dd1d42
- Size: 2,467,840 bytes

The context contains no model weights, recordings, .env files, or keys. The inference
Pod does not have Docker Buildx, so an OCI image was not built or pushed here. Use the
same Bazel run target on the image-builder host to build the runpod-stack target.

## Verification

The full Python regression suite passed 672 tests with 12 skipped. Ruff and repository
architecture checks passed. Mypy found no issues in 140 source files. One-hour warm
canaries completed in 188.4 seconds for Pipeline and 310.3 seconds for Studio Voice
cleaning on the repeated Track 8 fixture.

Before switching from simulation to the real backend, perform one authenticated
production canary through claim -> inference -> Backblaze upload/readback -> outcome
persistence -> approval. The simulation canaries intentionally did not contact the
production backend or real Backblaze.
