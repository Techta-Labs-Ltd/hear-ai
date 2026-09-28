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

Qwen and Fish are explicitly CUDA-resident. Host RAM usage is Python/native state,
audio buffers, memory-mapped/file-cache pages, and loader state. Fish NF4 temporarily
staged about 18.5 GiB RSS while a worker was loading, then settled near 3.1 GiB RSS
after GPU residency. That startup spike is not dynamic GPU-to-RAM offload.

After both Fish workers, Pipeline, and cleanup-capable Magic Clean workers had actually
run, the A40 used 35,459 MiB of 46,068 MiB. The four-cleanup-job canary also sampled
35,459 MiB peak. Do not start an additional standalone Qwen transcription worker in
this layout; it previously duplicated about 7.7 GiB of VRAM.

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
- /root/hear-ai-v11/builds/hear-runtime-context-e2080d52bdbf.tar
- SHA-256: e2080d52bdbff7b13cd4adee2de565731e26af5ba2475346455b3e47b5009dd1
- Size: 2,447,360 bytes

The context contains no model weights, recordings, .env files, or keys. The inference
Pod does not have Docker Buildx, so an OCI image was not built or pushed here. Use the
same Bazel run target on the image-builder host to build the runpod-stack target.

## Verification

The full Python regression suite passed 668 tests with 12 skipped. Ruff and repository
architecture checks passed. Mypy found no issues in 139 source files. One-hour warm
canaries completed in 188.4 seconds for Pipeline and 310.3 seconds for Studio Voice
cleaning on the repeated Track 8 fixture.

Before switching from simulation to the real backend, perform one authenticated
production canary through claim -> inference -> Backblaze upload/readback -> outcome
persistence -> approval. The simulation canaries intentionally did not contact the
production backend or real Backblaze.
