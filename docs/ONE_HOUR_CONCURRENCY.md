# One-hour, ten-job validation

## Execution policy

The tested candidate supports separate process and role limits. The configured
host ceiling is ten active jobs in total; pipeline and Magic Clean each have a
role ceiling of ten. These are not additive twenty-job reservations. Fish TTS and
the dedicated transcription role remain at one job each.

Pipeline uses two isolated workers with five in-flight workflows each. Each
worker shares its pinned ASR engine and serializes that engine's model windows,
preserving mutable VAD/model state. Ten active jobs therefore means two ASR
model processes, not ten replicas. The initial single-process stress run was
interrupted by the broker's default acknowledgement timeout and is not a pass.
60-second windows bound per-job sample buffers and permit fair model scheduling.

Magic Clean uses ten worker processes, each with its own DeepFilterNet instance
and a one-job process limit. Do not raise native thread concurrency on the shared
DeepFilter session: the engine explicitly permits only one active session.

HEAR_POD_ROLE_LIMITS controls cross-process per-type admission.
HEAR_POD_PROCESS_LIMITS controls workflows in each worker process.
HEAR_WORKER_REPLICAS controls how many isolated role workers the launcher starts.
HEAR_HOST_MAX_CONCURRENT_JOBS remains the total Pod admission ceiling.

## Fixture and test scope

The one-hour MP3 is made by repeating the user's Track 8 recording, resetting
sample timestamps and trimming to exactly 172,800,000 mono frames at 48 kHz.
It is a 3,600-second load fixture, not an independently recorded hour or ten
unique programmes. Every attempt runs the real model; outputs are not reused.
Original MP3s remain unchanged. Model weights stay on root storage.

## Disk and storage

Ten one-hour float-audio workspaces exceed the Pod root disk's available space.
Only decoded audio scratch is directed to /workspace/hear-ai-v11/.runtime-audio;
model weights remain in /models and /root/hear-ai-v11/models. Scratch is excluded
from Git and image build contexts and is removed by each attempt's cleanup.

The local S3 emulator now implements bounded multipart uploads and assembles
parts only after their ETags are checked. The real boto3 transfer path and full
public-URL readback are exercised. This is not a real Backblaze round-trip.

## Acceptance evidence

The canary accepts HEAR_CANARY_COPIES=10, selected job types and an explicit
one-hour source. It requires ten completed canonical outcomes, one claim per
attempt, the configured global ceiling, and streamed URL SHA-256 checks.
An additional verifier decodes every delivered audio file to confirm all
172,800,000 samples remain, checks finite audio, and checks pipeline transcript
bounds and presence of speech in the last minute. These are technical integrity
checks, not a manual word-accuracy or perceptual-quality certification.

Do not treat a successful 202 response, ten pending messages, or an unchanged
GPU-memory reading as a successful ten-job completion test. Record actual
claim/completion intervals, output validation and resource measurements.

The multi-role image launcher supports the replica configuration, but this
load test runs the existing Pod directly; it is not a fresh container-image
build or a worst-case startup test for every possible model combination.

## Failure found and corrected

The initial pipeline run hit RabbitMQ's 1,800,000 ms delivery-acknowledgement
timeout at 2026-09-28 06:58:43 UTC. All ten pipeline executions were interrupted
before the complete hour finished. This failed run is preserved separately.
A bounded 10,800,000 ms timeout is now persisted in broker/image configuration
and applied to inference queues through a live consumer-timeout policy.

A temporarily unavailable backend lease on a re-delivered message now waits in
the retry queue instead of being permanently acknowledged. The local simulator
now expires stale leases and labels its peak counter scope. Its earlier stale
'running' records were not evidence of twenty simultaneous model executions.
The interrupted fake attempts were explicitly marked failed, not completed,
before the fresh pipeline batch was submitted.
