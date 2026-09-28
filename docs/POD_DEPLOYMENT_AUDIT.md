# Single-Pod deployment audit — 28 September 2026

## Findings

The live Pod runs main 49d2195 with one host-wide admitted job, not one job per
role. Its ready queues are DeepFilterNet and the previous FFmpeg-only reconstruction.
The corrected branch uses Fish TTS reconstruction and a different queue. It has
not replaced the live checkout or changed backend/queue/replica configuration.

The A40 exposes 46,068 MiB GPU memory; the Pod RAM limit is 50 GB and effective
CPU quota 7.65 cores, despite a larger CPU affinity list. Measured engine results
are in verification/pod-deployment-fish-audit.json. Four standard cleaner processes,
two selected-overlap cleaning processes and two transcription processes completed
independent full-file tests. These are not simultaneous mixed-lane capacities.

**The four-process FFmpeg result is not Fish reconstruction capacity.** Fish has
no real inference benchmark here: package import works, the required model files
are absent and the existing permission-required model gate remains intact.
One Fish job per worker is the initial enforced policy, not a measured maximum.

## API and ownership release gates

The inspected deployed backend still posts /process, whereas this worker expects
/v1/attempts/stream. Required catalogue and attempt control integration is missing
from that deployment. Do not call healthz/readyz proof of a successful application
job: first verify job submission, claim/heartbeat, progress, storage, persisted
outcome, preview and approval with the real owning backend.

Prod and dev point at the same Pod but have DIFFERENT backend IDs, buckets, B2
regions and CDN domains. The registry code rejects cross-environment tokens,
callbacks and storage. Enable it only with deployment-owned origins/token digests
and a backend that supplies the matching validated envelope. Keep all cloud keys
server-side. Cloud-level key scope and a real upload/read test were not verified.

The corrected B2 key builder prevents a duplicated jobs/job_id folder. Grant expiry
is checked on use, and claimed upload digests are verified against local bytes.
HEAD validation verifies length/checksum metadata, not an independent download.
A timestamp in our grant does not itself revoke a long-lived B2 application key.

## Residual audio

The new optional reduce_stationary_noise processing detects persistent narrow
mains lines and can use speech-protected quiet material for bounded spectral
cleanup. It is not another AI denoiser or a change to DeepFilterNet defaults.
Raw remains denoise-only. On the sampled corrected recording, voice-region RMS
changed about -0.018 dB and unprotected intervals about -1.67 dB. No persistent
mains line was confirmed in the sampled real recordings. A controlled added
50/100-Hz test improved known-reference error by 26.34 dB. This does not establish
resolution of the user's audible humming complaint; listening acceptance remains
open. Latest complete candidate is in:
/workspace/hear-ai-v11/clean/background-audit-20260928/full-cleaner/

## Rollout order

Complete the backend v11 dispatcher/attempt-control routes, provision the approved
Fish model and verify a real text-edit job, configure environment-specific routing
and cloud key restrictions, then perform a real B2/approval canary. Only after
that raise Pod admission/replicas with mixed-job resource measurements. Do not
set all independently tested maxima concurrently. Retain one in-process Fish job.
Move the same envelope/workflow to Serverless later; it now reports authoritative
outcomes back to Hear before returning completion. Retry/lease/source-revision
checks remain necessary under either provider.

See FISH_TTS_RECONSTRUCTION.md for the request/result contract and explicit limits.
