# Job intake and result delivery

## Canonical path

The backend validates the signed-in user and creates a durable ProcessingJob.
Its v1 dispatcher binds a unique attempt to the source bytes/revision, exact
options, backend identity, deadline and storage scope. The worker receives
POST /v1/attempts and returns 202 only after broker publication is confirmed.
The request connection is not the lifetime of the job. Progress and the final
outcome are delivered to the owning backend; /v1/attempts/stream is optional.

A claim contains the worker identity and a canonical execution-scope hash. The
backend grants a lease only when the request matches the stored attempt. Duplicate
claims cannot execute concurrently, including duplicates from the same worker.
Worker-generation headers fence progress/outcomes. Source changes, cancellation,
wrong-bucket outputs, unmanifested links and conflicting duplicate outcomes fail.
The backend reads and hashes pipeline/transcription manifests before forwarding
results into its existing durable result-application/approval flow.

## Removed legacy code

The old FFmpeg-only reconstruction workflow and its unused synthesizer/DNSMOS/
pitch-processing/storage adapter were removed. Their tests were retired alongside
them; the Fish text-edit, timing, codec and provider tests remain. No recordings,
cloud objects, queue messages or unrelated worktrees were deleted.

## Configuration

Use the versioned backend adapter on branch fix/ai-v1-job-routing. Required backend
settings are HEAR_AI_RUNTIME_V1=true, HEAR_HTTP_URL pointing at the Pod gateway,
HEAR_AI_CALLBACK_BASE_URL including /api/v1, and HEAR_AI_INGRESS_TOKEN. Configure
matching deployment-owned policy/registry entries on the Pod, with distinct
backend tokens and their correct bucket/CDN/callback origins. Keep credentials
out of frontend requests. The backend already stores attempt state in its
ProcessingJob JSON column; this adapter does not require a database migration.

The backend feature flag is OFF by default. Do not enable it until both sides have
passed a real round-trip. Advanced Magic Clean options are supplied as
runtime_options; unsupported legacy music/voice-mixing sliders are rejected rather
than pretending DeepFilterNet can remix independent stems.

The entrypoint scripts must be run from this checkout's root in the locked role
environments. Configure HEAR_POD_STACK_ROLES explicitly. Dedicated transcription
is no longer silently omitted when pipeline is also requested. Keep one admitted
job initially; independent engine benchmarks do not establish mixed-lane capacity.

Fish requires its approved checkpoint and cannot use the removed FFmpeg substitute.
Pipeline requires the owning backend catalogue. Do not silently omit either missing
role and label the whole stack ready. The same execution-scope contract applies
to Serverless; canonical outcomes must be persisted before returning completion.

## Read-only checks

```bash
/opt/hear-ai-v11/venvs/pipeline/bin/python -m scripts.check_job_runtime \
  --env-file /root/hear-ai-v11/runtime.env
```

This checks configuration, asset presence and the backend protocol endpoint. It
is not a real inference, word-quality, broker-recovery or cloud-upload test.

## Release checks still required

Verify actual broker policy compatibility, queue saturation/recovery, the callback
lease/outcome path with a real backend row, and a scoped B2 upload/read/approval.
Preserve old queues until reconciled. Never purge them as source-code cleanup.
The currently observed RabbitMQ configuration uses reject-publish-dlx on quorum
queues; that policy needs correction before production sign-off. The attempted
policy edit was blocked by the tool and is NOT included in this branch.

Fish model provisioning/permission, live backend rollout, legacy dashboard control
migration and the user's residual-hum listening acceptance remain separate gates.
Neither unit tests nor readyz imply these have been completed.
