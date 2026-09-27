# Runtime ownership and dependency patches

## Runtime ownership

| Area | Responsibility | Main locations |
| --- | --- | --- |
| Contracts | Validate versioned jobs, attempts, events, outcomes, and worker identities | `hear/contracts/` |
| Execution | Claim attempts, report heartbeats/events/outcomes, and select workflows | `hear/execution/` |
| Workflows | Coordinate pipeline, transcription, reconstruction, and Magic Clean operations | `hear/workflows/` |
| Services | Implement inference, catalog, storage, moderation, categorization, and audio behavior | `hear/services/` |
| Runtime | Build role-specific dependencies, check readiness, manage Pod lifecycle, and enforce capabilities | `hear/runtime/`, `hear/bootstrap.py` |
| Pod API | Authenticate attempt submissions, stream execution events as SSE, and expose operational routes | `hear/api/`, `hear/runtime/pod.py` |
| Entrypoints | Start the Pod API/consumer or the native RunPod handler | `hear/entrypoints/` |
| Cleaner runtime | Load certified engines, enforce resource limits, and publish immutable attempt artifacts | `hear/runtime/cleaner/`, `hear/services/magic_clean/` |

Transport adapters deliver an `AttemptEnvelope` to the shared executor. Workflows do not own durable job state or provider retry policy. The backend owns job scheduling, attempt fencing, retry decisions, and user-visible progress history.

## Dependency groups

`deploy/runtime/pyproject.toml` and `deploy/runtime/uv.lock` define production dependency groups. Docker targets combine a workload group with either `pod` or `serverless`. Pipeline and transcription share model dependencies; optional Qwen targets add `pipeline-llm`. Reconstruction has its own common base. Magic Clean profiles use separate groups so each image contains only its engine dependencies.

## Dependency patch

The Qwen/WhisperX patch is specified in `patches/manifest.json`. The image build applies it with `python -m hear.tools.dependency_patches`; `--check` verifies the expected patched dependency files. Pipeline and transcription readiness also verify the patch. If the installed files do not match the manifest, the worker remains unready.

## Change boundaries

Put provider-specific delivery code in an entrypoint or transport adapter. Keep payload validation in contracts, attempt reporting in execution, and business processing in workflows and services. Avoid importing profile-specific heavy dependencies from shared runtime modules; profile factories load those dependencies only when the matching role is built.
