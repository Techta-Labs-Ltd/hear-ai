# Backend integration by job type

Hear Backend owns durable job state, provider selection, dispatch, retry policy, progress history, SSE, and final-state reconciliation. Hear AI executes one claimed attempt and reports attempt-scoped events and an outcome using the contract in `hear/contracts/`.

| Job type | Attempt payload | Runtime path | Result expected by backend |
| --- | --- | --- | --- |
| `pipeline` | Source revision, storage grant, pipeline options, deadline | `PipelineWorkflow`; pipeline-capable workers | Transcript, moderation/categorization/discovery outputs, and artifact references |
| `transcription` | Source revision, storage grant, transcription options, deadline | `TranscriptionWorkflow`; transcription or pipeline role | Transcript and timing data with artifact references |
| `reconstruction` | Source revision, storage grant, one reconstruction operation, operation-specific options | `ReconstructionWorkflow` | Reconstructed audio and metadata; preview operations return preview artifacts |
| `magic_clean` | Source revision, profile, profile-specific plan, storage grant, deadline | `MagicCleanWorkflow`; role is pinned to one profile | Immutable candidate artifacts and a verified result manifest |

Reconstruction operations are `replace_segments`, `edit_transcript`, `rebuild`, `remove_segments`, and `preview`. Magic Clean profiles are `natural`, `studio_voice`, `outdoor_mobile`, and `clean_raw`. All route to the existing Natural worker. See [the profile guide](DEEPFILTER_CLEANING_PROFILES.md) for boolean controls and measured-output semantics. SAM is retired.

## Dispatch and reporting

For Pod execution, POST the versioned envelope to the role-specific Pod API at `/v1/attempts/stream` and consume the `text/event-stream` response. For RunPod Serverless, submit to the role-specific endpoint and pass the same envelope to the native handler. Both providers invoke the shared `AttemptStream`, executor, and workflows. The worker claims the attempt before execution and sends heartbeats; the backend consumes the canonical events and terminal outcome from the provider stream, then validates the current attempt fence before persisting events or artifacts.

The backend must persist dispatch intent and accepted attempt results durably, make claim and outcome handling idempotent, and reconcile interrupted attempts. These are backend integration requirements; the repository implements the worker-side contract and client only.
