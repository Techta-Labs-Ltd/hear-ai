# V11 runtime migration status

## Current runtime shape

This repository contains Hear AI worker runtime code organized around four durable job types: pipeline, transcription, reconstruction, and Magic Clean. Workers receive a versioned attempt envelope, claim it through the backend reporting API, execute a shared workflow, and report ordered events and a terminal outcome.

Pod deployments expose authenticated HTTP/SSE attempt ingress plus health, readiness, capabilities, and drain endpoints. RunPod Serverless deployments use a native handler and provider progress updates. Both run the same attempt stream, executor, and workflow code. Backend-owned scheduling, persistence, retries, reconciliation, and browser-facing progress remain outside this runtime.

Role-specific Docker targets and dependency groups limit each image to the runtime dependencies it needs. Model assets are provisioned separately and checked at startup. Pipeline and transcription workers verify the pinned Qwen/WhisperX dependency patch. Magic Clean profile factories load optional engine dependencies lazily and require profile certification.

## Implemented migration work

- Versioned contracts cover job types, reconstruction operations, attempt claims, worker identity, events, outcomes, and Magic Clean profile options.
- Shared executor and workflow paths support Pod and Serverless transports.
- Readiness checks validate role capabilities, configured resources, model assets, and applicable dependency patches.
- Audio processing uses attempt-scoped workspaces, bounded download/decode operations, and cleanup on completion.
- Magic Clean validates pinned source identity, resource envelopes, attempt authorization, and immutable artifact manifests.
- CI defines lint, typing, architecture, test, and supported image-build gates.

## Remaining integration and release gates

- Connect and validate the backend implementation for durable dispatch, attempt fencing, report endpoints, retries, and restart reconciliation.
- Run provider integration checks for Pod HTTP/SSE and RunPod progress, status, cancellation, and webhook behavior.
- Provision production models, storage credentials, and profile certificates; complete audio golden-set and Pod/Serverless parity review.
- Measure supported workload limits and peak CPU, memory, GPU, scratch, latency, and cancellation behavior.

The migration plan and register are the release checklist. Passing repository CI verifies code and contract gates; it does not by itself complete backend integration or production audio certification.
