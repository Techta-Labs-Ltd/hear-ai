# gRPC interface

Ray Serve exposes both protobuf services on `GRPC_PORT` (default `50051`).
Clients must attach `application=hear` and `x-api-key` metadata.

The same Ray Serve application exposes FastAPI health and readiness probes on
`HTTP_PORT` (default `8000`). Ray owns both proxies; no standalone Uvicorn or
`grpc.aio.server` process is started.

Clients should set their maximum receive message size to at least 64 MiB for
long-recording results. Word timestamps are returned in
`transcription.segments[].words`; the redundant top-level `word_segments` list
is intentionally empty so multi-hour results remain compact.

## Pipeline

- `Subscribe`, `GetResult`, `CancelJob`, `GetQueueStats`
- `Moderate`, `Categorize`, `ListDiscovery`
- `CreatePreview`, `ConfirmPreview`, `RemoveSegment`,
  `RollbackPreview`, `GetPreview`
- `TrainCategorizer`, `IngestCategoryEvent`,
  `UpdatePlatformSettings`
- `Health`

`Subscribe` is a server stream and emits stage, heartbeat, completion, failure,
and cancellation events. Structured results use `google.protobuf.Struct`
where the underlying discovery, moderation, or job result is intentionally
open-ended.

Jobs are submitted only through authenticated `POST /process`. The Hear
backend then opens `Subscribe` with the returned `job_id`; reconnecting after a
terminal event replays the persisted completion, failure, or cancellation.

## Resolver

- `Resolve`
- `ResolverHealth`
- `Rebuild`
- `Apply`

Resolver state is owned by its Ray deployment. Rebuild operations no longer
coordinate replicas through an HTTP endpoint.

## Error handling

Invalid requests use `INVALID_ARGUMENT`, missing resources use `NOT_FOUND`,
duplicate active jobs use `ALREADY_EXISTS`, unavailable models or resolver
indexes use `UNAVAILABLE`, and invalid credentials use `UNAUTHENTICATED`.
Unexpected internal errors are logged and returned as `INTERNAL` without
leaking implementation details.
