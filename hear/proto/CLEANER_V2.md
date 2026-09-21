# Cleaner v2 wire boundary (not deployed)

`cleaner_v2.proto` is the proposed canonical internal execution envelope. It uses
the separate `hear.cleaner.v2` package/service and does not modify or fall back to
legacy `hear.pipeline.v1.Pipeline` methods. No server handler is registered yet.
Backend code generation and shared contract approval remain required.

The proposed result manifest also carries optional `validation.speech_activity`
evidence (source and pre-mastering PCM hashes, policy digests, activity totals,
source-frame risk intervals and a truncation flag). This lives in the fetched
manifest JSON, not the compact protobuf notice. Strict backend manifest parsers
must update together with this still-unreleased contract; the presence of VAD
evidence does not mean wanted-content validation passed or authorize application.

Regenerate Python bindings with `grpcio-tools==1.75.1`, the version in `uv.lock`:

```bash
python -m scripts.generate_cleaner_proto
```

The generator refuses a different compiler. Use a separate tooling environment
or an isolated `PYTHONPATH` package target if the host compiler differs. Generated
Python, type stubs and gRPC bindings are generated together, never hand edited.
Other backend languages must generate from this same `.proto`; do not hand-copy
field numbers. Do not renumber existing fields; reserve names/numbers on removal.

## Presence and validation

Scalar fields use explicit presence. `CleanerWireCodec` requires all non-nullable
semantic fields; omitted `false` options or zero seeds are not defaulted. Absence
maps to `None` only for the explicitly nullable semantic fields (sample, model
checkpoint for CPU processing, and profile-specific plan options). The complete
plan validators then enforce the correct profile/engine combination.

Decoding rejects unknown protobuf fields recursively. A client must negotiate a
supported version instead of silently losing a future option. Envelopes are
bounded to 256 KiB before parsing. Native integer values preserve uint64 precision;
timestamps are timezone-aware RFC3339 strings. Grants are separate from the
semantic ticket and decode into redacted `SecretStr` values. The protobuf object
and serialized bytes still contain raw tokens: never log them or project them to
the frontend.

## Required integration still outstanding

An opt-in `CleanerGrpcIngress` handler now connects this RPC to injected
authentication and dispatch callbacks. It is not registered by the running server.
Authentication must return `CleanerPrincipal` from verified transport credentials,
not ticket fields. Backend and tenant must match exactly before dispatch. The
handler bounds envelopes, passes a cancellation event and monotonic RPC deadline,
and permits one synchronous active dispatch per instance. A published execution
failure returns a verified failure-manifest reference, not a successful candidate.
Other errors use sanitized gRPC statuses without grant/native exception details.

The dedicated gRPC server must use `CleanerGrpcIngress.SERVER_OPTIONS` to limit
message sizes before protobuf deserialization, and must configure authenticated
TLS. The loopback test uses an insecure ephemeral socket only as a test fixture.
The injected dispatch adapter still must build an authenticated `ExecutionContext`,
enforce current fences and scoped grants, tighten the ticket deadline to the RPC
deadline, and supervise restart-required workers. It must not interpret this
synchronous RPC as a durable queue acknowledgment or add automatic retries.

Terminal notices use `ManifestReference` and are bounded to 16 KiB. The result
codec requires the exact ticket identity, fence and manifest key, explicit
outcome, a nonempty object version and consistent error code. Unknown fields are
rejected. Encoding first verifies the published manifest and receipt. After
fetching and verifying the remote bundle, consumers must call
`verify_result_bundle` to compare the complete notice against that bundle;
receiving a notice alone does not prove successful publication or authorize a
backend commit.

Before calling the executor, an authenticated ingress must check the service
identity, backend/tenant ownership, current attempt fence, source/grant scope,
expiry, provider and contract compatibility. Successful decoding proves none of
these. It must also construct allowlisted storage clients, handle grant refresh,
supervise worker deadlines, map typed failures/terminal references, and provide
progress/cancellation/reconciliation behavior. The initial execution RPC alone
does not implement that lifecycle. Existing production transport remains unchanged.
