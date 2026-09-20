# Implementation status — 20 September 2026

This is an incremental implementation against `a89ee0ff9b231351034d612b7c4dfa6f8e42b091`, not completion of P00–P09. No production rollout or cross-service migration has been performed. Existing unrelated worktree changes were retained.

The backend repository was not available locally, and an authenticated checkout could not be obtained. The requested backend deliverable is therefore [05_HEAR_BACKEND_HANDOFF.md](05_HEAR_BACKEND_HANDOFF.md), not an assertion that backend changes were made.

The earlier explicit instruction to remove training remains in effect pending clarification. A33 and the training-related portions of T23/T24/T38/T40 are not implemented or restored. AI resolver removal remains in place. Backend resolver functionality has not been verified.

## Implemented boundaries

| Boundary | Changed implementation | Migrated callers and checks |
|---|---|---|
| Legacy credential admission | `submission._validate_magic_clean_storage_ttl` raises the existing typed `StorageCredentialsExpiringError`; REST returns 422 with the stable code and gRPC returns FAILED_PRECONDITION | Gateway mapping and submission regression tests; actual backend issuer policy still requires repair |
| Legacy event fan-out | Orchestrator registers a separate bounded queue before snapshot, fans out to each subscriber, explicitly signals overflow, replays persisted terminal state, and unregisters in finally | `subscribe`, `_push_event`, `_schedule_job`; subscription tests cover multiple readers, overflow and cancellation |
| Reconstruction controls | `_process_edit_transcript` forwards stored `same_speaker`; false does not supply a voice reference | Both transport-presence tests and direct execution tests |
| Preview quality | Assessment exceptions return passed=false with a sanitized unavailable reason; confirmation rejects expired or nonpassing candidates | Injected quality assessor and preview-failure regression; exact-candidate backend approval remains outstanding |
| Health | `RayHealthSnapshot` and `ServiceHealth` report deployment states with bounded, coalesced cached probes; CPU ingress no longer measures its CUDA device as model health | Gateway → injected Operations → health reader; orchestrator stats call is bounded; health tests cover ready/cold/warming/unavailable/disabled and probe failure |
| Optional models | Graph respects LLM and Fish enablement; Fish import failure is explicit at model construction; configured codec path is actually used | `ApplicationBuilder`, runtime validation and composition tests |
| Import-safe entry point | `main` imports the lightweight composition module without importing the model graph; concrete deployments resolve only when the builder is invoked | Confirmed in a separate Python process; all Python import statements remain at module scope |
| Database import boundary | Missing database configuration is rejected on engine initialization, not module import | Missing-URL and lazy-engine regression tests; legacy startup still requires PostgreSQL |
| Sync client guard | Legacy synchronous model calls reject event-loop execution and cancel the submitted response rather than blocking the loop | Async-context and worker-thread regression tests; remaining synchronous consumers still require migration |
| Shared-cluster lifecycle | Attaching a driver to an external Ray cluster no longer calls global Serve shutdown on driver exit | `main.run`; real shared-cluster failure drill remains outstanding |
| ASR service injection | `TranscriptionService` receives its model client and awaits async inference; obsolete sync ASR entry removed | Orchestrator, gateway and synthesis composition updated; no implicit transcriber singleton |
| Bounded ASR transport | Full-recording orchestration uses `transcribe_file`; sequential disk-backed windows are downmixed/resampled for the model, then timestamps are offset once | Pipeline/audio-tag/discovery/edit-transcript callers migrated; window, final-short-window, stereo/resampling and invalid-result tests |
| Source conversion | ASR source preparation retains channels in the local source; RF64 can represent large decoded WAV output | Downloader callers use explicit preserve_channels; temporary-file cancellation tests updated |
| Acquisition bounds | Finite download read timeout, encoded byte limit and FFmpeg conversion timeout | Typed configuration and environment example; this does not implement source authorization or complete SSRF protection |
| Native execution | `NativeWorker` owns a serial thread lane; cancellation waits for actual native completion; model adapters expose close and bounded Serve queues | ASR, Fish, small models and LLM; native cancellation tests; batch LLM execution is explicitly sequential |
| Model error truthfulness | ASR no longer maps arbitrary IndexError/ValueError to no speech; malformed ASR results and empty/nonfinite Fish output fail explicitly | ASR result/window tests; actual Fish inference still requires provisioning |
| Shared DSP | NoiseReducer moved unchanged to `core/noise.py`; duplicate Magic Clean blocking re-export removed | Both Magic Clean and synthesis imports, plus DSP/cancellation tests migrated |
| Packaging | WhisperX declaration pinned to the already-locked commit `68a6634d1b9dfbc3ec4ca3ba800d998fc682b56b` | Lock regenerated offline without a model/package upgrade; tracked `venv` symlink removed without deleting its target |
| Protobuf presence | Progress/time measurements distinguish absent from explicit zero; health exposes control/capability state additively | All three generated files regenerated with grpc_tools; backend descriptors must be regenerated from the same proto |

## Package gates

| Package | Status | Remaining gate |
|---|---|---|
| P00 | Partial | GPU audio baselines, immutable model manifests, backend snapshot/runtime and image capture |
| P01 | Partial | Backend TTL issuer/classification, exact-candidate approval replacement and cross-service tests |
| P02 | Not implemented | Backend ProcessingJobAttempt, current-attempt CAS, durable dispatch/result/event ownership |
| P03 | Not implemented | ExecuteAttempt protocol, BackendClient, claims/leases/grants, manifests and backend ingestion |
| P04 | Partial | JobExecutor/PipelineService, fully injected policies/model services, independent execution profiles |
| P05 | Partial | ASR is windowed; Magic Clean remote transport and reconstruction splicing remain whole-recording paths; workspace quotas and all model budgets are unfinished |
| P06 | Not implemented | Backend durable preview/source revision ownership and exact-candidate idempotent approval |
| P07 | Partial | AI capability reader exists; backend incident/outbox/reconciliation, busy telemetry and planned drain are unfinished |
| P08 | Not eligible | Live AI SQL, previews, catalogue mutation, lineage and cleanup still have consumers; do not delete them before ownership migration |
| P09 | Not eligible | Real GPU/B2 integration, crash/rollback drills, canary and measured load/quality gates |

## File-map coverage

| Entries | Outcome |
|---|---|
| A01–A04 | Partial: startup, flags, graph import boundary and package pin; profiles/image build/settings injection unfinished |
| A05 | Partial: health and error mapping; legacy DB/bootstrap responsibilities retained |
| A06–A08 | Legacy fixes only; new execution wrapper, JobExecutor and PipelineService not implemented |
| A09–A10 | Partial acquisition bounds and NativeWorker; full AudioIO/workspace ownership not implemented |
| A11–A13 | Existing storage/SQL ownership retained; no v2 attempt/schema migration |
| A14 | Async injected ASR and synthesis model access; other global/sync client consumers remain |
| A15 | Pipeline extraction not implemented |
| A16–A17 | Partial: disk windows, async service, native lane, truthful errors, global pad monkeypatch removed; hardware continuity/longest-input validation outstanding |
| A18–A23 | Shared noise relocation and wrapper deletion only; existing Magic Clean service/model/DSP ownership retained |
| A24–A27 | Partial injection, quality safety, native Fish lane/codec/output validation; durable approval and bounded reconstruction remain |
| A28 | Native lanes, explicit sequential batch, queue limits and missing type imports fixed; token budgets and full-text model-window coverage remain |
| A29–A31 | Training remains removed and import-time warning configuration removed; full immutable tenant-policy/global-client migration remains |
| A32 | Additive health/presence protocol changes; ExecuteAttempt and backend-owned compatibility routing remain |
| A33 | Conflicts with prior explicit removal instruction; not restored |
| A34 | Prior AI resolver removal retained; backend resolver verification not performed |
| A35 | Live synthesis tool migrated to explicit clients; complete operational/tooling consolidation remains |
| A36 | Focused regression tests added; T01–T44 are not all satisfied |

## Environment and evidence

- Python 3.12; observed installed Ray 2.56.0, grpcio 1.82.1, protobuf 6.33.6, torch 2.8.0+cu128, transformers 4.57.6.
- Protocol generation used grpcio-tools 1.75.1; existing field numbers and oneof variants were retained.
- GPU detected: NVIDIA A40, 46068 MiB, driver 580.159.04. Detection is not model validation.
- Installed system `libsox-dev` and its runtime dependencies to enable existing tempo-regression tests. A release image must provision this explicitly; application startup does not install packages.
- `uv lock --offline` and `uv lock --check --offline` completed. WhisperX stayed on the same commit.
- `python -m compileall -q main.py hear tests scripts`, focused Ruff checks and `git diff --check` completed during implementation.
- First focused run: 125 passed. Second focused run: 62 passed.
- Follow-up window, temporary-file, control and startup tests: 39 passed.
- Final full suite, including database initialization and sync-client regression tests: **473 passed, 56 warnings in 130.85 seconds**. Output: `/tmp/hear-final-pytest.log`. Earlier full runs passed 468 and then 469 tests as coverage was added.
- Broader interim run excluding reconstruction tests: 437 passed, two test-double signature failures. The two fakes were updated for the explicit channel-preservation argument; both pass in the final suite.
- `main.py --validate-only` correctly exits 2 in this environment: Fish Speech package/weights, backend registry, storage-encryption key and database URL are missing. No credentials were fabricated.
- No real model inference, scoped B2 upload, backend transaction/CAS test, production deployment or crash/rollback drill was performed.
- Full Ruff was run, not just the focused checks: 591 findings remain across the repository. Existing and remaining migration issues are not claimed clean. Output: `/tmp/hear-final-ruff.log`.
- Full mypy initially reported 131 errors in 13 files. After annotation corrections, the rerun reports 115 errors in 8 files (82 source files checked). Output: `/tmp/hear-final-mypy.log`. This is not a type-clean repository.
- AST scan of `main.py`, `hear`, `tests` and `scripts` found no import statements nested inside functions or classes.

## Release restriction

Do not enable backend-owned v2 attempts against this code yet: ExecuteAttempt, its execution owner and manifest/control APIs are not shipped. The live path remains v1 with the correctness and resource-bound changes above. The residual SQL/global-client/whole-file references are known unfinished migration work, not justified deletions.

## Follow-up: class ownership and dependency patches

The subsequent ownership refactor moves application-level standalone functions into named classes and reusable helpers into `hear/utils`, with callers and tests migrated together. Database and client-provider state now belongs to its class. Generated protobuf APIs and pytest functions retain their framework-required organization. An AST architecture check and CI job enforce ownership and import placement. Dependency setup automatically applies the pinned WhisperX patch and verifies revision/source/patch hashes; runtime validation only checks and never installs or patches dependencies. See [CLASS_OWNERSHIP_AND_PATCHES.md](docs/CLASS_OWNERSHIP_AND_PATCHES.md) for setup commands and verification. This does not complete the cross-service migration described above.
