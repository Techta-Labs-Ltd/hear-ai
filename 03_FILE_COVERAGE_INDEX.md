# File coverage index

This index maps identified source files to the section that directs their implementation. It is a navigation aid, not a claim of production validation or full-body review of every module. New files are identified in the implementation documents. The master document defines execution order and acceptance tests.

## Hear-AI service files: 45 identified files

Read section IDs in [01_HEAR_AI_FILE_BY_FILE.md](01_HEAR_AI_FILE_BY_FILE.md).

| Existing file | Section | Disposition |
|---|---|---|
| `hear/services/__init__.py` | A35 | Import-safe exports |
| `hear/services/categorization/__init__.py` | A35 | Import-safe exports |
| `hear/services/categorization/discovery.py` | A30 | Keep generation; move catalogue ownership |
| `hear/services/categorization/service.py` | A29 | Inject; remove worker persistence |
| `hear/services/jobs/__init__.py` | A35 | Import-safe exports |
| `hear/services/jobs/scheduler.py` | A08 | Migrate owner; delete after legacy drain |
| `hear/services/jobs/submission.py` | A08 | Migrate owner; delete after legacy drain |
| `hear/services/llm.py` | A14 | Async injected model access |
| `hear/services/magic_clean/__init__.py` | A23 | Explicit per-file retain/move/delete map |
| `hear/services/magic_clean/blocking.py` | A23 | Explicit per-file retain/move/delete map |
| `hear/services/magic_clean/cleanup.py` | A23 | Explicit per-file retain/move/delete map |
| `hear/services/magic_clean/lineage.py` | A23 | Explicit per-file retain/move/delete map |
| `hear/services/magic_clean/models.py` | A23 | Explicit per-file retain/move/delete map |
| `hear/services/magic_clean/pipeline.py` | A20 | Preserve bounded DSP; correct guards through tests |
| `hear/services/magic_clean/processing/__init__.py` | A22 | Explicit per-processor ownership, dependencies and tests |
| `hear/services/magic_clean/processing/audio_io.py` | A22 | Explicit per-processor ownership, dependencies and tests |
| `hear/services/magic_clean/processing/dynamics.py` | A22 | Explicit per-processor ownership, dependencies and tests |
| `hear/services/magic_clean/processing/helpers.py` | A22 | Explicit per-processor ownership, dependencies and tests |
| `hear/services/magic_clean/processing/mossformer.py` | A22 | Explicit per-processor ownership, dependencies and tests |
| `hear/services/magic_clean/processing/noise.py` | A22 | Explicit per-processor ownership, dependencies and tests |
| `hear/services/magic_clean/processing/quality.py` | A22 | Explicit per-processor ownership, dependencies and tests |
| `hear/services/magic_clean/processing/silence.py` | A22 | Explicit per-processor ownership, dependencies and tests |
| `hear/services/magic_clean/processing/speech.py` | A22 | Explicit per-processor ownership, dependencies and tests |
| `hear/services/magic_clean/processing/stems.py` | A22 | Explicit per-processor ownership, dependencies and tests |
| `hear/services/magic_clean/processing/validation.py` | A22 | Explicit per-processor ownership, dependencies and tests |
| `hear/services/magic_clean/service.py` | A18 | Refactor existing CPU execution service |
| `hear/services/magic_clean/streaming.py` | A21 | One bounded disk-backed production path |
| `hear/services/model_client.py` | A14 | Async injected model access |
| `hear/services/moderation/__init__.py` | A35 | Import-safe exports |
| `hear/services/moderation/service.py` | A31 | Immutable policy; declared model ownership |
| `hear/services/reconstruction/__init__.py` | A26 | Explicit support-file changes and tests |
| `hear/services/reconstruction/audio_buffer.py` | A26 | Explicit support-file changes and tests |
| `hear/services/reconstruction/diff.py` | A26 | Explicit support-file changes and tests |
| `hear/services/reconstruction/dnsmos.py` | A26 | Explicit support-file changes and tests |
| `hear/services/reconstruction/quality.py` | A26 | Explicit support-file changes and tests |
| `hear/services/reconstruction/service.py` | A24 | Refactor execution; move durable approval |
| `hear/services/reconstruction/synthesizer.py` | A25 | Inject; bounded synthesis/splicing |
| `hear/services/reconstruction/tts_post_processor.py` | A26 | Explicit support-file changes and tests |
| `hear/services/reconstruction/voice_profile.py` | A26 | Explicit support-file changes and tests |
| `hear/services/transcription/__init__.py` | A35 | Import-safe exports |
| `hear/services/transcription/chunks.py` | A16 | Async bounded source/windows; preserve timestamps |
| `hear/services/transcription/service.py` | A16 | Async bounded source/windows; preserve timestamps |
| `hear/services/transport/__init__.py` | A32 | Transport adapters; remove misplaced persistence |
| `hear/services/transport/grpc.py` | A32 | Transport adapters; remove misplaced persistence |
| `hear/services/transport/operations.py` | A32 | Transport adapters; remove misplaced persistence |

## Backend AI-service files: 18 identified files

Read section IDs in [02_HEAR_BACKEND_FILE_BY_FILE.md](02_HEAR_BACKEND_FILE_BY_FILE.md). All paths below are relative to `src/app/services/ai/`.

| Existing file | Section | Disposition |
|---|---|---|
| `__init__.py` | B05 | Exports only; remove singleton and empty inheritance |
| `audio_joiner.py` | B17 | Required bounded assembly only |
| `b2_validator.py` | B09 | Source and destination authorization |
| `callbacks.py` | B15 | Pure typed normalization |
| `cleanup.py` | B20 | Backend remote candidate cleanup owner |
| `client.py` | B08 | Injected pooled transport |
| `constants.py` | B10 | One set of state/kind/error groups |
| `handlers.py` | B14 | Consolidate; remove empty inheritance structure |
| `job_result_processor.py` | B13 | Single result application owner |
| `media.py` | B16 | Durable exact-candidate preview/approval |
| `notifications.py` | B21 | Durable deduplicated delivery |
| `scheduler.py` | B07 | Existing fair scheduler becomes AIJobDispatcher |
| `service.py` | B06 | Real explicit job lifecycle owner |
| `sse_publisher.py` | B22 | Committed-event adapter only |
| `storage.py` | B09 | Actual scoped grant lifetime/refresh |
| `submission_policy.py` | B10 | Pure policy |
| `tagging.py` | B18 | Idempotent metadata persistence |
| `track_state.py` | B19 | Publication-aware projections |

## Other Hear-AI sections

| Section | Coverage |
|---|---|
| A01–A03 | main.py, settings/environment, packaging, lockfile, patch, tracked venv reference and image/build |
| A04–A08 | graph, gateway, common execution/lifecycle, orchestrator extraction and scheduler removal |
| A09–A12 | acquisition/conversion, workspace, blocking/native cancellation, storage, auth, context, all identified core loaders/helpers and new control/health adapters |
| A13 | schemas, stages, discovery values and migration/removal of database models |
| A17/A19/A27/A28 | transcription, enhancement, Fish Speech, small-model and LLM deployments |
| A32 | pipeline proto, generated .py/.pyi files, transport compatibility and resolver protocol removal |
| A33 | all seven identified training files and dataset/model ownership |
| A34 | resolver package/deployment removal and playback-prompt retention |
| A35 | all eight identified scripts, cleanup tooling, docs and package imports |
| A36 | unit/contract/GPU/storage/failure testing and final deletion scan |

## Other backend boundaries

| Section | Coverage |
|---|---|
| B01–B04 | internal control routes, existing ProcessingJob/attempt additions, existing StreamEvent/EventJournal and job data access |
| B11–B12 | worker/enqueue adapters and gRPC lifecycle |
| B23 | durable backend reconciler |
| B24 | all six identified audio_source files; existing audio_revision and canonical mutation |
| B25 | health/probe/policies/repository and incident dedupe/delivery |
| B26–B27 | creator/schema/approval routes, protocol/stubs, configuration, composition and migrations |
| B28 | preserved backend resolver, waveform, catalog and unrelated product owners |
| B29 | final dispositions and production gate |

## Status discipline

Every listed file needs an implementation outcome: kept and tested, changed and tested, moved with callers migrated, or removed after its gate. A file does not count as complete merely because a class was added. Report actual tests and retained legacy exceptions. No implementation or production tests were performed by creating this plan pack.
