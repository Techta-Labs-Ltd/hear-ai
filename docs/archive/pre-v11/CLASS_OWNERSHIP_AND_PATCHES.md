# Class ownership and automated patches

Application behavior belongs to named classes. Standalone reusable helpers live in `hear/utils`; generated protobuf bindings and pytest functions are excluded from this rule. Imports stay at module scope before executable code. Method-local callbacks remain inside their owning class methods.

The refactor updates production callers and tests together. Internal Python imports from the old utility modules and old module-level workflow functions are intentionally replaced, not retained as compatibility wrappers. It does not change the public gRPC schema or complete the pending backend-owned execution migration.

| Responsibility | Owner |
|---|---|
| Startup and validation | `RuntimeApplication` |
| Deployment composition | `ApplicationBuilder` |
| Database engine, sessions, initialization | `DatabaseRuntime` |
| Submission validation and request identity | `SubmissionPolicy` |
| Streaming cleanup and validation | `StreamingAudioCleaner`, `AudioValidator` |
| Artifact lineage and cleanup | `MagicCleanLineageResolver`, `MagicCleanCleanup` |
| Voice profile persistence | `VoiceProfileStore` |
| Storage grants and backend registry | `StorageContexts`, `BackendRegistry` |
| Temp workspace and acquisition | `TempWorkspace`, `AudioDownloader` |
| Reusable audio/text/timing functions | `hear/utils` |

## Environment setup

Run from the repository root:

```bash
python scripts/setup_runtime.py
```

This runs `uv sync --locked`, then automatically applies the checked-in dependency patch manifest. Run it again after rebuilding or replacing the environment. It does not download model weights, invent credentials, or start Ray.

Use `python scripts/setup_runtime.py --no-dev` for production image builds, or `python scripts/setup_runtime.py --check` to verify an existing environment without installing or patching anything.

For an already-installed environment:

```bash
uv run --no-sync python -m hear.tools.dependency_patches
uv run --no-sync python -m hear.tools.dependency_patches --check
```

The patch manager verifies the installed WhisperX Git revision, patch checksum and original/patched source checksums. An already-patched installation is a no-op. Unknown or partially modified source fails without overwriting it. A verified replacement is written atomically; startup validation checks the result but never modifies dependencies itself. Patch content or dependency upgrades require reviewing and updating the manifest hashes together.

## Automated architecture checks

```bash
python -m hear.tools.check_architecture
python -m pytest tests/test_architecture.py tests/test_dependency_patches.py tests/test_runtime_setup.py -q
```

`.github/workflows/architecture.yml` runs these checks on pushes and pull requests. This lightweight job does not validate real GPU inference or replace the full regression suite.

Operational tools can be invoked as modules, for example `python -m scripts.smoke_test`, `python -m scripts.live_test`, and `python -m hear.tools.clean_temp`. Live tools now execute through class-owned entry points instead of running network requests merely on import.

## Verification

The final complete regression run passed **487 tests**, with 56 warnings, in 139.89 seconds (`/tmp/hear-class-final-tests.log`). The architecture, patch and setup checks account for 14 tests. The installed WhisperX patch was applied once and verified on repeated runs, including `setup_runtime.py --check`. Live/smoke/playback module imports were checked without starting Ray. The ownership/import guard, undefined-name checks, formatting and focused lint pass. Broader repository lint and the previously documented production/backend migration work are not claimed complete.
