# Magic Clean runtime

Magic Clean is part of the shared Hear AI runtime. The canonical job contract, executor, workflow, and artifact publication code live in `hear/contracts/`, `hear/execution/`, `hear/workflows/magic_clean.py`, and `hear/services/magic_clean/`. Profile engines are loaded only for the matching worker role.

## Profile status

| Profile | Worker role | Build target | Status |
| --- | --- | --- | --- |
| Natural | `magic_clean_natural` | `magic-clean-natural-pod` or `magic-clean-natural-serverless` | Implemented; requires a deployment certificate and approved audio parity evidence |
| SAM Audio | `magic_clean_sam_audio` | `magic-clean-sam-audio-pod` or `magic-clean-sam-audio-serverless` | Text-prompt removal and isolation through the official SAM Audio API; SAM License review, deployment certificate, and approved audio parity evidence required |

Build images from the repository root. The production dependency groups and lock are in `deploy/runtime/pyproject.toml` and `deploy/runtime/uv.lock`. Do not use the historical cleaner-specific package files as the image dependency source.

## Runtime admission

Provision pinned model assets and a profile certification file before startup. Configure `HEAR_CLEANER_CERTIFICATION_PATH` and pin its bytes with `HEAR_CLEANER_CERTIFICATION_SHA256`. Each profile limit record points to an absolute evidence file and binds its bytes with `evidence_sha256`. Pinned assets and evidence are opened without following symlinks, checked as regular files, and hashed through stable file descriptors. GPU certificates include `certified_peak_device_bytes`, measured with the pinned runtime and certified input limit; CPU certificates set it to zero. Before advertising GPU readiness and before loading a GPU engine, the worker checks live `nvidia-smi` memory and requires the certified peak plus a 2 GB reserve to fit. Normal job startup does not download model files.

Build the immutable aggregate certificate from a completed draft with:

```bash
uv run --project deploy/runtime --group dev python scripts/build_cleaner_certification.py --draft /secure/cleaner/certification-draft.json --output /models/cleaner/certification.json
```

Set `HEAR_CLEANER_CERTIFICATION_SHA256` to the `certification_sha256` value printed by the builder. The output path is create-only; changing evidence or profile policy requires a new output path and digest.

SAM Audio provisions the pinned `facebook/sam-audio-base` checkpoint and `google-t5/t5-base` text encoder through `hear/model_manifest.json`. The SAM model requires accepted Hugging Face access. Its SAM License status remains `review_required` until the release review is recorded.

Each attempt binds a source revision and digest, profile plan, deadline, attempt authorization, and scoped artifact prefix. The runtime publishes checksummed immutable artifacts and a result manifest. The backend remains responsible for durable job state, attempt fencing, retries, and accepting results.

## Release evidence

Runtime contract and failure-path tests do not certify audio quality. Before enabling a profile in production, approve its versioned golden set, Pod/Serverless comparison, resource measurements, interrupted-upload checks, and certificate. Current implementation and open evidence are tracked in [the delivery ledger](../../docs/CLEANER_V2_DELIVERY_LEDGER.md); the repository-wide release gates are in [the migration plan](../../HEAR_AI_FULL_MIGRATION_MASTER_PLAN_V11.md).
