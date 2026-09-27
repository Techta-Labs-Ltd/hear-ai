# Magic Clean delivery ledger

This ledger records the V11 runtime implementation and its release evidence. The name is retained for continuity with the earlier cleaner delivery plan.

## Runtime status

| Profile | Engine path | Runtime status | Production evidence still needed |
| --- | --- | --- | --- |
| Natural | Pinned DeepFilterNet 3 runtime | Implemented behind certification and resource checks | Golden audio parity, long-file stability, and deployment certificate evidence |
| SAM Audio | Pinned official SAM Audio Base API with local T5 text encoder | Implemented for text-prompt removal and isolation | SAM License review, gated asset provisioning, golden audio parity, and deployment evidence |

The implementation records runtime, checkpoint, precision, and long-form policy digests in the cleaner plan and result manifest. Profile workers require a certification file at startup. A valid contract or present model file alone does not certify output quality.

DeepFilterNet 3 loads from the pinned local config and checkpoint through its supported initializer. The model stays resident for the worker lifetime, attempt sessions borrow it, and worker shutdown closes the engine cache before releasing lane ownership.

The SAM Audio profile uses Meta's API from a pinned source commit and locally provisioned Base checkpoint and T5 assets. The model manifest records the SAM License as `review_required`; certified mode remains disabled until license review, text-prompt certification, audio quality review, and deployment evidence are approved.

## Delivery gates

- Contract validation binds the input revision, source digest, profile options, attempt fence, output prefix, and deadline.
- Resource guards account for scratch reservations and high-water usage; the result manifest records validation resource measurements.
- Artifact publication is immutable and verifies object key, size, digest, and manifest contents.
- Cancellation, deadline expiry, subprocess termination, and workspace cleanup are covered by runtime behavior and tests.
- Release evidence must include approved golden samples, audio quality review, peak CPU/GPU/RAM/scratch measurements, interrupted-storage cases, and Pod/Serverless comparison.

Production certification and external parity approval remain open release gates. The current repository state must not be represented as a completed parity sign-off.
