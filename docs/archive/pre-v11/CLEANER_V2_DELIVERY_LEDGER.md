# Cleaner v2 implementation evidence

Baseline inspected: `a12117c5c9d9d66423536ff533fec2295e3b33c9`.
Scope remains the complete `HEAR_CLEANER_V2_AI_IMPLEMENTATION_PLAN.md`.
This ledger records incremental work, not release approval or certification.

Entries below preserve historical observations, including early missing-package
and missing-token reports. Those are not current availability assertions: later
real DF3/Silero tests, package/factory builds and the authenticated SAM 403 probe
supersede them. Historical wheel/GPU evidence remains bound to its recorded
source hashes and must not be relabeled as certification of newer source.

## Current implementation

### Opt-in cleaner-v2 gRPC execution handler (2026-09-21)

Added `CleanerGrpcIngress` with injected transport authentication and dispatch.
Trusted principal backend/tenant scope must exactly match the decoded ticket;
authentication precedes semantic decoding. Request/response size server options,
post-deserialization bounds, unknown-field/semantic validation, inactive/deadline
rejection, cancellation-event propagation and a nonblocking single-dispatch slot
are explicit. Terminal bundles are verified by the existing codec. Published
execution failures return their compact failure reference; other errors map to
sanitized gRPC status messages, without native or grant details. The handler owns
no durable job state, fallback, automatic retry or production credential policy.

46 ingress/wire/packaging/architecture tests passed in 6.74s, including a real
ephemeral loopback gRPC call returning a verified published failure. Tests also
cover authentication-before-decoding, backend/tenant rejection, expired/inactive
calls, occupied worker, callback propagation, invalid/oversized envelopes, native
error redaction and slot release. Ruff/format checks passed after import ordering
was corrected. Added the handler to the cleaner package allowlist (51 files).

No live registration, listener, TLS configuration, service-key change or serving
restart occurred. Production authentication, executor-context/provider dispatch,
fence/grant refresh and backend reconciliation remain required integration work.
The handler's injected boundaries are not evidence those integrations are finished.
The wire guide documents the required deployment conditions and remaining scope.

### Full cleaner-v2 regression after CUDA/preflight work (2026-09-21)

Ran `OMP_NUM_THREADS=1 .venv/bin/pytest -q tests/test_cleaner_v2_*.py
tests/test_architecture.py`: **697 passed, 15 skipped, two upstream oneDNN warnings
in 136.02s**. Explicit asset-backed followups covered the default opt-in skips:
15 SAM-core/codec/speech-activity tests passed in 29.72s with the pinned source,
config and Silero ONNX paths; 10 real DF3 checkpoint tests passed in 1.79s using
the isolated DF3 dependency environment and the current workspace loader. Those
runs emitted only the existing weight-norm/torchaudio deprecation warnings.
The skip-site audit also identified the noise-reference and speech-risk real
Silero cases; an explicit ONNX-backed followup of those suites passed all 33 tests
in 17.61s. Together these followups cover all 15 opt-in skips from the broad run.

The broad Ruff sweep found import-spacing/formatting drift in sam_checkpoint.py,
sam_noise.py and worker_lease.py. Corrected formatting without logic changes;
AST comparisons against the previously audited CUDA wheel matched for all three.
The final lint sweep passed and all 83 files in the format sweep passed. Whitespace
checks passed. Existing private backend-document edits were left untouched.

This is broad regression evidence, not long-form, audio-quality, aggregate GPU
memory, backend integration, image-build or production-deployment certification.
Prior wheel evidence still retains its exact original source hashes. No Ray
serving process was changed.

### Early SAM codec-resource admission (2026-09-21)

Added `SamSeparationPipeline.preflight`, called by the executor after source
inspection and before registry/session loading. It accounts for resampling to
48kHz and codec padding to a 1,920-frame boundary, then rejects inadequate frame
or scratch reservations. The scratch calculation is explicitly a necessary lower
bound: two batch-2, 96-channel, FP32 files coexist at the final decoder stage's
first activation (`padded_frames * 1536` bytes), in addition to existing workspace
files. Other live intermediates require additional space; passing this check is
not a full peak-space guarantee, and per-allocation runtime checks remain active.

Tests cover padding, resampling, existing occupied bytes, invalid geometry and
executor rejection before registry/model loading. The real pinned meta-codec test
also checks final-stage output channels/stride and activation shape to support the
lower-bound geometry. An initial 42 plan/executor/architecture tests passed in
57.47s; the final 26 plan/pre-load-order/architecture tests passed in 19.96s, and
two explicit real-source codec-loader tests passed in 22.99s with the existing
upstream weight-norm warning. Ruff/format and whitespace checks passed.

This changes early resource rejection, not model numerics or certified limits.
Prior wheel/inference evidence remains bound to its recorded source hashes.
Complete scratch-peak estimation, long-form certification and production deployment
remain unfinished; no serving process was changed.

### Real CUDA execution across the solver-window boundary (2026-09-21)

Extended the installed-package diagnostic with bounded fixture lengths and codec
tile selection. A 576,001-frame mono 48kHz sine fixture (just over 12 seconds) maps
to 301 latent frames, exceeding the 250-frame solver window and exercising
overlapping global-state evaluations with codec tile 4096. The new tile policy
has its own runtime/long-form identities. No independent CPU/full-runtime reference
was computed for this fixture, so this is not a numerical-parity result.

Two preliminary attempts failed typed resource checks: first the probe's exact
input-frame allowance excluded codec padding, then its 512MB scratch allowance
was insufficient. Added a 1,920-frame padding allowance and explicitly provisioned
2GB disk scratch after checking free space; the 9GB CUDA cap was never increased.
The completed run returned exactly 576,001 finite samples at 48kHz, preserved input
and cleaned scratch/module namespaces. Torch peaks were 3,387,147,776 allocated
and 3,575,644,160 reserved bytes. Post-close allocation remained 8,519,680 bytes.
Scratch peak was not measured. One execution does not establish repeatability.

Full result, hashes and limitations:
`deploy/cleaner/evidence/sam-cuda-window-crossing-2026-09-21.json`.
36 solver/factory/architecture tests passed in 21.01s; Ruff/format checks passed.
Multi-hour, aggregate NVML and listening/quality certification remain unfinished.
Existing Ray serving was untouched.

### Retained CUDA allocation attributed to cuBLAS workspace (2026-09-21)

Inspected the pinned installed Torch declarations for its private cuBLAS-workspace
clear hook, then added an explicit `--diagnose-cublas` probe option. The hook is
called only after all diagnostic sessions complete and CUDA is synchronized,
following independent GC/allocator-cache observations. No serving code uses it.

Three fresh-process A40 sessions again produced identical hashes and stable
8,519,680-byte post-close allocation. GC and empty-cache did not reduce allocated
bytes. Clearing cuBLAS workspaces then reduced Torch allocation to zero; a final
empty-cache reduced reservation to zero. This controlled observation attributes
the previously unexplained retained allocation to cuBLAS workspace in this test,
not live SAM model parameters. Keep this overhead within the worker memory budget;
do not add a private workspace-clear call to routine production cleanup.

Full evidence: `deploy/cleaner/evidence/sam-cublas-retention-2026-09-21.json`.
The installed wheel and numerical policy are unchanged. Four architecture tests
passed in 2.29s; Ruff/format checks passed. This resolves the short-fixture retention
question but not aggregate NVML, long-form, quality or release certification.
Existing Ray serving remains unchanged.

### Three-session CUDA retention diagnostic (2026-09-21)

Extended the isolated installed-wheel probe with bounded repeat counts (1–5),
per-session output hashes and post-close allocation/reservation observations.
It does not force GC or empty the CUDA allocator cache between sessions; final
GC and empty-cache observations are explicitly diagnostic, not production cleanup.
43 engine/factory/architecture tests passed in 21.83s; Ruff/format checks passed.

Three real A40 sessions in one process produced identical CUDA output hashes and
passed the prior CPU tolerance. Each session retained exactly 8,519,680 allocated
bytes and 3,038,773,248 reserved bytes after close. No growth was observed over
these three attempts. Final GC left allocation unchanged; empty-cache reduced
reserved memory to 20,971,520 bytes but allocation remained 8,519,680 bytes.
The owner of that allocation is not yet proven; do not call it a diagnosed leak
or assert indefinite leak-free behavior. Peak allocated memory remained
2,890,976,768 bytes. Per-session source/scratch/namespace checks passed.

Recorded installed-wheel identity, probe hash and full observations in
`deploy/cleaner/evidence/sam-installed-cuda-repeat-2026-09-21.json`.
This remains short synthetic repeated-session evidence, not long-form, aggregate
NVML, approved prompt or quality certification. Existing serving was unchanged.

### Real installed-wheel A40 short-fixture inference (2026-09-21)

Extended the isolated installed-wheel probe with explicit CPU/CUDA selection,
CPU known-answer comparison and Torch allocator measurements. Built and audited
the 50-file CUDA-capable wheel, SHA-256
`64f1de9910f7f719263f2a8883a6c9486533448373efa3aa13a4760eb5c75448`.
51 targeted regressions passed in 23.37s; 37 installed dependencies were compatible,
and Ruff/format checks passed. An initial diagnostic attempt failed resetting peak
statistics before CUDA initialization, after passing CPU inference; corrected the
probe initialization and completed a fresh-process run without changing the wheel.

On the A40, the real checkpoint/seed-42/3,841-sample fixture passed comparison with
the CPU result at unchanged atol=1e-6/rtol=1e-5; maximum error was
1.0477378964424133e-8. CUDA output is not byte-identical to CPU. Peak Torch allocated
memory was 2,890,976,768 bytes, reserved 3,038,773,248 bytes, under the configured
9,000,000,000-byte allocator cap for this short fixture. Source, scratch and module
namespace checks passed. 8,519,680 bytes remained Torch-allocated after close:
this needs characterization before claiming leak-free repeated-job operation.

Full wheel hashes, runtime identity and result are in
`deploy/cleaner/evidence/sam-installed-cuda-short-2026-09-21.json`.
Existing GPU baseline was 10,872 MiB and an intermediate whole-device observation
was 13,625 MiB; these are not sampled aggregate peak certification. The test does
not prove long-form memory, approved speech semantics, listening quality, complete
upstream parity or release readiness. Existing Ray workloads were not restarted.

### Explicit SAM CUDA FP32 loading path (2026-09-21)

`PinnedSamFactory` now accepts explicit `device="cuda:0"`; CPU remains the default.
Its CUDA identity is separate (`sam-offline-meta-cuda-fp32-v1`) and pins FP32,
autocast/TF32 off, cuDNN enabled with benchmarking/deterministic flags off, plus
the existing CPU staging policy. Admission rejects conflicting settings without
silently changing process-global precision configuration. CPU v3 identity remains
unchanged. The CUDA path is not registered as a ready/certified capability.

After source verification, CPU meta construction and strict CPU checkpoint load,
the factory requires one visible CUDA device, validates the parameter/buffer
reservation and sets the per-process allocator cap before transferring only the
required core and codec. Post-transfer parameter and buffer device/dtype checks
reject incomplete transfers. Optional vision/ranker modules are not constructed
or transferred. This cap is not proof of aggregate NVML process-memory compliance.

51 factory/engine/packaging/architecture tests passed in 24.54s; Ruff/format and
whitespace checks passed. Mocked CUDA tests verify identity separation, explicit
device selection, unchanged rejected backend flags, no premature CUDA availability
probe during identity validation, cap-before-transfer order, unavailable/multiple
devices, insufficient device reservation and incomplete transfer. No real GPU
model execution or peak-memory measurement was performed in this increment.
Read-only nvidia-smi reported an A40 with 46,068 MiB total and 10,872 MiB already
used by existing workloads; those workloads were not changed. Real CUDA parity,
allocator/NVML peaks, long-form and quality certification remain release gates.

### SAM native-fault quarantine policy v3 (2026-09-21)

SAM engine admission and sessions now translate native MemoryError/Torch CUDA
OOM into typed resource exhaustion and other native RuntimeError into typed engine
unavailability, with worker restart required. The affected engine is quarantined
and refuses new sessions even after its lease is released. Existing typed errors
retain their restart requirement; ordinary cancellation does not quarantine.
Failures are raised outside native exception handlers to avoid retaining native
traceback contexts containing partially loaded tensors. Native diagnostic details
are not exposed in the typed failure message. Backend cleanup and lease release
remain mandatory; this is not automatic worker respawning.

The factory runtime descriptor is now `sam-offline-meta-cpu-v3` and includes the
fault policy. Existing v2 wheel/model evidence remains historical, not v3 package
certification. 54 engine/factory/executor/architecture tests passed in 64.69s.
Injected loading/inference failures cover host OOM, CUDA OOM, native runtime errors,
typed restart requests and cancellation, including lease release, reuse/quarantine,
cleanup and absence of native exception contexts. These are injected fault tests,
not real GPU OOM or A40 certification. Ruff and formatting checks passed after
formatting the new tests. GPU factory/device admission remains unfinished.

### Installed raw-conditioning CPU inference (2026-09-21)

The installed-package probe now provisions its checksum-pinned engineering T5
fixture into the documented raw format and admits it through `from_files`.
It deletes the provisioned file before inference, exercising cache ownership.
The broader regression initially found an architecture-rule failure for a nested
NumPy import; this was corrected using the existing importlib convention, followed
by a new build and complete rerun. 57 targeted tests passed in 19.14s, with Ruff,
format and whitespace checks passing and 37 installed dependencies compatible.

Corrected 50-file wheel SHA-256:
`bf9fbb62b14572d2e2a74af522dbecebf0b6a2f4a4ebfd981a0a245bcf2c20af`.
From outside the repository, isolated Python with CUDA hidden loaded the real
checkpoint and produced the exact existing 3,841-sample target hash. Source
preservation, scratch/namespace cleanup and uninitialized CUDA checks passed.
The wheel includes the autocast guard and corrected conditioning-file loader.
Complete source hashes, fixture identity and result are recorded in
`deploy/cleaner/evidence/sam-installed-raw-conditioning-2026-09-21.json`.
This is CPU engineering package evidence, not prompt approval, GPU/long-form/
quality certification or production readiness. Existing serving was untouched.

### Offline SAM conditioning-file admission (2026-09-21)

Added `SamPromptCache.from_files` to admit deployment-provisioned raw embeddings
and masks against externally supplied `SamPromptIdentity` values. Each asset has
an exact bounded length (maximum 1,573,376 bytes), little-endian FP32 embeddings
and canonical binary mask bytes. Checksum validation precedes tensor creation;
existing shape/finite/nonzero validation and immutable cache snapshots still apply.
No pickle, T5 load, prompt selection or network access occurs. Nonblocking,
no-follow file opening rejects FIFOs and symlinks; only regular files qualify.

53 prompt-cache/factory/engine/packaging tests passed in 15.27s. New fixtures cover
roundtrip/source removal, missing/truncated/extra/corrupt files, symlink/FIFO
rejection and a noncanonical mask even when its supplied checksum matches.
Targeted Ruff and formatting checks passed. The deployment README documents the
exact byte format and trusted identity requirements.

This closes file admission only, not approved prompt generation/catalogue binding,
quality certification or GPU readiness. Existing wheel evidence remains bound to
its recorded source hashes; no rebuilt wheel or serving restart is claimed.

### SAM CPU autocast admission guard (2026-09-21)

The FP32 CPU precision descriptor already required autocast off, but admission
did not check the thread-local CPU autocast context. The factory now rejects an
enabled CPU autocast context before constructing modules. Session revalidation
uses the same guard before processing. A real Torch BF16 autocast-context test
checks rejection, no construction, preserved caller settings and successful
admission after leaving the context. No policy descriptor or numerical algorithm
changed; this enforces the existing declared policy.

31 factory/engine/packaging/architecture tests passed in 24.40s. Targeted Ruff
and format checks passed for both changed Python files.

Updated loader SHA-256:
`eb2766a59b5e4772125e1fc95594afcb5a9f01309bbb05b2aba8d8502d69ba2c`.
The earlier installed-wheel evidence remains tied to its original source hash;
it does not certify a rebuilt artifact containing this guard. No serving process
or production readiness registration changed.

### Installed-wheel SAM policy-v2 real inference (2026-09-21)

Added `scripts/benchmark_cleaner_sam_installed.py`, which refuses non-isolated
Python/repository imports, checks a bounded conditioning fixture checksum and its
known-answer embedding identity, and executes the concrete factory/engine/session
through the installed package. Built and audited a new 50-file v2 wheel, then
replaced only the project wheel in the existing isolated SAM environment. Both
historical and new wheel artifacts are retained. New wheel SHA-256:
`cdf492478523ab67a057991f83110e4d9d8845a72c1be4404705bb34d0f8604c`.

The separate offline T5 fixture matched the previously recorded embedding digest;
T5 was not installed into the serving-test environment. From outside the repository,
Python `-I` with CUDA hidden and one Torch thread loaded the real checkpoint and
processed the 3,841-sample seed-42 fixture. Output samples matched the established
target hash exactly, with source preservation, scratch cleanup and module namespace
cleanup passing; CUDA remained uninitialized. The v2 runtime and precision digests
are recorded in `deploy/cleaner/evidence/sam-installed-v2-inference-2026-09-21.json`
alongside all wheel source hashes and the fixture hash.

30 targeted factory/engine/packaging/architecture tests passed in 18.77s; dependency
validation found all 37 installed packages compatible. Ruff/format and whitespace
checks passed. The native run emitted the upstream weight-norm warning only.
This closes installed-wheel CPU inference for this engineering fixture, not a
container build, approved prompt provisioning, A40 peak-memory/long-form/listening
certification, backend integration or production readiness. No serving process changed.

### Explicit SAM CPU backend-policy admission (2026-09-21)

Closed a provenance gap in the CPU factory: its earlier precision identity named
the default Torch backend without checking process-wide settings. Policy v2 now
records and requires CPU default device, FP32 default dtype, MKLDNN enabled with
its deterministic flag off, deterministic algorithms off and float32 matmul
precision `highest`. Mismatches fail typed without changing global settings.
`SamSession.process` revalidates runtime admission immediately before pipeline
work, catching configuration drift after session creation.

30 factory/engine/packaging/architecture tests passed in 17.82s. Six altered-setting
fixtures fail admission and preserve their settings; a session revalidation failure
never reaches pipeline/file work. Ruff/format and whitespace checks passed. This
is not synchronization against concurrent global-setting mutation during a native
call: the dedicated worker must retain exclusive configuration ownership.

Factory source SHA-256:
`dc3c1a74077f26b317d628209cb860a85fbae974ffa5a6abbc5e5b8a68559285`;
engine source SHA-256:
`273249f3d09110388f3d1ed8084992e16aeb9373ac4127a2706ff5fb648e4dba`.
The precision/runtime identities intentionally changed. Earlier real inference and
wheel evidence stays bound to v1 and its source hashes; no v2 real-model, packaged
inference, GPU or quality certification is claimed by these tests.

### Isolated SAM wheel dependency alignment (2026-09-21)

Added the cleaner `sam` extra with pinned Torch 2.8.0+cu128 and einops 0.8.2;
NumPy 1.26.4 and SoundFile 0.12.1 were already aligned with the factory. Resolved
52 lock packages offline. Fresh SAM-only installation initially lacked the cached
einops wheel; downloaded that pinned artifact and completed the frozen sync without
changing the root or serving environment. T5 remains separate provisioning, not
a runtime dependency of precomputed conditioning.

Built and byte-audited the 50-file cleaner wheel (SHA-256
`4e440b656189178a94c818a56de86a4607cb7ee3ced094e6f03194152b72e50e`), installed it
without dependencies into `/tmp/hear-cleaner-sam-package.8MWUFr/venv`, then ran
`pip check` successfully over 37 installed packages. Python `-I` from outside the
repository imported the installed factory and matched every dependency pin with
CUDA uninitialized. No transformers, SQLAlchemy, ClearVoice, Demucs, DACVAE,
descript-audio-tools or DeepFilter distribution was present in this SAM-only env.
The installed real-source builders produced 247 core/317 codec FP32 meta tensors
and closed successfully, with the upstream weight-norm warning only.

18 packaging/factory/architecture tests passed in 19.77s; Ruff/format and whitespace
checks passed. Full wheel source hashes and verification scope:
`deploy/cleaner/evidence/sam-isolated-wheel-2026-09-21.json`.
This establishes dependency/import/construction isolation, not real-checkpoint
inference inside the fresh environment, a built container, GPU/quality certification
or production registration. Upstream sources/assets still need separately pinned
deployment provisioning. The temporary environment and audited wheel are retained.

### Real offline factory and engine-session verification (2026-09-21)

Extended the CPU-only Ray probe to open `SamEngine` through the concrete
`PinnedSamFactory`, process a resolved engineering Voice Focus plan with the real
checkpoint and known-answer T5 cache, and close the session. The output float32
sample hash matched the prior direct pipeline exactly:
`862bb0f8c69581effe20473b5156d374bf237155e269a44ded7ae694d3b8897c`.
The fixture is 3,841 mono 48 kHz samples, seed 42, 16 steps and codec tile 257.
Input bytes, output shape/finiteness, intermediate cleanup, module-namespace
cleanup and preservation of the borrowed prompt cache all passed. CUDA remained
uninitialized and no optional vision/ranking modules were imported.

31 factory/engine/probe/packaging/architecture tests passed in 19.61s with two
existing Torch warnings. Ruff/format and whitespace checks passed. Exact runtime
digests and full results are recorded in
`deploy/cleaner/evidence/sam-factory-real-cpu-2026-09-21.json`.
This closes the real CPU factory/session smoke gap, not production admission.
The prompt remains an engineering fixture, prior diagnostic objects make this
unsuitable as a cold-load peak measurement, and A40/device admission, clean-image
dependency alignment, long-form/quality, backend integration and release gates
remain open. No readiness entry or serving restart was added.

### Concrete offline CPU SAM factory assembly (2026-09-21)

Added `PinnedSamAssets`, `PinnedSamFactory` and `LoadedSamBackend`, composing the
verified source builders, exact optional-key manifest, strict checkpoint loader,
borrowed pinned prompt cache and concrete separation pipeline. The factory checks
explicit package versions before loading, verifies the manifest bytes, admits the
prompt before model construction, verifies 247/317 key counts and CPU residency,
and closes both builders on partial failure. Backend close retires the pipeline
and owned module namespaces without clearing the shared immutable prompt cache.

Runtime identity now binds source/config/checkpoint/manifest hashes, package pins,
prompt-cache identity, CPU FP32 policy and the full long-form descriptor (solver,
RNG, codec tile frames, resampling, mean route, retained watermark and mono policy).
Changing solver or tile policy changes the identity and mismatched admission fails.
The supported factory variant is explicitly CPU; this does not substitute CPU
results for the required A40 release runtime or enable a serving capability.

22 factory/engine/packaging/architecture tests passed in 26.02s; Ruff/format and
whitespace checks passed. Fixture builders cover successful ownership transfer,
codec/checkpoint/post-load cancellation failures, close idempotency and missing
prompt rejection before construction. Source SHA-256:
`e1c96ab827bd13c8f43e209154bcb3820dca5f07e2cb11cd56b6ab940d21368d`.
The package allowlist now has 50 files. A real-model run through the assembled
factory, GPU/device admission, approved prompt provisioning, packaging dependencies,
long-form/speech quality and production registration remain open.

### Meta-built codec real checkpoint/waveform verification (2026-09-21)

Switched the CPU Ray diagnostic's codec construction to `SamCodecBuilder` with
explicit lifetime cleanup. Both core and codec now begin with meta checkpoint
tensors, and the strict loader admits 247 core/317 codec keys with the exact 601
vision exclusions. The real codec's weight-normalization hooks and recurrent
decoder executed successfully after assignment. Its canonical namespaces were
removed after the run without changing the serving process.

The seeded 3,841-sample mono fixture preserved exact feature/file target equality
and reference tolerance: target maximum difference 5.8498699218034744e-9, paired
maximum difference 1.4901161193847656e-8. Finite samples, exact shape, borrowed-input
preservation and owned scratch cleanup passed. CUDA remained uninitialized and
optional modules were absent. All 32 codec-builder/checkpoint/probe/packaging/
architecture tests passed in 31.10s with three upstream warnings. Ruff/format and
whitespace checks passed. Full result:
`deploy/cleaner/evidence/sam-meta-codec-real-cpu-2026-09-21.json`.
This is short-fixture checkpoint-loaded codec evidence, not final cold-load peak
measurement, complete factory/readiness wiring, speech/no-target quality, long-form
or GPU certification. Earlier conditioning-parity failures remain open.

### Pinned meta-device codec construction (2026-09-21)

Added `SamCodecBuilder`: verifies the original layers, VAE bottleneck, codec model
source and Small config before execution, then constructs the original encoder,
VAE bottleneck and watermark decoder directly on meta. This avoids the upstream
DACVAE constructor's temporary vector-quantizer allocation, downloader imports,
training-loss package imports and global audiotools registration. Original decoder
defaults are extracted as literal constants from the verified module. The retained
watermark decoder remains alpha 0.25 with 16-bit messages.

The codec's canonical module names are protected by a lifetime lease because the
bounded graph validates original class identity. Existing foreign DACVAE imports
are rejected without modification; owned namespaces and the verified TorchScript
source cache are removed on close. TorchScript receives the verified source bytes
through linecache, not an unchecked source-path reread. The ordinary bottleneck
dummy-loss scalar is CPU; all 317 checkpoint tensors are FP32 meta.

The first real-source test exposed missing module-level decoder constants; these
were added from pinned source rather than substituted. The final real-source,
packaging and architecture run passed **13 tests** in 23.50s with one upstream
weight-norm deprecation warning. Repeated build/close, unchanged RNG, exclusive
ownership, foreign-module preservation and key count were checked. Ruff/format and
whitespace checks passed. Source SHA-256:
`bf2f8c747e0191e36e93da4096479e4fe33eaff5de3adb9614026339afac5cd0`.
The package now contains 49 files. Real checkpoint/codec waveform parity through
this constructor, complete factory/device policy and certification remain pending.

### Meta-built SAM core real inference verification (2026-09-21)

The CPU-only Ray diagnostic now obtains its core from `SamCoreBuilder`, checks
that checkpoint tensors start on meta and frequency buffers start on CPU, and
closes the builder-owned namespace in a finally block. Real checkpoint assignment
and the subsequent strict combined reload passed. Original forward-method reference
comparisons at 2/7/13 latent frames remained exact; the 3,841-sample seeded waveform
target still matched the feature pipeline exactly and the native reference within
unchanged tolerance (maximum target difference 5.8498699218034744e-9). Paired maximum
difference was 1.4901161193847656e-8. CUDA stayed uninitialized, optional modules
were absent, and input-preservation/intermediate-cleanup checks passed.

Regression initially caught three invalid-step tests reaching source loading before
argument rejection. Restored early validation; all 30 targeted tests then passed
in 19.89s with two existing Torch warnings. The valid Ray run was not restarted;
its numerical path was unchanged by the validation fix. Both launch and final
probe hashes are recorded in `deploy/cleaner/evidence/sam-meta-core-real-cpu-2026-09-21.json`.
Ruff/format and whitespace checks passed. This verifies the meta-built core's
inference path, not codec construction, clean cold-load peaks, long-form/speech
quality, GPU behavior or the complete production factory.

### Pinned-source meta-device SAM core construction (2026-09-21)

Added `SamCoreBuilder`: verifies all six upstream source files and the Small config
against fixed digests before executing any supplied code. It executes the already
verified bytes in a unique namespace with no source-directory import search path,
extracts only the original TransformerConfig/anchor/timestep classes where upstream
top-level imports pull in optional models, and runs the pinned core module bodies.
No multimodal constructor, vision encoder or ranker is instantiated.

Learned parameters are constructed on the meta device, avoiding a full random CPU
parameter allocation before strict checkpoint assignment. Nonpersistent rotary,
transformer-timestep and outer timestep frequency buffers are explicitly recreated
on CPU using the pinned methods/formula; they are not left as uninitialized meta
storage. `SamCoreModules.close` drops the core reference and private module names.
Construction failures remove their namespace and report typed unavailability.

24 source-builder/checkpoint/packaging/architecture tests passed in 28.23s with
explicit real pinned source/config paths, including the real constructor test.
It verified 247 FP32 meta checkpoint tensors, finite CPU buffers, unchanged global
Torch RNG, no optional package imports, eval mode and namespace cleanup. Missing/
mismatched source fails before execution. Ruff/format and whitespace checks passed.
The package now contains 48 files. This proves construction only: combined strict
loading/output parity through this builder, codec construction, full factory/device
policy and memory/quality certification remain unfinished.

### Real strict SAM checkpoint-loader verification (2026-09-21)

Verified the full local checkpoint SHA-256, then recorded its 601 excluded vision
keys in `deploy/cleaner/sam-small-optional-keys.json`. The manifest is tied to the
previously pinned checkpoint, not regenerated dynamically during admission. Its
SHA-256 is `df43a1d8d8306fab6d9e9d70c991eb364efbdacdddac51bdc2e981a914bbed7e`;
a regression test locks the checksum, checkpoint identity, unique sorted inventory
and vision-only exclusion scope.

Updated the CPU-only Ray diagnostic to reload both constructed audio modules
through `SamCheckpointLoader` before reference/file-pipeline comparisons. Strict
admission loaded 247 core and 317 codec keys and excluded exactly the pinned 601
vision keys. The 3,841-sample seeded fixture retained exact file/feature target
equality, target/reference maximum difference 5.8498699218034744e-9 and paired
maximum difference 1.4901161193847656e-8, within unchanged tolerance. Input/cleanup
checks passed, CUDA stayed uninitialized, and no optional modules were imported.

30 checkpoint/probe/packaging/architecture tests passed in 17.40s, with two existing
Torch backend warnings. Ruff/format checks passed. Full result:
`deploy/cleaner/evidence/sam-strict-checkpoint-real-cpu-2026-09-21.json`.
The diagnostic still constructs modules using source extraction and had already
loaded core weights before the strict reload; it is not clean cold-load/memory
certification or a completed production factory. Source pinning, approved prompt
provisioning, device policy, long-form/speech quality and serving gates remain open.

### Strict SAM CPU checkpoint admission (2026-09-21)

Added `SamCheckpointLoader` as a reusable production-loader component. It hashes
the offline checkpoint in bounded reads with guard checks, uses CPU mmap and
`weights_only=True`, and requires an exact expected key set composed of constructed
core/codec state plus an explicitly pinned vision-key allowlist. Extra vision keys
are rejected, not swallowed by a prefix filter. Only vision keys may be omitted;
all retained shapes and every checkpoint tensor's CPU/FP32 type are checked before
either module receives assignments. Both loads use strict assignment, then eval.
Malformed/restricted-unpickling failures are typed without exposing native details.

25 checkpoint/engine/packaging/architecture tests passed in 14.54s; targeted Ruff,
formatting and whitespace checks passed. Synthetic modules cover hash, missing/
extra/extra-optional key, shape and dtype mismatch before core mutation, successful
strict loading, forbidden audio-key exclusion and restricted-unpickling failure.
Source SHA-256:
`c18b361d4075e84583714ceca8d9b02108339d836a922c52533cb99ed26eb6b3`.
The package allowlist now has 47 files. No new real checkpoint run occurred here.

Trusted deployment must supply the expected checkpoint digest and exact optional
manifest, keep the asset immutable through hash/load, verify source/dependencies,
construct the correct audio-only modules and discard modules after any load error.
The complete concrete factory, prompt provisioning, device policy, memory evidence
and serving registration remain unfinished. This component is not load-peak or
production certification, and does not promote the diagnostic AST loader to serving.

### SAM executor/session adapter (2026-09-21)

Added `SamEngine`/`SamSession` implementing the existing engine protocol and
calling the concrete `SamSeparationPipeline.separate_plan` route. Construction
and each session require factory identity validation; loading is explicit, not
an automatic model download or fallback. Sessions bind the exact plan and resource
guard, accept one processing call only, and hold an engine-wide nonblocking lease.
Closing during active inference is rejected. Successful close is idempotent;
unload failure marks the engine unavailable and releases the session lock. Rejected
post-load prompt/channel policy closes loaded state before releasing the lease.

31 engine/plan/packaging/architecture tests passed in 16.93s; targeted Ruff,
formatting and whitespace checks passed. Fixture factories test concurrent-session
rejection, exact plan/seed-conditioning route binding, successful/failed inference,
one-shot processing, close-once behavior, load rejection cleanup and unload-failure
quarantine. Source SHA-256:
`1681bf95dec5bcce9153bdfbc2f731201e740dccb80949670440ba4a827d3d5a`.
The isolated package now includes 46 files.

The concrete production `SamBackendFactory` is still missing: its implementation
must verify pinned assets/source/precision/solver/RNG/codec/resampling policy and
own partial-load cleanup. These fixture tests do not prove those loader duties or
GPU failure recovery. No certified runtime, readiness entry or production serving
registration was added, and all quality/memory/backend acceptance gates remain open.

### Broad cleaner regression and explicit model checks (2026-09-21)

Ran `OMP_NUM_THREADS=1 .venv/bin/pytest -q tests/test_cleaner_v2_*.py
tests/test_architecture.py`: **629 passed, 13 skipped**, two existing Torch backend
warnings, 136.13s. This covers the current cleaner unit/integration suite, not the
root application's unrelated legacy workflows or production deployment.

Resolved every default-run skip in explicit follow-up runs:

- Ten DF3 checkpoint tests initially errored at setup in the root environment
  because `deepfilternet` was absent. No dependency was installed into that shared
  environment. Re-running with the existing isolated
  `/tmp/hear-cleaner-df3.wRIG8t/venv/bin/python`, the pinned offline checkpoint and
  the root site-packages appended only as a fallback for pytest produced **10
  passed**, one torchaudio deprecation warning, 5.31s. Existing isolated packages
  take precedence; this is a test harness, not a clean-image packaging proof.
- Explicit `HEAR_TEST_SILERO_ONNX` using the local pinned Silero asset enabled
  the three real tests in noise-reference, speech-activity and speech-risk files:
  **3 passed, 40 deselected**, 24.33s. These verify adapter behavior, not listening
  quality or speech-detection accuracy on an approved corpus.

Corrected import spacing in resource guard, codec graph and subprocess runner.
A later check observed compacted imports/spacing in `sam_pcm.py`; its functional
body was inspected, formatting corrected, and **24 PCM/architecture tests passed**
in 4.78s. The source-changing actor is not established; historical hashes are not
reassigned to these formatted sources. Targeted broad Ruff/format checks passed.
No Ray serving restart, production GPU change, approval of prompts, or release
certification follows from these results. C12/C13 production loader/long-form,
C17/C18 backend integration, C25/C26 certification and migration/removal gates
remain open.

### DeepFilter session output-ownership repair (2026-09-21)

Completed the follow-up audit of `DeepFilterSession.process`: its prior broad
failure cleanup could delete a concurrent caller-owned destination even after
the resampler preserved it. Both native-48 kHz and resampled execution now target
an owned temporary directory. Only after successful processing and a guard check
does the session publish via create-only hard link. Conflicts preserve the existing
destination. Post-publication failure cleanup checks the recorded inode, matching
the resampler's trusted-workspace ownership assumption. Inner unconditional cleanup
now applies only to paths inside the session-owned directory.

72 DeepFilter/resampling/executor/packaging/architecture tests passed in 52.49s;
targeted Ruff, formatting and whitespace checks passed. Four new race cases cover
48/44.1 kHz input and successful/failed native inference: all preserve the competing
file and clean owned scratch, with session closure still releasing the backend.
Source SHA-256:
`79877e863e79af4199bc415a65ef173acc2fcedc85ad13e5f2a01d9bb4df506f`.
DSP policy is unchanged; this is output-ownership evidence, not renewed real-model
quality, memory or release certification. Other artifact boundaries remain subject
to their own ownership tests and the full implementation plan remains open.

### Create-only resampler output publication (2026-09-21)

Fixed the shared `AudioResampler.convert` boundary: FFmpeg now writes only into
an owned temporary directory, and validated output is published using an exclusive
hard link. A concurrent destination produces `ARTIFACT_CONFLICT` without overwrite
or deletion. Native failure cleans private staging rather than the public path.
Post-publication failure cleanup checks the published device/inode before removal.
The workspace remains trusted attempt-local storage; this is not a defense against
an adversary continuously replacing paths during cleanup. Filesystems without
hard-link support fail typed; no overwrite fallback is used. DSP policy is unchanged.

62 resampling/SAM-plan/DeepFilter/packaging/architecture tests passed in 23.02s.
New real-FFmpeg race fixtures verify preservation of a destination introduced
during conversion, both when the native call succeeds and when it fails. A final
native-call cancellation case verifies staging cleanup. Targeted Ruff, formatting
and whitespace checks passed. Source SHA-256:
`11890049ef6992f9560cdc1b91d290d70b8ae3f5ce987d8768cf73be3ce54bfa`.
This evidence covers the resampler boundary; broader callers with unconditional
destination cleanup (notably `DeepFilterSession.process`) still require an ownership
audit. No production restart or new model/quality certification occurred.

### SAM plan-path source-rate restoration (2026-09-21)

Connected the existing bounded cancellable `AudioResampler` to `separate_plan`.
Mono PCM at 8–96 kHz is preflighted; non-48 kHz input is converted to model rate,
processed with the pinned seed/cache binding, then converted back to the original
sample rate with exact source frame count. The 48 kHz path bypasses resampling.
Import checks workspace paths, input size, PCM format, channel count and frame
bounds. Conversion files are attempt-local and removed after success or failure;
a post-export failure removes the newly created destination.

59 plan/resampling/file-pipeline/pipeline/packaging/architecture tests passed in
21.81s; targeted Ruff, formatting and whitespace checks passed. New tests use real
FFmpeg conversion and a fixture inference path at 8/16/22.05/24/32/44.1/48/96 kHz,
including non-round frame counts. They verify model-rate input, exact restored
rate/channel/frame shape, finite samples, unchanged borrowed input and cleanup
after inference failure. This is conversion/lifecycle evidence, not SAM speech
quality at those rates or release certification. Production runtime identity must
bind `AudioResampler.POLICY`; loader/session wiring, stereo policy, quality gates
and certification remain open.

### Resolved-plan binding for SAM conditioning (2026-09-21)

Added `SamSeparationPipeline.separate_plan`, binding a resolved Voice Focus plan
to caller-supplied expected runtime and prompt identities before audio work.
It requires exact runtime equality, a matching plan prompt digest, an exact
verified cache entry and the currently implemented mono channel policy. It passes
the plan seed and isolated cached conditioning to file processing. Unsupported
dual-mono policy is rejected rather than silently treated as mono or downmixed.

47 plan/cache/file-pipeline/pipeline/packaging/architecture tests passed in 15.07s;
targeted Ruff, formatting and whitespace checks passed. Runtime/prompt mismatch,
wrong cached model identity, closed cache and unsupported channel policy all fail
before audio processing; repeated valid calls receive fresh conditioning copies.
The caller must obtain expected identities from trusted loader configuration,
not echo them from the incoming request. This is a plan-binding layer, not the
missing production loader, full CleanEngine/session adapter, registry certification,
stereo preflight or quality gate. Existing real-model probe evidence remains tied
to its earlier source hash; no new real-model run was made in this step.

### Pinned CPU text-conditioning cache (2026-09-21)

Added `SamPromptIdentity` and `SamPromptCache` for bounded, precomputed CPU FP32
conditioning snapshots. Keys bind prompt/model/precision; exact lookup additionally
requires embedding/mask hashes and token count. The identity digest covers all
fields with a versioned schema. Construction verifies actual tensor bytes, finite
nonzero embeddings, nonempty boolean masks and exact [1,L,768]/[1,L] shapes, with
L<=512 and at most 16 entries. Duplicate keys fail rather than replace an entry.
Snapshots isolate encoder-owned tensors; reads clone them so one invocation cannot
mutate later conditioning. Closing clears entries and later lookups fail closed.

34 prompt-cache/file-pipeline/packaging/architecture tests passed in 16.83s;
targeted Ruff, formatting and whitespace checks passed. Coverage includes identity
component mismatch, corrupt/nonfinite/zero tensors, mask corruption, shape/dtype
rejection, mutation isolation, strict size bounds and closed-cache behavior.
Evidence: `deploy/cleaner/evidence/sam-prompt-cache-2026-09-21.json`.
The isolated package now has 45 files. This implements cache storage and admission,
not prompt approval, encoder provisioning, production loader/engine integration or
quality certification. No engineering prompt is promoted to an approved preset.

### Real seeded SAM file-pipeline CPU probe (2026-09-21)

Extended the existing CPU-only Ray diagnostic to compare `separate_file` against
the feature-file pipeline and an independently assembled native reference. The
reference now uses independently generated PCG64 seed-42 latent/message streams
matching the pinned noise contract, replacing the prior sine-noise fixture for
new runs only. Historical evidence remains unchanged.

Strict real core (247 keys), codec (317 keys) and offline T5 processed 3,841 mono
48 kHz synthetic sine samples with 16 midpoint steps. Exported target samples
matched the feature-file target exactly and matched the native reference within
unchanged atol 1e-6 / rtol 1e-5 (maximum absolute difference
5.8498699218034744e-9). Paired waveform maximum difference was
1.4901161193847656e-8. Input preservation, exact mono RF64 FLOAT shape and
intermediate cleanup passed. CUDA remained uninitialized and no optional ranker
or vision modules were imported. The worker terminated successfully.

25 targeted probe/file-pipeline/packaging/architecture tests passed (two existing
Torch backend warnings); Ruff/format checks passed. Evidence:
`deploy/cleaner/evidence/sam-file-pipeline-real-cpu-2026-09-21.json`.
This short single-window numerical fixture is not speech/no-target/long-form or
GPU certification, does not waive earlier conditioning parity failures, and does
not supply approved prompts or production registration. No serving restart occurred.

### Prepared-file SAM orchestration (2026-09-21)

Added `SamSeparationPipeline.separate_file`: imports prepared mono 48 kHz PCM,
generates one global seeded latent file plus independent watermark messages,
runs the existing separation pipeline, and exports only the target as mono RF64.
It returns the seed/frame/policy noise identity for later runtime provenance
integration. Existing output conflicts fail before PCM/model work. Scratch input
and noise files live in an attempt-local temporary directory; all mappings are
closed and owned intermediates removed. A failure after export (including cleanup
failure) removes that newly owned waveform instead of leaving a success-looking
output. The borrowed source and any pre-existing destination remain untouched.

Verification: 69 file-pipeline/pipeline/PCM/noise/packaging/architecture tests passed
in 15.64s; targeted Ruff, formatting and whitespace checks passed. Lifecycle tests
use a fixture separator with real PCM/noise I/O: repeated calls yield identical
noise, watermark messages are checked against the independent stream, only target
samples are exported, and injected separation/cancellation/cleanup failures leave
only the borrowed input. Evidence:
`deploy/cleaner/evidence/sam-file-pipeline-2026-09-21.json`.
This does not add a real-model parity run, approved prompt cache, engine registration,
quality gates, resampling/stereo policy, GPU admission or release certification.

### Bounded SAM PCM file boundary (2026-09-21)

Added `SamPCM` to import prepared 48 kHz mono WAV/RF64 PCM into bounded feature
files and export explicitly selected target or residual as mono RF64 float32.
There is no implicit downmix, resampling, target/residual stereo interpretation,
normalization or clipping. Source files are borrowed; output creation is exclusive
and failed partial outputs are removed. Import enforces input byte limits; feature
allocation and export reservations account for existing workspace usage. Guards
check each tile and completion. The package allowlist now contains 44 files.

Verification: 64 PCM/pipeline/noise/packaging/architecture tests passed in 14.71s;
targeted Ruff and formatting passed. Three tile sizes preserve exact 257-frame
float32 samples for both output streams. Tests cover wrong rates/stereo rejection,
nonfinite samples, incomplete/wrong-batch export, invalid tile sizes, scratch/input
limits, existing-file preservation and cancellation cleanup. Evidence:
`deploy/cleaner/evidence/sam-pcm-adapter-2026-09-21.json`.
This supplies the prepared-file boundary; complete engine orchestration, stereo
preflight policy, source-rate conversion, production loader and certification
remain open. No model or serving process was changed by this verification.

### SAM noise allocation-failure handling (2026-09-21)

Hardened the existing seeded sampler: RNG-state, watermark and partial-noise
allocation failures now report typed `RESOURCE_EXHAUSTED` errors. A partial
noise file is removed without touching unrelated files. Frame and tile counts
must be positive integers (booleans and fractional values are rejected), and
identity calculation validates the seed without allocating an RNG. The guard
is checked before RNG construction as well as during and after tile generation.

Verification: 44 noise/pipeline/packaging/architecture tests passed in 19.36s;
targeted Ruff, formatting and tracked whitespace checks passed. Existing fixed
policy, noise, watermark and invocation digests remain unchanged. Evidence:
`deploy/cleaner/evidence/sam-noise-resource-failures-2026-09-21.json`.
These are injected allocation failures, not physical memory-exhaustion or GPU
certification. Production SAM integration and the broader acceptance gates remain
open; the preceding noise evidence retains its historical source hash.

### Versioned bounded SAM noise generation (2026-09-21)

Added `SamNoise` / `SamNoisePolicy`: pinned NumPy 1.26.4, explicit PCG64 and
SeedSequence domain identifiers, float32 normal latent draws in global time-major
target/residual order, and an independent stream for two 16-bit watermark messages.
Policy digest records algorithm/layout/version; invocation identity adds plan seed
and latent frame count. The 63-bit plan-seed range is enforced. Global NumPy RNG
state is untouched. This deliberately does not promise Torch.randn bit identity
for the same integer seed; reference comparisons must receive identical explicit
noise and messages.

Noise is written in bounded tiles through exclusive feature-file ownership.
Failures/cancellation remove only the newly created output. A final guard check
also catches cancellation after the final write. Existing destinations are preserved
and incompatible NumPy versions fail closed. The isolated package has 43 files.

All 126 targeted noise/pipeline/feature/graph/conditioning/solver/packaging and
architecture tests passed; Ruff/format checks passed. Five write-tile sizes
(1,7,128,250,1024) produced identical bytes for a 257-frame seed-42 fixture.
Fixed noise/message/identity digests are asserted in tests; seed changes, independent
message generation, global-state isolation and first/final-tile cancellation passed.
Evidence: `deploy/cleaner/evidence/sam-noise-reproducibility-2026-09-21.json`.
Production runtime-identity and engine invocation wiring remain open; this supplies
explicit inputs to the existing pipeline, not a certified serving RNG policy or
new speech/GPU/long-form quality evidence.

### Attempt-local connected SAM separation pipeline (2026-09-21)

Added `SamSeparationPipeline`, composing bounded codec mean encoding, frozen text
and watermark snapshots, the 16-step global solver, joint target/residual decode
and original-length trimming. Inputs are borrowed and must be complete, compatible
feature files from the same resource guard. Initial noise remains explicit rather
than introducing an unvalidated RNG policy. Solver files live in an attempt-owned
temporary directory; mean/field/solver intermediates are cleaned on failure, and
a decoded result is also removed if cancellation occurs before return.

All 215 targeted tests passed (two existing Torch warnings), including lifecycle
failures during encoding, solving, decoding and after decoding, frozen caller
inputs, incomplete-noise rejection and cleanup preserving borrowed files. Ruff
and formatting passed after final import sorting. Package allowlist has 42 files.

A CPU-only Ray probe strictly loaded all 247 core and 317 codec keys plus real
local T5 conditioning. The connected pipeline processed a 3,841-sample synthetic
mono sine with explicit noise and distinct watermark messages, using 16 steps.
Both returned streams had exact [2,1,3841] shape and were finite. Outputs matched
the independently assembled original full-codec/core-midpoint/decoder reference
at unchanged atol 1e-6 / rtol 1e-5; maximum absolute difference was
1.210719347000122e-8. Borrowed input bytes and intermediate cleanup were verified.
CUDA stayed uninitialized; optional vision/ranking modules were not imported.
Evidence: `deploy/cleaner/evidence/sam-separation-pipeline-cpu-2026-09-21.json`.

This is connected numerical execution on one approximately 80 ms fixture, not
speech quality or release approval. It does not waive earlier mean-conditioning
differences. Production asset/loading policy, PCM file adapters, seeded-noise
identity, quality/no-target checks, engine registration, long-form bounds, GPU
certification and backend/publication/deployment gates remain unfinished.

### Joint target/residual decoder handoff (2026-09-21)

Added bounded `paired_latents` conversion from joint [1,256,T] features to
[2,128,T] in the original target-then-residual order. `decode_joint` composes that
conversion with existing out-projection, watermark-preserving decoding and exact
original-frame trimming, retiring the paired intermediate on success or failure.
The returned batch contains two semantic streams, not stereo channels. Residual
is decoded from its own model latents, never synthesized by subtracting target.

All 78 focused feature/graph/packaging/architecture tests passed. They cover
different tile sizes, partial tails, stream ordering, invalid layouts and failed
decode cleanup without touching the borrowed joint input. Ruff/format checks
passed. A real CPU-only Ray probe strictly loaded 317 codec keys and compared
the complete paired decoder against the original reshape/out-projection/decoder
route, with different explicit watermark messages for each stream. Both 3,839-frame
outputs matched at unchanged atol 1e-6 / rtol 1e-5; maximum absolute difference
was 8.940696716308594e-8. Intermediate cleanup passed and CUDA stayed uninitialized.
Evidence: `deploy/cleaner/evidence/sam-joint-decoder-cpu-2026-09-21.json`.

This closes the latent-layout handoff primitive, not the complete audio-separation
engine. Pipeline lifecycle, seeded-noise identity, production loading, conditioning
parity, long-duration behavior, quality and GPU/release certification remain open.

### Real-core sixteen-step solver validation (2026-09-21)

Extended the offline core probe to default to the specified 16 midpoint steps,
while retaining explicit `--steps 2` for the earlier smoke fixture. Other step
counts are rejected before source/model loading. The isolated worker's execution
deadline is 600 seconds for this CPU diagnostic; no serving timeout or configuration
was changed. Corrected the remaining import-format failure and verified it stayed
clean after tests and the completed real-model run.

The real CPU-only run strictly loaded 247 core keys and used the pinned real T5
embedding. Across seven global latent frames, five-frame windows and two-frame
overlaps, all 16 steps completed (96 native field evaluations including a one-frame
tail). The final disk-backed state matched independent in-memory integration of
the original forward bodies exactly; outputs were finite and intermediate cleanup
passed. CUDA was uninitialized and optional modules were not imported.

All 46 focused conditioning/probe/solver/packaging/architecture tests passed with
two existing upstream Torch backend warnings. Ruff, format and whitespace checks
passed. Exact source hashes and results are recorded in
`deploy/cleaner/evidence/sam-conditioned-solver-16step-cpu-2026-09-21.json`.
This verifies the required step count on a small integration fixture, not complete
audio separation, production 250-frame windows, long-form quality or GPU memory.
The broader objective and earlier codec-conditioning parity gates remain open.

### Disk-conditioned real SAM solver binding (2026-09-21)

Added `SamConditionedField` to bind the existing global midpoint solver to
`AudioOnlySamForward`. It reads exactly the requested global mean-feature interval,
owns frozen text/mask snapshots, copies solver input/output arrays, validates
window shape/dtype/time/offset and releases snapshots on close while preserving
borrowed feature files. Tests cover offset correctness, caller mutation isolation,
closed-field behavior and invalid windows.

All 200 targeted conditioning/probe/graph/feature/recurrent/convolution/solver/
packaging/architecture tests passed, with two known upstream Torch warnings.
The real CPU-only Ray probe strictly loaded 247 core keys and real T5 text features.
A seven-frame, two-step solve with five-frame windows and two-frame overlaps made
12 native evaluations, including a one-frame final window. It matched an independent
in-memory midpoint implementation using original forward bodies exactly; no
solver intermediate files remained. Earlier direct-forward comparisons still
matched exactly. CUDA stayed uninitialized and optional modules were not imported.

Evidence: `deploy/cleaner/evidence/sam-conditioned-solver-cpu-2026-09-21.json`.
Final Ruff import sorting is recorded separately from the observed probe source
hash rather than relabeling historical bytes. This is a short two-step integration
fixture, not the default 16-step generation policy or end-to-end audio separation.
Production source loading, noise/RNG identity, codec-to-solver lifecycle, conditioning
parity, long-form quality and GPU/release certification remain unfinished.

Follow-up: 37 conditioning/solver/packaging/architecture tests passed. The
conditioning source changed again after import formatting (cause not established),
so final Ruff I001/import-format verification is not clean. The later file hash and
failed lint observation are recorded separately in the evidence; earlier source
hashes must not be presented as current-byte certification.

### Bounded audio-only SAM core forward (2026-09-21)

Added `AudioOnlySamForward` with one-batch, at-most-250-frame FP32 conditioning,
real nonzero text features, explicit valid text masks/time range, original
target/residual audio duplication, trained no-video alignment, timestep/text
memory and original DiT execution. Output shape/finiteness and post-native
cancellation are checked. Caller autocast is disabled. Correctly shaped zero
video features are passed through learned alignment rather than omitting its
bias, normalization and gate. No zero-vector T5 substitute is accepted.

A CPU-only Ray probe strictly matched and loaded all 247 non-codec/non-vision
keys, including alignment and anchor weights. It used real local T5 embeddings
for the engineering-only `speech` fixture and original pinned DiT/align/position
definitions. Adapter outputs matched unchanged upstream forward-method bodies
exactly at 2/7/13 frames and times 0/0.5/1. No optional vision/ranking modules were
imported and CUDA remained uninitialized. Source checkout was clean at
bb4c6999d2677c7402360e426afc01ddfad6dce0. Raw evidence and exact source hashes:
`deploy/cleaner/evidence/sam-audio-only-core-cpu-2026-09-21.json`.

All 194 targeted conditioning/probe/graph/feature/recurrent/convolution/solver/
packaging/architecture tests passed, with two existing upstream Torch backend
warnings. Ruff/format checks passed. Package allowlist now has 41 files.
The source-isolation AST mechanism is diagnostic-only, not production loading.
This proves forward-method assembly on short synthetic features, not whole-SAM
separation parity, prompt approval, long-form behavior or memory certification.
Codec/solver binding, deployment-safe loading and all release gates remain open.

### Per-operation CPU backend diagnostic (2026-09-21)

Added diagnostic-only `--mixed-cpu-backend` in the single-use CPU Ray worker:
MKLDNN is disabled outside LSTM calls, and LSTM hooks temporarily enable it and
restore prior flags even when a forward fails. Restoration tests passed. This
uses process-global backend flags and is not thread-safe serving policy; no
production flags, loader, precision identity or readiness setting was changed.

All tested candidate same-policy convolution, recurrent, graph, decoder and
mean/waveform roundtrip comparisons passed the unchanged tolerances. An independent
original-default-backend reference was also computed so this result cannot hide a
changed reference. Both original-default mean comparisons FAILED: maximum absolute
differences were 1.5832483768463135e-5 and 6.440281867980957e-5. Both waveform
comparisons against the original default passed, with maximum differences about
2.14e-8 and 2.05e-8. The candidate is not accepted as equivalent separator
conditioning. Original-default parity remains unresolved.

All 183 targeted tests passed, with two upstream oneDNN/Intel-GPU TF32 warnings
during CPU backend-context tests. Ruff/format/whitespace checks passed. Detailed
results: `deploy/cleaner/evidence/sam-mixed-cpu-backend-2026-09-21.json`.
Next validation must address actual separator conditioning/output and longer
fixtures, rather than treating a same-policy reference change as release approval.

### Bounded mean-latent codec roundtrip (2026-09-21)

Implemented right-only reflect padding to the pinned 1,920-sample hop and bounded
frame/channel selection. Padding matches original Torch semantics; unsupported
ultra-short inputs fail explicitly. Tests cover exact hops, partial tails, tile
boundaries, batch two, rejected shapes and preservation of borrowed inputs.
`SamCodecGraph.encode_mean` now runs padding, encoder, in-projection and first-half
mean selection, with no random VAE posterior sampling. `decode_latents` applies
out-projection, retained watermark decoding and original-length trimming, checking
latent/output geometry and cleaning intermediate files. A synthetic projection
test verifies Torch RNG state is unchanged and failed geometry leaves no output.

All 181 targeted probe/graph/feature/recurrent/convolution/solver/packaging and
architecture tests passed; Ruff and formatting passed. A real CPU-only Ray probe
strictly loaded 317 codec keys and roundtripped 3,840 / 3,841 mono 48 kHz sine
samples using fixed watermark bits. Output lengths and intermediate cleanup passed.
Both waveform comparisons passed unchanged atol 1e-6 / rtol 1e-5, with maximum
differences 1.7695128917694092e-8 / 1.3969838619232178e-8.

Both mean-latent comparisons FAILED the same tolerance: maximum differences
9.894371032714844e-6 / 3.8951635360717773e-5. Waveform agreement cannot certify
separator conditioning or waive those failures. Evidence:
`deploy/cleaner/evidence/sam-codec-roundtrip-cpu-2026-09-21.json`.
Fixtures are only about 80 ms; long-form behavior, accepted numerical policy,
speech/listening tests, separator integration and GPU certification remain open.

### Real-checkpoint watermark failure and ownership checks (2026-09-21)

Extended the CPU-only Ray probe with five real-checkpoint fault cases: mutation
of the caller-owned message after the first message-processor tile, cancellation
inside message processing, early decoder cancellation, late decoder cancellation
at the message processor, and rejection of a nonbinary decoder message. Each case
uses a fresh workspace, guard and cancellation event; cancelled work is not resumed.

All five passed. Caller message mutation did not alter subsequent tiles, which
matched the original-message full reference exactly. Cancellation produced typed
CANCELLED errors. Workspace and raw input-byte checks verified no partial outputs
and unchanged borrowed inputs after each case. The complete short decoder parity
probe still passed, and all 162 targeted regression/architecture tests passed.
Ruff and formatting passed after binding the cancellation hook's loop variable.
Evidence: `deploy/cleaner/evidence/sam-watermark-faults-cpu-2026-09-21.json`.

These checks cover cooperative cancellation around bounded native calls, not
interrupting a running native call, process death or host failure. GPU cancellation,
long-form behavior, source-audio roundtrip and full SAM certification remain open.

### Complete bounded watermark decoder path (2026-09-21)

Added frozen explicit-message processing and the pinned 0.25 blend to the feature
runner, rejecting unsupported processors, nonbinary/non-FP32 messages, shape
broadcasting and disabled/rescaled watermarking. `SamCodecGraph.decode` composes
the inspected full Decoder route, including both watermark LSTMs, reverse
upsample and forward downsample groups, pre/post operations and final blend.
The base path executes the original pre layers except the final convolution,
matching upstream forward_no_conv without temporarily mutating shared modules.
All created intermediates are cleaned on failure; borrowed input is preserved.

All 162 probe/graph/feature/recurrent/convolution/solver/packaging/architecture
tests passed; Ruff and format checks passed. CPU-only Ray execution strictly
loaded all 317 codec keys. With synthetic [1,1024,2] input and a fixed alternating
16-bit message, the complete disk-backed decoder returned [1,1,3840] finite
samples and matched native full decoding at unchanged atol 1e-6 / rtol 1e-5.
Maximum absolute difference was 7.078051567077637e-8. Workspace inspection found
only the borrowed input and returned output after successful completion.
Evidence: `deploy/cleaner/evidence/sam-watermark-decoder-cpu-2026-09-21.json`.

This is one short decoder fixture, not source-audio roundtrip, long-duration or
whole SAM certification. Earlier feature-level parity failures remain recorded.
Mean-latent projection/padding integration, more decoder/message/failure fixtures,
long-form bounds, real separator integration and release gates remain open.

### CPU numerical-mismatch diagnosis (2026-09-21)

Extended the reproducible probe with a fixed-effective-weight FP64 convolution
oracle and an isolated-worker `--disable-mkldnn` comparison. For the previously
failing layer/input, FP64 full and tiled calculations matched exactly. Default
FP32 full and tiled paths differed from the FP64 reference by maxima of
9.19790561049183e-7 and 1.4044548102276622e-6 respectively. This supports
backend-dependent FP32 rounding rather than support/crop error for that fixture.

With MKLDNN disabled, all tested convolution cases passed and all five composed
encoder/decoder-block cases matched exactly, but three of six recurrent cases
failed their unchanged 1e-6 absolute/relative tolerance (worst 3.0100345611572266e-6).
The first disabled-backend run stopped on that recurrent assertion. The diagnostic
now records each activation/recurrent comparison status, like its convolution
and graph checks, so a completed diagnostic does not imply accepted parity.
Default-backend recurrent comparisons continued to pass. A global backend toggle
is therefore not an accepted solution; no production runtime flags or tolerances
were changed. A per-operation precision/backend candidate would need validation
against full codec/waveform behavior before adoption.

All 155 probe/graph/feature/recurrent/convolution/solver/packaging/architecture
tests passed; targeted Ruff and formatting passed. Evidence, including both
worker results and failed-run history, is in
`deploy/cleaner/evidence/sam-cpu-rounding-diagnostic-2026-09-21.json`.
This narrows the CPU discrepancy without certifying other inputs, GPU kernels,
complete SAM outputs, long-duration behavior or release readiness.

### Bounded codec block composition (2026-09-21)

Added `SamCodecGraph` for exact Sequential containers and inspected upstream
Encoder, EncoderBlock, ResidualUnit, DecoderBlock and residual LSTMBlock types.
It invokes existing bounded primitives, preserving decoder alternating-chunk
selection instead of treating its ModuleList as a sequential graph. Unsupported
blocks fail explicitly. Intermediate outputs have unique exclusive paths; inputs
are borrowed, intermediate files are retired promptly, and a run's failure cleans
all created intermediates. Empty sequences return independently owned copies.
The isolated package allowlist now has 40 files.

All 151 graph/feature/recurrent/convolution/solver/packaging/architecture tests
passed. Targeted Ruff passed. A CPU-only Ray probe strictly loaded 317 codec keys
and compared the full encoder and all four decoder blocks. Output shapes and
intermediate cleanup passed. Numerical parity did not: the encoder and first two
decoder blocks exceeded unchanged atol 1e-6 / rtol 1e-5. Their maximum absolute
differences were 1.4662742614746094e-5, 2.384185791015625e-6 and
1.6987323760986328e-6 respectively. The final two decoder blocks matched exactly
on this short fixture (their outputs fit one 257-frame tile). This is not evidence
that their long tiled execution is exact. Previous primitive failures remain
recorded, not reclassified as passing.

Raw evidence: `deploy/cleaner/evidence/sam-codec-graph-cpu-2026-09-21.json`.
The real-model probe exposes accumulated numerical differences requiring further
diagnosis and validation. Mean-latent codec wiring, complete watermark message/
blending execution, long-form bounds, full SAM parity and certification remain open.

### Bounded SAM activation and residual passes (2026-09-21)

Added disk-backed activation execution for original pinned Snake1d and exact
Torch ELU/Tanh/Identity classes. Unsupported temporal modules and training mode
are rejected; learned activation weights must be colocated FP32. Residual
execution preserves branch-plus-shortcut order and upstream even centered
shortcut cropping, with explicit batch/channel/length and shared-guard checks.
True-skip mismatches, odd crops and shorter shortcuts cannot silently broadcast.
Both operations use bounded tiles, exclusive output creation and failure cleanup.

All 146 feature/recurrent/convolution/solver/packaging/architecture tests passed;
targeted Ruff and formatting passed. The CPU-only Ray checkpoint probe tested
89 real activation modules with batch-two, 103-frame features and 17-frame tiles.
All matched full-sequence output within atol/rtol 1e-6; maximum absolute
difference was 5.960464477539063e-8. The recurrent checks still passed and the
previous convolution full-sequence tolerance failure still reproduced unchanged.
Evidence: `deploy/cleaner/evidence/sam-feature-activation-disk-cpu-2026-09-21.json`.

Residual behavior is covered by synthetic crop/tail/rejection tests, not a full
learned residual-block parity claim. Full codec composition, fixed watermark
message state, long-duration memory, whole-model parity and certification remain
unfinished. No production service restart or GPU serving changes were made.

### Disk-backed SAM recurrent execution (2026-09-21)

Connected `SamLSTMStream` to the feature runner with fresh per-invocation state,
contiguous bounded reads/writes, colocated FP32 weight validation, exclusive
destination creation and finally-based state cleanup. Cancellation/native failures
remove the new output without removing input or pre-existing destination files.
Unsupported bidirectional configurations fail before output creation. This keeps
the upstream residual connection and does not replace watermark generation.

The combined feature/recurrent/convolution/solver/packaging/architecture suite
passed 136 tests. An architecture failure in the new probe script was corrected
by assigning its function to a class and removing nested imports; targeted Ruff
and format checks passed. CPU-only Ray execution strictly loaded all 317 codec
keys and compared both real LSTMs at batch two, 103 frames and chunk sizes 8, 17
and 64. All six disk-backed results matched full-sequence output exactly.
The real convolution probe still reports the same full-sequence tolerance failure;
the original threshold was not changed. Detailed evidence is recorded in
`deploy/cleaner/evidence/sam-feature-recurrent-disk-cpu-2026-09-21.json`.

Full codec composition, residual/activation passes, watermark message handling,
numerical parity, whole-model integration and certification remain unfinished.

### Disk-backed SAM codec convolution execution (2026-09-21)

Added bounded frame-major feature storage and a convolution runner using the
existing exact support/crop planner. Creation is exclusive, writes are contiguous,
in-process incomplete reads are refused, and scratch/input/output halo limits are
checked. Failure/cancellation removes only the operation's new output. Extracted
the existing solver's flush-before-madvise behavior into shared `MappedResidency`;
the solver retains its tested eviction wrapper. Native workspace/GPU allocations
are not certified by these host tile limits. The package allowlist has 39 files.

All 130 feature/convolution/recurrent/solver/packaging/architecture tests passed.
Ruff and formatting checks passed. The real CPU-only Ray probe strictly loaded
317 codec keys and exercised 90 convolution modules at two feature lengths.
All 180 disk-backed results exactly matched identical contiguous in-memory tiled
execution; reopening the files also matched exactly and scratch cleanup passed.
Against full-sequence inference, one case exceeded atol 1e-6 / rtol 1e-5:
`decoder.model.1.block.4.block.1`, 32 input frames. Its maximum absolute difference
was 1.4901161193847656e-6; the maximum across all cases was 1.9073486328125e-6.
The stricter full-sequence parity gate remains failed, not silently relaxed.
This isolates the observed difference from disk persistence, but does not prove
the numerical cause or acceptable composed-codec error. The different fixture
does not invalidate the earlier geometry probe's recorded passing result.

Reproducible diagnostic: `scripts/benchmark_cleaner_sam_features.py`; raw evidence:
`deploy/cleaner/evidence/sam-feature-disk-cpu-2026-09-21.json`.
The composed codec graph, waveform parity, fixed watermark message handling,
long-duration bounds, GPU certification and full SAM integration remain open.

### SAM codec convolution window alignment (2026-09-21)

Added integer-only `SamConvolutionGeometry` to compute bounded input support and
output crops for the pinned DACVAE convolution wrappers. It handles ordinary and
transposed convolution, dilation, stride phase, automatic causal/noncausal padding
and transposed output unpadding. Automatic-padding slices retain the original
input-length residue so their wrapper behavior matches full-sequence execution.
Unsupported configurations fail explicitly. Original module forward calls and
weight-normalization hooks are retained rather than reconstructing or dropping
weights. This does not implement a new codec or disable its watermark path.

All 99 convolution/packaging/architecture tests passed, including odd strides,
output padding, dynamic-padding phase, partial tiles and hour-length tile-size
arithmetic. A separate zero-GPU Ray probe strictly loaded the real codec and
tested all 90 Conv1d/ConvTranspose1d layers at two feature lengths (180 cases).
Tiled output matched full output within atol 1e-6 / rtol 1e-5; the worst absolute
difference was 4.172325134277344e-7. Raw geometry counts, source/checkpoint hashes
and scope are in `deploy/cleaner/evidence/sam-convolution-cpu-2026-09-21.json`.
Ruff/format/whitespace checks passed. The intermediate-feature disk runner,
composed recurrent/watermark codec graph, whole-codec parity and long-duration
GPU/quality gates remain open. The package allowlist now contains 37 files;
earlier wheel evidence remains bound to its original source inventory.

### Bounded SAM codec recurrent-state primitive (2026-09-21)

Implemented `SamLSTMStream` for the retained watermark LSTM path: per-instance
hidden/cell state, explicit contiguous feature offsets, bounded block size,
FP32 inference outside caller autocast, original residual connection, finite
input/output/state checks and teardown on in-call failure/cancellation. Duplicate
or skipped timesteps cannot silently reset/advance state. This helper does not
generate watermark messages or claim complete decoder streaming.

All 18 recurrent/packaging/architecture tests passed after correcting the lazy
Torch import to comply with repository architecture. A separate zero-GPU Ray
probe strictly loaded both real 512-wide, two-layer watermark LSTMs from the
checkpoint. Full-sequence versus 8/17/64-step chunk execution on 103-step,
batch-two synthetic features had maximum absolute difference 0 in all six cases.
Each layer retained 4,096 FP32 state elements (16 KiB). CUDA stayed uninitialized.
Raw evidence and source/checkpoint hashes are in
`deploy/cleaner/evidence/sam-recurrent-cpu-2026-09-21.json`.
Convolution streaming, sample alignment, watermark RNG/message coordination,
whole-codec parity and long-duration/GPU certification remain open. The package
allowlist now includes 36 files; previous wheel evidence remains historical.

### Strict real SAM codec CPU probe and watermark/RNG finding (2026-09-21)

Pinned DACVAE source to `414c20785fc3a28373073ea8ef7a1316eeeaca6e` in an isolated
unmodified checkout. A one-use CPU Ray task constructed the codec using SAM's
exact configuration, compared its complete state-key set to the checkpoint's
`audio_codec` subset and loaded all 317 keys with strict validation. No missing
or unexpected codec keys were accepted. It used SAM's deterministic mean latent,
not DACVAE's random variational sample. No optional vision/ranking model was loaded.

The initial repeatability assertion failed because the unmodified decoder draws
a random watermark message. Source inspection also found LSTM blocks in the
watermark path. This changes long-form requirements: preserve watermark modules,
coordinate message/RNG state and recurrent state; independent context-window
decoding cannot simply be assumed equivalent. No watermark was removed, disabled
or replaced to make the test pass.

A repeat with controlled per-call RNG passed: all real encode/decode values were
finite; 48,000-frame and 48,017-frame synthetic inputs produced expected latent
and padded output shapes, with explicit crop back to original length and nonzero
tails. Same-seed decodes were bitwise equal, while another seed produced measured
differences of about 0.000454. Decoder watermark blend alpha remained 0.25 and
CUDA stayed uninitialized. Exact versions, shapes, output hashes and limitations
are in `deploy/cleaner/evidence/sam-codec-cpu-2026-09-21.json`.
This is standalone real codec compatibility evidence, not a SAM separator,
long-form codec, full-runtime parity, ultra-short-input or A40 certification.

### Pinned SAM source and real offline T5 conditioning probe (2026-09-21)

Cloned upstream SAM source at exact commit
`bb4c6999d2677c7402360e426afc01ddfad6dce0` into an isolated temporary checkout;
the checkout remains unmodified. Audited model construction, text/vision paths,
codec construction and base checkpoint loading. The T5 encoder/tokenizer are
loaded separately; optional rankers/span modules are constructed eagerly when
configured. No-video features are zeros of the configured visual dimension,
not a substitute for required T5 features. The remote base loader uses
`cls.revision` instead of the supplied revision argument, reinforcing the need
for verified local model paths. Upstream unpinned git dependencies are not a
cleaner lock or authority to install optional models into production.

Downloaded `google-t5/t5-base` at
`a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1` through a CPU-only Ray task into
gitignored `models/t5-base/<revision>/`. Selected safetensors, model config and
tokenizer files rather than redundant TensorFlow/Flax/PyTorch copies. All file
sizes and hashes were checked; the 891,646,390-byte safetensors file matches LFS
SHA-256 `a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4`.

A fresh one-use CPU Ray task loaded the real T5 encoder/tokenizer with local-only
loading and Hugging Face/Transformers offline flags. FP32 single-thread encoding
of engineering fixture `speech` produced finite `[1, 2, 768]` embeddings with
bitwise-identical repeated output. The loader reported no missing/mismatched
weights and CUDA remained uninitialized. This verifies required text computation,
not the choice of a production prompt or retention of all speakers. Exact asset
hashes, package versions, token IDs and embedding digest are recorded in
`deploy/cleaner/evidence/sam-t5-offline-cpu-2026-09-21.json`.
The audio-only SAM loader, DACVAE/source dependency pins, validated embedding
cache, full-runtime parity, bounded codec route and A40 certification remain open.

### SAM Small access approved, pinned checkpoint downloaded (2026-09-21)

Rechecking gated access now succeeded, superseding the earlier 401/403 blockers.
Downloaded the complete `facebook/sam-audio-small` snapshot at
`20b65f56888142eebe7c37448c6f6b3b32600e9b` through the existing Ray cluster using
one CPU, zero GPUs, no retries and a one-use worker. The token was read locally
from the server `.env`, never embedded in task arguments or evidence. Assets are
under the gitignored `models/sam-audio-small/<revision>/` directory.

Streamed SHA-256 over every downloaded file and checked upstream file sizes.
The 5,100,547,943-byte checkpoint matched its published LFS digest:
`8c44fda9821fd9f2ec8977304e3c0f55290d9eacb6bbf25b4b8fb1f69c2a8c06`.
All file hashes/sizes and raw inspection metadata are in
`deploy/cleaner/evidence/sam-small-download-2026-09-21.json`.

A separate CPU-only Ray task inspected the checkpoint with `weights_only=True`,
`mmap=True` and CPU mapping. It is a flat 1,165-tensor FP32 state dictionary:
audio codec, vision encoder, transformer and conditioning projections/anchors.
CUDA remained uninitialized. The config names required `t5-base` conditioning,
but no text-encoder tensor prefix is present in the downloaded state. Next work
is to pin/audit upstream loader code and required external text assets, construct
the narrow audio-only route and validate parity before any GPU certification.
Checkpoint storage bytes are not an inference-memory claim. No model was
constructed for inference, no capability was advertised ready and no running
production service was restarted.

### Real CPU noise-profile one-hour synthetic probe (2026-09-21)

Added `scripts/benchmark_cleaner_noise_profile.py` and ran 30-second then one-hour
synthetic stereo cases through the existing Ray cluster: one CPU, zero GPUs,
2 GB scheduling memory reservation, no retries, one task per worker process.
The concrete CPU noise engine used real pinned Silero reference review and actual
reference-digest revalidation, not an injected analyser. The generated first
noise-only second was explicitly confirmed for this fixture; music uncertainty
was retained and no user-media confirmation was invented.

Both cases passed full finite-output/frame/rate/channel checks and preserved the
17-frame partial tail. The hour contained 172,800,017 frames at 48 kHz; processing
took 141.48 seconds (RTF 0.03930), with 158.50 seconds overall probe time. The
1,382,400,240-byte RF64 output was hashed in bounded blocks. Torch was not imported.
The task reported 167,936,000 bytes via `getrusage` peak RSS, but external `ps`
samples reached 185,577,472 bytes. Both observations are preserved rather than
claiming the lower value as a ceiling. Neither is aggregate process-tree/native
memory certification; Ray's reservation is not hard enforcement. The worker PID
and generated temporary audio directory were confirmed absent after completion.

Exact runtime, source/script, speech-policy and output hashes plus raw measurements
are in `deploy/cleaner/evidence/noise-profile-cpu-synthetic-hour-2026-09-21.json`.
Four architecture tests, script Ruff/format and whitespace checks passed. The
existing GPU services were not restarted. This advances long-duration CPU engine
evidence, but does not include mastering/publication, real music/transient or
speech-retention listening evaluation, concurrent-load certification or release
approval. CPU profile readiness remains gated on those missing requirements.

### Attempt-local loading/inference and pipeline timing (2026-09-21)

Added bounded `StageTimings` to execution contexts. Eight fixed stage names use
monotonic durations; repeated registry/session loading spans sum independently
of inference, with separate cleanup, inspection, validation, mastering, optional
download and upload. Measurements reset after admission and survive failed spans.
Missing measurements are omitted rather than fabricated as zero. The validation
report includes `stage_seconds_before_publication`; final upload timing stays in
the context for an authenticated transport metrics exporter. No backend queue
time, model-memory metric, billing value or performance certification is inferred.
Mastering and inspection timings include their internal codec/process work.

All 42 timing/executor/factory/packaging/architecture tests passed in 46.63 seconds.
Deterministic clock tests verify loading accumulation, inference separation,
exception preservation, bounded stage vocabulary and per-instance isolation.
Executor tests verify report timing coverage, post-publication upload timing and
failed-cleanup/cancellation measurements without candidate publication. Ruff lint,
format and whitespace checks passed. Packaging allowlist now contains 35 files;
earlier wheel evidence is historical and must be rebuilt for this revision.
Production metrics export and backend queue metrics remain unwired.

### Full cleaner regression with real CPU models (2026-09-21)

Ran the complete `tests/test_cleaner_v2_*.py` suite plus architecture tests after
the worker factory, restricted subprocess environment and teardown-retirement
changes: **352 passed, no skips, one upstream warning, 182.52 seconds**.
`HEAR_DF3_TEST_CHECKPOINT` selected the pinned real DF3 checkpoint and
`HEAR_TEST_SILERO_ONNX` selected the local Silero ONNX asset; both opt-in real-model
paths ran. DF3 dependencies were supplied from the isolated probe package path
alongside the checkout. This is a current-source CPU regression, not an installed
wheel or container test. The warning is upstream DF3's deprecated torchaudio
metadata import; no third-party code was edited to suppress it.

Ruff passed for all cleaner runtime/engine modules, contracts, artifacts,
inspection, mastering, quality and cleaner tests. The independent 51-package
deployment lock passed its offline consistency check; whitespace checks passed.
Current transport inspection still finds no cleaner-v2 registration in the
production deployment/entrypoint surfaces, and immutable storage writes remain
an injected protocol rather than a live B2 writer. Backend repository location
was requested to proceed with the shared authenticated integration. Tests do not
close those gaps, SAM repository approval, listening evaluation, A40 certification
of the latest source, migration/drain or canary gates.

### Failed engine teardown retires the worker (2026-09-21)

Corrected executor cleanup handling: an exception from session close previously
could mask the primary typed error and escape without retiring the worker.
Close failures now produce a sanitized typed restart-required failure, preserving
the original error code when processing had already failed. Failed teardown
after otherwise successful inference prevents validation/mastering/publication.
Cancellation and deadline retain their codes and publish no competing terminal
manifest. Eligible processing failures may publish only a reauthorized failure
manifest with no candidate artifacts; ownership is marked unhealthy before
another attempt can be admitted. No automatic retry or model fallback is added.

All 39 executor/factory/worker-lease/architecture tests passed. New injected-fault
cases cover cleanup after success, cancellation, deadline and resource exhaustion,
sanitized diagnostics, no candidate publication and subsequent admission rejection.
Existing tests still exercise real inspection/mastering and failure reconciliation
with fake inference/in-memory storage. Ruff lint/format and whitespace checks
passed. This is not physical CUDA fault certification or deployed Ray recycling.
The previous wheel evidence retains its original hashes; this later executor
source change must be rebuilt before release verification.

### Explicit cleaner worker assembly and lifecycle readiness (2026-09-21)

Added `CleanerWorkerFactory.build` to assemble the shared executor from explicit
certified runtimes, offline loaders/readiness probes, scoped immutable storage
and optional speech-risk analysis. It acquires CPU/GPU lifetime ownership, rejects
mixed-lane registrations and releases ownership on construction failure. It does
not load/probe models, contact storage, read credentials or start transport.
Worker capability reporting now combines registry readiness with live ownership:
retired, closed or ownership-lost workers cannot advertise a ready runtime.
Close refuses active attempts; session-owned models must finish cleanup first.
External model caches are deliberately unsupported without explicit teardown.

The executor integration fixture now uses this assembly rather than duplicating
constructor wiring. After correcting an architecture violation (standalone build
function moved onto a factory class), all 47 factory/capability/lease/packaging/
executor/architecture tests passed. These include real source inspection,
FLAC/MP3 mastering and manifest generation, but fake inference/in-memory storage.

Rebuilt the allowlisted wheel with 34 source files, including the preceding codec
environment fix. Installed it in a separate environment, then verified every
installed source byte against the wheel under isolated Python execution outside
the checkout. Empty CPU worker assembly/shutdown and subsequent ownership
reacquisition passed; all uncertified profiles remained unavailable. Torch, Ray,
database and legacy model packages were absent; ONNX Runtime was not imported by
assembly. Wheel SHA-256:
`6a620d197437295a02271f1496c1f1cdcb653d3a13e81bcda46c8c9ad0c0cefa`.
Exact source hashes, package inventory and observations are in
`deploy/cleaner/evidence/cleaner-worker-factory-2026-09-21.json`.
Production configuration, grant refresh/immutable writes, authenticated transport,
Ray supervision, model certification and backend end-to-end integration remain
open. This is the shared composition layer, not a deployed production factory.

### Restricted codec child environment (2026-09-21)

Closed an environment boundary gap: source inspection already supplied a
restricted environment, but the shared runner otherwise let FFmpeg inherit the
server environment. The default now contains only the system executable path,
locale and single-thread OpenMP/OpenBLAS settings. Credentials, proxy settings,
Python/library injection variables and FFREPORT are not passed to codec children.
Explicit environments are copied without merging with the parent, and remain
trusted worker configuration rather than ticket fields. The server's own `.env`
loading and credentials are unchanged.

Actual child-process tests verify absent credential/control variables and a real
FFmpeg invocation confirms no inherited report file is created. Runtime,
resampling and packaging tests passed (54 tests), including real codec work.
The separate runtime/mastering/inspection/executor/architecture run passed all
81 tests, covering real FLAC/MP3 mastering and cancellation paths.
Ruff lint and format checks passed. This is environment minimization, not an OS
sandbox or network/filesystem denial; deployment isolation remains required.
Earlier wheel evidence preserves its original hashes and predates this source
change; rebuild the package before certifying or deploying this revision.

### Built and exercised cleaner-only Python distribution (2026-09-21)

Added a separate cleaner package definition and generated 51-package Python lock
covering core, optional CPU analysis and optional DF3/CUDA dependencies. Its graph
excludes legacy cleaner, business database and transcription model dependencies.
An explicit 33-file allowlist feeds an offline setuptools 83.0.0 wheel builder;
the actual archive is checked for exact source bytes and permitted metadata only.
Private docs, environment files, old pipeline protobuf, checkpoints and legacy
application modules are not copied. Nonempty output directories are preserved.
Two builds produced identical wheel SHA-256
`8183920ae12b3b7a78c123cebb4f269abca71400f5cf7f227f07a85f817f31ad`.

Installed that wheel plus hash-checked locked core/analysis packages into a fresh
virtual environment, outside the checkout and with isolated Python import mode.
The initial offline install correctly failed for uncached wheels; those exact
hash-checked dependencies were then fetched. `--no-config` was required to avoid
the legacy root project's NumPy override affecting the new install. Root runtime
packages and the production environment were not changed.

The isolated runtime imported the executor/wire boundary and ran real Silero
selected-reference review, CPU spectral noise reduction, speech-risk comparison
and exact FLAC/MP3 mastering over 16,017 synthetic stereo frames. Validation stayed
`review_required`. SQLAlchemy, psycopg2, Demucs, ClearVoice, WhisperX, Qwen ASR,
Torch and Ray were confirmed absent from that environment. The installed-package
inventory, runtime origin and all source/wheel hashes are preserved in
`deploy/cleaner/evidence/cleaner-wheel-isolation-2026-09-21.json`.

Eleven packaging/architecture tests passed; lock consistency, targeted Ruff and
whitespace checks passed. No Docker/Podman executable is available, so no image
build or image inventory is claimed. GPU execution from this package, SAM,
container/system dependency pinning, production factory and authenticated
transport remain open. The package shares the `hear` namespace and must never
be co-installed into the retained legacy application environment. C27 legacy
removal is not authorized by this isolated packaging result.

A second fresh environment installed the frozen lock with both extras, then the
same audited wheel. Binary-only installation failed for deepfilterlib 0.5.6;
an isolated Rust toolchain successfully built it from source. No legacy package
path was borrowed. Python isolated mode confirmed imports from the new installed
wheel and absence of Ray, database clients and legacy model packages. Real DF3
CPU execution processed 576,017 stereo 48 kHz frames in two contextual blocks,
with finite output and a nonzero final 17-frame tail. Inference took 0.852 seconds;
peak process RSS was 881,139,712 bytes. CUDA was not initialized, despite the CUDA
dependency distributions being installed. Exact inventory, runtime descriptor and
output hash are in
`deploy/cleaner/evidence/cleaner-wheel-df3-isolation-2026-09-21.json`.
This is current v3 loader CPU evidence, not new A40 certification, a Rust/system
dependency lock, listening acceptance or a deployed cleaner service. The first
CPU-only environment and running Ray services were unchanged.

### Typed allocation failure and worker retirement (2026-09-21)

Fixed an inference fault gap: CUDA OOM previously fell through as a generic
processing error. DF3 load/inference now classify host/CUDA OOM as
`resource_exhausted`, close failed backends, and require process replacement
after OOM or CUDA runtime faults. The factory rejects subsequent loads in a
faulted process. Sanitized errors are raised outside native exception handlers
so native tracebacks do not keep failed model/output tensors alive. Cancellation
inside loading/inference receives the same reference cleanup but preserves its
typed code without automatically poisoning a healthy worker.

Added internal `worker_restart_required` control state, preserved across Python
worker serialization and published-failure wrapping. Registry loading preserves
typed resource/fatal errors instead of relabeling them. The executor marks its
worker lease unhealthy, retains ownership and rejects further admissions. Under
a live authorized ticket it can publish only a failure marker, never candidate
artifacts. The supervisor must still recycle the actual worker process; neither
the factory nor executor retries, raises the cap or substitutes another profile.

Verification: full cleaner-v2/architecture suite passed 329 tests with both real
DF3 and Silero explicitly enabled (123.24 seconds). After the cancellation
reference-cleanup refinement, 67 loader/real-DF3/executor/lease/capability/
architecture tests passed (54.00 seconds). A separate registry propagation test
also passed. Weak-reference tests hold the public error while confirming failed
model objects are collectible; injected failures verify terminal-only
publication, session closure, no readmission and serialization of the restart
marker. Targeted Ruff/formatting/whitespace checks passed. The real DF3 tests
emit one known upstream torchaudio metadata-import deprecation warning.

Fault tests are injected, not physical A40 OOM/recovery certification, and the
running Ray services were untouched. Loader policy advanced to v3, changing its
runtime/source identity. Earlier v2 A40 records remain unchanged historical
measurements; final-image release gates must be rerun rather than relabeled.
Supervisor/readiness wiring and real fault/model-switch/overlap drills remain open.

### One-hour real A40 DF3 duration probe (2026-09-21)

Revalidated an idle A40 (34,617 MiB free), Ray GPU availability and 83 GiB scratch
free before admitting a separate 1-CPU/0.1-GPU/4-GB-memory Ray task with no retry.
The child retained the prior 1 GB Torch allocator cap. External supervision
sampled per-process VRAM at requested 200 ms intervals, with a 3 GB observed
stop threshold and 900-second deadline. The existing service actors stayed up.

The pinned DF3 adapter processed 172,800,017 stereo frames at 48 kHz using 361
contextual calls, at most 576,000 input frames per call. Output was finite,
frame-exact and retained the nonzero 17-frame tail. Inference took 34.569 seconds;
benchmark elapsed time was 45.983 seconds and supervisor elapsed time 57.984
seconds. Sampled child VRAM peaked at 834,666,496 bytes, identical to the short
probe. Torch peak allocated/reserved counters were 293,311,488/501,219,328 bytes;
host peak RSS was 1,181,962,240 bytes. There were 164 GPU-process samples.

Exact observed output, a short progress trace, supervision settings and caveats
are preserved directly from the tool result in
`deploy/cleaner/evidence/df3-a40-synthetic-hour-2026-09-21.json`. Verified schema,
runtime identity equality with the short probe, adapter code hash, frame count
and block/window bounds. After exit, only the original transcription/small-model
GPU PIDs remained and scratch availability returned to its pre-test display.

This strengthens duration evidence but does not certify all memory peaks,
absence of leaks, perceptual speech retention, SAM, busy-service coexistence,
model switching, faults, deployment overlap or the full pipeline. No production
readiness changed and no universal 12 GB hard-partition claim is made.

### Initial real A40 DF3 CUDA probe (2026-09-21)

Read-only inspection found an idle A40 with 34,617 MiB free, two existing GPU
service processes and 0.65 Ray GPU scheduling units available. Ran one separately
admitted Ray task (1 CPU/0.1 GPU/no retries) spawning the offline DF3 benchmark
with a 1 GB Torch allocator cap. Supervision sampled per-PID NVIDIA CLI memory,
required 20,000 MiB free at admission and enforced a 120-second deadline and
3 GB sampled process stop threshold. No production actor was restarted/unloaded.

The actual pinned checkpoint processed 1,440,017 stereo frames at 48 kHz in four
contextual calls (maximum 576,000 input frames) with finite output, exact frame
count and a nonzero partial tail. Inference took 3.125 seconds. Sampled process
VRAM peaked at 834,666,496 bytes; Torch allocated/reserved peaks were respectively
293,311,488 / 501,219,328 bytes. These counters must not be added. The child exited
successfully and GPU usage returned to the original 10,872 MiB. Exact runtime,
code/PCM hashes and metrics are in
`deploy/cleaner/evidence/df3-a40-synthetic-smoke-2026-09-21.json`.

The benchmark gained explicit device/cap options; CPU remains its default.
Nineteen loader/architecture tests and targeted Ruff passed. The evidence is a
synthetic initial hardware probe only: sampled/quantized process memory can miss
peaks, and this does not certify the complete 12 GB release gate, long-duration
CUDA behavior, SAM, real wanted-content retention, simultaneous service load,
deployment overlap, or model switching/fault recovery. Those gates remain open.

### CPU noise runtime provenance enforcement (2026-09-21)

Closed a provenance gap: the noise engine previously accepted any supplied
runtime identity naming `noise_profile`. `NoiseProfileEngine.describe` now
derives runtime/precision/long-form digests from dependency versions, reference
analysis policy, precision, window/hop, channel-linking, smoothing and tail
settings. The reference analyser's policy digest includes the pinned speech
analysis policy (model/dependencies/threshold). The engine validates all digests
at construction, session opening and before processing; mismatches or malformed
reference policies fail typed as unavailable before output creation. Backend
plans are never silently relabeled. The actual smoothing coefficients remain
the existing explicit 0.8/0.2 values.

Verification: 86 noise-engine/reference/contract/capability/wire/architecture
tests passed (16.11 seconds), including the real Silero reference check and
new tests for wrong runtime/precision/long-form digests, changed NumPy/SoundFile/
libsndfile versions, window/smoothing drift and changed/malformed reference
policies after session creation. Targeted Ruff/format/whitespace checks passed.
Descriptors are not image/source certification; release catalogue approval,
real listening/host-memory evidence and production factory wiring remain open.
No running service or production readiness changed.

### Concrete selected noise-reference review (2026-09-21)

Added `SpeechAwareNoiseReferenceAnalyser` for the existing CPU noise engine.
Preview review accepts a revision and selected frame bounds without needing a
placeholder hash or prior confirmation. It hashes the actual complete source,
extracts precisely the selected interval using bounded reads, preserves channel
separation and runs the pinned CPU speech adapter. The returned digest binds
source bytes, revision, interval, rate/layout, analysis policy and activity
counts. Runtime review recomputes that digest; `NoiseProfileSession` rejects
stale evidence or speech in either channel before writing cleaned output.

The reviewer always reports uncertainty because music analysis is unavailable.
Negative VAD cannot establish a noise-only sample; explicit backend-owned
confirmation remains required and cannot override detected speech. No automatic
first/quietest-segment choice is introduced. Invalid bounds, silence, invalid
PCM, contradictory analysis identity, cancellation and analysis failures are
typed; attempt-local crop/resampling files are cleaned on exit.

Verification: 67 noise-reference/noise-engine/contract/wire/architecture tests
passed (9.22 seconds), including an explicitly enabled real Silero repeatability
check, changed-source/revision/interval/policy rejection and a real CPU spectral
noise-engine run with injected speech evidence. The injected speech cases prove
orchestration, not classifier accuracy. Targeted Ruff/formatting/whitespace checks
passed. Preview ingress/authorization, backend confirmation storage, frontend
warning presentation, production factory wiring, music analysis and corpus/
native-call certification remain outstanding. No running service was changed.
Full regression follow-up: 300 cleaner-v2/architecture tests passed and 10
opt-in DF3 tests skipped in 149.66 seconds, with all real Silero tests explicitly
enabled. This includes added silent/NaN/infinite-reference rejection tests.
At that checkpoint, token availability was rechecked without printing credentials
and no token was configured. The later authenticated SAM download entry supersedes
this diagnosis: a token is now configured, but repository access is denied (403).

### Source/output speech-risk integration (2026-09-21)

`AudioQualityGate` now accepts an explicit `SpeechRiskComparison` dependency,
and `CleanExecutor` passes the authenticated pinned source identity. The
comparison scans the source and independently hashed pre-mastering output PCM.
Reports include both hashes, analysis/comparison policy digests, full activity
totals per channel and at most 128 suspected-loss intervals on the source grid.
Intervals omitted by the reporting cap are never treated as silence; incomplete
evidence is explicitly flagged. Mono output is compared to each source channel
without downmixing or claiming that the same speaker/words survived.

Initial engineering warnings use at least 100 ms of missing activity or an output
activity total below half its source-channel total. Equal total activity at the
wrong time still produces a loss warning. These thresholds need corpus
calibration and never grant approval: matching activity stays `review_required`.
Absent analysis is explicit; configured analysis failures are typed and cannot
silently fall back to energy-only validation. Manifest validation binds evidence
to source hash/frame bounds/channel layout and requires associated warning codes.
The executor publishes the same evidence in both validation report and manifest.

This replaces the adapter-only status below, but does not deploy a production
factory or update an external backend. Shared contract approval, factory wiring,
noise-reference integration, native-call supervision and real speech/listening
calibration remain open. The proposed manifest field is documented for backend
consumers; legacy transport and running Ray services are unchanged.
Verification: full cleaner-v2 plus architecture suite passed with 282 tests and
10 opt-in DF3 tests skipped (103.05 seconds), with the real Silero parity and
quality-comparison tests explicitly enabled. A subsequently added executor
manifest/report propagation test also passed. Targeted Ruff, formatting and
whitespace checks passed. Synthetic parity/identity fixtures do not establish
speech-recognition accuracy or perceptual preservation.

### Pinned CPU speech-activity adapter (2026-09-21)

Added `CpuSpeechActivity` using the existing Silero support model through ONNX
Runtime's CPU provider only. Construction verifies exact dependency versions,
bounded model size, model SHA-256 and expected model interface; ONNX receives
the verified bytes. No Torch import, model download or GPU provider is used.
The adapter verifies the pinned source, uses the common resampler for other
rates and scans 512-frame/16-kHz blocks with fresh per-channel recurrent state
and 64-frame context. Reports bind source/policy digests, retain source-frame
positions and complete per-channel active-frame totals, and cap intervals at
128 with an explicit truncation flag. Temporary resampling files are removed.

The installed Silero 6.2.1 model SHA-256 is
`1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.
An opt-in real-model test compares every probability against the installed
upstream `OnnxWrapper` on a seeded stereo fixture with a partial tail; results
match with absolute tolerance 1e-7 and zero relative tolerance. All 17 targeted
speech-activity, quality and architecture tests passed, including that real
model test (34.08 seconds); targeted Ruff/formatting/whitespace checks passed.

This is support evidence, not another cleaner engine or a quality certificate.
It is not yet integrated into source/output risk comparison or the noise
reference reviewer. Threshold calibration on speech/music, native-call
supervision, deployment dependency/host-memory certification and the actual
quality decision integration remain open. No profile readiness was changed.

### Supervised native source inspection (2026-09-21)

Source checksum and bounded libsndfile scans now execute in a fresh child process
instead of inside the model-owning process. Parent supervision enforces the
attempt deadline, cancellation and scratch budget even during a stuck native
decoder call, and kills/reaps the child process group. The child uses a restricted
environment without inherited service/storage/model tokens and returns a bounded,
sanitized structured response. Existing source/hash/format/frame/channel/finite
checks remain in the child; the parent also validates returned metadata/metrics.

New tests cover real audio inspection outside the parent, credential environment
isolation, cancellation/deadline termination of a stuck child and malformed
responses. The model-stage deadline regression pre-inspects its fixture so it
continues to exercise model-session teardown rather than decoder startup timeout.
This is not a native-memory limit or an OS sandbox. Additional in-process native
calls elsewhere and dedicated model-worker supervision remain outstanding.
Verification: full cleaner-v2 plus architecture suite passed with 252 tests and
10 opt-in real-model tests skipped (158.59 seconds). Targeted Ruff/formatting,
standalone architecture and whitespace checks passed. No production deployment
or GPU-memory certification is implied.

### SAM Audio Small download attempted through Ray (2026-09-21)

Update after the user supplied a token: configured `HF_TOKEN` in the gitignored
server `.env`, then retried the same pinned snapshot through the existing Ray
cluster (one CPU, zero GPUs, no task retries). The worker read the token locally;
it was not embedded in task arguments, runtime environment configuration or this
ledger. Download failed with HTTP 403 `GatedRepoError`. An independent CPU-only
Ray check confirmed the token authenticates successfully but gated `config.json`
access returns 403 with the account-not-authorized condition. This is no longer
a missing-token failure: the token owner's account must receive repository
access before another download attempt. No model payload files were downloaded;
only local Hugging Face cache/lock bookkeeping appeared. Running GPU services
were not restarted or changed. The shared workspace filesystem did not honor
`chmod 600` on `.env` (it still reports 0666), so deployment secret-file access
controls also remain unresolved. Never copy the token into source or evidence.

Submitted a CPU-only task to the existing Ray cluster with zero GPU reservation,
pinning `facebook/sam-audio-small` to revision
`20b65f56888142eebe7c37448c6f6b3b32600e9b`. Repository metadata reports a
5,100,547,943-byte checkpoint and manual access gating. The Ray worker found no
Hugging Face token in its environment/cache or the project's `.env` and received
HTTP 401 `GatedRepoError` on the first gated file. No model files were downloaded,
no model was loaded, and no production service was restarted. An account with
approved repository access and its read token are required to resume. Place the
token in server-side `.env` as `HF_TOKEN`; never put it in frontend configuration
or paste it into this ledger. Intended local destination is the gitignored
`models/sam-audio-small/<revision>/` directory.

### Terminal reference wire validation (2026-09-21)

Added bounded terminal-reference encoding/decoding with strict ticket identity,
fence, manifest key, version and outcome/error consistency checks. Encoding checks
the published manifest and receipt; post-fetch verification compares the notice
against the verified bundle. Unknown fields and contradictory notices fail closed.
Verification: 80 wire/artifact/contract/architecture tests passed; targeted Ruff
checks passed. This is a local codec, not a deployed RPC or backend integration.

### Historical initial implementation snapshot (superseded by later evidence)

The following bullets describe the first component implementation, not current
asset availability, supported rates or latest test counts. In particular, DF3
has since been installed and exercised on CPU and A40, and bounded resampling
has been implemented. Release certification remains distinct from those probes.

- C11 loader: `PinnedDeepFilterFactory` verifies local config/checkpoint SHA-256
  and exact configured package versions before imports, explicitly constructs
  DF3 on CPU, uses `weights_only=True` and strict state loading, rejects non-finite
  weights, disables postfilter and applies the allocator cap before model CUDA
  transfer. Fresh filter/STFT state accompanies each contextual inference call.
  A process-global lease prevents simultaneous use of upstream mutable config.
  No checkpoint download, permissive shape filtering or model fallback exists.
  Six loader failure/asset tests plus eight adapter tests and four architecture
  tests cover the component boundaries; actual model loading remains untested
  because the DF package/checkpoints are absent. No approved descriptor/package
  inventory has yet been installed, and the strict checkpoint key compatibility
  must be checked against real pinned weights before release.
  Source audit: https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/df/checkpoint.py.

- C11: bounded `DeepFilterEngine` contextual file adapter implemented with
  explicit attenuation, policy-derived digest, per-attempt backend state,
  context cropping, exact tail/layout checks, float/finiteness validation,
  session exclusion and partial-output cleanup. Eight adapter tests pass with
  a fake backend; these prove block orchestration only. Input currently requires
  prepared 48 kHz audio. The actual pinned DF3 loader/weights, approved
  resampling route, physical delay tests, stereo listening and certification
  are still outstanding. DeepFilter package and checkpoint inventory confirmed
  absent on this host, and Natural remains unadvertised/unavailable.
  Upstream inspection confirmed `enhance()` resets hidden state and uses
  STFT delay compensation; contextual calls must pass real boundary/parity tests:
  https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/df/enhance.py.

- C09/C16 integration: `hear/runtime/cleaner/executor.py` now coordinates an
  authenticated attempt, source inspection, certified-registry admission,
  per-attempt engine session, quality gate, mastering, report and manifest-last
  publication. Authorization is injected and rechecked before uploading; the
  production transport authorizer is not yet implemented. The session closes
  before mastering/upload and on inference failure/cancellation. No business DB
  imports or approval-state changes are introduced. This provisional module
  avoids importing the legacy `service.py` model chain; move it to the final
  service location during the planned consumer cutover.
- Sample execution retains full-source inference context and crops a bounded
  exact-frame output before mastering. It does not yet optimize inference to a
  contextual sample interval. The sample interval remains bound into the
  manifest. Samples never emit application/review decisions.
- `AudioQualityGate` checks finite PCM, exact timebase/channels, severe channel
  disappearance, silence hallucination and no-target output. All surviving
  results remain `review_required`; numeric energy checks do not prove speech
  preservation. Production speech evidence, interval warnings, boundary and
  musical-transient risk calibration remain open.
- Executor integration fixtures use a test-only engine/store with real codec
  inspection/mastering. Tests cover full/sample attempts, missing-channel
  rejection, engine failure/session closure, authorization rejection/revocation,
  and cancellation after inference. They are not production engine certification.
  Current combined verification: 91 tests passed; targeted Ruff checks passed.

- C15: `AudioMasteringService` now measures loudness/true peak, applies bounded
  linear gain, writes 24-bit FLAC and encodes 128 kbps mono / 192 kbps stereo MP3.
  Master rate follows the float processing input; MP3 delivery is explicitly
  48 kHz. Reported processing/delivery rates do not imply bandwidth recovery.
  Original source/model rates must still be connected through executor provenance.
  Both exact files are decoded and checked for finite PCM, layout and timeline;
  both are measured against -1 dBTP. At most two codec corrections rebuild from
  float input. Short/silent loudness is null with a reason; no fake -99 LUFS.
  Dither policy is explicitly none. No additional compressor/mastering chain.
  The existing legacy mastering path remains until cutover; this is the one v2
  owner, not a completed replacement deployment.
- C15 evidence: FFmpeg `6.1.1-3ubuntu5`; 83 combined tests passed, plus one
  separate forced codec-overshoot test passed after its addition. Actual codec
  fixtures cover mono/stereo and 8/22.05/48/96 kHz float input, exact FLAC frames,
  silence/short clips, non-finite rejection, loudness-off transparency, and gain
  limits. Forced correction is injected; it proves the retry source path, not
  natural overshoot incidence or listening quality. Targeted lint/whitespace
  checks passed. Filter reference: https://ffmpeg.org/ffmpeg-filters.html#loudnorm.

- C01: proposed strict Python attempt/profile/runtime schema implemented in
  `hear/services/magic_clean/contracts.py`. Unknown fields and unresolved options
  fail validation. No backend agreement or canonical protobuf generation yet.
- C09: engine/session protocols and explicit certified-runtime registry introduced.
  No model is registered or advertised as certified. No transport entrypoint yet.
- C10: attempt scratch preflight, decimal GPU budget configuration, cancellation,
  monotonic deadline and process-group termination implemented. Diagnostic capture
  is bounded. Scratch polling detects overruns; it is not an OS disk quota or a
  complete aggregate host reservation system. GPU configuration does not itself
  impose a physical memory partition. `SourceInspector` now hashes and inspects
  attempt-local audio in bounded blocks, verifies pinned size/rate/channels/frames,
  rejects non-finite PCM, and reports per-channel peaks/RMS/correlation. Authorized
  network download, sandboxed decoding and aggregate reservations remain.
- C17: result manifest schema, manifest-last artifact writer, and bounded manifest
  verifier implemented. Artifact identities include remote versions, byte counts
  and SHA-256. Verification compares authenticated attempt/source/plan/prefix
  identities. The storage protocol requires atomic create-only writes and remote
  byte verification. Its production B2 adapter, grant refresh, failure-marker
  ingress delivery and backend reconciliation are still outstanding. Failure-marker
  writing and executor integration are now implemented (see evidence below).
  Legacy B2 upload is intentionally not used because it does not enforce
  create-only writes.
- C14: CPU `NoiseProfileEngine` and per-attempt sessions now implement bounded
  spectral noise-profile attenuation. The explicit selected interval supplies the
  profile, 3/6 dB presets bound attenuation, and one linked gain mask preserves
  dual-mono/anti-phase relationships. Overlap-add writes exact partial tails.
  No CUDA/model fallback is present. Production CPU reference-analysis evidence,
  real music/transient listening calibration and profile certification remain
  open; the injected analyser is exercised with test evidence only. No capability
  has been marked ready or deployed.
- Prior verification: 60 contract/runtime/artifact/inspection/architecture tests
  passed; targeted Ruff and whitespace checks passed. Storage failure tests use
  a test store, not live B2. Inspection tests use actual WAV codec fixtures at
  8/16/22.05/24/32/44.1/48/96 kHz, partial blocks, anti-phase stereo and invalid
  samples. These are inspection tests, not certified engine input coverage. No real
  model, audio-quality, A40 memory, long-duration or deployment certification run.
- Host inventory reports NVIDIA A40, 46068 MiB. This is inventory, not cleaner
  peak-memory evidence.
- CPU adapter verification: 10 additional synthetic audio tests passed, covering
  partial final windows, mono-correlated and anti-phase stereo, reduction bounds,
  contaminated/silent references and cancellation. These do not replace the real
  recordings and listening evidence required by C26.

## Deletion ledger and protected consumers

| Old surface | Observed consumers | Replacement / deletion gate |
| --- | --- | --- |
| MossFormer adapter | cleaner service, pipeline, streaming | DeepFilter/SAM invariant and real-model acceptance; then delete |
| Demucs stem adapter | cleaner service, pipeline, streaming | three explicit profile adapters; then delete |
| ContentMode / StemLevels | models, pipeline, dynamics, service, streaming, content_context | resolved CleanPlan; reconcile shared content analysis before deleting |
| Model paths and provisioning | config, main, model_provisioning, deployment environment | capability-specific pinned assets and offline startup tests |
| clearvoice / demucs packages | root dependency set and above adapters | remove after caller cutover; verify built image inventory |
| NoiseReducer | reconstruction/synthesizer | preserve reconstruction consumer and its regression tests |
| AI PostgreSQL | shared orchestrator/job and other workflow consumers | backend-owned attempts for every retained workflow before removal |

Old engine execution remains in production until the plan's replacement,
consumer migration, certification and drain requirements are satisfied. No old
model directories have been deleted. Unrelated documentation edits were preserved.

## Outstanding release evidence

### Real DF3 compatibility evidence (2026-09-21)

The isolated CPU probe now loads the actual pinned DF3 checkpoint with strict
state-dict validation and runs the production loader/enhance wrapper. The
original config failed strict configuration loading because two 0.5.6 defaults
were absent; `deploy/cleaner/deepfilter3.ini` now makes those values explicit.
The loader rejects ambient environment overrides before model construction and
before each inference call. Asset revisions, hashes and the repeatable opt-in
test command are in `deploy/cleaner/README.md`.

Seven real-checkpoint tests passed: mono/stereo exact-length finite outputs,
one-sample and hop-boundary tails, repeat-call state isolation, and silence.
Nineteen loader/adapter unit tests also passed; the wider Cleaner V2 and
architecture regression run passed all 110 tests. Targeted Ruff, formatting and
whitespace checks passed. The real test emitted one upstream
torchaudio metadata deprecation warning. No production dependencies, Ray workers,
capability registrations, or GPU allocations were changed by this CPU probe.
This advances C11 but does not certify resampling, long-form seams, speech
quality, or A40 memory. SAM and the other release gates below remain open.

C00 baseline fixtures and complete consumer/job inventory; C01 shared contract
approval and generated fixtures; C09 isolated image/factory/executor; C10 safe I/O
and aggregate resource admission; C11 DeepFilter; C12 SAM audio-only parity;
C13 bounded codec/separator/decode; C14 CPU reference processing; C15 mastering;
C16 content protection; C17 manifest-last storage; C18 authenticated transport;
C19 shared persistence migration; C25 actual memory/coexistence tests; C26 human
listening; C27 drain/removal/image audit; C28 canary/rollback. C29 remains future
provider work as specified by the plan. None of these gates may be inferred from
passing mock/unit tests.

### Natural resampling implementation (2026-09-21)

Added `AudioResampler` with explicit SWR parameters, finite-block validation,
cancellable FFmpeg execution, scratch preflight, exact frame-grid checks and
float RF64-capable output. Natural file sessions now prepare 48 kHz input and
restore original sample rate/frame count before the existing quality/sample/
mastering flow; already-48 kHz input retains the direct path. Per-attempt
temporary resampling files are cleaned on success/failure. The policy is included
in the long-form identity digest, requiring new runtime certification identities.

36 resampling/DeepFilter/architecture tests passed, including real FFmpeg
roundtrips at six rates from 8 through 96 kHz, one-sample/tiny/tail lengths,
channel isolation and impulse alignment within one sample. The expanded offline
real-model suite passed 9 tests, including 44.1 and 96 kHz file sessions through
the real DF3 checkpoint. One upstream torchaudio deprecation warning remains.
FFmpeg build and policy details are in `deploy/cleaner/README.md`.
The broader Cleaner V2/architecture run passed 134 tests. A subsequent added
resampled-model-failure cleanup test passed in the 12-test DeepFilter suite.
Targeted Ruff, formatting and whitespace checks passed.

This establishes structural rate/timing behavior, not listening or long-form
boundary acceptance. No deployment is certified or switched by these tests.

### Typed terminal failure publication (2026-09-21)

`ArtifactWriter.publish_failure` now creates a bounded, identity-bound terminal
manifest with a typed code, no raw exception diagnostics, no candidate artifacts,
and no claims of successful validation. Failure and success compete for the same
immutable terminal key; neither can overwrite the other. Terminal serialization
uses bounded in-memory bytes instead of a mutable scratch file. The schema rejects
inconsistent cancellation/error combinations and failure validation claims.

`CleanExecutor` rechecks deadline/cancellation and authorization before attempting
failure publication for eligible processing/integrity/resource failures. A
`PublishedExecutionError` preserves the typed code and carries the terminal bundle
reference for the future ingress. Failure-publication errors never replace the
original execution error. No cancellation flag is cleared and no deadline is
extended. Authorization/conflict errors and uncertain storage acceptance do not
trigger competing terminal uploads. Backend reconciliation must still handle
absent markers after cancellation, deadline, storage outage or process death.

Tests cover every error code, success/failure immutable conflicts, expired and
cancelled guards, schema incoherence, reauthorization, publication failure and
uncertain upload handling. These use the contract test store: actual remote
create-only storage, production transport delivery and backend reconciliation
are still open C17/C18 gates.
Verification: 65 targeted tests and 154 broader Cleaner V2/architecture tests
passed. Targeted Ruff, formatting and whitespace checks passed.

### Large PCM container correction (2026-09-21)

Natural's model-rate output, CPU noise-profile output and executor preview
cropping now explicitly write float RF64 instead of RIFF WAV. This removes the
32-bit RIFF data-length limitation from those bounded-output paths. Resampling
already uses FFmpeg RF64 auto-selection. Regression assertions inspect the actual
container produced by both engines and preview cropping; mastering still consumes
these `.wav`-named float files through the existing codec path.

A separate temporary sparse-file probe on this host wrote/read a stereo float
sample at frame 536,870,912 using libsndfile RF64: 536,870,913 total frames,
4,294,967,408 logical bytes, 8,192 allocated bytes, exact recovered values
`[0.125, -0.25]`. The temporary file was removed automatically. This proves
container/addressing support only; it is not a multi-hour inference, disk-budget,
memory, cancellation, or listening soak. Those release gates remain open.
Verification: 50 engine/executor/mastering/architecture tests passed, including
the actual RF64 container assertions. Targeted Ruff and whitespace checks passed.

### Bounded wanted-content warning intervals (2026-09-21)

The quality summary now carries `source_warning_intervals` with typed risk code,
start/end frames on the pinned source timeline, and minimum per-block RMS ratio.
Adjacent matching intervals merge. At most 128 intervals are retained, with
`warning_intervals_truncated` explicitly reporting omitted intervals. Warnings
are computed from the existing 32,768-frame signal comparisons, not a speech
detector. They locate coarse loss/no-target risks; all candidates still require
review and no speech-retention claim is made. Full-source coordinates also apply
to sample jobs; consumers must intersect them with the declared preview interval
when displaying preview-local positions.

The shared validation report/manifest already serializes this summary. Schema
checks reject interval codes missing from the summary, intervals past the pinned
source, and signal-risk evidence on unvalidated failures. Regression fixtures
cover separated and adjacent attenuation regions, exact partial tails, interval
caps/truncation and manifest bounds. Real VAD, speech-loss calibration, spectral
and musical-transient evidence, and frontend interval rendering remain open.
Verification: 69 quality/contract/artifact/executor/architecture tests passed;
targeted Ruff and whitespace checks passed.

### Per-profile capability snapshots and readiness admission (2026-09-21)

`EngineRegistry.capabilities()` now returns an internal versioned snapshot of all
three profiles. Unregistered engines are explicitly `not_certified`. Registered
engines expose pinned runtime/evidence identities, input-byte limits, channel
counts and maximum frames/duration for each certified input rate. A registered
engine is not ready without an explicit successful lightweight readiness probe.
Probe exceptions produce a generic unavailable reason without leaking local paths
or credentials. Capability reads never invoke model loaders.

Dispatch rechecks the same readiness probe before model allocation, so a stale
healthy snapshot is not accepted as readiness evidence. Probes are injected by
the eventual isolated runtime factory and must inspect local assets/dependencies
without model allocation/downloads. No production probe or certification is
registered here. This snapshot is not a concurrency reservation or a guarantee
that an allocation cannot fail. Canonical protobuf publication, authenticated
ingress, backend projection and real deployment probes remain open.
Verification: 33 capability/contract/executor/architecture tests passed, including
missing probes, per-profile independence and readiness revocation between snapshot
and dispatch. Targeted Ruff and whitespace checks passed.

### SAM global solver groundwork (2026-09-21)

No SAM Audio Small assets or installation were found in the inspected local
workspace. The upstream [SAM model implementation](https://github.com/facebookresearch/sam-audio/blob/main/sam_audio/model/model.py)
constructs optional vision/ranking modules and encodes/decodes whole input feature
tensors in `separate()`. It is not used as a production bounded long-form path.

`hear/runtime/cleaner/longform_sam.py` implements a disk-backed global midpoint
solver stage with a bounded vector-field interface. All windows at one solver
evaluation read a common state; weighted overlapping predictions are accumulated
and normalized before computing midpoint/final updates. State, midpoint,
derivative and weights use memory-mapped files. Scratch is reserved before
allocation; cancellation/deadline checks occur between bounded operations.
Inputs/derivatives/updates require finite float32 values and exact shapes.
Temporary files and failed partial output are removed; the input noise remains
unchanged. Solver settings, grid origin, blend and precision have a policy digest.

Eleven solver/architecture tests passed, including analytic midpoint behavior,
per-window shared-state consistency, multiple overlaps, partial tails, invalid
derivatives, cancellation and pre-allocation budget rejection. Targeted Ruff and
whitespace checks passed. These use analytic vector fields, NOT the SAM model.

This module is not registered as a runnable engine. The pinned audio-only loader,
real conditioning/noise policy, bounded codec encode/decode, window positional
semantics, real-model parity, and listening tests remain to be implemented.
Memory mapping bounds array operations but is NOT evidence of bounded resident
host memory: mapped-page residency/eviction and RSS need measurement/enforcement.
No CUDA execution, A40 measurement or production readiness is claimed.

### Solver mapped-page residency control (2026-09-21)

The solver now flushes dirty mappings and calls `MADV_DONTNEED` after bounded
initialization, derivative accumulation/normalization, and update blocks.
Read-only noise mappings are evicted without writes; duplicate state/output
aliases are flushed/evicted only once. Platforms without `madvise` fail before
allocation. Page-cache/cgroup usage is not a physical memory partition and still
needs deployment-level accounting. Eviction failure cleans partial output.

Added `python -m scripts.benchmark_cleaner_solver --frames N`, a CPU analytic
zero-field probe with bounded input generation and exact output validation.
It reports Linux `/proc/self/status` VmHWM, not launcher-inherited `ru_maxrss`.
Discarded initial getrusage measurements because their pre-solver baselines were
inconsistent across processes. Corrected independent runs (8 latent channels,
8192-frame windows, 2048 overlap, 2 midpoint steps) measured:

| Latent frames | Latent bytes | Baseline peak RSS | Final peak RSS | Elapsed |
| --- | --- | --- | --- | --- |
| 131072 | 4194304 | 48955392 | 49078272 | 0.451 s |
| 1048576 | 33554432 | 48807936 | 49573888 | 3.096 s |
| 4194304 | 134217728 | 48992256 | 50356224 | 12.426 s |

Thirteen solver/architecture tests passed, including flush-before-eviction,
alias handling, partial cleanup, and prior solver arithmetic tests. Targeted
Ruff/whitespace checks passed. These measurements concern this analytic solver
only, not real SAM conditioning, codec, GPU use, or the complete worker image.

### Real DF3 hour-long CPU soak (2026-09-21)

Added `scripts/benchmark_cleaner_deepfilter.py` and ran 3600 seconds plus a
17-frame tail through the real pinned DF3 checkpoint, stereo 48 kHz, CPU float32,
one Torch thread, 10-second blocks and 1-second contextual margins. The first
harness invocation failed plan validation before inference because it disabled
the required comparison setting; the corrected harness passed.

The complete output scan verified 172,800,017 frames, stereo layout, 48 kHz,
RF64, finite PCM and nonzero partial tail. Inference took 221.658 seconds
(RTF 0.06157); total probe time was 233.228 seconds. Maximum model input was
576,000 frames across 361 calls. Peak process RSS was 991,264,768 bytes; it rose
from roughly 942 MB early in inference, so repeated-job leak testing remains
necessary. Output logical size was 1,382,400,240 bytes. Temporary input/output
audio was automatically removed after validation.

Exact results, code/checkpoint/policy hashes and limitations are recorded in
`deploy/cleaner/evidence/df3-cpu-synthetic-hour-2026-09-21.json`. This is structural
CPU engine evidence, not real-speech listening, A40 memory, full-pipeline soak or
production-image certification. No readiness/certified-duration setting changed.
The broader Cleaner V2/architecture suite also passed all 171 tests. Benchmark
Ruff, formatting and whitespace checks passed.

### Version-pinned source staging and executor integration (2026-09-21)

Added `S3SourceStager` on the verified bounded-read path. It requests the ticket's
exact object key/version, enforces input/scratch limits before network I/O,
hashes streamed bytes, and publishes an attempt-local source only after size and
SHA-256 verification. Download scratch is private and temporary; a same-filesystem
create-only hard link prevents destination replacement. Corrupt/truncated/extra
bytes and wrong versions leave no visible source or temporary file. Source MIME
metadata is not treated as proof of codec validity; `SourceInspector` still checks
the complete staged audio afterward.

`CleanExecutor.execute(..., stager=...)` now stages after authorization, exact-plan
checking and worker-attempt admission, then follows the existing inspection,
engine, quality, mastering and manifest path. Unauthorized attempts never fetch
the source. Existing already-staged local execution remains available for trusted
ingress. Tests exercise >1 MiB downloads, rejection/cleanup and an executor flow
with test storage/engine plus real audio inspection/mastering.

Production scoped-client construction, endpoint/bucket allowlisting, grant
refresh, authenticated transport and live B2 evidence remain open. This path
does not fetch arbitrary URLs, certify untrusted decoder isolation, or resolve
the still-unverified B2 create-only upload guarantee.
Verification: 36 source-staging/remote-verification/executor/architecture tests
passed; targeted Ruff, formatting and whitespace checks passed.

### Initial Cleaner v2 protobuf envelope (2026-09-21)

Added separate `hear.cleaner.v2` protobuf definitions for the typed attempt,
source/runtime/plan, scoped grants and compact terminal reference, plus initial
execution RPC. Existing Pipeline fields/services were not changed. Python message,
typing and gRPC bindings were generated with locked `grpcio-tools==1.75.1` in an
isolated package target because the installed host compiler was 1.82.1. A pinned
regeneration script reproduced identical hashes for all three generated files.

`CleanerWireCodec` bounds requests before parsing, rejects unknown fields at any
defined nesting level, preserves explicit optional-scalar presence and uint64
precision, and runs the existing strict semantic validators. Required absent
booleans/seeds do not become defaults. Grants remain separate/redacted after
conversion; malformed-grant errors suppress raw validation context. Grant expiry
now requires a timezone. Raw protobuf bytes still contain secrets and must not
be logged or sent to the browser.

31 wire/contract/architecture tests passed; targeted Ruff/formatting/whitespace
checks passed. `hear/proto/CLEANER_V2.md` documents the wire rules and regeneration.
No v2 handler was registered. Backend codegen/approval, capability negotiation,
authenticated ingress, progress/cancel/reconciliation and worker lifecycle remain
open C01/C18 gates; this envelope is not a deployed or complete transport.
The complete current Cleaner V2/architecture regression run passed 220 tests
with 9 opt-in real-checkpoint tests skipped. Those skips do not replace the
separately recorded real-checkpoint smoke and hour-long CPU soak evidence.

### DeepFilter asset/provenance binding (2026-09-21)

Found a provenance gap between `DeepFilterEngine`'s supplied identity and the
actual pinned loader configuration. `PinnedDeepFilterFactory.identity()` now
derives the adapter runtime digest from config/checkpoint hashes, canonical
package pins, device and loader policy, plus an explicit precision digest.
The engine validates the binding at construction and again before loading.
Mismatched runtime/checkpoint/precision identities fail before allocation.
Descriptor package ordering does not alter the digest; relevant changes do.

Inference explicitly disables caller autocast and loader startup disables TF32
for the dedicated worker. The benchmark now records the asset-bound identity and
a separate adapter-code hash. This descriptor is not a full image/source digest:
immutable build provenance and image certification still need independent evidence.
Old certification identities must not be silently reused. The earlier hour-long
soak remains historical evidence for its recorded code/policy, not certification
of this changed loader or new identity format.

31 loader/adapter/architecture tests passed. The expanded real-model suite passed
10 tests, including exact output equality inside/outside an outer CPU BF16
autocast context. One upstream torchaudio deprecation warning remains. Targeted
Ruff, formatting and whitespace checks passed. Production service was not changed.

### Typed subprocess launch failures (2026-09-21)

`CancellableProcessRunner` previously launched children outside its typed error
handling. Missing/non-executable binaries, exhausted process/file-descriptor
limits and other spawn errors could escape as raw OS errors. Launch failures now
map to `engine_unavailable`, `resource_exhausted`, or `process_failed`, with no
executable/workspace paths or raw exception context in the public error.
Tests cover eight OS error classes plus a real missing executable. Existing
process-group deadline/cancellation, mastering and executor tests remain in scope.
Verification: 56 runtime/mastering/executor/architecture tests passed; targeted
Ruff, formatting and whitespace checks passed.

### In-process attempt admission (2026-09-21)

The process lease alone did not serialize concurrent requests inside one worker:
different engine loaders could otherwise overlap before their engine-local
session locks. `WorkerLease.attempt` now adds a shared nonblocking attempt slot,
held by `CleanExecutor` across inspection, model loading, processing, validation
and publication. Admission rejection happens before execution/failure-publication
handling, so queue contention does not create a terminal failure artifact.

Attempt exceptions release only the request slot; the process/model-cache lease
remains held. Worker close acquires the same slot and refuses active work, avoiding
release while an attempt can still allocate models. Ownership is validated under
the admission lock. Tests cover contention, error cleanup, shutdown refusal,
cross-thread requests and executor rejection before inspection. External queue
handling, actual worker lifecycle integration and aggregate GPU budgets remain
separate uncompleted gates.
Verification: 25 worker-lease/executor/architecture tests passed; targeted Ruff,
formatting and whitespace checks passed.

### Version-pinned remote bundle verification (2026-09-21)

Reviewed Backblaze's current [Put Object reference](https://www.backblaze.com/apidocs/s3-put-object):
the published header list does not establish a conditional create-only guarantee.
This absence is not proof of universal lack of support, but it is insufficient
to enable a v2 immutable upload adapter. No live write probe, bucket modification,
or provider substitution was made. B2 create-only publication remains gated on
authoritative support and actual concurrent-write validation, or an approved
backend-coordinated immutable publication design.

Implemented `S3BundleVerifier` for the independent read/reconciliation path.
Backblaze documents version-pinned [Get Object](https://www.backblaze.com/apidocs/s3-get-object).
The verifier explicitly supplies `VersionId` for the manifest and every artifact,
checks returned version/length/content type, streams SHA-256 over actual bytes,
and closes response bodies. It never relies on HEAD metadata or ETag as content
verification. Manifest collection is capped; large audio objects are not collected
in RAM. Partial/range and unfinished Live Read responses are rejected.

Recovery uses a separately authorized read deadline; it does not reuse an expired
execution deadline. The manifest's completion timestamp must still satisfy its
original ticket. Ingress must authenticate/revalidate fence and grant scope, and
provide an allowlisted client with bounded network timeouts. No new HTTP endpoint,
production credential client, write implementation or backend commit is implied.
Tests use deterministic client responses, not live B2, and cover remote byte
corruption, version/metadata mismatch, cancellation, response closure and delayed
reconciliation after an on-time completion.
Verification: 51 remote-verification/artifact/architecture tests passed; targeted
Ruff, formatting and whitespace checks passed.

### Cross-process cleaner worker ownership (2026-09-21)

Added Linux `WorkerLease` with separate GPU/CPU lanes, exclusive nonblocking
file locks, close-on-exec descriptors and stable lock inodes. Ownership lasts for
the worker/model-cache lifetime, not a single job. `CleanExecutor` now requires
that ownership and checks the matching lane before work and model loading.
Closed, foreign-process, wrong-lane and replaced-inode ownership is rejected.
Lock symlinks/nonregular files are refused and lock files are never unlinked on
release. Process exit releases ownership automatically.

Twenty lease/executor/architecture tests passed, including actual subprocess
contention and termination/reacquisition, CPU/GPU lane independence, symlink
rejection and executor rejection before processing. Targeted Ruff/whitespace
checks passed. Deployment mount/lifetime requirements are in the cleaner README.
The production factory/launcher still needs to acquire this lease, and legacy
actors do not participate. This is not aggregate admission for other AI workloads
or a hardware memory partition; C10/C25/deployment-drain gates remain open.

### Ticket deadline propagation into inner work (2026-09-21)

Found and corrected a deadline gap: executor stage checks used the authenticated
ticket deadline, while bounded inner scans and child processes used only the
locally configured monotonic timeout. `ResourceGuard.bind_deadline` now tightens
that timeout and retains the wall deadline. `ExecutionContext.check` and both
artifact-publication paths bind it before work. A later binding cannot extend
the original budget; forward wall-clock jumps expire work, while backward jumps
cannot grant extra monotonic runtime. Non-finite worker deadlines and timezone-
naive ticket deadlines are rejected.

An executor regression runs a sleeping child with a 30-second workload under a
0.3-second ticket and a longer local timeout, then verifies typed deadline
failure, process termination, session closure and no publication. Clock-jump
and publication tests cover the corresponding guard behavior. This does not
make an individual blocking model/native call interruptible: dedicated worker
supervision and bounded native-call certification remain required.
Verification: 67 runtime/executor/artifact/architecture tests passed; targeted
Ruff, formatting and whitespace checks passed.
