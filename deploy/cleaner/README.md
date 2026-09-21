# Cleaner runtime assets (not release-certified)

The package allowlist now includes an opt-in cleaner-v2 gRPC handler (51 files).
It requires injected verified authentication/dispatch and explicit bounded TLS
server configuration; it is not registered on the live service. See
`hear/proto/CLEANER_V2.md` for remaining integration requirements.

## Offline SAM conditioning assets

`SamPromptCache.from_files(((identity, path), ...))` loads precomputed conditioning
without T5, network access or pickle. Provision each file as exactly
`tokens * 768` little-endian float32 embedding values in token-major order,
followed by `tokens` mask bytes (each exactly 0 or 1). The implicit tensor shapes
are `[1, tokens, 768]` and `[1, tokens]`; there is no header or embedded metadata.
Each file is at most 1,573,376 bytes (512 tokens), with at most 16 cache entries.
Loading rejects symlinks, nonregular files, wrong sizes, invalid masks and checksum
mismatches; the resulting cache snapshots remain valid after source files disappear.

Supply `SamPromptIdentity` from trusted deployment configuration. Its embedding
and mask digests cover those exact byte sections. Its model identity must bind
the encoder, tokenizer and encoding runtime; its prompt and precision identities
must match the approved catalogue. Do not derive expected checksums from an
untrusted input file, request or adjacent unverified manifest. Provision files
offline from the approved encoder output and publish assets immutably.

This implements artifact admission, not prompt approval or encoder provisioning.
The engineering `speech` fixture is not an approved all-wanted-speech preset.

## Runtime evidence

Latest broad regression: 697 tests passed, with 15 default opt-in skips subsequently
covered by explicit SAM/Silero and real-DF3 asset runs. Lint and the 83-file formatting
sweep passed. See the delivery ledger for commands and scope; these checks do not
constitute release certification.

The executor now checks SAM's padded model-frame allowance and a pinned decoder
scratch lower bound before registry/model loading. Existing occupied bytes count
against the reservation. This catches impossible budgets early; it does not replace
runtime allocation checks or establish complete peak scratch requirements.

A real 576,001-frame (~12s) CUDA fixture now crosses the 250-latent-frame solver
window and completes with finite, exact-length output and cleanup checks. Torch
peaks were 3.39 GB allocated / 3.58 GB reserved with codec tile 4096. It required
codec-padding frame allowance and a 2GB scratch budget; there is no independent
reference or quality verdict for this fixture. See
`evidence/sam-cuda-window-crossing-2026-09-21.json`.

The retained 8,519,680-byte allocation has now been attributed to cuBLAS workspace
in a controlled diagnostic: after three sessions, GC/cache clearing left allocation
unchanged, while an explicit private cuBLAS-workspace clear reduced it to zero.
See `evidence/sam-cublas-retention-2026-09-21.json`. Serving does not use this private
hook; account for library workspace in the memory budget. Long-form and aggregate
memory certification remain pending.

Three consecutive CUDA sessions now produced identical output hashes with stable
post-close allocation (8,519,680 bytes), without forced collection/cache clearing
between attempts. Final GC did not reduce allocation; empty-cache reduced reserved
memory only. Allocation ownership remains unproven. See
`evidence/sam-installed-cuda-repeat-2026-09-21.json`; this is not long-run leak or
release certification.

The CUDA installed wheel has now completed real A40 short-fixture inference and
CPU comparison at unchanged tolerance (maximum error about 1.05e-8). Torch peaks
were 2.89 GB allocated / 3.04 GB reserved; 8.52 MB remained allocated after close
and needs characterization. See `evidence/sam-installed-cuda-short-2026-09-21.json`.
This is not aggregate NVML, repeated-job, long-form, quality or release certification.

An explicit `PinnedSamFactory(..., device="cuda:0")` path now stages on CPU and
caps the CUDA allocator before transferring the required audio core and codec.
It requires one visible GPU and rejects conflicting FP32/backend settings; CPU
remains the default. Its separate CUDA identity is not registered as ready.
51 targeted tests passed, including mocked transfer ordering and rejection cases.
Real CUDA inference, parity and aggregate A40 memory certification remain pending.

Current source uses SAM loader policy v3, adding typed native-fault quarantine and
worker-restart requirements. 54 engine/factory/executor/architecture tests passed,
including injected host/CUDA OOM and cancellation. No GPU readiness is implied;
the v2 installed-wheel evidence below predates this fault-policy change.

The current autocast guard and raw-conditioning loader passed real CPU inference
from the corrected installed wheel, with an exact known-answer output and source/
cleanup checks. See `evidence/sam-installed-raw-conditioning-2026-09-21.json`.
57 targeted tests and the 37-package dependency check also passed. This does not
establish approved prompt semantics, GPU memory limits or release readiness.

Current source also rejects active thread-local CPU autocast before loading and
processing, enforcing the existing FP32 policy without changing caller settings.
The installed-wheel evidence below predates this admission-only guard and retains
its original artifact/source hashes.

Policy v2 has now passed real factory/session inference from the audited installed
wheel using isolated Python outside the repository. The target samples matched the
existing engineering known answer exactly; source and cleanup checks passed with
CUDA uninitialized. See `evidence/sam-installed-v2-inference-2026-09-21.json`.
This is CPU short-fixture package validation, not GPU/quality/release certification.

SAM CPU loader policy v2 explicitly checks the declared Torch backend/default dtype
and device settings. Sessions revalidate before inference and never silently change
process-global flags. Earlier v1 wheel/model evidence retains its original hashes;
it is not automatically certification of this updated runtime identity.

The `sam` extra pins Torch and einops for the precomputed-conditioning runtime;
it does not install T5 or optional rankers. A fresh frozen SAM-only environment
passed dependency checks, isolated installed-wheel imports and real-source meta
construction. See `evidence/sam-isolated-wheel-2026-09-21.json` for the audited
50-file wheel and source hashes. This is not container or inference certification.

`PinnedSamFactory` now assembles the source-verified meta builders, strict checkpoint
loader, pinned CPU conditioning cache and file pipeline. Its runtime identity binds
the complete loading/processing policy. It currently supports only an explicitly
identified CPU validation runtime, with no default readiness/serving registration.
Factory lifecycle tests and real CPU factory/engine-session inference pass; see
`evidence/sam-factory-real-cpu-2026-09-21.json`. The short synthetic target matched
the direct pipeline byte-for-byte. Required GPU/device admission, approved prompts,
clean-image dependency alignment and certification remain pending.

`SamCodecBuilder` constructs the original encoder, VAE bottleneck and retained
watermark decoder directly on meta from checksum-verified source. It avoids the
temporary vector quantizer and upstream package initialization. A lifetime lease
owns canonical DACVAE module names; foreign imports fail closed. Real-source tests
confirm 317 meta checkpoint tensors and unchanged RNG. A subsequent real-checkpoint
CPU Ray probe preserved short-fixture waveform parity through this constructor;
see `evidence/sam-meta-codec-real-cpu-2026-09-21.json`. Production factory wiring,
peak-memory measurement and quality/long-form certification remain pending.

`SamCoreBuilder` verifies six fixed upstream source hashes and the Small config
before constructing only the audio core. Learned tensors start on the meta device;
nonpersistent frequency buffers are recreated on CPU. Its private module namespace
is explicitly owned and cleaned. Real-source construction tests and a subsequent
real-checkpoint CPU inference probe passed; see
`evidence/sam-meta-core-real-cpu-2026-09-21.json`. Full production factory integration,
and memory/quality certification are still pending.

`SamCheckpointLoader` verifies the offline file digest and exact core/codec/optional
key inventory before strict CPU FP32 assignment. The optional inventory must be
an externally pinned list of vision keys, not a list inferred from the incoming
checkpoint. Immutable trusted assets and separately verified module construction
remain factory requirements; the complete production loader is still unfinished.

The exact 601-key exclusion inventory is pinned in `sam-small-optional-keys.json`.
A real CPU Ray probe admitted 247 core/317 codec keys through the strict loader
and preserved the seeded file-pipeline parity result; see
`evidence/sam-strict-checkpoint-real-cpu-2026-09-21.json`. This diagnostic reload
does not certify production module construction or cold-load memory peaks.

`SamEngine`/`SamSession` connect the pinned-plan pipeline to the common executor
protocol. They require an explicitly supplied factory, serialize sessions, bind
plan/resource ownership, and quarantine the engine after unload failure. The
CPU factory is implemented and smoke-tested but not registered as a certified
capability. These tests do not establish GPU safety or release readiness.

`AudioResampler.convert` stages and validates output privately, then creates the
destination using an exclusive hard link. Existing or racing destinations are
preserved at this boundary; unsupported hard-link publication fails closed.
Callers must also preserve ownership when cleaning up their own failures.
`DeepFilterSession.process` now applies the same private-staging/create-only
publication rule on both native-rate and resampled paths, so its cleanup no longer
undoes the resampler's competing-file protection.

`SamSeparationPipeline.separate_plan` binds a resolved Voice Focus plan to the
expected runtime and pinned prompt-cache entry, then uses its seed for file
processing. Supply expected identities from trusted loader configuration, never
from the request itself. Only the implemented mono policy is accepted; dual-mono
preflight is still pending. This layer does not replace registry admission or the
production loader/session adapter.

The plan path now prepares mono PCM at 8–96 kHz for the 48 kHz model and restores
the original sample rate and exact frame count using `AudioResampler.POLICY`.
48 kHz input bypasses conversion. Real FFmpeg/fixture-inference tests cover eight
rates and failure cleanup; these are not speech-quality certifications. Production
runtime provenance must bind the resampling policy before serving admission.

`SamPromptCache` holds at most 16 CPU FP32 conditioning snapshots, each bounded to
512 tokens. `SamPromptIdentity` binds prompt, encoder/tokenizer/runtime model,
precision, embedding bytes, mask bytes and token count. Construction verifies
tensor bytes against externally pinned identities; lookup requires an exact
identity and returns independent clones. There is no prompt selection, model
download or implicit encoding. Provisioning must supply approved prompt identities
and verified encoder outputs; the cache is not itself approval or serving wiring.

`SamSeparationPipeline.separate_file(source, destination, seed=..., text=...,
text_mask=..., guard=...)` connects prepared PCM import, seeded global noise,
independent watermark messages, separation and target-only RF64 export. It returns
the noise invocation identity and removes owned intermediate files. The caller
must supply pinned conditioning; this is not an approved prompt cache or a serving
engine registration. Lifecycle tests use a fixture separator with real PCM/noise
I/O; see `evidence/sam-file-pipeline-2026-09-21.json`.

A subsequent real-model CPU Ray probe of 3,841 synthetic samples matched the
feature-file target exactly and the independently assembled native reference
within unchanged tolerance (maximum target difference 5.8498699218034744e-9).
See `evidence/sam-file-pipeline-real-cpu-2026-09-21.json`. This short seeded fixture
does not establish speech quality, long-form behavior or deployment readiness.

`SamPCM` imports prepared mono 48 kHz WAV/RF64 PCM into feature files using
bounded reads, and exports an explicitly selected target or residual waveform
as mono RF64 float32. It never treats the two outputs as stereo, downmixes input
or resamples implicitly. Existing destinations are preserved; failures remove
only newly owned outputs. Scratch reservations include existing workspace files.
Stereo preflight/channel policy, source resampling and engine orchestration still
need integration; this adapter alone does not register a serving capability.

`SamSeparationPipeline` now connects bounded mean encoding, frozen conditioning,
the 16-step global solver and joint watermark-preserving decoding. It borrows
prepared 48 kHz mono feature input and explicit initial noise, returning a
target/residual waveform batch after exact-length trimming. Owned intermediates
are retired on success/failure; it is not registered as a certified serving engine.

The first full connected CPU fixture (3,841 sine samples, real core/codec/T5
weights, explicit noise/messages) matched an independently assembled original
reference within unchanged tolerance, maximum difference 1.210719347000122e-8.
See `evidence/sam-separation-pipeline-cpu-2026-09-21.json`. This approximately
80 ms fixture does not establish speech retention, no-target/hallucination safety,
long-form quality, GPU memory or deployment readiness, and does not waive the
earlier feature-conditioning parity failures.

`SamNoise` can now produce the explicit latent-noise file and independent
target/residual watermark messages from a plan seed. It pins NumPy 1.26.4, PCG64,
stream domains and float32 layout, with a policy digest and seed/frame identity.
Five tile sizes produced identical bytes and fixed known-answer digests are tested.
See `evidence/sam-noise-reproducibility-2026-09-21.json`. This is not Torch.randn
bit identity for the same seed, cryptographic randomness, automatic serving wiring
or quality certification; pass the generated explicit inputs to reference runs.

Frame/tile counts require positive integers. RNG-state, watermark and partial-noise
allocation failures report `RESOURCE_EXHAUSTED`; partial owned noise files are
removed. Identity calculation does not allocate an RNG. The fault-injection tests
and unchanged known-answer checks are recorded in
`evidence/sam-noise-resource-failures-2026-09-21.json`; this is not physical
memory-exhaustion certification.

`AudioOnlySamForward` now supplies a bounded FP32 conditioning/DiT forward path
using borrowed core modules and real text embeddings. It duplicates mean audio
features into joint target/residual conditioning and preserves no-video alignment:
zero video features still pass through the learned alignment bias, normalization
and gate. It does not omit that layer or replace T5 with zero features.

A CPU-only Ray probe strictly loaded all 247 non-codec/non-vision checkpoint keys
and matched the original forward-method body exactly at 2/7/13 latent frames and
times 0/0.5/1. Real offline T5 conditioning was used; optional vision/ranking modules
were not imported and CUDA remained uninitialized. The diagnostic extracts
unchanged definitions from the pinned source to bypass eager package constructors;
this is not production loading or whole-separation parity. See
`evidence/sam-audio-only-core-cpu-2026-09-21.json` and
`scripts/benchmark_cleaner_sam_core.py`. Core/codec/solver integration and GPU,
long-form, prompt and quality certification remain unfinished.

`SamConditionedField` binds disk-backed mean features and frozen text/mask snapshots
to the global solver. Global window offsets are preserved and solver arrays are
copied so backend calls cannot mutate shared solver state. A real CPU seven-frame,
two-step overlapping-window probe matched an independent original-forward midpoint
reference exactly across 12 evaluations, including its one-frame tail. See
`evidence/sam-conditioned-solver-cpu-2026-09-21.json`. This does not certify the
production 16-step policy or complete codec-to-separator audio execution.
The subsequent `--steps 16` run also matched exactly across 96 real field
evaluations on that seven-frame fixture; see
`evidence/sam-conditioned-solver-16step-cpu-2026-09-21.json`. Sixteen steps are now
the probe default (`--steps 2` reproduces the smoke-step scope). This confirms the
step count, not production window sizes, long-duration input or complete separation.

`SamCodecGraph.decode_joint` now preserves the original joint-latent handoff:
[1,256,T] becomes [2,128,T], target first and residual second, then both streams
receive full watermark decoding and exact-length trimming. The returned batch is
not stereo audio; neither stream is computed by subtracting the other. A real
CPU 3,839-frame paired-output fixture with distinct watermark messages matched
the original route within tolerance, maximum difference 8.940696716308594e-8.
See `evidence/sam-joint-decoder-cpu-2026-09-21.json`; this is not full separation
or long-form/GPU certification.

SAM Small snapshot `20b65f56888142eebe7c37448c6f6b3b32600e9b` is now downloaded
under `models/sam-audio-small/<revision>/` (gitignored). The checkpoint SHA-256 is
`8c44fda9821fd9f2ec8977304e3c0f55290d9eacb6bbf25b4b8fb1f69c2a8c06`, verified
against upstream LFS metadata. See `evidence/sam-small-download-2026-09-21.json`
for all file hashes and the CPU-only tensor inventory. Earlier access-denied
reports are historical. This does not supply a working/certified Voice Focus
adapter: production asset wiring, runtime pinning, strict optional-module
omissions, audio-only parity, bounded codec execution and A40 gates remain open.
Do not load the entire multimodal checkpoint onto CUDA to test availability.

Required T5 assets are now available at
`models/t5-base/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1/`, with safetensors SHA-256
`a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4`.
The real encoder/tokenizer passed an offline CPU FP32 repeatability probe;
`evidence/sam-t5-offline-cpu-2026-09-21.json` records its exact versions and hashes.
SAM source was inspected at `bb4c6999d2677c7402360e426afc01ddfad6dce0`.
Its eager multimodal constructor and unpinned optional dependencies must not be
used as production cleaner wiring. The `speech` probe is an engineering fixture,
not an approved prompt or a claim of all-speaker preservation. Required assets
being present does not make the audio-only adapter or its embedding cache complete.

DACVAE source is pinned for the compatibility probe at
`414c20785fc3a28373073ea8ef7a1316eeeaca6e`. Real CPU construction and strict loading
of all 317 SAM codec keys passed, as did exact/partial-tail mean-latent encode/decode
probes. See `evidence/sam-codec-cpu-2026-09-21.json`. Keep the decoder's watermarking
intact: it uses random messages and recurrent LSTM layers, so deterministic and
bounded long-form decoding must preserve message/RNG and recurrent state. The
first unseeded repeatability probe failed; controlled same-seed probes passed.
These short CPU cases do not certify independent window decoding or GPU memory.

`SamLSTMStream` supplies one bounded recurrent primitive for the eventual decoder.
Each instance owns one layer's hidden/cell state, accepts only contiguous feature
timesteps (not PCM offsets), preserves the upstream residual connection and drops
state on close or an in-call failure/cancellation. Bidirectional/projected/dropout
or training configurations are rejected. Caller autocast is disabled for the
FP32 path. It does not own model weights or replace watermark generation.
Both real checkpoint LSTMs matched full-sequence CPU inference exactly on the
tested 103-step, batch-two feature fixture with chunks of 8, 17 and 64 steps.
Each held 4,096 FP32 state elements (16 KiB), independent of sequence duration.
See `evidence/sam-recurrent-cpu-2026-09-21.json`. Convolution halos, alignment,
message/RNG state, full decoder parity and GPU execution still need implementation
and validation; do not treat this primitive as a complete long-form codec.

`SamConvolutionGeometry` now plans exact input support/output crops for the
pinned codec's regular and transposed convolutions, preserving stride phase,
dilation, dynamic padding and output unpadding. It keeps the original layer
forward call and weight-normalization hooks. Auto-padding tiles preserve the
full input-length residue; applying the wrapper to arbitrary slices would not.
All 90 real codec convolution layers passed 180 CPU FP32 tile comparisons at
31/32 input steps with 13-step output tiles. The largest absolute difference was
4.172325134277344e-7 (tolerance atol 1e-6, rtol 1e-5). Details are recorded in
`evidence/sam-convolution-cpu-2026-09-21.json`. This integer-only planner is not
the intermediate-feature disk runner or the complete streaming codec graph.

`SamFeatureFile` and `SamFeatureRunner` now provide frame-major float32 scratch
files and bounded convolution tiles. Outputs are exclusively created, writes must
be contiguous, and in-process incomplete outputs cannot be read. Scratch and
input/output halo tile limits are checked, writable mappings are flushed before
page eviction, and failed operations remove only their newly created output.
These limits do not certify native activation/workspace or physical GPU memory.
Complete codec admission and long-form validation remain unfinished.
The real CPU-only Ray probe covered 90 layers / 180 cases: disk output exactly
matched equivalent contiguous in-memory tiles and persisted reads. One case
failed the unchanged full-sequence tolerance (atol 1e-6 / rtol 1e-5), so codec
parity is not approved. See `evidence/sam-feature-disk-cpu-2026-09-21.json` and
`scripts/benchmark_cleaner_sam_features.py` for the diagnostic and its limitations.

The feature runner also executes each residual watermark LSTM over contiguous
disk-backed tiles, owns a fresh stream per invocation and clears hidden/cell state
on success or failure. Both real checkpoint LSTMs matched full-sequence CPU FP32
output exactly for 103-frame, batch-two features with 8/17/64-frame tiles. This
primitive does not itself coordinate the graph or watermark message.
See `evidence/sam-feature-recurrent-disk-cpu-2026-09-21.json`.

Bounded activation passes now retain the original Snake1d/ELU/Tanh/Identity
forward calls; other module types cannot be passed as pointwise activations.
Residual passes preserve `branch + shortcut`, including even centered shortcut
cropping, and reject incompatible lengths/channels rather than broadcasting or
inventing alignment. Both paths preserve inputs and remove newly created outputs
on failure. Graph composition is implemented separately below.
All 89 real activation modules passed the CPU-only disk-tile probe with maximum
absolute difference 5.960464477539063e-8; the convolution tolerance failure remains.
See `evidence/sam-feature-activation-disk-cpu-2026-09-21.json` for exact scope.

`SamCodecGraph` composes inspected encoder, residual, decoder and residual-LSTM
blocks from the bounded operations. Decoder blocks preserve upstream alternating
chunk selection rather than executing every entry of their ModuleList. Each run
borrows its input, returns one owned result and removes intermediate files as they
become unnecessary or on failure. Unsupported blocks fail closed. Certified codec
admission and complete separator integration remain unfinished.
The first real block-composition probe has three full-sequence tolerance failures
(encoder and first two decoder blocks), with maximum encoder difference about
1.47e-5. This is unaccepted numerical parity, not a certified codec. Exact cases,
source hashes and limitations are in `evidence/sam-codec-graph-cpu-2026-09-21.json`.
Further CPU diagnostics found exact full/tiled FP64 agreement for the isolated
failing convolution. Disabling MKLDNN fixed the tested convolution/block
comparisons but introduced three recurrent tolerance failures. It is not an
accepted global fix, and production flags remain unchanged. See
`evidence/sam-cpu-rounding-diagnostic-2026-09-21.json`. Successful exit from the
diagnostic script means measurements completed; inspect all per-case tolerance
statuses and failure lists before drawing any parity conclusion.

`SamCodecGraph.decode` now retains the full watermark route: original pre/post
layers, reverse upsample groups, recurrent state, one frozen explicit binary
message, forward downsample groups, and `base + 0.25 * watermark`. It does not
temporarily replace shared model modules or generate a new message per tile.
One real-checkpoint CPU fixture decoded [1,1024,2] features to [1,1,3840] samples
and matched the original decoder at unchanged atol 1e-6 / rtol 1e-5, with maximum
absolute difference 7.078051567077637e-8. Intermediate cleanup passed. See
`evidence/sam-watermark-decoder-cpu-2026-09-21.json`. This is not source-audio
roundtrip, full SAM parity, long-form or GPU certification; prior feature-level
tolerance failures remain open.
Five additional real-checkpoint fault cases verified frozen caller-message
ownership, message-stage cancellation, early/late decoder cancellation and
nonbinary-message rejection. Borrowed inputs stayed byte-identical and no partial
outputs remained. See `evidence/sam-watermark-faults-cpu-2026-09-21.json`; this is
cooperative CPU cancellation evidence, not host-crash or GPU certification.

`encode_mean` and `decode_latents` now connect bounded right-reflect padding,
encoder/in-projection, first-half mean extraction, out-projection, retained
watermark decoding and explicit original-length trimming. The posterior is never
sampled. Inputs too short for the upstream reflect rule fail explicitly; no
unvalidated padding substitute is used. Feature inputs must represent prepared
48 kHz mono audio and the scratch/frame budget must include padding.

Real CPU exact-hop and partial-tail roundtrips (3,840 / 3,841 source frames)
returned the exact requested sample counts and passed waveform tolerance with
maximum differences 1.7695128917694092e-8 / 1.3969838619232178e-8. Both mean-latent
comparisons failed unchanged atol 1e-6 / rtol 1e-5, with maximum differences about
9.89e-6 / 3.90e-5. Good decoded waveform agreement does not approve altered
separator conditioning. See `evidence/sam-codec-roundtrip-cpu-2026-09-21.json`;
these approximately 80 ms sine fixtures do not certify long-form or speech quality.

The diagnostic `--mixed-cpu-backend` keeps MKLDNN only for LSTM calls in an
isolated worker. Same-policy comparisons passed, but mean latents still differed
from the original default backend beyond tolerance (maximum about 1.58e-5 /
6.44e-5). Independent default-backend references are recorded separately in
`evidence/sam-mixed-cpu-backend-2026-09-21.json`. This has not been adopted into
runtime code or certified as equivalent separator conditioning.

`deepfilter3.ini` is an explicit inference configuration for DeepFilterNet 0.5.6.
It preserves the upstream checkpoint configuration and adds the two defaults
required by that library: `emb_gru_skip_enc = none` and `pf_beta = 0.02`.
Postfilter remains disabled. The loader stages the model on CPU and rejects
environment overrides of configuration keys. No checkpoint key is dropped,
renamed, or loaded permissively.
Build the adapter identity with `PinnedDeepFilterFactory.identity(policy.digest)`.
The engine validates this against the factory at construction and before loading.
Its runtime digest covers config/checkpoint hashes, package pins, device and
loader policy; the precision digest records float32, disabled autocast and disabled
TF32. Source/image revision evidence remains a separate certification requirement.
Do not substitute arbitrary hashes or reuse pre-change certification identities.

## Pinned DF3 smoke-test assets

Upstream archive:
https://raw.githubusercontent.com/Rikorose/DeepFilterNet/d375b2d8309e0935d165700c91da9de862a99c31/models/DeepFilterNet3.zip

| Asset | SHA-256 |
| --- | --- |
| Upstream archive | `49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2` |
| Original `DeepFilterNet3/config.ini` | `415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290` |
| Repository `deepfilter3.ini` | `0a926b0471793d7ba7446b07a8bdc10eafa5c9e3b93de4d65496e2cbcacc40d3` |
| `DeepFilterNet3/checkpoints/model_120.ckpt.best` | `23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003` |

The loader does not download assets. Supply the verified checkpoint locally.
The opt-in test requires `deepfilternet==0.5.6`, `deepfilterlib==0.5.6`,
`torch==2.8.0+cu128`, `torchaudio==2.8.0+cu128`, and `numpy==1.26.4`.
The probe also required `appdirs==1.4.4`. This is not a complete deployment lock.
On this Python 3.12 host, deepfilterlib required a Rust source build. Probe
dependencies were installed in an isolated temporary target; the running Ray
service's packages were not changed.

Run inside the prepared test environment:

```bash
HEAR_DF3_TEST_CHECKPOINT=/absolute/path/model_120.ckpt.best \
  python -m pytest -q tests/test_cleaner_v2_deepfilter_real.py
```

Without the checkpoint variable, the tests skip explicitly. With it, missing or
different packages/assets fail rather than being skipped or downloaded. These
CPU tests cover strict loading, finite exact-length mono/stereo tails, fresh
state between calls, and silence. They do not certify speech retention,
long-form boundary quality, resampling, GPU peaks, or production readiness.

## Natural resampling policy

Natural now converts non-48 kHz input through a bounded FFmpeg file pipeline,
then restores the original sample rate and exact source frame count after DF3.
The internal grid uses integer ceiling division; the return trip only permits
the corresponding rounding correction. A finite 4096-frame zero extension
flushes the resampler on tiny inputs, and the output is trimmed to the declared
grid. Every input/output is scanned in bounded blocks for finite samples.
Intermediate files are attempt-local and removed after success or failure.

The explicit SWR filter parameters are encoded in `AudioResampler.POLICY` and
included in the contextual policy digest, so older runtime identities cannot
silently acquire this behavior. Timestamp stretching, dither, and channel
remixing are not requested. See the official
[FFmpeg resampler options](https://ffmpeg.org/ffmpeg-resampler.html).
Probe binary: FFmpeg `6.1.1-3ubuntu5`, libswresample `4.12.100`. The eventual
certified runtime image still needs to pin this binary and its dependencies.

`tests/test_cleaner_v2_resampling.py` exercises real FFmpeg at 8, 16, 22.05,
44.1, 88.2, and 96 kHz with one-sample, tiny, and partial-tail inputs. Checks
include exact roundtrip frame counts, impulse alignment within one sample,
channel isolation, non-finite rejection, pre-cancellation, and conflict handling.
The opt-in real DF3 suite also exercises resampling through file sessions at
44.1 and 96 kHz. These structural checks are not perceptual certification.

## Offline long-duration CPU probe

In the prepared DF3 environment, run:

```bash
python -m scripts.benchmark_cleaner_deepfilter \
  --checkpoint /absolute/path/model_120.ckpt.best --seconds 3600
```

This generates stereo synthetic tone/noise in bounded chunks, appends a
17-frame partial tail, processes it with the real checkpoint (10-second blocks,
1-second context each side), and scans the complete output for finite samples
and exact rate/layout/frame count. It prints progress/RSS and a final JSON record
with code/checkpoint/policy hashes, PCM checksum, tail peak, timing, and maximum
model window length. Scratch audio is temporary and automatically removed.
No production registry, Ray worker, GPU or remote storage is changed.

The probe is not a speech corpus, listening test, codec/mastering acceptance,
A40 certification, or supported-duration advertisement. Its CPU deadline is
two hours and its input duration is bounded to 1–7200 seconds. The shared
comparison-loudness setting is retained in the plan even though this direct
engine benchmark does not invoke mastering/comparison generation.

## Worker lane ownership

The v2 executor now requires a held `WorkerLease`. Acquire `gpu` before creating
or loading a Natural/Voice Focus worker, and `cpu` for the separate bounded
Music & Atmosphere worker. Hold the lease through model teardown and process
shutdown, not just through each attempt: idle CUDA caches still count toward
the memory budget. Startup contention fails typed instead of loading a second
worker. The operating system releases the lock when the owning process exits.
The executor also holds one nonblocking attempt slot per worker across the
complete attempt. A second concurrent request is rejected before model loading;
the backend must handle queue/admission retry without treating it as completed
processing. Worker shutdown refuses an active slot. Finishing an attempt releases
that slot but retains process ownership and its model-cache lifetime protection.

All cleaner worker containers sharing a device must see the same trusted local
lock directory/inode (bind mount it across replacements). Use one directory per
device, keep it inaccessible to untrusted writers, and do not unlink lock files
on shutdown. A replaced lock is rejected on ownership checks. Use spawn/exec for
child workers; do not fork a live model owner and retain its descriptors. The
lock descriptor is close-on-exec. Existing legacy workers do not participate:
they must still drain before the new runtime starts.

This serializes cleaner worker ownership only. It is neither a VRAM partition
nor admission for unrelated transcription/TTS processes. Shared GPU budgets,
certified coexistence and physical memory controls remain separate release gates.

## Source inspection supervision

`SourceInspector.inspect` launches a fresh Python child (exec, not fork) for the
checksum and bounded libsndfile scan. The parent continuously checks the attempt
deadline, cancellation and scratch budget, and kills/reaps the child process
group on failure. The child receives only a restricted environment, not inherited
service/storage/Hugging Face credentials. Its structured diagnostic response is
bounded to 16 KiB and is not logged. Decoder import/startup time counts against
the attempt deadline. Package code and the active interpreter must be available
inside the worker container; no model weights are loaded by this child.

The shared subprocess runner also defaults to a minimal environment for FFmpeg
resampling/mastering: system executable search path, `C.UTF-8` locale and single
OpenMP/OpenBLAS thread settings. Parent credentials, proxies, Python/library
injection variables and `FFREPORT` are not inherited. Install codec binaries on
the image's system path (or pass a trusted absolute executable path); do not
depend on the server's ambient PATH. Explicit child environments, such as the
inspection package path, replace rather than merge with the parent environment
and must come only from trusted worker wiring, never an attempt ticket.

This is not an OS sandbox or a native-memory limit. Container isolation and host
memory accounting remain deployment requirements. Engine, quality and mastering
code still perform additional native operations; this change does not certify
all native calls as interruptible or replace dedicated model-worker supervision.

## Optional CPU speech-activity evidence (not wired into production)

`CpuSpeechActivity` uses the existing Silero ONNX support model, not a cleaner
engine. Construct it explicitly with a trusted `SpeechActivityPolicy`: exact
model SHA-256, ONNX Runtime version, NumPy version and threshold. It loads the
verified bytes (not a second unverified file read), uses CPUExecutionProvider
only, and limits both ONNX thread pools to one. It does not import Torch, download
weights or load a CUDA model. Retain one session in the bounded CPU worker;
recurrent state and context are local to each scan and separate for each channel.

The source must match its pinned identity. Other supported rates use the shared
explicit resampling policy. Reports retain source-frame intervals, full active
frame totals per channel, source/policy digests, at most 128 intervals and an
explicit truncation flag. Analysis and resampling temporaries are attempt-local.
Silence/negative VAD is not proof of a noise-only reference, and matching VAD
activity is not proof that words or speakers were preserved. The 0.5 threshold is
an engineering starting point, not a calibrated quality gate.

The installed Silero 6.2.1 `data/silero_vad.onnx` used in the local adapter parity
test has SHA-256
`1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.
Run the opt-in real-model parity test with an explicit local path:

```bash
HEAR_TEST_SILERO_ONNX=/path/to/silero_vad.onnx \
  .venv/bin/pytest -q tests/test_cleaner_v2_speech_activity.py
```

To enable source/output risk comparison in the v2 factory, inject
`SpeechRiskComparison(scanner)` into `AudioQualityGate`. The executor passes the
pinned source identity; the comparison independently hashes and scans the exact
processed PCM before mastering and sample cropping. Evidence is included in the
validation report and result manifest, with source/output hashes, analysis and
comparison policy digests, per-channel activity totals and bounded source-frame
loss intervals. These refer to the complete pre-mastering PCM, not MP3 bytes.

The initial comparison flags missing activity intervals of at least 100 ms or
an output active-frame total below half the corresponding source-channel total.
For the acknowledged mono route, each source channel is compared with the mono
target; this cannot prove individual-speaker retention. Truncated interval lists
are never interpreted as silence; they produce an incomplete-evidence warning.
Matching activity still requires review. An absent analyser is explicitly
reported as `speech_activity_unavailable`; a configured analyser that fails
raises a typed error rather than silently falling back to energy-only checks.

Production factory wiring, noise-reference review, speech/music corpus and
threshold calibration, native-call supervision, image dependency pinning and
deployment host-memory accounting remain required. This evidence must not turn
any profile's readiness on by itself. Backend consumers must accept and validate
the proposed manifest's optional `validation.speech_activity` field before
deploying this contract; no backend release is implied by the local integration.

## Selected noise-reference review

`SpeechAwareNoiseReferenceAnalyser(scanner)` now implements the noise engine's
reference-analysis interface. An authenticated preview ingress can call `review`
with a `NoiseReferenceSelection` (revision ID and source-frame bounds only).
It receives an immutable review with the actual analysis digest and warnings;
no placeholder digest, confirmation or completed cleaner plan is needed to ask
for a review. The ingress must authorize the source/revision before this call.

The reviewer hashes the whole source, reads only the selected interval into a
bounded attempt-local float file, preserves separate channels, and checks speech
activity on that interval. The digest binds source bytes, revision, bounds, rate,
channels, analysis policy and speech activity. It does not bind user confirmation:
confirmation is a separate backend-owned decision in the resolved plan. At
execution, `NoiseProfileSession` recomputes the review and rejects a mismatched
digest or detected speech before producing cleaned output. Cancellation/errors
remove only the review's temporary files.

The backend should present `speech_in_noise_reference` when detected and must
present `music_analysis_unavailable` / `noise_reference_requires_confirmation`.
This implementation has no validated music classifier. It always reports
uncertainty, including when VAD is negative; it never automatically certifies a
noise-only interval or chooses the first/quietest segment. Speech detections are
not overridden by confirmation. Reference review endpoints, persistence of user
confirmation, frontend presentation and production factory wiring remain with
the integration owners. Real speech/music calibration and native-call
supervision remain release gates.

## CPU noise runtime identity

`NoiseProfileEngine.describe(reference_analyser)` builds the local runtime
descriptor from NumPy/SoundFile/libsndfile versions, the reference analyser's
policy digest, explicit float/FFT/output precision, window/hop, channel linking,
gain smoothing and tail policy. The concrete speech-aware reference policy also
binds the speech model/dependency/threshold policy. This is an adapter descriptor,
not proof of a certified runtime image or listening quality.

Use this descriptor when preparing an approved catalogue/certification entry.
Do not rewrite a backend-authorized plan to the current descriptor at execution.
The engine checks the supplied identity at construction, session opening and
before processing; a wrong digest or changed dependency/reference/processing
policy fails typed as unavailable before output creation. There is no fallback
to an arbitrary runtime label. Existing reference-review digests must be renewed
when the reference policy changes. Registration/readiness still requires the
separate certification evidence and approved input limits.

## Initial A40 DF3 probe (not a release certificate)

`evidence/df3-a40-synthetic-smoke-2026-09-21.json` records a real 30-second stereo
DF3 CUDA probe with a partial tail, launched in a separately admitted Ray task
(one CPU, 0.1 GPU scheduling units, no retry). Existing services were not stopped
or unloaded. The supervisor required at least 20,000 MiB free before launch,
used a 120-second timeout and sampled the child's GPU memory; a 3 GB observed
process threshold would stop the probe. The model's Torch allocator cap was
1,000,000,000 bytes, lower than the initial production ceiling.

The measured child high-water sample was 834,666,496 bytes (796 MiB), with Torch
peaks of 293,311,488 allocated and 501,219,328 reserved bytes. Allocated is already
included in reserved. Sampling every requested 100 ms can miss faster peaks;
the NVIDIA CLI reports quantized MiB values. This is not a hard physical-memory
partition or a complete cold-load/inference peak certification. GPU usage
returned to the original 10,872 MiB after the child exited.

The benchmark now accepts an explicit CUDA device and allocator ceiling:

```bash
python -m scripts.benchmark_cleaner_deepfilter \
  --checkpoint /path/to/model_120.ckpt.best \
  --seconds 30 --device cuda:0 --allocator-cap-bytes 1000000000
```

This command alone does not perform Ray admission or whole-process supervision.
Run it only in an admitted, monitored test worker with the pinned dependencies;
the default remains CPU. Long-duration CUDA, real speech/listening, all layouts,
model switching, error/recovery, deployment overlap, SAM and concurrent service
load still need certification. No production readiness was enabled by this probe.

### One-hour A40 follow-up

`evidence/df3-a40-synthetic-hour-2026-09-21.json` records the same pinned runtime
processing one hour of synthetic 48 kHz stereo plus 17 tail frames. All
172,800,017 frames remained finite and the partial tail remained nonzero. The
361 contextual calls each used at most 576,000 input frames; no full-file GPU
allocation was introduced. Inference took 34.569 seconds (RTF 0.00960), while the
benchmark itself took 45.983 seconds, excluding interpreter/Ray overhead.

The sampled process VRAM peak remained 834,666,496 bytes, and Torch peak
allocated/reserved counters remained 293,311,488 / 501,219,328 bytes—matching the
short probe. Host peak RSS was 1,181,962,240 bytes. The separate Ray task reserved
1 CPU, 0.1 GPU scheduling units and 4 GB scheduling memory; those reservations
are not physical memory partitions. Its supervisor used a 900-second timeout,
3 GB sampled process threshold and requested 200 ms sampling cadence. There were
164 samples containing the GPU child process. The child exited successfully,
temporary audio was removed, and GPU usage returned to the pre-test snapshot.

This is duration/integrity evidence for one measured synthetic workload, not a
leak proof or a production latency promise. It does not include mastering,
speech-risk scans, storage transfer, busy-service coexistence, all input rates/
layouts, intended maximum duration, model switching, OOM or deployment overlap.
The other A40 release gates above remain open.

## Allocation faults and worker retirement

DF3 loader policy `df3-offline-strict-v3` treats host/CUDA OOM as
`resource_exhausted`; CUDA runtime faults also require process replacement.
The failed backend drops its model reference, releases its configuration lease
and returns sanitized typed errors without retaining the native exception's
traceback/tensor locals. No allocation-cap increase, immediate retry, precision
change or profile fallback occurs. The DF3 factory refuses subsequent loads in
that process after a restart-required fault.

Cancellation/deadline errors raised after model allocation also discard native
traceback/model references while retaining their original typed code. Ordinary
cancellation does not by itself mark a healthy process as OOM-damaged. These
checks bound references owned by the adapter; cached CUDA allocations and all
other process state still require proper worker teardown.

`CleanExecutionError.worker_restart_required` is an internal control signal,
preserved across Python worker serialization and terminal failure wrapping.
The executor marks its `WorkerLease` unhealthy, retains the ownership lock and
blocks further admissions. It may publish a failure-only manifest if the ticket
and grants remain valid; it never publishes a candidate on that path. The Pod/
Ray supervisor must then recycle the process, releasing ownership only after
model/worker teardown. There is no reset/retry method for the unhealthy lease.
Production supervisor/readiness wiring and real fault-injection certification
remain required. The unit tests inject faults; they do not exhaust the shared GPU.

Session teardown is also fail-closed: an exception from `EngineSession.close()`
requires process replacement because native resource ownership is uncertain.
Successful inference followed by failed cleanup cannot reach validation or
candidate publication. If cleanup fails while handling a typed processing error,
the original error code is retained, including cancellation/deadline; those paths
still do not publish a terminal failure manifest. The internal restart marker
and unhealthy ownership prevent another attempt on that worker. A failure-only
manifest may be published for eligible processing errors after reauthorization.

The earlier A40 probe records intentionally retain their measured v2 loader
runtime and code hashes. They are historical hardware evidence, not certification
of the newer v3 fault-handling build. Re-run release gates against the final
approved image/code descriptor; do not relabel old evidence as a new measurement.

## Cleaner-only Python package boundary

`deploy/cleaner/pyproject.toml` and `uv.lock` define an independent Python 3.12
dependency graph. Core imports do not require model frameworks. The `analysis`
extra adds CPU ONNX Runtime; `deepfilter` pins DF3 and the CUDA 12.8 Torch/audio
distributions. SAM is not packaged as a working engine yet. The lock contains no
ClearVoice, Demucs, business PostgreSQL/SQLAlchemy or transcription model packages.
This does not remove those dependencies from the legacy application: its retained
consumers must still migrate and drain before root dependency removal.

Build the wheel from an environment containing setuptools 83.0.0 and the `uv`
CLI, from the repository root:

```bash
python -m scripts.build_cleaner_wheel --output /path/to/new-empty-build-directory
```

The builder stages only `package-files.json`'s 50 allowlisted files, not the root
application, private docs, `.env`, checkpoints or old protobuf service. It builds
offline, fixes archive timestamps, then checks the actual wheel file list and
every source byte. It refuses nonempty output directories instead of deleting or
overwriting artifacts. Two builds of the tested source produced identical bytes.

Install core plus CPU analysis in a **new, separate environment**:

```bash
uv export --project deploy/cleaner --frozen --no-emit-project --no-dev \
  --extra analysis --output-file /path/to/cleaner-requirements.txt
uv venv --python 3.12 /path/to/new-cleaner-venv
uv pip install --no-config --python /path/to/new-cleaner-venv/bin/python \
  --require-hashes -r /path/to/cleaner-requirements.txt
uv pip install --no-config --python /path/to/new-cleaner-venv/bin/python \
  --no-deps /path/to/new-empty-build-directory/hear_cleaner_runtime-0.1.0-py3-none-any.whl
```

`--no-config` prevents the legacy root project's dependency overrides from
changing these locked requirements. Do not co-install this wheel with `hear-ai`:
they share the `hear` namespace and belong in separate processes/environments.
Do not install the metadata-only `deploy/cleaner` directory directly; use the
allowlist builder to include the actual implementation files.

The DF3 extra was also installed into a second fresh Python 3.12 environment
using the frozen project lock and its explicit PyTorch CUDA 12.8 index:

```bash
UV_PROJECT_ENVIRONMENT=/path/to/new-df3-venv uv sync --project deploy/cleaner \
  --frozen --all-extras --no-install-project --no-default-groups --python 3.12
uv pip install --no-config --python /path/to/new-df3-venv/bin/python \
  --no-deps /path/to/new-empty-build-directory/hear_cleaner_runtime-0.1.0-py3-none-any.whl
```

This required a Rust source build of deepfilterlib 0.5.6; binary-only installation
failed because no matching wheel was available. Supply an approved Rust build
toolchain for that step. The installed wheel then ran the pinned real DF3 model
on CPU over 576,017 synthetic stereo frames in two contextual blocks, preserving
finite output and the final 17 frames. Legacy application packages and Ray were
absent. CUDA dependencies installed, but CUDA was never initialized or executed.
Evidence is in `evidence/cleaner-wheel-df3-isolation-2026-09-21.json`.
The Python lock does not certify Rust transitive dependencies or native/system
libraries. Model assets and FFmpeg are external and must be pinned in the image.

The fresh-environment smoke loaded the wheel (not the checkout), ran real Silero
reference review, spectral noise reduction, speech-risk checks and FLAC/MP3
mastering on 16,017 synthetic stereo frames. Torch, Ray, old cleaner models and
business database packages were absent. Validation remained `review_required`.
Exact inventory and wheel/source hashes are in
`evidence/cleaner-wheel-isolation-2026-09-21.json`. Docker/Podman were unavailable,
so this is not an image build, GPU-extra certification, deployed API or completion
of the old-runtime removal gate. Production factory/transport wiring remains open.

## Explicit worker assembly

`CleanerWorkerFactory.build` in `hear.runtime.cleaner.factory` now wires source
inspection, the approved engine registry, optional speech-risk analysis, quality,
mastering and immutable artifact writing into a shared executor. Call it inside
a dedicated worker process with a pre-existing shared local lock directory,
an explicit CPU/GPU lane, approved `CertifiedRuntime` records, offline loaders,
readiness probes and a scoped immutable store. Assembly acquires the lane but
does not call probes/loaders/storage, read `.env`, start Ray, download assets or
invent runtime certification. Mixed CPU/GPU runtime registrations are rejected.

Use the returned worker as a context manager, or call `close` after all attempts
have stopped. Shutdown refuses an active attempt; failed construction releases
ownership. Supported engine models are attempt-session-owned and must be closed
by session teardown; this factory does not support external model caches. A
future shared cache needs explicit teardown before releasing ownership.

Ingress should report `worker.capabilities()`, not the registry alone: worker
retirement, lost ownership and closure override runtime readiness. Missing
certifications remain unavailable, and healthy capability snapshots are not
resource reservations. The ingress must still authenticate capabilities and
provide attempt authorizers, bounded workspaces/budgets, cancellation and progress
sinks. Production configuration, actual storage grants/writes, Ray supervision
and backend result integration are not supplied by this composition layer.

## Attempt stage durations

`ExecutionContext.timings` holds at most eight named wall-duration totals using
the monotonic clock: download, inspection, loading, inference, cleanup, validation,
mastering and upload. The executor resets them after attempt admission; contexts
must remain attempt-local. Loading includes registry/asset readiness checks and
session construction, separately from actual inference. Mastering includes its
encoding and exact-file checks. Inspection includes supervised child startup and
source scanning. These are operational spans, not pure kernel benchmarks.

Failures and cancellation retain completed/failed span durations. Missing stages
mean not measured, not zero work. The immutable validation report records
`stage_seconds_before_publication`; it cannot contain its own future upload time.
Ingress can read the final `context.timings.snapshot()` after return or exception
to export upload/failure timing with authorized attempt identity. No transport
metrics exporter is wired yet. Queue time must come from backend scheduling;
these measurements do not invent it, measure GPU memory or prove real-time audio
quality. Timings contain no paths, credentials or arbitrary diagnostic strings.

## CPU noise-profile duration probe

`scripts/benchmark_cleaner_noise_profile.py` creates deterministic synthetic stereo
RF64 audio in a private temporary directory, reviews its first noise-only second
with pinned real CPU Silero, and runs the concrete noise-profile engine at 3 dB
reduction. The reference confirmation applies only to that generated fixture,
never to user media. Music uncertainty remains explicit. Output is scanned in
bounded blocks for exact frame count, rate, stereo layout, finiteness and a
nonzero 17-frame tail; hashes bind output PCM, runtime and relevant source code.

```bash
PYTHONPATH=/workspace/hear-ai .venv/bin/python scripts/benchmark_cleaner_noise_profile.py \
  --speech-model /absolute/path/to/pinned/silero_vad.onnx --seconds 3600
```

This probe expects ONNX Runtime 1.27.0, NumPy 1.26.4 and the documented Silero
asset hash. It checks free scratch space and uses a 900-second processing guard;
supervise the process separately for native-call hangs. Ray can admit it as a
one-CPU, zero-GPU, no-retry task. Its memory reservation is not a hard memory
limit. Reported peak RSS belongs to the probe process, not the combined process
tree or host. Processing time includes reference revalidation, not only FFT work.
Synthetic tones/noise do not certify music transient retention or speech quality.

Recorded 30-second and one-hour Ray runs both passed. The hour processed
172,800,017 stereo frames in 141.48 seconds and preserved the partial tail. See
`evidence/noise-profile-cpu-synthetic-hour-2026-09-21.json`. In-task `getrusage`
reported 167,936,000 bytes peak RSS, while external `ps` RSS samples reached
185,577,472 bytes. This discrepancy is retained explicitly: do not use the lower
reported high-water value, sparse samples or this synthetic workload as a
certified memory upper bound. Aggregate/native memory limits remain unverified.
