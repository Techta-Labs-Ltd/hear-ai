# Fish Speech S2 Pro NF4 runtime

The model is groxaxo/s2-pro-BnB-4Bits at revision
5c09659b9dbea2f64b90c1a611c4824560619ce7, a community NF4 quantization of
Fish Audio S2 Pro, not a separate replacement TTS model.

The matching loader is groxaxo/fish-speech-int4-patch at
fc4e1e24ff3b8d7d28fdd66e6789f23acb63c5bb. Reconstruction's locked environment
includes bitsandbytes 0.49.2 and inflect 7.5.0. CUDA inference explicitly enables
bnb4, float16 compute and a 4096-token cache. The codec is not quantized. No
MossFormer, SAM or changes to DeepFilterNet are included.

## Provisioning

Run outside job execution, in the reconstruction environment:

```bash
HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 python -m scripts.provision_fish_nf4 \
  --model-root /root/hear-ai-v11/models
```

Provisioning verifies every required file's SHA-256. The runtime view hardlinks
the unchanged model/codec/tokenizer weights and uses the official pinned base
model tokenizer metadata, because the community TokenizersBackend metadata is
not supported by the installed Transformers. All 4096 semantic IDs were checked.
A generic model download alone does not prepare this required runtime view.

Pod configuration is saved at /root/hear-ai-v11/fish-nf4.env:

```text
FISH_SPEECH_HOME=/root/hear-ai-v11/models/fish-speech/source-nf4
FISH_SPEECH_MODEL_ROOT=/root/hear-ai-v11/models
FISH_SPEECH_BNB_MODE=nf4
```

The model root is independently configurable for reconstruction, leaving the
other role model paths unchanged. On Serverless provision the same artifacts
onto its attached volume and set FISH_SPEECH_MODEL_ROOT accordingly. Both image
variants pin the compatible source and dependency versions. Apply these settings
to the corrected worker source, not the obsolete live FFmpeg reconstruction.

## Real model verification

The A40 generated a 7.3839456-second recording from text in 21.2933 seconds after
a 125.742-second cold load. The same warm model then generated a same-speaker
text edit through the real reconstruction renderer, producing a 3.483-second
master and MP3 in 12.2889 seconds including local rendering/mastering. Source bytes
were unchanged. These are real Fish outputs, not synthetic model doubles.

Peak sampled device memory was 9743 MiB (9.51 GiB). PyTorch peak allocation was
9578704896 bytes; allocator reservation was 9854517248 bytes. These measurements
include different accounting scopes and must not be added together. They are
single-job observations, not a guarantee of maximum use for arbitrary prompts.

The runs used no compilation, no batching and no cloud services. They prove that
the quantized model loads and generates with our adapter, not a production latency
or multi-job capacity certification. Independently checked word accuracy and
speaker similarity are not claimed.

Listen in /workspace/hear-ai-v11/clean/fish-nf4-20260928:
01_real_fish_nf4.wav, 02_edited_master.flac and 02_edited_delivery.mp3.
Verification details and hashes are in verification.json.

The previous missing-checkpoint problem is resolved by this provisioning.
Commercial license metadata remains unchanged; downloading or quantizing does
not supply commercial permission. The production API/backend/B2 round trip was
not deployed by this model-install step.

Final regression: 630 passed, 12 skipped, 3 warnings. Mypy reported no issues
across 136 source files; Ruff and architecture checks passed.
