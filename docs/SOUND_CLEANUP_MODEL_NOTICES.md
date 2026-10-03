# Sound Cleanup model provenance

No model weights or source audio are committed to this repository.

- PANNs Cnn14 DecisionLevelMax: Qiuqiang Kong and contributors. Official weights:
  https://zenodo.org/records/3987831 (record metadata: CC BY 4.0).
  https://github.com/qiuqiangkong/audioset_tagging_cnn
  Source checkpoint MD5: `70539c43c18b6a289b3199c503a82c5a`.
  Provisioning records SHA-256 for the exported runtime asset.
- Silero VAD 6.2.1: Silero team, MIT-licensed source/model project.
  https://github.com/snakers4/silero-vad
  Pinned JIT SHA-256: `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`.
- AudioSep: Xubo Liu and contributors. Source license: MIT, copyright Xubo Liu.
  https://github.com/Audio-AGI/AudioSep
  Source revision: `944583f18b84589dc965de3ad77525c945334252`.
  Author checkpoint SHA-256: `f8cda01bfd0ebd141eef45d41db7a3ada23a56568465840d3cff04b8010ce82c`.
  Preserve the author's notices/model provenance; confirm applicable checkpoint
  redistribution terms before distributing compiled model bundles externally.
- Controlled-test bark: ESC-10 clip `1-100032-A-0.wav`, derived from
  `rose_bark.wav` by nfrae, Freesound sound 100032, CC0. The reference speech
  remains the user's local material and was never uploaded to a model API.

Model provenance is separate from acceptance of speech quality. No performance
score or licensing review can replace the [audio acceptance checks](AUDIO_JOBS.md#acceptance).
