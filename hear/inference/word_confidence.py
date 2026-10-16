"""wav2vec2 CTC posteriors for per-word transcript confidence.

Loads `facebook/wav2vec2-base-960h` offline, in float16 on CUDA, and turns a
16 kHz transcription window into CTC log-probabilities in overlapping pieces.
The scoring itself is torch-free in `hear.services.transcription.word_confidence`.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from hear.services.reconstruction.verification import CtcAligner
from hear.services.transcription.word_confidence import Piece, WordConfidence

MODEL_RATE = 16000
PIECE_SECONDS = 30.0
OVERLAP_SECONDS = 2.0


class WordConfidenceScorer:
    def __init__(self, model_dir: Path, *, device: str | None = None) -> None:
        if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
            raise RuntimeError("word_confidence_model_not_provisioned")
        self._model_dir = model_dir
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        self._lock = threading.Lock()
        self._loaded = False
        self._failed = False
        self.source = "wav2vec2_ctc"

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            try:
                self._processor = Wav2Vec2Processor.from_pretrained(
                    str(self._model_dir), local_files_only=True
                )
                self._model = (
                    Wav2Vec2ForCTC.from_pretrained(
                        str(self._model_dir), local_files_only=True, torch_dtype=self._dtype
                    )
                    .to(self._device)
                    .eval()
                )
                vocab = self._processor.tokenizer.get_vocab()
                self._vocab = {str(k): int(v) for k, v in vocab.items()}
                self._blank = int(self._processor.tokenizer.pad_token_id)
                if CtcAligner.WORD_DELIMITER not in self._vocab:
                    raise RuntimeError("word_confidence_vocab_missing_delimiter")
            except Exception:
                self._failed = True
                raise
            self._loaded = True

    def check_health(self) -> None:
        if self._failed:
            raise RuntimeError("word_confidence_scorer_failed")
        if not (self._model_dir / "config.json").is_file():
            raise RuntimeError("word_confidence_model_not_provisioned")

    def warmup(self) -> None:
        self._load()
        self._log_probs(np.zeros(MODEL_RATE, dtype="float32"))

    def _log_probs(self, audio16: np.ndarray) -> np.ndarray:
        inputs = self._processor(audio16, sampling_rate=MODEL_RATE, return_tensors="pt")
        with torch.inference_mode():
            logits = self._model(inputs.input_values.to(self._device, self._dtype)).logits[0]
            return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

    def score_window(self, samples: np.ndarray, words: list[dict]) -> list[float | None]:
        """Per-word probabilities for words whose times are relative to `samples` (16 kHz mono)."""
        self._load()
        if not words:
            return []
        audio = np.asarray(samples, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        window_seconds = len(audio) / MODEL_RATE
        pieces = []
        for start, end in WordConfidence.pieces_for(window_seconds, PIECE_SECONDS, OVERLAP_SECONDS):
            clip = audio[int(start * MODEL_RATE) : int(end * MODEL_RATE)]
            if len(clip) < MODEL_RATE // 10:
                continue
            log_probs = self._log_probs(clip)
            pieces.append(Piece(start, log_probs, (len(clip) / MODEL_RATE) / log_probs.shape[0]))
        return WordConfidence.score_words(pieces, words, self._vocab, self._blank)
