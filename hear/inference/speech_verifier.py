"""Word and speaker verification of generated speech with two small local models.

wav2vec2-base-960h (CTC) provides a greedy transcript and forced alignment of the
requested text; WavLM-base-plus-sv provides x-vector speaker embeddings for a
cosine similarity between a generated take and the voice reference. Both load
offline from the manifest directories and share the reconstruction worker's GPU.

Speed and memory: weights are held in float16 on CUDA (about 0.2 GB each), one
CTC forward pass serves both the transcript and the alignment, and the
reference embedding is cached per reference so takes and edits of one job pay
for it once. `last_metrics` records wall time and peak CUDA memory per call.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WavLMForXVector,
)

from hear.services.reconstruction.verification import CtcAligner, VerificationResult, WordSpan

SOURCE_RATE = 48000
MODEL_RATE = 16000
MAX_ALIGN_SECONDS = 60
DECODE_WINDOW_SECONDS = 30
MAX_REFERENCE_SECONDS = 20
REFERENCE_CACHE_SIZE = 8


class SpeechVerifier:
    def __init__(self, aligner_dir: Path, speaker_dir: Path, *, device: str | None = None):
        for directory in (aligner_dir, speaker_dir):
            if not directory.is_dir() or not (directory / "config.json").is_file():
                raise RuntimeError("speech_verifier_models_not_provisioned")
        self._aligner_dir = aligner_dir
        self._speaker_dir = speaker_dir
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        self._lock = threading.Lock()
        self._loaded = False
        self._failed = False
        self._references: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.last_metrics: dict[str, float | int] = {}

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            try:
                self._processor = Wav2Vec2Processor.from_pretrained(
                    str(self._aligner_dir), local_files_only=True
                )
                self._ctc = (
                    Wav2Vec2ForCTC.from_pretrained(
                        str(self._aligner_dir), local_files_only=True, torch_dtype=self._dtype
                    )
                    .to(self._device)
                    .eval()
                )
                self._extractor = AutoFeatureExtractor.from_pretrained(
                    str(self._speaker_dir), local_files_only=True
                )
                self._speaker = (
                    WavLMForXVector.from_pretrained(
                        str(self._speaker_dir), local_files_only=True, torch_dtype=self._dtype
                    )
                    .to(self._device)
                    .eval()
                )
                vocab = self._processor.tokenizer.get_vocab()
                self._vocab = {str(k): int(v) for k, v in vocab.items()}
                self._id_to_char = {v: k for k, v in self._vocab.items()}
                self._blank = int(self._processor.tokenizer.pad_token_id)
                self._delimiter = self._vocab[CtcAligner.WORD_DELIMITER]
            except Exception:
                self._failed = True
                raise
            self._loaded = True

    def warmup(self) -> dict[str, float | int]:
        """Load both models and run one short pass so the first job pays no cold start."""
        self._load()
        silence = np.zeros(MODEL_RATE, dtype="float32")
        started = time.monotonic()
        self._reset_peak()
        self._log_probs(silence)
        self._embedding(silence)
        return self._metrics(started, "warmup")

    def check_health(self) -> None:
        if self._failed:
            raise RuntimeError("speech_verifier_failed")
        for directory in (self._aligner_dir, self._speaker_dir):
            if not (directory / "config.json").is_file():
                raise RuntimeError("speech_verifier_models_not_provisioned")

    @property
    def device(self) -> str:
        return self._device

    def _reset_peak(self) -> None:
        if self._device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

    def _metrics(self, started: float, phase: str) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            "phase": phase,  # type: ignore[dict-item]
            "seconds": round(time.monotonic() - started, 4),
        }
        if self._device.startswith("cuda"):
            metrics["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            metrics["allocated_bytes"] = int(torch.cuda.memory_allocated())
        self.last_metrics = metrics
        return metrics

    @staticmethod
    def to_model_rate(audio: np.ndarray) -> np.ndarray:
        values = np.asarray(audio, dtype="float32")
        if values.ndim == 2:
            values = values[:, 0]
        return resample_poly(values, MODEL_RATE // 8000, SOURCE_RATE // 8000).astype("float32")

    def _log_probs(self, audio16: np.ndarray) -> np.ndarray:
        inputs = self._processor(audio16, sampling_rate=MODEL_RATE, return_tensors="pt")
        with torch.inference_mode():
            values = inputs.input_values.to(self._device, self._dtype)
            logits = self._ctc(values).logits[0]
            return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

    def _decode(self, audio16: np.ndarray, log_probs: np.ndarray | None) -> list[str]:
        if log_probs is not None:
            return CtcAligner.greedy_decode(log_probs, self._id_to_char, self._blank)
        window = DECODE_WINDOW_SECONDS * MODEL_RATE
        words: list[str] = []
        for start in range(0, len(audio16), window):
            piece = audio16[start : start + window]
            if len(piece) < MODEL_RATE // 10:
                continue
            words += CtcAligner.greedy_decode(self._log_probs(piece), self._id_to_char, self._blank)
        return words

    def _align(
        self, log_probs: np.ndarray, seconds: float, text: str
    ) -> tuple[WordSpan, ...] | None:
        words = [cleaned for _, cleaned in CtcAligner.normalize_words(text)]
        if not words or any(not word for word in words):
            return None
        tokens = CtcAligner.tokens_for(words, self._vocab)
        if tokens is None:
            return None
        path = CtcAligner.viterbi(log_probs, tokens, self._blank)
        if path is None:
            return None
        frame_seconds = seconds / log_probs.shape[0]
        spans = CtcAligner.word_spans(
            path, log_probs, tokens, words, self._blank, self._delimiter, frame_seconds
        )
        return tuple(spans) if len(spans) == len(words) else None

    def _embedding(self, audio16: np.ndarray) -> torch.Tensor:
        inputs = self._extractor(
            audio16, sampling_rate=MODEL_RATE, return_tensors="pt", padding=True
        )
        with torch.inference_mode():
            batch = {
                k: v.to(self._device, self._dtype) if v.is_floating_point() else v.to(self._device)
                for k, v in inputs.items()
            }
            output = self._speaker(**batch)
            return torch.nn.functional.normalize(output.embeddings.float(), dim=-1)[0].cpu()

    def _reference_embedding(self, reference: np.ndarray) -> torch.Tensor:
        voice = self.to_model_rate(reference)[: MAX_REFERENCE_SECONDS * MODEL_RATE]
        key = hashlib.sha256(voice.tobytes()).hexdigest()
        cached = self._references.get(key)
        if cached is not None:
            self._references.move_to_end(key)
            return cached
        embedding = self._embedding(voice)
        self._references[key] = embedding
        while len(self._references) > REFERENCE_CACHE_SIZE:
            self._references.popitem(last=False)
        return embedding

    def similarity(self, audio: np.ndarray, reference: np.ndarray) -> float:
        self._load()
        generated = self._embedding(self.to_model_rate(audio))
        return float(torch.dot(generated, self._reference_embedding(reference)))

    def verify(
        self, audio: np.ndarray, text: str, reference: np.ndarray | None
    ) -> VerificationResult:
        """Transcript, alignment and speaker similarity for one generated take."""
        self._load()
        started = time.monotonic()
        self._reset_peak()
        audio16 = self.to_model_rate(audio)
        seconds = len(audio16) / MODEL_RATE
        # One CTC pass serves the transcript and the alignment for takes up to a
        # minute; longer takes are transcribed in windows and not aligned.
        log_probs = self._log_probs(audio16) if seconds <= MAX_ALIGN_SECONDS else None
        decoded = tuple(self._decode(audio16, log_probs))
        words = self._align(log_probs, seconds, text) if log_probs is not None else None
        similarity = None
        if reference is not None:
            similarity = float(
                torch.dot(self._embedding(audio16), self._reference_embedding(reference))
            )
        self._metrics(started, "verify")
        return VerificationResult(words, decoded, similarity, words is not None)
