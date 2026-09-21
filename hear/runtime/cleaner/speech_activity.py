import hashlib
import json
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from hear.runtime.cleaner.resampling import AudioResampler
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode, SourceIdentity
from hear.services.magic_clean.inspection import SourceInspector


@dataclass(frozen=True)
class SpeechInterval:
    channel: int
    start_frame: int
    end_frame: int
    peak_probability: float


@dataclass(frozen=True)
class SpeechActivityReport:
    source_sha256: str
    policy_sha256: str
    sample_rate: int
    frames: int
    channels: int
    active_frames: tuple[int, ...]
    intervals: tuple[SpeechInterval, ...]
    intervals_truncated: bool


@dataclass(frozen=True)
class SpeechActivityPolicy:
    model_sha256: str
    onnxruntime_version: str
    numpy_version: str
    threshold: float = 0.5

    def __post_init__(self):
        if len(self.model_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.model_sha256
        ):
            raise ValueError("speech model requires a SHA-256 digest")
        if not 0 < self.threshold < 1:
            raise ValueError("invalid speech probability threshold")
        if not self.onnxruntime_version or not self.numpy_version:
            raise ValueError("speech runtime versions must be explicit")

    @property
    def digest(self) -> str:
        descriptor = {
            "model_sha256": self.model_sha256,
            "onnxruntime": self.onnxruntime_version,
            "numpy": self.numpy_version,
            "threshold": self.threshold,
            "policy": "silero-6.2.1-onnx-cpu-16k-512-context64-channel-batch-v1",
            "resampling": AudioResampler.POLICY,
            "threads": 1,
            "max_intervals": 128,
            "tail": "zero-pad-final-frame-clip-to-source",
        }
        return hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class CpuSpeechActivity:
    """One offline ONNX session; recurrent state is fresh for every scan.

    Thresholds require corpus calibration. Negative VAD is never sufficient to
    approve a noise-only reference, and VAD agreement cannot approve word retention.
    Container memory limits/native-call supervision remain deployment requirements.
    """

    def __init__(
        self,
        model: Path,
        policy: SpeechActivityPolicy,
        resampler: AudioResampler,
        guard: ResourceGuard,
    ):
        guard.check()
        if model.is_symlink() or not model.is_file():
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "speech model unavailable")
        try:
            if (
                version("onnxruntime") != policy.onnxruntime_version
                or version("numpy") != policy.numpy_version
            ):
                raise ValueError("speech runtime version mismatch")
            # Small, explicit model ceiling prevents an accidental large model load.
            if not 0 < model.stat().st_size <= 16 * 1024 * 1024:
                raise ValueError("speech model size mismatch")
            digest = hashlib.sha256()
            payload = bytearray()
            with model.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    guard.check()
                    if len(payload) + len(chunk) > 16 * 1024 * 1024:
                        raise ValueError("speech model grew beyond size limit")
                    digest.update(chunk)
                    payload.extend(chunk)
            if digest.hexdigest() != policy.model_sha256:
                raise ValueError("speech model checksum mismatch")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                bytes(payload), sess_options=options, providers=["CPUExecutionProvider"]
            )
            if self.session.get_providers() != ["CPUExecutionProvider"]:
                raise ValueError("speech model provider mismatch")
            if [(v.name, v.type) for v in self.session.get_inputs()] != [
                ("input", "tensor(float)"),
                ("state", "tensor(float)"),
                ("sr", "tensor(int64)"),
            ] or [v.name for v in self.session.get_outputs()] != ["output", "stateN"]:
                raise ValueError("speech model interface mismatch")
        except CleanExecutionError:
            raise
        except Exception:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "pinned CPU speech model could not load"
            ) from None
        self.policy = policy
        self.resampler = resampler
        guard.check()

    def scan(
        self, source: Path, expected: SourceIdentity, guard: ResourceGuard
    ) -> SpeechActivityReport:
        SourceInspector.inspect(source, expected, guard)
        with tempfile.TemporaryDirectory(prefix="speech-analysis-", dir=guard.workspace) as temp:
            prepared = source
            if expected.sample_rate != 16000:
                prepared = Path(temp) / "analysis.wav"
                self.resampler.convert(source, prepared, 16000, guard)
            try:
                return self._scan(prepared, expected, guard)
            except CleanExecutionError:
                raise
            except Exception:
                raise CleanExecutionError(
                    ErrorCode.INVALID_AUDIO, "CPU speech analysis failed"
                ) from None

    def _scan(
        self, prepared: Path, expected: SourceIdentity, guard: ResourceGuard
    ) -> SpeechActivityReport:
        intervals: list[SpeechInterval] = []
        last_index: dict[int, int] = {}
        active_frames = [0] * expected.channels
        truncated = False
        # Each channel has its own recurrent state and audio context. Never mix
        # anti-phase or split-speaker stereo into an apparently silent mono input.
        state = np.zeros((2, expected.channels, 128), dtype=np.float32)
        context = np.zeros((expected.channels, 64), dtype=np.float32)
        with sf.SoundFile(prepared) as audio:
            total = AudioResampler.frame_count(expected.frames, expected.sample_rate, 16000)
            if (audio.samplerate, audio.channels, audio.frames) != (
                16000,
                expected.channels,
                total,
            ):
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "speech input grid mismatch")
            offset = 0
            while offset < total:
                guard.check()
                data = audio.read(min(512, total - offset), dtype="float32", always_2d=True)
                if not len(data) or not np.isfinite(data).all():
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid speech input")
                block = np.zeros((expected.channels, 512), dtype=np.float32)
                block[:, : len(data)] = data.T
                model_input = np.concatenate((context, block), axis=1)
                probability, next_state = self.session.run(
                    ["output", "stateN"],
                    {"input": model_input, "state": state, "sr": np.array(16000, dtype=np.int64)},
                )
                guard.check()
                if (
                    probability.shape != (expected.channels, 1)
                    or next_state.shape != state.shape
                    or probability.dtype != np.float32
                    or next_state.dtype != np.float32
                    or not np.isfinite(probability).all()
                    or not np.isfinite(next_state).all()
                    or np.any((probability < 0) | (probability > 1))
                ):
                    raise CleanExecutionError(
                        ErrorCode.INVALID_AUDIO, "invalid speech model output"
                    )
                state = next_state
                context = model_input[:, -64:].copy()
                start = min(expected.frames, offset * expected.sample_rate // 16000)
                offset += len(data)
                end = min(expected.frames, offset * expected.sample_rate // 16000)
                for channel, value in enumerate(probability[:, 0]):
                    if value < self.policy.threshold or end <= start:
                        last_index.pop(channel, None)
                        continue
                    active_frames[channel] += end - start
                    previous = last_index.get(channel)
                    if previous is not None and intervals[previous].end_frame == start:
                        old = intervals[previous]
                        intervals[previous] = SpeechInterval(
                            channel, old.start_frame, end, max(old.peak_probability, float(value))
                        )
                    elif len(intervals) < 128:
                        last_index[channel] = len(intervals)
                        intervals.append(SpeechInterval(channel, start, end, float(value)))
                    else:
                        truncated = True
                        last_index.pop(channel, None)
            guard.check()
        return SpeechActivityReport(
            expected.sha256,
            self.policy.digest,
            expected.sample_rate,
            expected.frames,
            expected.channels,
            tuple(active_frames),
            tuple(intervals),
            truncated,
        )
