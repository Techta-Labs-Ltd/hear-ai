from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from hear.runtime.cleaner.pool import WorkerPool
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode
from hear.services.magic_clean.loudness import BlockMeasurement, KWeightedMeter

RATE = 48000
FRAME_SAMPLES = 1152
ENCODER_DELAY = 576
HANDLE_FRAMES = 24
VERIFY_MARGIN_FRAMES = 2
PIECE_SECONDS = 600
BITRATES = {1: 128, 2: 192}
_BITRATE_KBPS = (None, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
_SAMPLE_RATES = (44100, 48000, 32000)

@dataclass(frozen=True)
class Mp3Piece:
    index: int
    first_frame: int
    frame_count: int

    @property
    def start(self) -> int:
        return self.first_frame * FRAME_SAMPLES

    @property
    def end(self) -> int:
        return (self.first_frame + self.frame_count) * FRAME_SAMPLES


@dataclass(frozen=True)
class EncodeTask:
    master: str
    workspace: str
    deadline_epoch: float
    piece: Mp3Piece
    channels: int
    frames: int
    gain_db: float


@dataclass(frozen=True)
class VerifyTask:
    delivery: str
    workspace: str
    deadline_epoch: float
    piece: Mp3Piece
    frame_bytes: int
    total_frames: int
    channels: int


@dataclass(frozen=True)
class VerifiedPiece:
    measurement: BlockMeasurement


class Mp3Frames:
    @staticmethod
    def header_length(header: bytes) -> int | None:
        if len(header) < 4 or header[0] != 0xFF or header[1] & 0xE6 != 0xE2:
            return None  # sync + MPEG-1 + layer III
        bitrate = _BITRATE_KBPS[header[2] >> 4] if (header[2] >> 4) < 15 else None
        rate_index = (header[2] >> 2) & 3
        if bitrate is None or rate_index == 3:
            return None
        padding = (header[2] >> 1) & 1
        return 144 * bitrate * 1000 // _SAMPLE_RATES[rate_index] + padding

    @classmethod
    def offsets(cls, data: bytes) -> list[int]:
        offsets = []
        position = 0
        while position < len(data):
            length = cls.header_length(data[position : position + 4])
            if length is None or position + length > len(data):
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "mp3 frame stream is damaged")
            offsets.append(position)
            position += length
        offsets.append(position)
        return offsets


class ParallelMp3:
    @staticmethod
    def plan(frames: int, piece_seconds: int = PIECE_SECONDS) -> list[Mp3Piece]:
        total = max(2, -(-frames // FRAME_SAMPLES))
        per_piece = max(1, piece_seconds * RATE // FRAME_SAMPLES)
        pieces: list[Mp3Piece] = []
        first = 0
        while first < total:
            count = min(per_piece, total - first)
            pieces.append(Mp3Piece(len(pieces), first, count))
            first += count
        return pieces

    @staticmethod
    def encode_piece(task: EncodeTask) -> tuple[int, str]:
        guard = WorkerPool.guard(Path(task.workspace), task.deadline_epoch)
        piece = task.piece
        handle = HANDLE_FRAMES * FRAME_SAMPLES
        input_start = piece.start - handle + ENCODER_DELAY
        input_end = piece.end + handle + ENCODER_DELAY
        audio = np.zeros((input_end - input_start, task.channels), dtype=np.float32)
        read_start, read_end = max(0, input_start), min(task.frames, input_end)
        if read_end > read_start:
            with sf.SoundFile(task.master) as master:
                master.seek(read_start)
                data = master.read(read_end - read_start, dtype="float32", always_2d=True)
            audio[read_start - input_start : read_end - input_start] = data
        wav = Path(task.workspace) / f"piece-{piece.index:04d}.wav"
        sf.write(wav, audio, RATE, subtype="FLOAT")
        mp3 = Path(task.workspace) / f"piece-{piece.index:04d}.mp3"
        CancellableProcessRunner().run(
            [
                "ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-v", "error",
                "-threads", "1", "-i", str(wav), "-map_metadata", "-1", "-n",
                "-af", f"volume={task.gain_db:.8f}dB:precision=double",
                "-c:a", "libmp3lame", "-b:a", f"{BITRATES[task.channels]}k",
                "-reservoir", "0", "-ar", str(RATE), "-write_xing", "0",
                "-id3v2_version", "0", "-threads", "1", str(mp3),
            ],
            guard,
        )
        wav.unlink()
        data = mp3.read_bytes()
        offsets = Mp3Frames.offsets(data)
        needed = HANDLE_FRAMES + piece.frame_count
        if len(offsets) - 1 < needed:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "mp3 piece is shorter than planned")
        kept = Path(task.workspace) / f"kept-{piece.index:04d}.mp3"
        kept.write_bytes(data[offsets[HANDLE_FRAMES] : offsets[needed]])
        mp3.unlink()
        return piece.index, str(kept)

    @staticmethod
    def splice(kept: list[str], delivery: Path) -> int:
        with delivery.open("xb") as stream:
            for path in kept:
                stream.write(Path(path).read_bytes())
        return len(Mp3Frames.offsets(delivery.read_bytes())) - 1

    @staticmethod
    def verify_piece(task: VerifyTask) -> VerifiedPiece:
        guard = WorkerPool.guard(Path(task.workspace), task.deadline_epoch)
        piece = task.piece
        lead = min(VERIFY_MARGIN_FRAMES, piece.first_frame)
        trail = min(VERIFY_MARGIN_FRAMES, task.total_frames - piece.first_frame - piece.frame_count)
        first, count = piece.first_frame - lead, piece.frame_count + lead + trail
        with Path(task.delivery).open("rb") as stream:
            stream.seek(first * task.frame_bytes)
            data = stream.read(count * task.frame_bytes)
        if len(Mp3Frames.offsets(data)) - 1 != count:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "mp3 verification range is damaged")
        clip = Path(task.workspace) / f"verify-{piece.index:04d}.mp3"
        clip.write_bytes(data)
        try:
            with sf.SoundFile(clip) as stream:
                if stream.samplerate != RATE or stream.channels != task.channels:
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "encoded layout mismatch")
                decoded = stream.read(dtype="float32", always_2d=True)
        except (RuntimeError, ValueError, OSError) as exc:
            if isinstance(exc, CleanExecutionError):
                raise
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "encoded audio decode failed") from exc
        finally:
            clip.unlink(missing_ok=True)
        guard.check()
        if len(decoded) != count * FRAME_SAMPLES or not np.isfinite(decoded).all():
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid encoded PCM")
        measurement = KWeightedMeter.measure(
            decoded, RATE, discard_frames=lead * FRAME_SAMPLES, keep_frames=piece.frame_count * FRAME_SAMPLES
        )
        return VerifiedPiece(measurement)
