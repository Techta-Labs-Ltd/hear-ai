import importlib
from pathlib import Path

import numpy as np

from hear.runtime.cleaner.mapped_residency import MappedResidency
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.sam_convolution import SamConvolutionGeometry
from hear.runtime.cleaner.sam_recurrent import SamLSTMStream
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class SamFeatureFile:
    MAX_TILE_BYTES = 64 * 1024 * 1024

    def __init__(
        self,
        path: Path,
        *,
        frames: int,
        batch: int,
        channels: int,
        guard: ResourceGuard,
        create: bool = False,
    ):
        guard.check()
        MappedResidency.require_supported()
        if not (0 < frames <= guard.budget.max_frames and batch in (1, 2) and 0 < channels <= 2048):
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "unsupported feature shape")
        if path.is_symlink() or not path.resolve().is_relative_to(guard.workspace.resolve()):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "feature path outside workspace")
        self.path, self.guard = path, guard
        self.frames, self.batch, self.channels = frames, batch, channels
        self._map = None
        self._owned = False
        self._written_frames = 0 if create else frames
        size = frames * batch * channels * 4
        try:
            if create:
                occupied = sum(p.stat().st_size for p in guard.workspace.rglob("*") if p.is_file())
                if occupied + size > guard.budget.scratch_bytes:
                    raise CleanExecutionError(
                        ErrorCode.RESOURCE_EXHAUSTED, "feature scratch reservation failed"
                    )
                with path.open("xb") as output:
                    self._owned = True
                    output.truncate(size)
            elif not path.is_file() or path.stat().st_size != size:
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "feature file size mismatch")
            self._map = np.memmap(
                path,
                dtype=np.float32,
                mode="r+" if create else "r",
                shape=(frames, batch, channels),
            )
            guard.check()
        except FileExistsError:
            self.close(remove=self._owned)
            raise CleanExecutionError(
                ErrorCode.ARTIFACT_CONFLICT, "feature output exists"
            ) from None
        except (OSError, MemoryError):
            self.close(remove=self._owned)
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "feature allocation failed"
            ) from None
        except BaseException:
            self.close(remove=self._owned)
            raise

    @property
    def complete(self) -> bool:
        return self._map is not None and self._written_frames == self.frames

    def _check_window(self, start: int, end: int) -> None:
        self.guard.check()
        if self._map is None:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "feature file is closed")
        if not 0 <= start < end <= self.frames:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid feature window")
        if (end - start) * self.batch * self.channels * 4 > self.MAX_TILE_BYTES:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "feature tile exceeds memory policy"
            )

    def read(self, start: int, end: int) -> np.ndarray:
        self._check_window(start, end)
        if end > self._written_frames:
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "feature input is incomplete")
        result = np.array(self._map[start:end].transpose(1, 2, 0), copy=True, order="C")
        MappedResidency.evict(self.guard, self._map)
        if not np.isfinite(result).all():
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "nonfinite feature input")
        return result

    def write(self, start: int, values: np.ndarray) -> None:
        if (
            values.ndim != 3
            or values.shape[:2] != (self.batch, self.channels)
            or values.dtype != np.float32
        ):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid feature output shape")
        end = start + values.shape[2]
        self._check_window(start, end)
        if not self._owned or self._map.mode == "r":
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "feature input is read-only")
        if start != self._written_frames:
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "noncontiguous feature output")
        if not np.isfinite(values).all():
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "nonfinite feature output")
        self._map[start:end] = values.transpose(2, 0, 1)
        MappedResidency.evict(self.guard, self._map)
        self._written_frames = end

    def close(self, *, remove: bool = False) -> None:
        if self._map is not None:
            self._map._mmap.close()
            self._map = None
        if remove and self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False


class SamFeatureRunner:
    @staticmethod
    def paired_latents(source: SamFeatureFile, destination: Path, *, tile_frames: int):
        """Convert joint [1,256,T] into target/residual [2,128,T], in that order."""
        if source.batch != 1 or source.channels != 256:
            raise ValueError("invalid joint SAM latent layout")
        SamFeatureRunner._check_tile(source, tile_frames)
        output = SamFeatureFile(
            destination,
            frames=source.frames,
            batch=2,
            channels=128,
            guard=source.guard,
            create=True,
        )
        try:
            for start in range(0, source.frames, tile_frames):
                end = min(source.frames, start + tile_frames)
                values = source.read(start, end)
                output.write(start, values.reshape(2, 128, end - start))
            return output
        except BaseException:
            output.close(remove=True)
            raise

    @staticmethod
    def reflect_pad(
        source: SamFeatureFile, destination: Path, *, tile_frames: int, hop: int = 1920
    ):
        """Match SAM's right-only reflect padding without reading the entire waveform."""
        if hop != 1920 or source.channels != 1:
            raise ValueError("unsupported SAM waveform padding policy")
        padding = (-source.frames) % hop
        if padding >= source.frames:
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "waveform too short for codec reflect padding"
            )
        SamFeatureRunner._check_tile(source, tile_frames)
        output = SamFeatureFile(
            destination,
            frames=source.frames + padding,
            batch=source.batch,
            channels=1,
            guard=source.guard,
            create=True,
        )
        try:
            for start in range(0, output.frames, tile_frames):
                end = min(output.frames, start + tile_frames)
                values = np.empty((source.batch, 1, end - start), dtype=np.float32)
                original = max(0, min(end, source.frames) - start)
                if original:
                    values[..., :original] = source.read(start, start + original)
                count = end - start - original
                if count:
                    offset = max(0, start - source.frames)
                    reflected = source.read(
                        source.frames - 1 - offset - count, source.frames - 1 - offset
                    )
                    values[..., original:] = reflected[..., ::-1]
                output.write(start, values)
            return output
        except BaseException:
            output.close(remove=True)
            raise

    @staticmethod
    def select(
        source: SamFeatureFile,
        destination: Path,
        *,
        frame_start: int,
        frame_end: int,
        channel_start: int,
        channel_end: int,
        tile_frames: int,
    ):
        """Copy an explicit frame/channel interval, for mean latents and output trimming."""
        if (
            not 0 <= frame_start < frame_end <= source.frames
            or not 0 <= channel_start < channel_end <= source.channels
        ):
            raise ValueError("invalid codec feature selection")
        SamFeatureRunner._check_tile(source, tile_frames)
        output = SamFeatureFile(
            destination,
            frames=frame_end - frame_start,
            batch=source.batch,
            channels=channel_end - channel_start,
            guard=source.guard,
            create=True,
        )
        try:
            for start in range(frame_start, frame_end, tile_frames):
                values = source.read(start, min(frame_end, start + tile_frames))
                output.write(start - frame_start, values[:, channel_start:channel_end, :])
            return output
        except BaseException:
            output.close(remove=True)
            raise

    @staticmethod
    def message(layer, message, source: SamFeatureFile, destination: Path, *, tile_frames: int):
        """Apply one explicit, frozen watermark message over every feature tile."""
        torch = importlib.import_module("torch")
        if type(layer).__module__ != "dacvae.nn.layers" or type(layer).__name__ != "MsgProcessor":
            raise ValueError("unsupported codec message processor")
        upstream = importlib.import_module("dacvae.nn.layers").MsgProcessor
        if type(layer) is not upstream or layer.training or layer.hidden_size != source.channels:
            raise ValueError("invalid codec message processor")
        if (
            not isinstance(message, torch.Tensor)
            or message.shape != (source.batch, layer.nbits)
            or message.dtype != torch.float32
            or not torch.isfinite(message).all()
            or not ((message == 0) | (message == 1)).all()
        ):
            raise ValueError("watermark message must contain explicit FP32 bits")
        parameter = layer.msg_processor.weight
        if parameter.dtype != torch.float32:
            raise ValueError("watermark message weights require FP32")
        frozen = message.detach().to(parameter.device).clone()
        SamFeatureRunner._check_tile(source, tile_frames)
        output = SamFeatureFile(
            destination,
            frames=source.frames,
            batch=source.batch,
            channels=source.channels,
            guard=source.guard,
            create=True,
        )
        try:
            for start in range(0, source.frames, tile_frames):
                values = source.read(start, min(source.frames, start + tile_frames))
                with torch.inference_mode(), torch.autocast(parameter.device.type, enabled=False):
                    predicted = layer(torch.from_numpy(values).to(parameter.device), frozen)
                    source.guard.check()
                    if predicted.shape != values.shape or predicted.dtype != torch.float32:
                        raise CleanExecutionError(
                            ErrorCode.INVALID_AUDIO, "invalid watermark features"
                        )
                    output.write(start, predicted.cpu().numpy())
            return output
        except BaseException:
            output.close(remove=True)
            raise

    @staticmethod
    def blend(
        base: SamFeatureFile,
        watermark: SamFeatureFile,
        destination: Path,
        *,
        alpha: float,
        tile_frames: int,
    ):
        if (
            alpha != 0.25
            or base.guard is not watermark.guard
            or (base.frames, base.batch, base.channels)
            != (watermark.frames, watermark.batch, watermark.channels)
        ):
            raise ValueError("unsupported watermark blend")
        SamFeatureRunner._check_tile(base, tile_frames)
        output = SamFeatureFile(
            destination,
            frames=base.frames,
            batch=base.batch,
            channels=base.channels,
            guard=base.guard,
            create=True,
        )
        try:
            for start in range(0, base.frames, tile_frames):
                end = min(base.frames, start + tile_frames)
                values = base.read(start, end) + np.float32(alpha) * watermark.read(start, end)
                output.write(start, values)
            return output
        except BaseException:
            output.close(remove=True)
            raise

    @staticmethod
    def activation(layer, source: SamFeatureFile, destination: Path, *, tile_frames: int):
        """Run only the pinned codec's elementwise activation families."""
        torch = importlib.import_module("torch")
        supported = type(layer) in (torch.nn.ELU, torch.nn.Tanh, torch.nn.Identity)
        if type(layer).__module__ == "dacvae.nn.layers" and type(layer).__name__ == "Snake1d":
            snake = importlib.import_module("dacvae.nn.layers").Snake1d
            supported = type(layer) is snake and layer.alpha.shape == (1, source.channels, 1)
        if not supported or layer.training:
            raise ValueError("unsupported codec activation")
        parameters = tuple(layer.parameters())
        device = parameters[0].device if parameters else torch.device("cpu")
        if any(p.dtype != torch.float32 or p.device != device for p in parameters):
            raise ValueError("codec activation requires colocated FP32 weights")
        SamFeatureRunner._check_tile(source, tile_frames)
        output = SamFeatureFile(
            destination,
            frames=source.frames,
            batch=source.batch,
            channels=source.channels,
            guard=source.guard,
            create=True,
        )
        try:
            for start in range(0, source.frames, tile_frames):
                values = source.read(start, min(source.frames, start + tile_frames))
                with torch.inference_mode(), torch.autocast(device.type, enabled=False):
                    predicted = layer(torch.from_numpy(values).to(device))
                    source.guard.check()
                    if predicted.shape != values.shape or predicted.dtype != torch.float32:
                        raise CleanExecutionError(
                            ErrorCode.INVALID_AUDIO, "invalid activation output"
                        )
                    output.write(start, predicted.cpu().numpy())
            return output
        except BaseException:
            output.close(remove=True)
            raise

    @staticmethod
    def residual(
        branch: SamFeatureFile,
        shortcut: SamFeatureFile,
        destination: Path,
        *,
        tile_frames: int,
        true_skip: bool = False,
    ):
        """Preserve DACVAE's branch + centered shortcut semantics, without broadcasting."""
        SamFeatureRunner._check_tile(branch, tile_frames)
        difference = shortcut.frames - branch.frames
        if (
            branch.guard is not shortcut.guard
            or (branch.batch, branch.channels) != (shortcut.batch, shortcut.channels)
            or difference < 0
            or difference % 2
            or (true_skip and difference)
        ):
            raise ValueError("incompatible codec residual features")
        offset = difference // 2
        output = SamFeatureFile(
            destination,
            frames=branch.frames,
            batch=branch.batch,
            channels=branch.channels,
            guard=branch.guard,
            create=True,
        )
        try:
            for start in range(0, branch.frames, tile_frames):
                end = min(branch.frames, start + tile_frames)
                values = branch.read(start, end)
                values += shortcut.read(start + offset, end + offset)
                output.write(start, values)
            return output
        except BaseException:
            output.close(remove=True)
            raise

    @staticmethod
    def _check_tile(source: SamFeatureFile, tile_frames: int) -> None:
        if not 0 < tile_frames <= 480000:
            raise ValueError("invalid codec feature tile size")
        if tile_frames * source.batch * source.channels * 4 > source.MAX_TILE_BYTES:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "feature tile exceeds memory policy"
            )

    @staticmethod
    def recurrent(layer, source: SamFeatureFile, destination: Path, *, tile_frames: int):
        """Run one upstream residual LSTM with attempt-local, contiguous state."""
        torch = importlib.import_module("torch")
        if not isinstance(layer, torch.nn.LSTM) or source.channels != layer.input_size:
            raise ValueError("unsupported codec recurrent layer")
        parameters = tuple(layer.parameters())
        if any(p.dtype != torch.float32 or p.device != parameters[0].device for p in parameters):
            raise ValueError("codec recurrent layer requires colocated FP32 weights")
        if tile_frames * source.batch * source.channels * 4 > source.MAX_TILE_BYTES:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "recurrent tile exceeds memory policy"
            )
        stream = SamLSTMStream(layer, max_frames=tile_frames, batch_size=source.batch)
        output = None
        try:
            output = SamFeatureFile(
                destination,
                frames=source.frames,
                batch=source.batch,
                channels=source.channels,
                guard=source.guard,
                create=True,
            )
            for start in range(0, source.frames, tile_frames):
                block = source.read(start, min(source.frames, start + tile_frames))
                features = torch.from_numpy(block).to(parameters[0].device)
                predicted = stream.process(features, start, source.guard)
                output.write(start, predicted.cpu().numpy())
            source.guard.check()
            return output
        except BaseException:
            if output is not None:
                output.close(remove=True)
            raise
        finally:
            stream.close()

    @staticmethod
    def convolution(layer, source: SamFeatureFile, destination: Path, *, tile_frames: int):
        torch = importlib.import_module("torch")
        if (
            not isinstance(layer, (torch.nn.Conv1d, torch.nn.ConvTranspose1d))
            or layer.training
            or layer.padding_mode != "zeros"
        ):
            raise ValueError("unsupported codec convolution")
        if source.channels != layer.in_channels or not 0 < tile_frames <= 480000:
            raise ValueError("invalid codec feature tile configuration")
        if tile_frames * source.batch * layer.out_channels * 4 > source.MAX_TILE_BYTES:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "output tile exceeds memory policy"
            )
        parameter = next(layer.parameters())
        if parameter.dtype != torch.float32:
            raise ValueError("codec convolution requires FP32 weights")
        transposed = isinstance(layer, torch.nn.ConvTranspose1d)
        geometry = SamConvolutionGeometry(
            layer.kernel_size[0],
            layer.stride[0],
            layer.dilation[0],
            layer.padding[0],
            layer.output_padding[0] if transposed else 0,
            transposed,
            getattr(layer, "pad_mode", "none") == "auto",
            getattr(layer, "causal", False),
        )
        output = SamFeatureFile(
            destination,
            frames=geometry.output_frames(source.frames),
            batch=source.batch,
            channels=layer.out_channels,
            guard=source.guard,
            create=True,
        )
        try:
            for start in range(0, output.frames, tile_frames):
                window = geometry.window(
                    source.frames, start, min(output.frames, start + tile_frames)
                )
                native_frames = geometry.output_frames(window.input_end - window.input_start)
                if native_frames * source.batch * layer.out_channels * 4 > source.MAX_TILE_BYTES:
                    raise CleanExecutionError(
                        ErrorCode.RESOURCE_EXHAUSTED, "convolution halo exceeds tile policy"
                    )
                block = source.read(window.input_start, window.input_end)
                with torch.inference_mode(), torch.autocast(parameter.device.type, enabled=False):
                    features = torch.from_numpy(block).to(parameter.device)
                    predicted = layer(features)
                    source.guard.check()
                    if (
                        predicted.shape != (source.batch, layer.out_channels, native_frames)
                        or predicted.dtype != torch.float32
                        or not torch.isfinite(predicted).all()
                    ):
                        raise CleanExecutionError(
                            ErrorCode.INVALID_AUDIO, "invalid codec convolution output"
                        )
                    values = predicted[..., window.crop_start : window.crop_end].cpu().numpy()
                    output.write(start, values)
                source.guard.check()
            return output
        except BaseException:
            output.close(remove=True)
            raise
