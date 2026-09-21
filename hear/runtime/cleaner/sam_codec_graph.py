import importlib
import uuid

from hear.runtime.cleaner.sam_features import SamFeatureFile, SamFeatureRunner


class SamCodecGraph:
    def __init__(self, *, tile_frames: int):
        if not 0 < tile_frames <= 480000:
            raise ValueError("invalid codec graph tile size")
        self.tile_frames = tile_frames

    def run(self, layer, source: SamFeatureFile) -> SamFeatureFile:
        """Return an owned output; preserve borrowed input and clean all intermediates."""
        created = []
        result = None
        try:
            result = self._execute(layer, source, created)
            source.guard.check()
            return result
        except BaseException:
            result = None
            raise
        finally:
            for feature in reversed(created):
                if feature is not result:
                    feature.close(remove=True)

    def _sequence(self, layers, source, created):
        current = source
        for layer in layers:
            following = self._execute(layer, current, created)
            if current is not source:
                current.close(remove=True)
            current = following
        if current is source:
            torch = importlib.import_module("torch")
            return self._execute(torch.nn.Identity().eval(), source, created)
        return current

    def encode_mean(self, codec, source: SamFeatureFile) -> SamFeatureFile:
        """Pinned SAM mean-latent route; never sample the DACVAE posterior."""
        created = []
        result = None
        try:
            padded = SamFeatureRunner.reflect_pad(
                source, self._destination(source), tile_frames=self.tile_frames
            )
            created.append(padded)
            encoded = self._execute(codec.encoder, padded, created)
            padded.close(remove=True)
            projected = self._execute(codec.quantizer.in_proj, encoded, created)
            encoded.close(remove=True)
            if projected.channels != 256 or projected.frames != (source.frames + 1919) // 1920:
                raise ValueError("unsupported SAM mean-latent geometry")
            result = SamFeatureRunner.select(
                projected,
                self._destination(source),
                frame_start=0,
                frame_end=projected.frames,
                channel_start=0,
                channel_end=128,
                tile_frames=self.tile_frames,
            )
            created.append(result)
            source.guard.check()
            return result
        except BaseException:
            result = None
            raise
        finally:
            for feature in reversed(created):
                if feature is not result:
                    feature.close(remove=True)

    def decode_joint(
        self, codec, source: SamFeatureFile, *, frames: int, message
    ) -> SamFeatureFile:
        """Return batched target/residual waveforms, not stereo or algebraic subtraction."""
        paired = SamFeatureRunner.paired_latents(
            source, self._destination(source), tile_frames=self.tile_frames
        )
        try:
            return self.decode_latents(codec, paired, frames=frames, message=message)
        finally:
            paired.close(remove=True)

    def decode_latents(
        self, codec, source: SamFeatureFile, *, frames: int, message
    ) -> SamFeatureFile:
        """Project mean latents, retain watermarking, then remove only codec padding."""
        if source.channels != 128 or frames < 1 or source.frames != (frames + 1919) // 1920:
            raise ValueError("invalid SAM latent/output geometry")
        projected = None
        decoded = None
        try:
            projected = self.run(codec.quantizer.out_proj, source)
            decoded = self.decode(codec.decoder, projected, message=message)
            if decoded.frames != source.frames * 1920 or decoded.channels != 1:
                raise ValueError("unsupported SAM decoded waveform geometry")
            return SamFeatureRunner.select(
                decoded,
                self._destination(source),
                frame_start=0,
                frame_end=frames,
                channel_start=0,
                channel_end=1,
                tile_frames=self.tile_frames,
            )
        finally:
            if decoded is not None:
                decoded.close(remove=True)
            if projected is not None:
                projected.close(remove=True)

    def decode(self, decoder, source: SamFeatureFile, *, message) -> SamFeatureFile:
        """Decode through the retained watermark graph with one explicit message."""
        upstream = importlib.import_module("dacvae.model.dacvae")
        if (
            type(decoder) is not upstream.Decoder
            or decoder.training
            or decoder.blending != "linear"
            or decoder.alpha != 0.25
        ):
            raise ValueError("unsupported SAM watermark decoder")
        torch = importlib.import_module("torch")
        processor = decoder.wm_model.msg_processor
        if (
            not isinstance(message, torch.Tensor)
            or message.dtype != torch.float32
            or message.shape != (source.batch, processor.nbits)
            or not ((message == 0) | (message == 1)).all()
        ):
            raise ValueError("decoder requires an explicit binary FP32 message")
        frozen = message.detach().clone()
        created = []
        result = None
        try:
            main = self._sequence(list(decoder.model), source, created)
            encoder = decoder.wm_model.encoder_block
            wm_decoder = decoder.wm_model.decoder_block
            hidden = self._execute(encoder.pre, main, created)
            stages = [list(block.upsample_group()) for block in reversed(list(decoder.model)[1:])]
            stages.append(list(encoder.post))
            for layers in stages:
                following = self._sequence(layers, hidden, created)
                hidden.close(remove=True)
                hidden = following
            following = SamFeatureRunner.message(
                processor, frozen, hidden, self._destination(source), tile_frames=self.tile_frames
            )
            created.append(following)
            hidden.close(remove=True)
            hidden = following
            stages = [list(wm_decoder.pre)]
            stages.extend(list(block.downsample_group()) for block in list(decoder.model)[1:])
            stages.append(list(wm_decoder.post))
            for layers in stages:
                following = self._sequence(layers, hidden, created)
                hidden.close(remove=True)
                hidden = following
            # Upstream forward_no_conv substitutes Identity for the last pre layer.
            # Execute the preceding layers without temporarily mutating shared modules.
            base = self._sequence(list(encoder.pre)[:-1], main, created)
            main.close(remove=True)
            result = SamFeatureRunner.blend(
                base,
                hidden,
                self._destination(source),
                alpha=decoder.alpha,
                tile_frames=self.tile_frames,
            )
            created.append(result)
            source.guard.check()
            return result
        except BaseException:
            result = None
            raise
        finally:
            for feature in reversed(created):
                if feature is not result:
                    feature.close(remove=True)

    def _execute(self, layer, source, created):
        torch = importlib.import_module("torch")
        source.guard.check()
        if layer.training:
            raise ValueError("codec graph requires evaluation mode")
        if type(layer) is torch.nn.Sequential:
            return self._sequence(list(layer), source, created)
        if type(layer).__module__ == "dacvae.model.dacvae":
            upstream = importlib.import_module("dacvae.model.dacvae")
            if type(layer) in (upstream.Encoder, upstream.EncoderBlock):
                return self._execute(layer.block, source, created)
            if type(layer) is upstream.ResidualUnit:
                branch = self._execute(layer.block, source, created)
                output = SamFeatureRunner.residual(
                    branch,
                    source,
                    self._destination(source),
                    tile_frames=self.tile_frames,
                    true_skip=layer.true_skip,
                )
                created.append(output)
                branch.close(remove=True)
                return output
            if type(layer) is upstream.DecoderBlock:
                # The upstream forward selects alternating chunks, not the full ModuleList.
                size = layer._chunk_size
                if size != 2:
                    raise ValueError("unsupported decoder block chunk layout")
                selected = [
                    child for index, child in enumerate(layer.block) if (index // size) % size == 0
                ]
                return self._sequence(selected, source, created)
            if type(layer) is upstream.LSTMBlock:
                if not layer.skip:
                    raise ValueError("unsupported nonresidual codec LSTM")
                output = SamFeatureRunner.recurrent(
                    layer.lstm,
                    source,
                    self._destination(source),
                    tile_frames=self.tile_frames,
                )
                created.append(output)
                return output
            raise ValueError("unsupported codec graph block")
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            output = SamFeatureRunner.convolution(
                layer,
                source,
                self._destination(source),
                tile_frames=self.tile_frames,
            )
        else:
            output = SamFeatureRunner.activation(
                layer,
                source,
                self._destination(source),
                tile_frames=self.tile_frames,
            )
        created.append(output)
        return output

    @staticmethod
    def _destination(source):
        return source.guard.workspace / f"codec-{uuid.uuid4().hex}.f32"
