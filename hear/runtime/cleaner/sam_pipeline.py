import importlib
import tempfile
from pathlib import Path

import soundfile as sf

from hear.runtime.cleaner.longform_sam import SolverPolicy, WindowedMidpointSolver
from hear.runtime.cleaner.resampling import AudioResampler
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.sam_codec_graph import SamCodecGraph
from hear.runtime.cleaner.sam_conditioning import AudioOnlySamForward, SamConditionedField
from hear.runtime.cleaner.sam_features import SamFeatureFile
from hear.runtime.cleaner.sam_noise import SamNoise
from hear.runtime.cleaner.sam_pcm import SamPCM
from hear.runtime.cleaner.sam_prompt_cache import SamPromptCache, SamPromptIdentity
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import (
    CleanExecutionError,
    CleanPlan,
    ErrorCode,
    RuntimeIdentity,
)


class SamSeparationPipeline:
    @staticmethod
    def preflight(frames: int, sample_rate: int, guard: ResourceGuard) -> int:
        """Reject impossible pinned-Small disk reservations before model loading.

        This is a necessary lower bound, not a complete peak-space estimator:
        the final decoder stage holds two batch-2, 96-channel, FP32 feature files
        while applying its first activation. Other live intermediates add space.
        """
        guard.check()
        if (
            type(frames) is not int
            or frames <= 0
            or type(sample_rate) is not int
            or not 8000 <= sample_rate <= 96000
        ):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid SAM preflight geometry")
        model_frames = AudioResampler.frame_count(frames, sample_rate, 48000)
        padded = ((model_frames + 1919) // 1920) * 1920
        if padded > guard.budget.max_frames:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "SAM codec padding exceeds frame reservation"
            )
        minimum = padded * 2 * 96 * 4 * 2
        occupied = sum(p.stat().st_size for p in guard.workspace.rglob("*") if p.is_file())
        if occupied + minimum > guard.budget.scratch_bytes:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "SAM minimum codec scratch reservation unavailable"
            )
        return minimum

    def __init__(
        self,
        core,
        codec,
        *,
        codec_tile_frames: int = 4096,
        policy: SolverPolicy | None = None,
    ):
        policy = policy or SolverPolicy(250, 50, 16)
        if policy.steps != 16 or policy.window_frames > 250:
            raise ValueError("unsupported SAM separation solver policy")
        self.core, self.codec = core, codec
        self.graph = SamCodecGraph(tile_frames=codec_tile_frames)
        self.solver = WindowedMidpointSolver(policy)

    def separate_plan(
        self,
        source: Path,
        destination: Path,
        *,
        plan: CleanPlan,
        expected_runtime: RuntimeIdentity,
        prompt: SamPromptIdentity,
        cache: SamPromptCache,
        guard: ResourceGuard,
    ) -> str:
        """Bind a resolved mono plan to loader-supplied runtime/prompt identities.

        The caller must obtain expected_runtime and prompt from trusted, certified
        loader configuration, not copy them from the incoming plan. This method
        does not replace registry admission or certify the loaded model.
        """
        guard.check()
        if (
            plan.profile != "voice_focus"
            or plan.runtime != expected_runtime
            or expected_runtime.engine != "sam_audio_small"
            or plan.prompt_sha256 != prompt.prompt_sha256
        ):
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM plan identity mismatch")
        if plan.channel_policy != "mono":
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "SAM channel policy is not implemented"
            )
        text, mask = cache.get(prompt)
        guard.check()
        for path in (source, destination):
            if path.is_symlink() or not path.resolve().is_relative_to(guard.workspace.resolve()):
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "SAM audio outside workspace")
        if destination.exists():
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "SAM output exists")
        try:
            if source.stat().st_size > guard.budget.max_input_bytes:
                raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "SAM input too large")
            with sf.SoundFile(source) as audio:
                rate, frames, channels = audio.samplerate, audio.frames, audio.channels
                if (
                    channels != 1
                    or not 8000 <= rate <= 96000
                    or not 0 < frames <= guard.budget.max_frames
                    or audio.format not in ("WAV", "WAVEX", "RF64")
                    or audio.subtype not in ("PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE")
                ):
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "unsupported SAM mono PCM")
        except (OSError, sf.LibsndfileError):
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "SAM input inspection failed"
            ) from None
        if rate == 48000:
            return self.separate_file(
                source, destination, seed=plan.seed, text=text, text_mask=mask, guard=guard
            )
        model_frames = AudioResampler.frame_count(frames, rate, 48000)
        guard.preflight_pcm(model_frames, 1, copies=2, output_bytes=frames * 4 + 16384)
        resampler = AudioResampler(CancellableProcessRunner())
        exported = False
        try:
            with tempfile.TemporaryDirectory(
                prefix="sam-resample-", dir=guard.workspace
            ) as directory:
                prepared, enhanced = (
                    Path(directory) / "prepared.wav",
                    Path(directory) / "target.wav",
                )
                resampler.convert(source, prepared, 48000, guard)
                identity = self.separate_file(
                    prepared, enhanced, seed=plan.seed, text=text, text_mask=mask, guard=guard
                )
                resampler.convert(enhanced, destination, rate, guard, exact_frames=frames)
                exported = True
            guard.check()
            return identity
        except BaseException:
            if exported:
                destination.unlink(missing_ok=True)
            raise

    def separate_file(
        self,
        source: Path,
        destination: Path,
        *,
        seed: int,
        text,
        text_mask,
        guard: ResourceGuard,
    ) -> str:
        """Process prepared mono PCM to target RF64; return the noise invocation identity.

        Conditioning must come from the pinned prompt cache. This low-level route
        does not certify or register a capability, master audio, or publish artifacts.
        """
        guard.check()
        if destination.exists() or destination.is_symlink():
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "SAM target output exists")
        if not destination.resolve().is_relative_to(guard.workspace.resolve()):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "SAM output outside workspace")
        sampler, pcm = SamNoise(), SamPCM()
        torch = importlib.import_module("torch")
        message = torch.from_numpy(sampler.watermark(seed))
        exported = False
        try:
            with tempfile.TemporaryDirectory(prefix="sam-pcm-", dir=guard.workspace) as directory:
                prepared = noise = result = None
                try:
                    prepared = pcm.read(source, Path(directory) / "input.f32", guard)
                    frames = (prepared.frames + 1919) // 1920
                    identity = sampler.identity(seed=seed, frames=frames)
                    noise = sampler.create(
                        Path(directory) / "noise.f32", frames=frames, seed=seed, guard=guard
                    )
                    result = self.separate(
                        prepared, noise, text=text, text_mask=text_mask, message=message
                    )
                    pcm.write(result, destination, stream="target")
                    exported = True
                finally:
                    # Close every mapping even if a preceding close fails.
                    try:
                        if result is not None:
                            result.close(remove=True)
                    finally:
                        try:
                            if noise is not None:
                                noise.close(remove=True)
                        finally:
                            if prepared is not None:
                                prepared.close(remove=True)
            guard.check()
            return identity
        except BaseException:
            if exported:
                destination.unlink(missing_ok=True)
            raise

    def separate(self, source: SamFeatureFile, noise: SamFeatureFile, *, text, text_mask, message):
        """Borrow prepared mono PCM and explicit noise; return target/residual batch."""
        guard = source.guard
        guard.check()
        frames = (source.frames + 1919) // 1920
        if (
            source.batch != 1
            or source.channels != 1
            or not source.complete
            or noise.guard is not guard
            or not noise.complete
            or (noise.frames, noise.batch, noise.channels) != (frames, 1, 256)
        ):
            raise ValueError("incompatible complete SAM pipeline inputs")
        torch = importlib.import_module("torch")
        if (
            not isinstance(text, torch.Tensor)
            or not isinstance(text_mask, torch.Tensor)
            or text.ndim != 3
            or text.shape[0] != 1
            or text.shape[2] != 768
            or not 1 <= text.shape[1] <= 512
            or text.dtype != torch.float32
            or text_mask.shape != text.shape[:2]
            or text_mask.dtype != torch.bool
            or not torch.isfinite(text).all()
            or not text.any()
            or not text_mask.any()
        ):
            raise ValueError("pipeline requires bounded nonempty FP32 text conditioning")
        if (
            not isinstance(message, torch.Tensor)
            or message.shape != (2, 16)
            or message.dtype != torch.float32
            or not ((message == 0) | (message == 1)).all()
        ):
            raise ValueError("pipeline requires explicit target/residual watermark messages")
        # Snapshot once before any native codec work, not after the solver finishes.
        frozen_message = message.detach().clone()
        frozen_text, frozen_mask = text.detach().clone(), text_mask.detach().clone()
        mean = field = solved = result = None
        try:
            mean = self.graph.encode_mean(self.codec, source)
            field = SamConditionedField(
                AudioOnlySamForward(self.core, guard, max_frames=self.solver.policy.window_frames),
                mean,
                frozen_text,
                frozen_mask,
            )
            with tempfile.TemporaryDirectory(
                prefix="sam-separation-", dir=guard.workspace
            ) as directory:
                destination = Path(directory) / "joint.f32"
                self.solver.solve(
                    noise.path, destination, frames=frames, channels=256, field=field, guard=guard
                )
                solved = SamFeatureFile(
                    destination, frames=frames, batch=1, channels=256, guard=guard
                )
                try:
                    result = self.graph.decode_joint(
                        self.codec, solved, frames=source.frames, message=frozen_message
                    )
                    guard.check()
                    return result
                finally:
                    solved.close()
        except BaseException:
            if result is not None:
                result.close(remove=True)
            raise
        finally:
            if field is not None:
                field.close()
            if mean is not None:
                mean.close(remove=True)
