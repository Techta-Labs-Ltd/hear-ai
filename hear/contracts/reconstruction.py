"""Text-edit contracts; supplied replacement audio is not a substitute for Fish TTS."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

# Fish S2 Pro locks timbre from roughly five seconds of continuous speech upwards;
# one-word edit intervals are far too short to clone from, so they must carry an
# explicit reference chosen elsewhere in the recording.
MIN_REFERENCE_SECONDS = 5.0
MAX_REFERENCE_SECONDS = 20.0
MAX_SEED = 2**31 - 1


class VoiceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    end_seconds: float = Field(gt=0, allow_inf_nan=False)
    text: str = Field(min_length=1, max_length=4000)
    channel: Literal[0, 1] | None = None

    @model_validator(mode="after")
    def bounds(self):
        duration = self.end_seconds - self.start_seconds
        if not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS or not self.text.strip():
            raise ValueError("voice_reference_requires_5_to_20_seconds_and_matching_text")
        return self


class SpeechEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    segment_start: float = Field(ge=0, allow_inf_nan=False)
    segment_end: float = Field(ge=0, allow_inf_nan=False)
    new_text: str = Field(default="", max_length=20000)
    original_text: str = Field(default="", max_length=4000)
    # Neighbouring words of the same sentence, outside the replaced interval. The
    # worker generates the whole sentence for natural prosody, then splices only
    # the new words back over [segment_start, segment_end].
    context_before: str = Field(default="", max_length=1000)
    context_after: str = Field(default="", max_length=1000)
    is_deletion: StrictBool = False

    @model_validator(mode="after")
    def bounds(self):
        if self.segment_end < self.segment_start:
            raise ValueError("invalid_reconstruction_interval")
        if self.is_deletion:
            if self.segment_end <= self.segment_start or self.new_text.strip():
                raise ValueError("deletion_requires_nonempty_interval_and_no_new_text")
            if self.context_before.strip() or self.context_after.strip():
                raise ValueError("deletion_takes_no_context")
        elif not self.new_text.strip():
            raise ValueError("new_text_required_empty_text_is_not_implicit_deletion")
        return self

    @property
    def generation_text(self) -> str:
        return " ".join(
            part.strip()
            for part in (self.context_before, self.new_text, self.context_after)
            if part.strip()
        )

    @property
    def context_word_counts(self) -> tuple[int, int, int]:
        return (
            len(self.context_before.split()),
            len(self.new_text.split()),
            len(self.context_after.split()),
        )

    @property
    def can_self_reference(self) -> bool:
        """Whether the replaced interval itself is long enough to clone the voice from."""
        return (
            bool(self.original_text.strip())
            and MIN_REFERENCE_SECONDS
            <= self.segment_end - self.segment_start
            <= MAX_REFERENCE_SECONDS
        )


class ReconstructionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    same_speaker: StrictBool = True
    language: str = Field(default="en", pattern=r"^en(?:-GB)?$")
    changes: list[SpeechEdit] = Field(default_factory=list, max_length=64)
    reference: VoiceReference | None = None
    # One take per seed: resend the seed from a preview to reproduce it on confirm,
    # or send a new one to ask for another take ("regenerate").
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED, strict=True)
    # Candidate takes per edit; the worker keeps the best verified one. Omit for
    # the worker default.
    takes: int | None = Field(default=None, ge=1, le=4, strict=True)
    edited_transcript: str | None = Field(default=None, max_length=20000)
    original_transcript: str | None = Field(default=None, max_length=20000)
    segment_start: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    segment_end: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @classmethod
    def validate_operation(cls, operation: str, raw: dict[str, Any]):
        result = cls.model_validate(raw)
        edits = sorted(result.changes, key=lambda x: (x.segment_start, x.segment_end))
        if any(
            a.segment_end > b.segment_start or a.segment_start == b.segment_start
            for a, b in zip(edits, edits[1:], strict=False)
        ):
            raise ValueError("overlapping_reconstruction_edits")
        if sum(len(e.new_text) for e in edits) > 20000:
            raise ValueError("reconstruction_text_budget_exceeded")
        if operation in {"replace_segments", "edit_transcript", "preview"}:
            if any(len(e.generation_text) > 2000 for e in edits):
                raise ValueError("segment_text_budget_exceeded")
            if (
                not edits
                or result.edited_transcript is not None
                or result.segment_start is not None
                or result.segment_end is not None
            ):
                raise ValueError("segment_operations_require_only_changes")
            if result.same_speaker and result.reference is None:
                for edit in edits:
                    if not edit.is_deletion and not edit.can_self_reference:
                        raise ValueError(
                            "same_speaker_requires_explicit_reference_or_5_to_20_second_aligned_interval"
                        )
        elif operation == "rebuild":
            if (
                not (result.edited_transcript or "").strip()
                or edits
                or result.segment_start is not None
                or result.segment_end is not None
            ):
                raise ValueError("rebuild_requires_only_edited_transcript")
            if result.same_speaker and result.reference is None:
                raise ValueError("same_speaker_rebuild_requires_explicit_aligned_reference")
        elif operation == "remove_segments":
            if (
                edits
                or result.edited_transcript is not None
                or result.segment_start is None
                or result.segment_end is None
                or result.segment_end <= result.segment_start
            ):
                raise ValueError("remove_segments_requires_valid_interval")
        else:
            raise ValueError("invalid_reconstruction_operation")
        return result
