"""Text-edit contracts; supplied replacement audio is not a substitute for Fish TTS."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class VoiceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    end_seconds: float = Field(gt=0, allow_inf_nan=False)
    text: str = Field(min_length=1, max_length=4000)
    channel: Literal[0, 1] | None = None

    @model_validator(mode="after")
    def bounds(self):
        if not 1 <= self.end_seconds - self.start_seconds <= 20 or not self.text.strip():
            raise ValueError("voice_reference_requires_1_to_20_seconds_and_matching_text")
        return self


class SpeechEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    segment_start: float = Field(ge=0, allow_inf_nan=False)
    segment_end: float = Field(ge=0, allow_inf_nan=False)
    new_text: str = Field(default="", max_length=20000)
    original_text: str = Field(default="", max_length=4000)
    is_deletion: StrictBool = False

    @model_validator(mode="after")
    def bounds(self):
        if self.segment_end < self.segment_start:
            raise ValueError("invalid_reconstruction_interval")
        if self.is_deletion:
            if self.segment_end <= self.segment_start or self.new_text.strip():
                raise ValueError("deletion_requires_nonempty_interval_and_no_new_text")
        elif not self.new_text.strip():
            raise ValueError("new_text_required_empty_text_is_not_implicit_deletion")
        return self


class ReconstructionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    same_speaker: StrictBool = True
    language: str = Field(default="en", pattern=r"^en(?:-GB)?$")
    changes: list[SpeechEdit] = Field(default_factory=list, max_length=64)
    reference: VoiceReference | None = None
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
            if any(len(e.new_text) > 2000 for e in edits):
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
                    if not edit.is_deletion and (
                        not edit.original_text.strip()
                        or not 1 <= edit.segment_end - edit.segment_start <= 20
                    ):
                        raise ValueError(
                            "same_speaker_requires_aligned_original_text_or_explicit_reference"
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
