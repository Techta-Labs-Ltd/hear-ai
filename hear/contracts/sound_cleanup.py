"""Versioned opt-in sound-repair contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SoundKind = Literal["handling", "impact", "animal", "cough", "click"]


class SelectedSound(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    kind: SoundKind
    confirmed_no_speech: bool = False

    @model_validator(mode="after")
    def interval(self):
        if not 0 < self.end_seconds - self.start_seconds <= 8:
            raise ValueError("sound_region_must_be_between_zero_and_eight_seconds")
        return self


class SoundCleanupOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    enabled: bool = Field(default=False, strict=True)
    auto_detect: bool = Field(default=True, strict=True)
    targets: list[SoundKind] = Field(
        default_factory=lambda: list[SoundKind](("handling", "impact", "animal")), max_length=5
    )
    remove_coughs: bool = Field(default=False, strict=True)
    preview_overlaps: bool = Field(default=False, strict=True)
    regions: list[SelectedSound] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def check_options(self):
        if len(self.targets) != len(set(self.targets)):
            raise ValueError("duplicate_sound_cleanup_targets")
        if (
            "cough" in self.targets or any(r.kind == "cough" for r in self.regions)
        ) and not self.remove_coughs:
            raise ValueError("cough_removal_requires_explicit_consent")
        if self.preview_overlaps and (
            not self.enabled or not self.regions or len(self.regions) > 4
        ):
            raise ValueError("overlap_preview_requires_one_to_four_selected_regions")
        if not self.enabled and (self.regions or self.remove_coughs):
            raise ValueError("sound_cleanup_is_disabled")
        if self.enabled and not self.auto_detect and not self.regions:
            raise ValueError("select_a_sound_or_enable_detection")
        if self.enabled and not self.targets and not self.regions and not self.remove_coughs:
            raise ValueError("sound_cleanup_has_no_targets")
        ordered = sorted(self.regions, key=lambda item: item.start_seconds)
        if any(a.end_seconds > b.start_seconds for a, b in zip(ordered, ordered[1:], strict=False)):
            raise ValueError("selected_sound_regions_overlap")
        if sum(r.end_seconds - r.start_seconds for r in self.regions) > 60:
            raise ValueError("selected_sound_regions_exceed_sixty_seconds")
        return self
