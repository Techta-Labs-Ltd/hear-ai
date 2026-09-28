"""Public, versioned one-click cleaning options shared by ingress and workers."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hear.contracts.sound_cleanup import SoundCleanupOptions


class MagicCleanProfile(StrEnum):
    NATURAL = "natural"
    STUDIO_VOICE = "studio_voice"
    OUTDOOR_MOBILE = "outdoor_mobile"
    CLEAN_RAW = "clean_raw"


@dataclass(frozen=True)
class ProfileSpec:
    label: str
    description: str
    attenuation: int
    auto_level: bool
    highpass_hz: int = 0
    presence_db: float = 0.0
    compression_ratio: float = 1.0


PROFILE_VERSION = "deepfilter-presets-v1"
PROFILES = {
    MagicCleanProfile.NATURAL: ProfileSpec(
        "Natural",
        "Reduce background noise while keeping a natural voice.",
        24,
        False,
    ),
    MagicCleanProfile.STUDIO_VOICE: ProfileSpec(
        "Studio Voice",
        "Clean and balance spoken recordings for listening.",
        18,
        True,
        70,
        1.5,
        2.0,
    ),
    MagicCleanProfile.OUTDOOR_MOBILE: ProfileSpec(
        "Outdoor & Mobile",
        "Reduce noise and low rumble in mobile recordings.",
        24,
        True,
        100,
        0.0,
        3.0,
    ),
    MagicCleanProfile.CLEAN_RAW: ProfileSpec(
        "Clean & Raw",
        "Gentle denoising only, without EQ or compression.",
        12,
        False,
    ),
}


class CleaningProfiles:
    @staticmethod
    def validate(options: dict[str, Any]) -> dict[str, Any]:
        try:
            profile = MagicCleanProfile(options.get("profile", ""))
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid_magic_clean_profile") from exc
        allowed = {
            "profile",
            "attenuation_limit_db",
            "auto_level",
            "remove_clicks",
            "trim_silence",
            "cleaner_ticket",
            "sound_cleanup",
            "reduce_stationary_noise",
        }
        if set(options) - allowed:
            raise ValueError("unsupported_magic_clean_options")
        spec = PROFILES[profile]
        attenuation = options.get("attenuation_limit_db", spec.attenuation)
        if type(attenuation) is not int or attenuation not in (12, 18, 24):
            raise ValueError("invalid_attenuation_limit_db")
        result: dict[str, Any] = {"profile": profile.value, "attenuation_limit_db": attenuation}
        for name, default in (
            ("auto_level", spec.auto_level),
            ("remove_clicks", False),
            ("trim_silence", False),
        ):
            value = options.get(name, default)
            if type(value) is not bool:
                raise ValueError(f"invalid_{name}")
            result[name] = value
        background = options.get("reduce_stationary_noise", False)
        if type(background) is not bool:
            raise ValueError("invalid_reduce_stationary_noise")
        if background and (profile == MagicCleanProfile.CLEAN_RAW or "cleaner_ticket" in options):
            raise ValueError("stationary_cleanup_requires_nonraw_available_profile")
        if "reduce_stationary_noise" in options:
            result["reduce_stationary_noise"] = background
        sound = SoundCleanupOptions.model_validate(options.get("sound_cleanup", {}))
        if sound.enabled:
            if profile == MagicCleanProfile.CLEAN_RAW:
                raise ValueError("clean_raw_is_denoise_only")
            if result["trim_silence"]:
                raise ValueError("sound_cleanup_preserves_timeline_disable_trimming")
            if "cleaner_ticket" in options:
                raise ValueError("sound_cleanup_requires_available_engine_mode")
        if "sound_cleanup" in options:
            result["sound_cleanup"] = sound.model_dump(mode="json")
        if profile == MagicCleanProfile.CLEAN_RAW and any(
            result[n] for n in ("auto_level", "remove_clicks", "trim_silence")
        ):
            raise ValueError("clean_raw_is_denoise_only")
        if "cleaner_ticket" in options:
            # A certificate for the old Natural chain does not certify new DSP.
            if profile != MagicCleanProfile.NATURAL or any(
                result[n] for n in ("auto_level", "remove_clicks", "trim_silence")
            ):
                raise ValueError("preset_requires_available_engine_mode")
            result["cleaner_ticket"] = options["cleaner_ticket"]
        return result

    @staticmethod
    def catalogue() -> dict[str, Any]:
        return {
            "profile_version": PROFILE_VERSION,
            "designed_for": "spoken_word",
            "recommended_profile": MagicCleanProfile.STUDIO_VOICE.value,
            "profiles": {
                profile.value: {
                    "label": spec.label,
                    "description": spec.description,
                    "engine": "deepfilternet3",
                    "defaults": CleaningProfiles.validate({"profile": profile.value}),
                    "options": {
                        "attenuation_limit_db": [12, 18, 24],
                        **(
                            {
                                "auto_level": [False, True],
                                "remove_clicks": [False, True],
                                "trim_silence": [False, True],
                            }
                            if profile != MagicCleanProfile.CLEAN_RAW
                            else {}
                        ),
                    },
                    "preserves_channels": True,
                    "preserves_internal_pauses": True,
                    "requires_approval": True,
                }
                for profile, spec in PROFILES.items()
            },
        }
