from dataclasses import dataclass
from enum import Enum

class ContentMode(str, Enum):
    SPEECH = "speech"
    MUSIC  = "music"
    AUTO   = "auto"


@dataclass(frozen=True, slots=True)
class StemLevels:
    """User-selected stem levels as percentages from the UI."""

    speech: int
    music: int
    background: int

    def __post_init__(self) -> None:
        for name, value in (
            ("speech", self.speech),
            ("music", self.music),
            ("background", self.background),
        ):
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be between 0 and 100")


DEFAULT_STEM_LEVELS = StemLevels(speech=100, music=10, background=10)


@dataclass
class EnhancementResult:
    b2_key:            str
    enhanced_url:      str
    local_path:        str
    quality_score:     float
    snr_db:            float
    peak_db:           float
    lufs:              float
    clipping_detected: bool
    mode_used:         str
    bucket_name:       str
