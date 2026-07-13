from dataclasses import dataclass
from enum import Enum

class ContentMode(str, Enum):
    SPEECH = "speech"
    MUSIC  = "music"
    AUTO   = "auto"

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
