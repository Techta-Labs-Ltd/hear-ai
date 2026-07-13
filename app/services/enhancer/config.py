from copy import deepcopy
from dataclasses import dataclass


def _preset_balanced() -> dict:
    return {
        "deepfilter_atten_lim_db": 12.0,
        "deepfilter_wet_dry": 0.70,
        "deepfilter_chunk_seconds": 30,
        "deepfilter_overlap_seconds": 3.0,
        "clearvoice_wet_dry": 0.30,
        "clearvoice_chunk_seconds": 30,
        "clearvoice_overlap_seconds": 3.0,
        "eq_highpass_hz": 80.0,
        "eq_bass_cut_hz": 250.0,
        "eq_bass_cut_db": -2.0,
        "eq_treble_boost_hz": 6000.0,
        "eq_treble_boost_db": 2.0,
        "deesser_freq_hz": 6000.0,
        "deesser_threshold_db": -20.0,
        "deesser_reduction_db": -6.0,
        "limiter_lookahead_ms": 5,
        "limiter_release_ms": 100,
        "limiter_ceiling_db": -2.5,
        "limiter_soft_clip_threshold": 0.9,
        "lufs_target": -16.0,
        "vad_threshold": 0.35,
        "vad_speech_gain": 1.0,
        "vad_silence_gain": 0.70,
        "vad_uncertain_gain": 0.80,
        "vad_fade_ms": 100,
        "output_fade_ms": 5,
    }


def _preset_high() -> dict:
    p = _preset_balanced()
    p.update({
        "deepfilter_atten_lim_db": 15.0,
        "deepfilter_wet_dry": 0.75,
        "clearvoice_wet_dry": 0.35,
        "vad_silence_gain": 0.60,
        "vad_uncertain_gain": 0.70,
        "limiter_ceiling_db": -3.0,
    })
    return p


def _preset_maximum() -> dict:
    p = _preset_high()
    p.update({
        "deepfilter_atten_lim_db": 20.0,
        "deepfilter_wet_dry": 0.80,
        "clearvoice_wet_dry": 0.40,
        "vad_silence_gain": 0.50,
        "vad_uncertain_gain": 0.60,
        "limiter_ceiling_db": -3.5,
    })
    return p


PRESETS = {
    "balanced": _preset_balanced,
    "high": _preset_high,
    "maximum": _preset_maximum,
}


def _apply_preset(config: "EngineConfig") -> "EngineConfig":
    builder = PRESETS.get(config.quality)
    if builder is None:
        return config
    overrides = builder()
    cfg = deepcopy(config)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@dataclass
class EngineConfig:
    target_sr: int = 48000
    quality: str = "balanced"

    deepfilter_atten_lim_db: float = 12.0
    deepfilter_wet_dry: float = 0.70
    deepfilter_chunk_seconds: int = 30
    deepfilter_overlap_seconds: float = 3.0

    clearvoice_wet_dry: float = 0.30
    clearvoice_chunk_seconds: int = 30
    clearvoice_overlap_seconds: float = 3.0

    eq_highpass_hz: float = 80.0
    eq_bass_cut_hz: float = 250.0
    eq_bass_cut_db: float = -2.0
    eq_treble_boost_hz: float = 6000.0
    eq_treble_boost_db: float = 2.0

    deesser_freq_hz: float = 6000.0
    deesser_threshold_db: float = -20.0
    deesser_reduction_db: float = -6.0

    limiter_lookahead_ms: int = 5
    limiter_release_ms: int = 100
    limiter_ceiling_db: float = -2.5
    limiter_soft_clip_threshold: float = 0.9

    lufs_target: float = -16.0

    vad_threshold: float = 0.35
    vad_speech_gain: float = 1.0
    vad_silence_gain: float = 0.70
    vad_uncertain_gain: float = 0.80
    vad_fade_ms: int = 100

    output_fade_ms: int = 5