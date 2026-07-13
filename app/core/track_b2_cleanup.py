from app.core.recording_fetcher import TrackData


def _asset_b2_key(storage, asset: dict | None) -> str | None:
    if not isinstance(asset, dict):
        return None
    key = asset.get("b2_key")
    if isinstance(key, str) and key.strip():
        return key.strip()
    url = asset.get("audio_url")
    if isinstance(url, str) and url.strip():
        return get_storage().public_url_to_key(url.strip())
    return None


def cleanup_track_ai_b2_assets(storage, track: TrackData, *, include_enhanced: bool) -> list[str]:
    keys: list[str | None] = []
    for layer in track.ai_speed_layers or []:
        if isinstance(layer, dict):
            keys.append(_asset_b2_key(storage, layer))
    keys.append(_asset_b2_key(storage, track.ai_compressed_audio))
    if include_enhanced:
        keys.append(_asset_b2_key(storage, track.ai_enhanced_audio))
    return get_storage().delete_keys_best_effort(keys)
