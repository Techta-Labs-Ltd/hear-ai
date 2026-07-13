import os
import tempfile

import torch
import torchaudio
import torchaudio.functional as F_audio

from app.core.audio_utils import save_as_mp3
from app.core.storage import get_storage


class AudioMixer:
    TARGET_SR = 44100

    def mix(self, track_paths: list[dict]) -> str:
        waveforms = []
        max_length = 0

        for tp in track_paths:
            if tp["is_muted"]:
                continue
            waveform, sr = torchaudio.load(tp["path"])
            if sr != self.TARGET_SR:
                waveform = F_audio.resample(waveform, sr, self.TARGET_SR)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform * tp["volume"]
            waveforms.append(waveform)
            max_length = max(max_length, waveform.shape[1])

        if not waveforms:
            return ""

        mixed = torch.zeros(1, max_length)
        for w in waveforms:
            padded = torch.zeros(1, max_length)
            padded[:, :w.shape[1]] = w
            mixed += padded

        peak = mixed.abs().max().item()
        if peak > 0.99:
            mixed = mixed * (0.99 / peak)

        out_path = save_as_mp3(mixed, self.TARGET_SR)
        return out_path

    async def mix_and_upload(self, track_paths: list[dict], track_id: str, job_id: str) -> dict:
        import asyncio
        mixed_path = self.mix(track_paths)
        if not mixed_path:
            return {}

        b2_key = f"masters/{track_id}/{job_id}.mp3"
        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(None, get_storage().upload_file, mixed_path, b2_key)
        os.unlink(mixed_path)

        return {
            "master_url": url,
            "b2_key": b2_key,
        }


mixer = AudioMixer()
