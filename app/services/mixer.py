import os
import tempfile

import torch
import torchaudio
import torchaudio.functional as F_audio

from app.core.storage import storage


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

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        torchaudio.save(out_path, mixed, self.TARGET_SR)
        return out_path

    async def mix_and_upload(self, track_paths: list[dict], recording_id: str, job_id: str) -> dict:
        import asyncio
        mixed_path = self.mix(track_paths)
        if not mixed_path:
            return {}

        b2_key = f"masters/{recording_id}/{job_id}.wav"
        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(None, storage.upload_file, mixed_path, b2_key)
        os.unlink(mixed_path)

        return {
            "master_url": url,
            "b2_key": b2_key,
        }


mixer = AudioMixer()
