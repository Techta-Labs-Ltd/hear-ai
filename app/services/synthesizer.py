import asyncio
import io
import os
import tempfile
import inspect
import wave
from dataclasses import dataclass
import importlib.util
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F_audio

from app.config import settings
from app.core.audio_utils import save_as_mp3
from app.core.hear_temp import (
    drop_temp_standalone,
    hear_temp_directory,
    hear_temp_job_dir,
    register_temp_standalone,
)
from app.core.storage import storage

@dataclass
class SynthesisResult:
    b2_key: str
    audio_url: str
    duration: float


class SpeechSynthesizer:
    TARGET_SR = 44100

    def __init__(self):
        self._loaded = False
        self._higgs_engine = None
        self._higgs_available = False

    def load(self):
        self._higgs_available = False
        if settings.HIGGS_AUDIO_ENABLED:
            module_name = (settings.HIGGS_AUDIO_MODULE or "higgs_audio").strip()
            self._inject_higgs_repo_path()
            if self._is_higgs_module_ready(module_name):
                self._higgs_available = True
                print(f"[STARTUP] Higgs audio ready ({module_name})")
            else:
                print(
                    f"[STARTUP] HIGGS_AUDIO_ENABLED=true but {module_name} not found under "
                    f"{settings.HIGGS_AUDIO_REPO_DIR} — rebuild/reconstruct jobs will fail until installed"
                )
        else:
            print("[STARTUP] Higgs audio disabled (pipeline, categorization, discovery unaffected)")
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def higgs_available(self) -> bool:
        return self._higgs_available

    async def reconstruct_segment(
        self,
        original_audio_path: str,
        segment_start: float,
        segment_end: float,
        new_text: str,
        track_id: str,
        same_speaker: bool = True,
    ) -> SynthesisResult:
        original_waveform, orig_sr = torchaudio.load(original_audio_path)

        if orig_sr != self.TARGET_SR:
            original_waveform = F_audio.resample(original_waveform, orig_sr, self.TARGET_SR)

        start_sample = int(segment_start * self.TARGET_SR)
        end_sample = int(segment_end * self.TARGET_SR)

        reference_path = None
        if same_speaker:
            reference_path = self._export_reference_clip(original_waveform, start_sample, end_sample, track_id=track_id)
        try:
            tts_bytes = await self._synthesize_higgs(new_text, reference_audio_path=reference_path)
        finally:
            if reference_path:
                drop_temp_standalone(reference_path)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=hear_temp_directory()) as tmp:
            tmp.write(tts_bytes)
            tts_path = tmp.name
        register_temp_standalone(tts_path, purpose="tts_segment", track_id=track_id)

        tts_waveform, tts_sr = torchaudio.load(tts_path)
        drop_temp_standalone(tts_path)

        if tts_sr != self.TARGET_SR:
            tts_waveform = F_audio.resample(tts_waveform, tts_sr, self.TARGET_SR)

        reconstructed = self._splice_segment(original_waveform, tts_waveform, start_sample, end_sample)

        out_path = save_as_mp3(reconstructed, self.TARGET_SR, track_id=track_id, purpose="reconstruct_mp3")

        duration = reconstructed.shape[1] / self.TARGET_SR
        b2_key = f"reconstructed/{track_id}/{os.urandom(8).hex()}.mp3"
        loop = asyncio.get_event_loop()
        audio_url = await loop.run_in_executor(None, storage.upload_file, out_path, b2_key)
        drop_temp_standalone(out_path)

        return SynthesisResult(
            b2_key=b2_key,
            audio_url=audio_url,
            duration=round(duration, 3),
        )

    async def reconstruct_segments(
        self,
        original_audio_path: str,
        track_id: str,
        changes: list,
        same_speaker: bool = True,
    ) -> SynthesisResult:
        original_waveform, orig_sr = torchaudio.load(original_audio_path)
        if orig_sr != self.TARGET_SR:
            original_waveform = F_audio.resample(original_waveform, orig_sr, self.TARGET_SR)
        merged = original_waveform
        normalized = self._normalize_changes(changes)
        if not normalized:
            raise ValueError("reconstruct requires non-empty segment changes")
        normalized = sorted(normalized, key=lambda c: float(c["segment_start"]), reverse=True)
        for change in normalized:
            start_sample = int(float(change["segment_start"]) * self.TARGET_SR)
            end_sample = int(float(change["segment_end"]) * self.TARGET_SR)
            start_sample = max(0, min(start_sample, merged.shape[1] - 1 if merged.shape[1] else 0))
            end_sample = max(start_sample + 1, min(end_sample, merged.shape[1]))
            reference_path = None
            if same_speaker:
                reference_path = self._export_reference_clip(merged, start_sample, end_sample, track_id=track_id)
            try:
                tts_bytes = await self._synthesize_higgs(change["new_text"], reference_audio_path=reference_path)
            finally:
                if reference_path:
                    drop_temp_standalone(reference_path)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=hear_temp_directory()) as tmp:
                tmp.write(tts_bytes)
                tts_path = tmp.name
            register_temp_standalone(tts_path, purpose="tts_segment", track_id=track_id)
            tts_waveform, tts_sr = torchaudio.load(tts_path)
            drop_temp_standalone(tts_path)
            if tts_sr != self.TARGET_SR:
                tts_waveform = F_audio.resample(tts_waveform, tts_sr, self.TARGET_SR)
            merged = self._splice_segment(merged, tts_waveform, start_sample, end_sample)
        peak = merged.abs().max().item()
        if peak > 0.99:
            merged = merged * (0.99 / peak)
        out_path = save_as_mp3(merged, self.TARGET_SR, track_id=track_id, purpose="reconstruct_mp3")
        duration = merged.shape[1] / self.TARGET_SR
        b2_key = f"reconstructed/{track_id}/{os.urandom(8).hex()}.mp3"
        loop = asyncio.get_event_loop()
        audio_url = await loop.run_in_executor(None, storage.upload_file, out_path, b2_key)
        drop_temp_standalone(out_path)
        return SynthesisResult(
            b2_key=b2_key,
            audio_url=audio_url,
            duration=round(duration, 3),
        )

    def _normalize_changes(self, changes: list) -> list[dict]:
        normalized: list[dict] = []
        for item in changes or []:
            if isinstance(item, dict):
                start_raw = item.get("segment_start", item.get("start"))
                end_raw = item.get("segment_end", item.get("end"))
                text_raw = item.get("new_text", item.get("text"))
            else:
                start_raw = getattr(item, "segment_start", getattr(item, "start", None))
                end_raw = getattr(item, "segment_end", getattr(item, "end", None))
                text_raw = getattr(item, "new_text", getattr(item, "text", None))
            try:
                start = float(start_raw)
                end = float(end_raw)
            except Exception:
                continue
            text = str(text_raw or "").strip()
            if end <= start or not text:
                continue
            normalized.append(
                {
                    "segment_start": start,
                    "segment_end": end,
                    "new_text": text,
                }
            )
        return normalized

    async def rebuild_track_audio(
        self,
        original_audio_path: str,
        edited_transcript: str,
        track_id: str,
        job_id: str,
        original_transcript: str = "",
    ) -> SynthesisResult:
        original_waveform, orig_sr = torchaudio.load(original_audio_path)
        if orig_sr != self.TARGET_SR:
            original_waveform = F_audio.resample(original_waveform, orig_sr, self.TARGET_SR)

        rebuilt_bytes = await self._synthesize_higgs(
            edited_transcript,
            reference_audio_path=original_audio_path,
        )
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=hear_temp_directory()) as tmp:
            tmp.write(rebuilt_bytes)
            rebuilt_path = tmp.name
        register_temp_standalone(
            rebuilt_path,
            purpose="rebuilt_track_intermediate",
            job_id=job_id,
            track_id=track_id,
        )

        rebuilt_waveform, rebuilt_sr = torchaudio.load(rebuilt_path)
        drop_temp_standalone(rebuilt_path)
        if rebuilt_sr != self.TARGET_SR:
            rebuilt_waveform = F_audio.resample(rebuilt_waveform, rebuilt_sr, self.TARGET_SR)

        start_sample, end_sample = self._detect_speech_bounds(original_waveform, self.TARGET_SR, original_transcript)
        merged = self._splice_segment(original_waveform, rebuilt_waveform, start_sample, end_sample)
        peak = merged.abs().max().item()
        if peak > 0.99:
            merged = merged * (0.99 / peak)

        out_path = save_as_mp3(
            merged, self.TARGET_SR, job_id=job_id, track_id=track_id, purpose="rebuilt_track_mp3"
        )
        duration = merged.shape[1] / self.TARGET_SR
        b2_key = f"rebuild/{track_id}/{job_id}.mp3"
        loop = asyncio.get_event_loop()
        audio_url = await loop.run_in_executor(None, storage.upload_file, out_path, b2_key)
        drop_temp_standalone(out_path)

        return SynthesisResult(
            b2_key=b2_key,
            audio_url=audio_url,
            duration=round(duration, 3),
        )

    async def _synthesize_higgs(self, text: str, reference_audio_path: str | None = None) -> bytes:
        if not text.strip():
            text = " "
        if not settings.HIGGS_AUDIO_ENABLED or not self._higgs_available:
            raise RuntimeError(
                "Audio rebuild/reconstruct requires HIGGS_AUDIO_ENABLED=true and the Higgs module "
                f"installed under {settings.HIGGS_AUDIO_REPO_DIR}"
            )
        
        prompt_name = None
        prompts_dir = os.path.join((settings.HIGGS_AUDIO_REPO_DIR or "/workspace/higgs-audio").strip(), "examples", "voice_prompts")
        if reference_audio_path:
            import uuid
            import shutil
            from app.services.registry import transcriber
            with open(reference_audio_path, "rb") as f:
                ref_bytes = f.read()
            t_res = await transcriber.transcribe(ref_bytes)
            ref_text = t_res.get("transcript", "").strip() or " "
            prompt_name = f"tmp_{uuid.uuid4().hex}"
            os.makedirs(prompts_dir, exist_ok=True)
            shutil.copy2(reference_audio_path, os.path.join(prompts_dir, f"{prompt_name}.wav"))
            with open(os.path.join(prompts_dir, f"{prompt_name}.txt"), "w") as f:
                f.write(ref_text)

        module_name = (settings.HIGGS_AUDIO_MODULE or "higgs_audio").strip()
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                self._run_local_higgs,
                text,
                reference_audio_path,
                prompt_name,
            )
        finally:
            if prompt_name:
                try:
                    os.remove(os.path.join(prompts_dir, f"{prompt_name}.wav"))
                    os.remove(os.path.join(prompts_dir, f"{prompt_name}.txt"))
                except Exception:
                    pass

    def _run_local_higgs(self, text: str, reference_audio_path: str | None = None, prompt_name: str | None = None) -> bytes:
        module_name = (settings.HIGGS_AUDIO_MODULE or "higgs_audio").strip()
        try:
            return self._run_boson_engine(text, prompt_name)
        except Exception as exc:
            raise RuntimeError(f"Local {module_name} module did not return audio bytes ({exc})") from exc

    def _run_boson_engine(self, text: str, prompt_name: str | None = None) -> bytes:
        import subprocess
        import tempfile
        from app.core.hear_temp import hear_temp_directory

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=hear_temp_directory()) as out_tmp:
            out_path = out_tmp.name
        
        script_path = os.path.join((settings.HIGGS_AUDIO_REPO_DIR or "/workspace/higgs-audio").strip(), "examples", "generation.py")
        cmd = [
            "python3", script_path,
            "--transcript", text,
            "--temperature", "0.3",
            "--out_path", out_path
        ]
        if prompt_name:
            cmd.extend(["--ref_audio", prompt_name])
        
        env = os.environ.copy()
        
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"generation.py failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
            with open(out_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def _wav_bytes_from_audio(self, audio: np.ndarray, sampling_rate: int) -> bytes:
        pcm = np.clip(audio.astype(np.float32), -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sampling_rate))
            wf.writeframes(pcm_i16.tobytes())
        return buffer.getvalue()

    def _ensure_higgs_module_available(self) -> str:
        module_name = (settings.HIGGS_AUDIO_MODULE or "higgs_audio").strip()
        self._inject_higgs_repo_path()
        if self._is_higgs_module_ready(module_name):
            return module_name
        raise RuntimeError(
            f"{module_name} module is not available from HIGGS_AUDIO_REPO_DIR="
            f"{settings.HIGGS_AUDIO_REPO_DIR}. Install Higgs in that path and restart."
        )

    def _is_higgs_module_ready(self, module_name: str) -> bool:
        if self._safe_find_spec(module_name) is None:
            return False
        if module_name == "boson_multimodal":
            serve_nested = self._safe_find_spec("boson_multimodal.serve.serve_engine") is not None
            serve_flat = self._safe_find_spec("boson_multimodal.serve_engine") is not None
            types_ok = self._safe_find_spec("boson_multimodal.data_types") is not None
            return (serve_nested or serve_flat) and types_ok
        return True

    def _safe_find_spec(self, module_name: str):
        try:
            return importlib.util.find_spec(module_name)
        except Exception:
            return None

    def _inject_higgs_repo_path(self):
        repo_dir = Path((settings.HIGGS_AUDIO_REPO_DIR or "/workspace/higgs-audio").strip())
        if repo_dir.exists():
            repo_path = str(repo_dir)
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
            importlib.invalidate_caches()

    def _export_reference_clip(
        self,
        waveform: torch.Tensor,
        start_sample: int,
        end_sample: int,
        *,
        track_id: str | None = None,
    ) -> str:
        start_sample = max(0, start_sample)
        end_sample = max(start_sample + 1, min(end_sample, waveform.shape[1]))
        clip = waveform[:, start_sample:end_sample].detach().cpu()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=hear_temp_directory()) as tmp:
            ref_path = tmp.name
        torchaudio.save(ref_path, clip, self.TARGET_SR)
        register_temp_standalone(
            ref_path, purpose="reference_clip", track_id=track_id
        )
        return ref_path

    def _detect_speech_bounds(self, waveform: torch.Tensor, sr: int, original_transcript: str) -> tuple[int, int]:
        mono = waveform.mean(dim=0).cpu().numpy()
        if mono.size < 2:
            return 0, waveform.shape[1]
        frame = max(256, int(sr * 0.02))
        kernel = np.ones(frame) / frame
        envelope = np.convolve(np.abs(mono), kernel, mode="same")
        threshold = max(float(np.percentile(envelope, 65) * 0.45), 1e-4)
        active = np.where(envelope >= threshold)[0]
        if active.size == 0:
            return 0, waveform.shape[1]
        pad = int(sr * (0.15 if original_transcript.strip() else 0.05))
        start = max(0, int(active[0]) - pad)
        end = min(waveform.shape[1], int(active[-1]) + pad)
        if end <= start:
            return 0, waveform.shape[1]
        return start, end

    def _splice_segment(
        self,
        original_waveform: torch.Tensor,
        replacement_waveform: torch.Tensor,
        start_sample: int,
        end_sample: int,
    ) -> torch.Tensor:
        if replacement_waveform.shape[0] != original_waveform.shape[0]:
            replacement_waveform = replacement_waveform.mean(dim=0, keepdim=True).expand(original_waveform.shape[0], -1)

        before = original_waveform[:, :start_sample]
        after = original_waveform[:, end_sample:]
        cross_len = min(int(0.02 * self.TARGET_SR), before.shape[1], replacement_waveform.shape[1], after.shape[1] if after.shape[1] else replacement_waveform.shape[1])
        if cross_len > 0:
            fade_out = torch.linspace(1.0, 0.0, cross_len).unsqueeze(0)
            fade_in = torch.linspace(0.0, 1.0, cross_len).unsqueeze(0)
            before[:, -cross_len:] = (before[:, -cross_len:] * fade_out) + (replacement_waveform[:, :cross_len] * fade_in)
            tail_out = torch.linspace(1.0, 0.0, cross_len).unsqueeze(0)
            tail_in = torch.linspace(0.0, 1.0, cross_len).unsqueeze(0)
            if after.shape[1] >= cross_len:
                replacement_waveform[:, -cross_len:] = (replacement_waveform[:, -cross_len:] * tail_out) + (after[:, :cross_len] * tail_in)
                after = after[:, cross_len:]
        return torch.cat([before, replacement_waveform[:, cross_len:], after], dim=1)
