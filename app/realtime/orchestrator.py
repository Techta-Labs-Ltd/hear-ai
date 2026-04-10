import time

from app.realtime.broadcaster import manager


class PipelineOrchestrator:
    def __init__(self, transcriber, enhancer, categorizer):
        self.transcriber = transcriber
        self.enhancer = enhancer
        self.categorizer = categorizer

    async def process_and_stream(
        self,
        job_id: str,
        recording_id: str,
        audio_bytes: bytes,
        input_path: str,
    ):
        await manager.broadcast(job_id, {
            "event": "pipeline_started",
            "job_id": job_id,
            "recording_id": recording_id,
            "timestamp": time.time(),
        })

        transcription_result = {}
        async for chunk in self.transcriber.stream(audio_bytes):
            if chunk["type"] == "segment":
                await manager.broadcast(job_id, {
                    "event": "transcript_segment",
                    "job_id": job_id,
                    "segment": chunk,
                    "timestamp": time.time(),
                })
                transcription_result.setdefault("segments", []).append(chunk)
            elif chunk["type"] == "done":
                await manager.broadcast(job_id, {
                    "event": "transcript_complete",
                    "job_id": job_id,
                    "language": chunk.get("language"),
                    "timestamp": time.time(),
                })
            elif chunk["type"] == "error":
                await manager.broadcast(job_id, {
                    "event": "transcript_error",
                    "job_id": job_id,
                    "message": chunk["message"],
                    "timestamp": time.time(),
                })

        await manager.broadcast(job_id, {
            "event": "enhancement_started",
            "job_id": job_id,
            "timestamp": time.time(),
        })

        try:
            result = await self.enhancer.enhance(
                job_id=job_id,
                input_path=input_path,
                recording_id=recording_id,
            )
            await manager.broadcast(job_id, {
                "event": "enhancement_complete",
                "job_id": job_id,
                "enhanced_url": result.enhanced_url,
                "quality_score": result.quality_score,
                "snr_db": result.snr_db,
                "lufs": result.lufs,
                "timestamp": time.time(),
            })
        except Exception as e:
            await manager.broadcast(job_id, {
                "event": "enhancement_error",
                "job_id": job_id,
                "message": str(e),
                "timestamp": time.time(),
            })

        if transcription_result.get("segments"):
            full_transcript = " ".join(s.get("text", "") for s in transcription_result["segments"])
            cat_result = await self.categorizer.categorize(
                transcript=full_transcript,
                segments=transcription_result["segments"],
            )
            await manager.broadcast(job_id, {
                "event": "categorization_complete",
                "job_id": job_id,
                "tags": cat_result["tags"],
                "categories": cat_result["categories"],
                "sentiment": cat_result["sentiment"],
                "timestamp": time.time(),
            })

        await manager.broadcast(job_id, {
            "event": "pipeline_complete",
            "job_id": job_id,
            "recording_id": recording_id,
            "timestamp": time.time(),
        })
