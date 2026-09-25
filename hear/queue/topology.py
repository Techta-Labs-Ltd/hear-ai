from dataclasses import dataclass

from hear.runtime.roles import WorkerRole


@dataclass(frozen=True, slots=True)
class QueueBinding:
    queue: str
    routing_key: str


class RabbitMQTopology:
    def __init__(
        self,
        *,
        exchange: str = "hear.ai.jobs",
        queue_prefix: str = "hear.ai",
        version: int = 1,
    ) -> None:
        self.exchange = exchange
        self._queue_prefix = queue_prefix
        self._version = version

    def binding(self, role: WorkerRole) -> QueueBinding:
        suffixes = {
            WorkerRole.PIPELINE: ("pipeline", "pipeline"),
            WorkerRole.TRANSCRIPTION: ("transcription", "transcription"),
            WorkerRole.RECONSTRUCTION: ("reconstruction", "reconstruction"),
            WorkerRole.MAGIC_CLEAN_NATURAL: ("magic_clean.natural", "magic_clean.natural"),
            WorkerRole.MAGIC_CLEAN_VOICE_FOCUS: (
                "magic_clean.voice_focus",
                "magic_clean.voice_focus",
            ),
            WorkerRole.MAGIC_CLEAN_MUSIC_ATMOSPHERE: (
                "magic_clean.music_atmosphere",
                "magic_clean.music_atmosphere",
            ),
            WorkerRole.MAGIC_CLEAN_STEM_MIX: ("magic_clean.stem_mix", "magic_clean.stem_mix"),
        }
        queue_suffix, routing_key = suffixes[role]
        return QueueBinding(
            queue=f"{self._queue_prefix}.{queue_suffix}.v{self._version}",
            routing_key=routing_key,
        )