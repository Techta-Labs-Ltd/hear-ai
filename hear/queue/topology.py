from dataclasses import dataclass

from hear.contracts.jobs import AttemptEnvelope, JobType, MagicCleanProfile
from hear.runtime.roles import WorkerRole


@dataclass(frozen=True, slots=True)
class QueueBinding:
    queue: str
    routing_key: str
    retry_queue: str
    dead_queue: str
    dead_routing_key: str


class RabbitMQTopology:
    def __init__(
        self,
        *,
        exchange: str = "hear.ai.jobs",
        queue_prefix: str = "hear.ai",
        version: int = 2,
        max_queue_messages: int = 1000,
        retry_delay_ms: int = 5000,
        max_retries: int = 5,
    ) -> None:
        self.exchange = exchange
        self.retry_exchange = f"{exchange}.retry"
        self.dead_exchange = f"{exchange}.dead"
        self._queue_prefix = queue_prefix
        self._version = version
        self.max_queue_messages = max_queue_messages
        self.retry_delay_ms = retry_delay_ms
        self.max_retries = max_retries

    def binding(self, role: WorkerRole) -> QueueBinding:
        suffixes = {
            WorkerRole.PIPELINE: "pipeline",
            WorkerRole.TRANSCRIPTION: "transcription",
            WorkerRole.RECONSTRUCTION: "reconstruction",
            WorkerRole.MAGIC_CLEAN_NATURAL: "magic_clean.natural",
            WorkerRole.MAGIC_CLEAN_SAM_AUDIO: "magic_clean.sam_audio",
        }
        suffix = suffixes[role]
        queue = f"{self._queue_prefix}.{suffix}.v{self._version}"
        return QueueBinding(
            queue=queue,
            routing_key=suffix,
            retry_queue=f"{queue}.retry",
            dead_queue=f"{queue}.dead",
            dead_routing_key=f"{suffix}.dead",
        )

    def role_for(
        self,
        envelope: AttemptEnvelope,
        available_roles: set[WorkerRole] | None = None,
    ) -> WorkerRole | None:
        if envelope.job_type == JobType.PIPELINE:
            return WorkerRole.PIPELINE
        if envelope.job_type == JobType.TRANSCRIPTION:
            if available_roles is not None and WorkerRole.PIPELINE in available_roles:
                return WorkerRole.PIPELINE
            return WorkerRole.TRANSCRIPTION
        if envelope.job_type == JobType.RECONSTRUCTION:
            return WorkerRole.RECONSTRUCTION
        roles = {
            MagicCleanProfile.NATURAL.value: WorkerRole.MAGIC_CLEAN_NATURAL,
            MagicCleanProfile.SAM_AUDIO.value: WorkerRole.MAGIC_CLEAN_SAM_AUDIO,
        }
        return roles.get(str(envelope.options.get("profile") or ""))

    def queue_arguments(self, binding: QueueBinding) -> dict[str, str | int]:
        return {
            "x-queue-type": "quorum",
            "x-max-length": self.max_queue_messages,
            "x-overflow": "reject-publish-dlx",
            "x-dead-letter-exchange": self.dead_exchange,
            "x-dead-letter-routing-key": binding.dead_routing_key,
            "x-delivery-limit": self.max_retries + 1,
        }

    def retry_queue_arguments(self, binding: QueueBinding) -> dict[str, str | int]:
        return {
            "x-message-ttl": self.retry_delay_ms,
            "x-dead-letter-exchange": self.exchange,
            "x-dead-letter-routing-key": binding.routing_key,
        }

    def dead_queue_arguments(self) -> dict[str, str | int]:
        return {
            "x-queue-type": "quorum",
            "x-max-length": self.max_queue_messages,
            "x-overflow": "drop-head",
        }
