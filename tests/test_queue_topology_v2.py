from hear.queue.topology import RabbitMQTopology
from hear.runtime.roles import WorkerRole


class TestRabbitMQTopology:
    def test_four_job_queues_and_magic_clean_profiles(self):
        topology = RabbitMQTopology()
        assert topology.binding(WorkerRole.PIPELINE).queue == "hear.ai.pipeline.v1"
        assert topology.binding(WorkerRole.TRANSCRIPTION).queue == "hear.ai.transcription.v1"
        assert topology.binding(WorkerRole.RECONSTRUCTION).queue == "hear.ai.reconstruction.v1"
        assert (
            topology.binding(WorkerRole.MAGIC_CLEAN_NATURAL).routing_key
            == "magic_clean.natural"
        )