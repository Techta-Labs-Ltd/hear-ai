from hear.deployments.audio_cleanup import AudioCleanupDeployment
from hear.deployments.fish_speech import FishSpeechDeployment
from hear.deployments.gateway import GrpcGateway
from hear.deployments.language_models import LLMDeployment, SmallModelsDeployment
from hear.deployments.magic_clean import MagicCleanDeployment
from hear.deployments.transcription import TranscriptionDeployment
from hear.orchestrator import Orchestrator


def build_application():
    """Build the complete Serve graph without side effects during import."""
    small_models = SmallModelsDeployment.bind()
    transcription = TranscriptionDeployment.bind()
    llm = LLMDeployment.bind()
    fish_speech = FishSpeechDeployment.bind()
    audio_cleanup = AudioCleanupDeployment.bind()
    magic_clean = MagicCleanDeployment.bind()

    orchestrator = Orchestrator.bind(
        transcription,
        llm,
        fish_speech,
        small_models,
        magic_clean,
    )

    return GrpcGateway.bind(
        orchestrator,
        audio_cleanup,
        transcription,
        fish_speech,
        small_models,
        llm,
    )
