from importlib import import_module

from hear.config import Settings, settings


class ApplicationBuilder:
    def __init__(self, runtime: Settings):
        self._runtime = runtime

    @staticmethod
    def _bind(module_name: str, class_name: str, *dependencies):
        deployment = getattr(import_module(module_name), class_name)
        return deployment.bind(*dependencies)

    def build(self):
        small_models = self._bind("hear.deployments.language_models", "SmallModelsDeployment")
        transcription = self._bind("hear.deployments.transcription", "TranscriptionDeployment")
        llm = (
            self._bind("hear.deployments.language_models", "LLMDeployment")
            if self._runtime.QWEN_LLM_ENABLED
            else None
        )
        fish_speech = (
            self._bind("hear.deployments.fish_speech", "FishSpeechDeployment")
            if self._runtime.FISH_SPEECH_TTS_ENABLED
            else None
        )
        audio_cleanup = self._bind("hear.deployments.audio_cleanup", "AudioCleanupDeployment")
        magic_clean = self._bind("hear.deployments.magic_clean", "MagicCleanDeployment")
        orchestrator = self._bind(
            "hear.orchestrator",
            "Orchestrator",
            transcription,
            llm,
            fish_speech,
            small_models,
            magic_clean,
        )
        return self._bind(
            "hear.deployments.gateway",
            "GrpcGateway",
            orchestrator,
            audio_cleanup,
            transcription,
            fish_speech,
            small_models,
            llm,
        )


def build_application(runtime: Settings | None = None):
    return ApplicationBuilder(runtime if runtime is not None else settings).build()
