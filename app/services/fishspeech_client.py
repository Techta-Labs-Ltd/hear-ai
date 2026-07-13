import logging

from ray import serve

logger = logging.getLogger(__name__)


class FishSpeechClient:
    def __init__(self):
        self._handle = None

    def _get_handle(self):
        if self._handle is None:
            self._handle = serve.get_deployment_handle("fish_speech", "default")
        return self._handle

    async def generate_speech(
        self,
        text: str,
        max_new_tokens: int = 1024,
        references: list[dict] | None = None,
        reference_id: str | None = None,
        language: str = "en",
    ) -> bytes:
        if not text:
            raise RuntimeError("Fish Speech requires text input")
        logger.info("Fish Speech TTS: text='%.60s' max_tokens=%d lang=%s", text, max_new_tokens, language)
        try:
            handle = self._get_handle()
            audio_bytes = await handle.generate_speech.remote(
                text, max_new_tokens, references, reference_id, language
            )
            if len(audio_bytes) < 100:
                raise RuntimeError("TTS generation failed")
            logger.info("Fish Speech returned %d bytes", len(audio_bytes))
            return audio_bytes
        except Exception as e:
            raise RuntimeError("TTS generation failed: %s" % str(e)) from e
