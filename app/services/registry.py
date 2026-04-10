from app.services.transcriber import TranscriptionService
from app.services.enhancer import AudioEnhancer
from app.services.categorizer import CategorizationService
from app.services.moderator import ModerationService
from app.services.synthesizer import SpeechSynthesizer
from app.realtime.orchestrator import PipelineOrchestrator
from app.worker import PipelineWorker

transcriber = TranscriptionService()
enhancer = AudioEnhancer()
categorizer = CategorizationService()
moderator = ModerationService()
synthesizer = SpeechSynthesizer()
orchestrator = PipelineOrchestrator(transcriber, enhancer, categorizer)
worker = PipelineWorker(enhancer, transcriber, categorizer, moderator)
