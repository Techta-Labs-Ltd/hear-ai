import sys
sys.path.append("/workspace/hear-ai")
from app.services.synthesizer import SpeechSynthesizer

synth = SpeechSynthesizer()
synth.load()
try:
    synth._run_local_higgs("Hello world")
except Exception as e:
    import traceback
    traceback.print_exc()
