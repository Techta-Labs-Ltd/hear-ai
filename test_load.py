import sys
sys.path.append("/workspace/hear-ai")
from app.services.synthesizer import SpeechSynthesizer

synth = SpeechSynthesizer()
synth.load()
print("Load called successfully. is_loaded=", synth.is_loaded, "higgs_available=", synth.higgs_available)
