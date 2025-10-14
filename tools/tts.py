# /home/taylo/mcp-agent/tools/tts.py
import kokoro
import sounddevice as sd
import numpy as np
import subprocess
import os
import logging

logger = logging.getLogger("mcp_server")

PIPELINE = None

def initialize_tts():
    global PIPELINE
    PIPELINE = kokoro.KPipeline(lang_code='a')  # American English
    logger.info("Kokoro TTS initialized with neutral voice.")

def tts(text: str, mood: str = "neutral") -> dict:
    if not PIPELINE:
        return {"error": "TTS not initialized."}

    try:
        # Synthesize (use 'heart' for prosody/neutral)
        wav, sample_rate = PIPELINE.synthesize(text, voice='heart')
        # Play with underrun fallback
        sd.default.blocksize = 2048
        sd.default.latency = 'high'
        sd.play(wav, sample_rate, blocking=True)
        sd.wait()
        return {"status": "TTS played successfully."}
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        temp_out = 'temp_tts.wav'
        np.save(temp_out, wav)
        subprocess.run(['aplay', temp_out])
        os.remove(temp_out)
        return {"error": str(e)}
