# /home/taylo/mcp-agent/mcp_server.py
import os
import asyncio
import json
import requests
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import paho.mqtt.client as mqtt
import tenacity
from dotenv import load_dotenv
import concurrent.futures
import psutil
from faster_whisper import WhisperModel
import threading
import queue
import numpy as np
import pyaudio
import soundfile as sf
from scipy.signal import butter, lfilter

from memory_manager import LocalMemoryManager
from offline_handler import set_connection_status, is_online, initialize_fallback_llm
from tools.calculator import calculate
from tools.pdf_extractor import extract_data_from_pdf
from tools.file_manager import manage_file
from tools.tts import tts, initialize_tts
from tools.query_knowledge import query_knowledge
from tools.obd import obd_query  # Future stub

from tools import pdf_extractor, calculator

load_dotenv()  # Optional .env load

# --- Configuration ---
DEVICE_ID = os.getenv("DEVICE_ID", "RASPI5")
PI_IP = os.getenv("PI_IP", "192.168.1.70")
PI_PORT = int(os.getenv("PI_PORT", 8000))
COGNITION_CLIENT_URL = os.getenv("COGNITION_CLIENT_URL", "http://192.168.1.69:8001")
MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "192.168.1.69")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", 1883))

BASE_PROJECT_PATH = "/home/taylo/mcp-agent"
EXTRACT_MODEL_PATH = os.path.join(BASE_PROJECT_PATH, "models/LFM2-350M-Extract-Q4_K_M.gguf")
MATH_MODEL_PATH = os.path.join(BASE_PROJECT_PATH, "models/LFM2-350M-Math-Q4_K_M.gguf")
FALLBACK_MODEL_PATH = os.path.join(BASE_PROJECT_PATH, "models/LFM2-1.2B-Tool-GGUF-Q4_K_M.gguf")

CONTROL_TOPIC = f"agents/control/{DEVICE_ID}"
STATUS_TOPIC = f"agents/status/{DEVICE_ID}"

STT_ENABLED = True  # Toggle for voice input
WHISPER_MODEL = "base.en"  # Lite STT
SAMPLE_RATE = 16000
AUDIO_CHUNK_SEC = 5

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp_server")

# --- Tool Definitions ---
AVAILABLE_TOOLS = {
    "calculate": {
        "description": "Solves a mathematical problem expressed in natural language.",
        "schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "handler": calculate
    },
    "extract_data_from_pdf": {
        "description": "Extracts structured data from a PDF file based on a JSON schema.",
        "schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "schema": {"type": "object"}}, "required": ["file_path", "schema"]},
        "handler": extract_data_from_pdf
    },
    "manage_file": {
        "description": "Manipulates local files: create, edit, delete, read, or list. Use 'operation' to specify action.",
        "schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["create", "edit", "delete", "read", "list"]},
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["operation", "path"]
        },
        "handler": manage_file
    },
    "tts": {
        "description": "Synthesizes and plays text as speech using lightweight TTS.",
        "schema": {"type": "object", "properties": {"text": {"type": "string"}, "mood": {"type": "string"}}, "required": ["text"]},
        "handler": tts
    },
    "query_knowledge": {
        "description": "Queries local embedded knowledge base for facts (e.g., from PDFs).",
        "schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "handler": query_knowledge
    },
    "obd_query": {
        "description": "Queries car OBD2 data (read-only).",
        "schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "handler": obd_query
    },
    "get_metrics": {
        "description": "Gets device metrics like CPU/temp for monitoring.",
        "schema": {"type": "object", "properties": {}},
        "handler": lambda: {"cpu": psutil.cpu_percent(), "temp": psutil.sensors_temperatures().get("cpu-thermal", [{}])[0].get("current", 0)}
    },
}

# --- Initialization ---
app = FastAPI()
memory_manager = LocalMemoryManager(device_id=DEVICE_ID, persist_directory=os.path.join(BASE_PROJECT_PATH, "chroma_db"))

class ToolExecutionRequest(BaseModel):
    params: dict

class MemoryAddRequest(BaseModel):
    text: str
    metadata: dict = None

class MemoryQueryRequest(BaseModel):
    query_text: str
    n_results: int = 5

# --- MQTT Client ---
@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=5, max=60), stop=tenacity.stop_after_attempt(10))
def connect_mqtt():
    mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Connected to MQTT Broker!")
        client.subscribe(CONTROL_TOPIC)
        client.publish(STATUS_TOPIC, json.dumps({"status": "online", "device_id": DEVICE_ID}), retain=True)
        set_connection_status(True)
    else:
        logger.error(f"Failed to connect to MQTT, return code {rc}")
        set_connection_status(False)

def on_disconnect(client, userdata, rc):
    logger.warning("Disconnected from MQTT Broker.")
    set_connection_status(False)
    connect_mqtt()  # Retry connect

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload)
        if "command" in payload:
            if payload["command"] == "sync_memory":
                ids, _ = memory_manager.get_unsynced_memories()
                # Sync logic...
                logger.info("Manual memory sync triggered.")
            elif payload["command"] == "add_knowledge":
                memory_manager.embed_knowledge(payload.get("file_path", ""))
            logger.info(f"Handled command: {payload}")
    except Exception as e:
        logger.error(f"MQTT message error: {e}")

mqtt_client = mqtt.Client(client_id=DEVICE_ID)
mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.on_message = on_message
lwt_payload = json.dumps({"status": "offline", "device_id": DEVICE_ID})
mqtt_client.will_set(STATUS_TOPIC, payload=lwt_payload, qos=1, retain=True)

# --- Enhanced VAD class ---
class AdaptiveVAD:
    def __init__(self, initial_threshold=0.01, alpha=0.99):
        self.threshold = initial_threshold
        self.alpha = alpha
        self.baseline_energy = 0.001  # Calibrate on init

    def calibrate(self, chunk, samples=3):
        energies = [np.max(np.abs(chunk)) for _ in range(samples)]  # Simplified for example
        self.baseline_energy = np.mean(energies) * 1.5  # Buffer above noise floor

    def is_speech(self, chunk):
        energy = np.max(np.abs(chunk))
        if energy < self.threshold:
            return False
        # Spectral check: FFT, check voice band ratio
        fft = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1/SAMPLE_RATE)
        voice_mask = (freqs >= 500) & (freqs <= 3000)  # Bumped lowcut for car rumble
        voice_energy = np.sum(fft[voice_mask]**2)
        total_energy = np.sum(fft**2) + 1e-10
        if voice_energy / total_energy < 0.2:  # <20% in voice band? Likely noise
            return False
        # Adapt threshold
        self.threshold = self.alpha * self.threshold + (1 - self.alpha) * energy
        return True

vad = AdaptiveVAD()

# Butter bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)

# STT listener thread
def stt_listener(q):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=1024)
    logger.info("STT listening started.")
    while True:
        data = stream.read(SAMPLE_RATE * AUDIO_CHUNK_SEC)
        chunk = np.frombuffer(data, dtype=np.int16) / 32768.0
        filtered_chunk = apply_filter(chunk, 500, 3000, SAMPLE_RATE)  # Voice band filter
        if vad.is_speech(filtered_chunk):
            temp_wav = "temp_input.wav"
            sf.write(temp_wav, filtered_chunk, SAMPLE_RATE)
            if os.path.exists(temp_wav):
                q.put(temp_wav)

# Background STT processing task
async def stt_background():
    if not STT_ENABLED:
        return
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    q = queue.Queue()
    listener_thread = threading.Thread(target=stt_listener, args=(q,))
    listener_thread.start()

    # Calibrate VAD (5s ambient)
    logger.info("Calibrating STT VAD... (5s silence)")
    p_temp = pyaudio.PyAudio()
    stream_temp = p_temp.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=1024)
    for _ in range(5):
        data_temp = stream_temp.read(SAMPLE_RATE)
        chunk_temp = np.frombuffer(data_temp, dtype=np.int16) / 32768.0
        vad.calibrate(chunk_temp)
    stream_temp.close()
    p_temp.terminate()

    while True:
        try:
            audio_file = q.get(timeout=1)
            if not os.path.exists(audio_file):
                continue
            segments, _ = whisper.transcribe(audio_file, beam_size=5, language="en")
            transcript = " ".join([s.text for s in segments if s.avg_logprob > -1.0]).strip()
            os.remove(audio_file)
            if not transcript or len(transcript) < 5 or transcript.strip('. ') == '':
                continue
            if "hey mcp" not in transcript.lower():  # Simple wake-word filter
                continue

            logger.info(f"STT transcript: {transcript}")
            # Send to central
            try:
                resp = requests.post(f"{COGNITION_CLIENT_URL}/process_query", json={"query": transcript}, timeout=10)
                if resp.status_code == 200:
                    response_text = resp.json().get("response")
                    if response_text:
                        # Call local TTS tool
                        await app.execute_tool("tts", ToolExecutionRequest(params={"text": response_text}))
            except Exception as e:
                logger.error(f"Central query failed: {e}")
        except queue.Empty:
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"STT loop error: {e}")

# --- Background Tasks ---
# ... existing register_with_cognition_client and sync_memory_periodically ...

async def prune_memory_periodically():
    while True:
        await asyncio.sleep(86400)  # Daily
        memory_manager.delete_old_memories()

# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("MCP Server starting up...")
    pdf_extractor.initialize_llm(EXTRACT_MODEL_PATH)
    calculator.initialize_llm(MATH_MODEL_PATH)
    initialize_fallback_llm(FALLBACK_MODEL_PATH)
    initialize_tts()
    connect_mqtt()
    mqtt_client.loop_start()
    asyncio.create_task(register_with_cognition_client())
    asyncio.create_task(sync_memory_periodically())
    asyncio.create_task(prune_memory_periodically())
    asyncio.create_task(stt_background())
    memory_manager.embed_knowledge("/path/to/car_fsm.pdf")  # Tweak to your FSM PDF path for KB
    logger.info("Startup complete.")

@app.on_event("shutdown")
def shutdown_event():
    mqtt_client.publish(STATUS_TOPIC, json.dumps({"status": "offline", "device_id": DEVICE_ID}), retain=True)
    mqtt_client.loop_stop()
    mqtt_client.disconnect()

# --- API Endpoints ---
@app.get("/status")
async def get_status():
    return {"device_id": DEVICE_ID, "tools": list(AVAILABLE_TOOLS.keys()), "online": is_online()}

@app.post("/tools/{tool_name}")
async def execute_tool(tool_name: str, request: ToolExecutionRequest):
    if tool_name not in AVAILABLE_TOOLS:
        raise HTTPException(status_code=404, detail="Tool not found")
    logger.info(f"Executing tool '{tool_name}' with params: {request.params}")
    mqtt_client.publish(STATUS_TOPIC, json.dumps({"event": "tool_started", "tool": tool_name}))
    try:
        handler = AVAILABLE_TOOLS[tool_name]["handler"]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await asyncio.get_event_loop().run_in_executor(executor, handler, **request.params)
        mqtt_client.publish(STATUS_TOPIC, json.dumps({"event": "tool_finished", "tool": tool_name, "status": "success"}))
        return result
    except Exception as e:
        logger.error(f"Tool execution failed for {tool_name}: {e}", exc_info=True)
        mqtt_client.publish(STATUS_TOPIC, json.dumps({"event": "tool_finished", "tool": tool_name, "status": "failure", "error": str(e)}))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/add")
async def add_memory(request: MemoryAddRequest):
    success = memory_manager.add_memory(request.text, request.metadata)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to add memory.")
    return {"status": "success"}

@app.post("/memory/query")
async def query_memory(request: MemoryQueryRequest):
    results = memory_manager.query_memory(request.query_text, request.n_results)
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PI_PORT)
