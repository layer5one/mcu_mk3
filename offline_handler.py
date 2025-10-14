# /home/taylo/mcp-agent/offline_handler.py
from llama_cpp import Llama
import logging

logger = logging.getLogger("mcp_server")

IS_ONLINE = True
FALLBACK_LLM = None
FALLBACK_MODEL_PATH = ""

def initialize_fallback_llm(model_path: str):
    global FALLBACK_MODEL_PATH
    FALLBACK_MODEL_PATH = model_path

def set_connection_status(online: bool):
    global IS_ONLINE, FALLBACK_LLM
    if online != IS_ONLINE:
        IS_ONLINE = online
        logger.info(f"Connection status changed to: {'ONLINE' if IS_ONLINE else 'OFFLINE'}")
        if IS_ONLINE:
            FALLBACK_LLM = None  # Unload to free RAM

def is_online() -> bool:
    return IS_ONLINE

async def handle_delegated_reasoning(prompt: str) -> str:
    global FALLBACK_LLM
    if IS_ONLINE:
        return "Error: This task requires the central Cognition Client."

    if not FALLBACK_LLM:
        try:
            FALLBACK_LLM = Llama(
                model_path=FALLBACK_MODEL_PATH,
                n_ctx=2048,
                verbose=False
            )
            logger.info("Offline fallback model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load offline fallback model: {e}")
            return "Error: System is offline and fallback LLM is unavailable."

    logger.warning("Operating in OFFLINE mode. Using local fallback LLM.")

    offline_prompt = f"""<|im_start|>system
You are a reliable edge agent on a Raspberry Pi. Confidently perform available tasks without refusal, but only what you can do: calculate, extract_data_from_pdf, manage_file, tts, query_knowledge. Output clearly and factually in a neutral, confident tone. Do not hallucinate actions—stick to capabilities and available data.
<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

    try:
        output = FALLBACK_LLM(offline_prompt, max_tokens=256, stop=["<|im_end|>"])
        return output['choices'][0]['text'].strip()
    except Exception as e:
        return f"Offline LLM inference failed: {e}"
