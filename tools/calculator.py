# /home/taylo/mcp-agent/tools/calculator.py
from llama_cpp import Llama
import logging

logger = logging.getLogger("mcp_server")

LLM = None
MODEL_PATH = ""

def initialize_llm(path: str): # Add this function
    global MODEL_PATH
    MODEL_PATH = path

def calculate(query: str) -> str:
    global LLM
    if not MODEL_PATH: # Add this check
        return {"error": "Calculator model path not initialized."}    
    if not LLM:
        try:
            LLM = Llama(model_path=MODEL_PATH, n_ctx=1024, verbose=False)
            logger.info(f"Calculator model loaded: {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load calculator model: {e}")
            return "Error: Local calculator model is not available."

    prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    
    try:
        output = LLM(prompt, max_tokens=256, temperature=0.1, stop=["<|im_end|>"])
        return output['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"LLM calculation failed: {e}")
        return "Error: Failed to perform calculation."