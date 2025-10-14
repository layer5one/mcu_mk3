# /home/taylo/mcp-agent/tools/pdf_extractor.py
import fitz  # PyMuPDF
from llama_cpp import Llama
import json
import logging

logger = logging.getLogger("mcp_server")

LLM = None
MODEL_PATH = ""

def initialize_llm(path: str): # Add this function
    global MODEL_PATH
    MODEL_PATH = path

def extract_data_from_pdf(file_path: str, schema: dict) -> dict:
    global LLM
    if not MODEL_PATH: # Add this check
        return {"error": "PDF Extractor model path not initialized."}
    if not LLM:
        try:
            LLM = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)
            logger.info(f"PDF Extractor model loaded: {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load PDF extractor model: {e}")
            return {"error": "Local LLM for extraction is not available."}

    try:
        doc = fitz.open(file_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        return {"error": f"Failed to read PDF file: {e}"}

    system_prompt = f"You are an expert data extraction assistant. Extract information from the provided text that matches the following JSON schema. Output ONLY the JSON object.\nSchema:\n{json.dumps(schema, indent=2)}"
    
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nText:\n{full_text[:3000]}\n<|im_end|>\n<|im_start|>assistant\n"

    try:
        output = LLM(prompt, max_tokens=512, temperature=0.0, stop=["<|im_end|>"])
        response_text = output['choices'][0]['text'].strip()
        
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()
        
        return json.loads(response_text)
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return {"error": "Failed to extract data using local LLM.", "raw_output": response_text}