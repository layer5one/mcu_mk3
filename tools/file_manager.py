# /home/taylo/mcp-agent/tools/file_manager.py
import os
import logging

logger = logging.getLogger("mcp_server")

SANDBOX_DIR = "/home/taylo/mcp-sandbox"  # Create this dir

def manage_file(operation: str, path: str, content: str = None) -> str:
    abs_path = os.path.abspath(os.path.join(SANDBOX_DIR, path.lstrip('/')))  # Restrict to sandbox
    if not abs_path.startswith(SANDBOX_DIR):
        return "Error: Path outside sandbox."
    try:
        if operation == "create":
            if os.path.exists(abs_path):
                return "File already exists."
            with open(abs_path, "w") as f:
                f.write(content or "")
            return "File created."
        elif operation == "edit":
            if not os.path.exists(abs_path):
                return "File does not exist."
            with open(abs_path, "w") as f:
                f.write(content or "")
            return "File edited."
        elif operation == "delete":
            if not os.path.exists(abs_path):
                return "Path does not exist."
            if os.path.isdir(abs_path):
                os.rmdir(abs_path)
                return "Directory deleted."
            else:
                os.remove(abs_path)
                return "File deleted."
        elif operation == "read":
            if not os.path.exists(abs_path) or os.path.isdir(abs_path):
                return "Invalid file path."
            with open(abs_path, "r") as f:
                return f.read()
        elif operation == "list":
            if not os.path.exists(abs_path) or not os.path.isdir(abs_path):
                return "Invalid directory path."
            return str(os.listdir(abs_path))
        else:
            return "Invalid operation."
    except Exception as e:
        logger.error(f"File operation failed: {e}")
        return f"Error: {str(e)}"
