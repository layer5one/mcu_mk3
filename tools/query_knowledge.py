# /home/taylo/mcp-agent/tools/query_knowledge.py
import logging

logger = logging.getLogger("mcp_server")

def query_knowledge(query: str) -> str:
    from mcp_server import memory_manager  # Import here to avoid circular
    results = memory_manager.query_knowledge(query)
    if not results:
        return "No relevant knowledge found."
    return "\n".join(results)  # Factually return chunks
