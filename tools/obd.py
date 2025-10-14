# /home/taylo/mcp-agent/tools/obd.py
import logging

logger = logging.getLogger("mcp_server")

def obd_query(query: str) -> str:
    # Stub: Replace with pyobd connection for real reads (safety: read-only)
    # e.g., import obd; connection = obd.OBD(); value = connection.query(obd.commands[query.upper()])
    return f"OBD read-only query stub: {query} (e.g., RPM = 2000)"  # Mock for testing
