#!/usr/bin/env python3
"""Run the DashScope-style HTTP API (development / single-process)."""
import os

import uvicorn

if __name__ == "__main__":
    host = os.environ.get("WAN_API_HOST", "0.0.0.0")
    port = int(os.environ.get("WAN_API_PORT", "8008"))
    uvicorn.run(
        "serve.api:app",
        host=host,
        port=port,
        workers=1,
        reload=os.environ.get("WAN_API_RELOAD", "").lower() in ("1", "true", "yes"),
    )
