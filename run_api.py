#!/usr/bin/env python3
"""
API Server Entry Point

Runs the FastAPI server for box detection and label processing.
Supports both local development and cloud deployment.
"""

import os
import uvicorn
from src.api_server import app

if __name__ == "__main__":
    # Get host and port from environment (for cloud deployment) or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # Check if running in production (Cloud Run sets PORT=8080)
    is_production = port == 8080 or os.getenv("K_SERVICE") is not None
    
    if is_production:
        # Production settings for Cloud Run
        uvicorn.run(
            app, 
            host=host, 
            port=port,
            log_level="info",
            access_log=True,
            workers=1
        )
    else:
        # Development settings with auto-reload
        uvicorn.run(
            "src.api_server:app", 
            host=host, 
            port=port,
            reload=True,
            log_level="debug"
        )