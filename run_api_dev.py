#!/usr/bin/env python3
"""
Development entry point for the YOLO Box Counting & Label Processing API server.

This script runs the server with hot reload enabled for development.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Change working directory to project root for relative paths to work
os.chdir(current_dir)

if __name__ == "__main__":
    import uvicorn
    
    # Run the server with hot reload for development
    uvicorn.run(
        "src.api_server:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,  # Hot reload enabled
        log_level="info",
        reload_dirs=[str(src_dir)]  # Watch src directory for changes
    )