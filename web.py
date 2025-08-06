#!/usr/bin/env python3
"""
Web interface launcher for SciPathBench
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.web_server import app, setup_logging
import config
import uvicorn

def main():
    """Launch the SciPathBench web interface."""
    
    # Setup logging
    setup_logging(config.LOG_FILE)
    
    # Ensure output directory exists
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Check if benchmark data exists
    benchmark_file = Path(config.BENCHMARK_DATA_FILE)
    if not benchmark_file.exists():
        print(f"⚠️  Warning: Benchmark data file not found at {config.BENCHMARK_DATA_FILE}")
        print("   You may need to run: uv run src/generate_data.py")
        print("   The web interface will still work but may have limited functionality.")
        print()
    
    print("🚀 Starting SciPathBench Web Interface...")
    print("📊 Features available:")
    print("   • Interactive Mode: Play the pathfinding game yourself")
    print("   • Live LLM Runs: Watch AI agents solve challenges in real-time")
    print("   • Leaderboard: View performance metrics and comparisons")
    print()
    print("🌐 Web interface will be available at: http://localhost:8001")
    print("📝 Logs will be saved to:", config.LOG_FILE)
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "src.web_server:app",
            host="0.0.0.0",
            port=8001,
            reload=False,  # Set to True for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down SciPathBench Web Interface...")
    except Exception as e:
        print(f"❌ Error starting web server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()