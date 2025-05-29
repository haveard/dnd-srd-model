#!/usr/bin/env python3
"""
D&D LoRA Dashboard Launcher
===========================

Quick launcher for the Streamlit dashboard with proper configuration.

Usage:
    python launch_dashboard.py
    python launch_dashboard.py --port 8501
    python launch_dashboard.py --host 0.0.0.0 --port 8080
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Launch D&D LoRA Streamlit Dashboard")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", default=8501, type=int, help="Port to bind to (default: 8501)")
    parser.add_argument("--browser", action="store_true", help="Open browser automatically")
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    dashboard_file = script_dir / "streamlit_dashboard.py"
    
    if not dashboard_file.exists():
        print(f"❌ Dashboard file not found: {dashboard_file}")
        sys.exit(1)
    
    # Build streamlit command
    cmd = [
        "streamlit", "run", 
        str(dashboard_file),
        "--server.address", args.host,
        "--server.port", str(args.port)
    ]
    
    if not args.browser:
        cmd.extend(["--server.headless", "true"])
    
    print(f"🚀 Starting D&D LoRA Dashboard on {args.host}:{args.port}")
    print(f"📂 Working directory: {script_dir}")
    print(f"🌐 Dashboard URL: http://{args.host}:{args.port}")
    print()
    
    try:
        subprocess.run(cmd, cwd=script_dir)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it:")
        print("   pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
