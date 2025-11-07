#!/usr/bin/env python3
"""
Auto-launcher for Australian Freight Export Analysis Dashboard
This script automatically starts the Streamlit app and opens it in your browser.
"""
import subprocess
import sys
import os
import webbrowser
import time
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    app_path = script_dir / "app.py"
    
    if not app_path.exists():
        print(f"Error: {app_path} not found!")
        sys.exit(1)
    
    print("=" * 60)
    print("Australian Freight Export Analysis Dashboard")
    print("=" * 60)
    print(f"Starting Streamlit app from: {app_path}")
    print("The app will automatically open in your browser...")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Start Streamlit with auto-open browser
    try:
        # Run streamlit with auto-open browser enabled
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.headless=false",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

