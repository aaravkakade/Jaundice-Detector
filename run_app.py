#!/usr/bin/env python3
"""
Script to run the Streamlit app.
Usage: python run_app.py
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).parent / "app" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
