#!/usr/bin/env python3
"""
DICOM Processing Pipeline GUI Launcher
======================================

Simple launcher script that ensures the GUI runs properly.
Checks for tkinter availability and provides helpful error messages.
"""

import sys
import os
from pathlib import Path

def check_tkinter():
    """Check if tkinter is available."""
    try:
        import tkinter as tk
        return True
    except ImportError:
        return False

def main():
    """Main launcher function."""
    print("DICOM Processing Pipeline - GUI Launcher")
    print("=" * 45)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7 or higher is required.")
        print(f"Current version: {sys.version}")
        input("Press Enter to exit...")
        return 1
    
    # Check tkinter
    if not check_tkinter():
        print("ERROR: tkinter is not available.")
        print("tkinter is required for the GUI interface.")
        print("\nSolutions:")
        print("- On Ubuntu/Debian: sudo apt-get install python3-tk")
        print("- On CentOS/RHEL: sudo yum install tkinter")
        print("- On macOS: tkinter should be included with Python")
        print("- On Windows: tkinter should be included with Python")
        input("Press Enter to exit...")
        return 1
    
    # Check if gui_app.py exists
    gui_app_path = Path(__file__).parent / "gui_app.py"
    if not gui_app_path.exists():
        print(f"ERROR: gui_app.py not found at {gui_app_path}")
        input("Press Enter to exit...")
        return 1
    
    # Check if run.py exists
    run_py_path = Path(__file__).parent / "run.py"
    if not run_py_path.exists():
        print(f"ERROR: run.py not found at {run_py_path}")
        print("Make sure you're running this from the project directory.")
        input("Press Enter to exit...")
        return 1
    
    # Check if runMulti.py exists
    runmulti_path = Path(__file__).parent / "runMulti.py"
    if not runmulti_path.exists():
        print(f"ERROR: runMulti.py not found at {runmulti_path}")
        print("Make sure you're running this from the project directory.")
        input("Press Enter to exit...")
        return 1
    
    print("[OK] Python version check passed")
    print("[OK] tkinter is available")
    print("[OK] All required files found")
    print("\nStarting GUI application...")
    print("-" * 45)
    
    # Import and run the GUI
    try:
        from gui_app import main as gui_main
        gui_main()
        return 0
    except ImportError as e:
        print(f"ERROR: Failed to import GUI application: {e}")
        input("Press Enter to exit...")
        return 1
    except Exception as e:
        print(f"ERROR: Failed to start GUI: {e}")
        print("\nTechnical details:")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        return 1

if __name__ == "__main__":
    sys.exit(main())