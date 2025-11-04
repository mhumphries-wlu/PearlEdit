#!/usr/bin/env python3
"""
Build script for creating a standalone executable of PearlEdit.
This script uses PyInstaller to create a windowed (no console) executable.
"""

import subprocess
import sys
from pathlib import Path

def build_exe():
    """Build the standalone executable using PyInstaller."""
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print("PyInstaller found.")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Warn about pathlib issue if present
    try:
        import pathlib
        if hasattr(pathlib, '__version__'):
            print("WARNING: Obsolete pathlib backport detected. This may cause issues.")
            print("If build fails, try: conda remove pathlib")
    except ImportError:
        pass
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Determine path separator for --add-data
    if sys.platform == 'win32':
        path_sep = ';'
    else:
        path_sep = ':'
    
    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--name=PearlEdit",
        "--windowed",  # No console window
        "--onefile",   # Single executable file
        f"--icon={project_root / 'util' / 'icons' / 'PearlEdit.png'}",
        f"--add-data=util{path_sep}util",  # Include icons directory
        "--hidden-import=tkinterdnd2",  # Ensure tkinterdnd2 is included
        "--hidden-import=PIL._tkinter_finder",  # PIL/Tkinter bridge
        "--clean",  # Clean PyInstaller cache
        str(project_root / "main.py")
    ]
    
    print("Building PearlEdit executable...")
    print("Command:", " ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        if result.returncode == 0:
            print("\n" + "="*60)
            print("Build completed successfully!")
            print(f"Executable location: {project_root / 'dist' / 'PearlEdit.exe'}")
            print("="*60)
        else:
            print("Build output:")
            print(result.stdout)
            print("Build errors:")
            print(result.stderr)
            print("\n" + "="*60)
            print("Build failed. See errors above.")
            print("="*60)
            sys.exit(1)
    except FileNotFoundError:
        print("PyInstaller not found. Please install it manually:")
        print("  pip install pyinstaller")
        sys.exit(1)

if __name__ == "__main__":
    build_exe()

