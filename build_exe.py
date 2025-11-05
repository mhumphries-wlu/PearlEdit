#!/usr/bin/env python3
"""
Build script for creating a standalone executable of PearlEdit.
This script uses PyInstaller to create a windowed (no console) executable.
"""

import subprocess
import sys
import shutil
from pathlib import Path

def find_tkdnd_files():
    """Find tkinterdnd2 package and its tkdnd directory."""
    try:
        import tkinterdnd2
        tkdnd_path = Path(tkinterdnd2.__file__).parent / "tkdnd"
        if tkdnd_path.exists():
            return str(tkdnd_path)
    except ImportError:
        pass
    return None

def build_exe():
    """Build the standalone executable using PyInstaller."""
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print("PyInstaller found.")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Determine path separator for --add-data
    if sys.platform == 'win32':
        path_sep = ';'
    else:
        path_sep = ':'
    
    # Find tkdnd files
    tkdnd_path = find_tkdnd_files()
    if not tkdnd_path:
        print("WARNING: Could not find tkinterdnd2 tkdnd directory.")
        print("Drag-and-drop functionality may not work in the executable.")
    else:
        print(f"Found tkdnd at: {tkdnd_path}")
    
    # Build PyInstaller command
    cmd = [
        "pyinstaller",
        "--name=PearlEdit",
        "--windowed",  # No console window - TEMPORARILY DISABLED FOR DEBUGGING
        "--onefile",   # Single executable file
        f"--icon={project_root / 'util' / 'icons' / 'PearlEdit.png'}",
        
        # Add data files
        f"--add-data=util{path_sep}util",  # Include util directory
        
        # Collect entire packages (includes submodules, data, and binaries)
        "--collect-all=tkinterdnd2",  # All tkinterdnd2 files and DLLs
        "--collect-all=fitz",          # All PyMuPDF files
        "--collect-submodules=pearl_edit",  # All pearl_edit submodules
        
        # Hidden imports (cv2 needs to be hidden import, not collect-all)
        "--hidden-import=cv2",
        "--hidden-import=PIL._tkinter_finder",  # PIL/Tkinter bridge
        "--hidden-import=PIL.Image",
        "--hidden-import=PIL.ImageTk",
        "--hidden-import=numpy",
        "--hidden-import=pandas",
        "--hidden-import=platformdirs",
        
        "--clean",  # Clean PyInstaller cache
        str(project_root / "main.py")
    ]
    
    # Add tkdnd directory if found
    if tkdnd_path:
        cmd.insert(-2, f"--add-data={tkdnd_path}{path_sep}tkinterdnd2/tkdnd")
    
    print("\nBuilding PearlEdit executable...")
    print("Command:", " ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        
        # Always print output for debugging
        if result.stdout:
            print("Build output:")
            print(result.stdout)
        
        if result.returncode == 0:
            print("\n" + "="*60)
            print("Build completed successfully!")
            print(f"Executable location: {project_root / 'dist' / 'PearlEdit.exe'}")
            print("="*60)
        else:
            if result.stderr:
                print("\nBuild errors:")
                print(result.stderr)
            print("\n" + "="*60)
            print("Build failed. See errors above.")
            print("="*60)
            sys.exit(1)
    except FileNotFoundError:
        print("PyInstaller not found. Please install it manually:")
        print("  pip install pyinstaller")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during build: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_exe()
