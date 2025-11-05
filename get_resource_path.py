"""
Utility function to get resource paths that work in both development and PyInstaller bundle.
This should be imported by modules that need to access bundled resources.
"""

import sys
from pathlib import Path


def get_resource_path(relative_path: str) -> Path:
    """
    Get absolute path to a resource, works for both dev and PyInstaller.
    
    Args:
        relative_path: Path relative to project root (e.g., "util/icons")
        
    Returns:
        Absolute Path to the resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        # Running in development mode
        # Get project root (3 levels up from this file: get_resource_path.py -> root)
        base_path = Path(__file__).parent
    
    return base_path / relative_path


def get_project_root() -> Path:
    """
    Get the project root directory.
    Works in both development and PyInstaller bundle.
    
    Returns:
        Path to project root
    """
    try:
        # PyInstaller bundle
        return Path(sys._MEIPASS)
    except AttributeError:
        # Development mode
        return Path(__file__).parent



