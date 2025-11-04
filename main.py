#!/usr/bin/env python3
"""
Main entry point for PearlEdit application.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import pearl_edit
sys.path.insert(0, str(Path(__file__).parent))

from pearl_edit.cli import main

if __name__ == "__main__":
    sys.exit(main())
