#!/usr/bin/env python3
"""
Main entry point for PearlEdit application.
"""

from pearledit import ImageSplitter
import sys

if __name__ == "__main__":
    # Get directory from command line argument or use current directory
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    
    app = ImageSplitter(directory)
    app.run()

