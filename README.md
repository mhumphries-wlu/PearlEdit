# PearlEdit

Transcription Pearl Image Preprocessing Tool v0.9 beta

A Python-based image editing tool for preprocessing transcription images with features for splitting, cropping, rotating, and adjusting images. Now runs as a standalone application with modular architecture.

## Features

- **Image Splitting**: Split images vertically, horizontally, or at custom angles
- **Auto Cropping**: Automatic cropping to largest white area with threshold adjustment
- **Image Rotation**: Rotate images by 90 degrees or custom angles
- **Batch Processing**: Process multiple images in batch mode
- **Image Navigation**: Navigate through image collections with ease
- **Standalone Operation**: No master program dependency - runs independently
- **Settings Persistence**: User preferences saved between sessions
- **Headless Mode**: Command-line interface for automated processing

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd PearlEdit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note for Drag and Drop Support:**
The `tkinterdnd2` package is included in requirements but may need additional setup on some systems. If drag and drop doesn't work:
- Make sure `tkinterdnd2` is installed: `pip install tkinterdnd2`
- On Windows, this should work out of the box
- If issues persist, you can still use File > Import Images (Ctrl+O)

## Requirements

- Python 3.7 or higher
- OpenCV (opencv-python)
- NumPy
- Pillow (PIL)
- Pandas
- platformdirs (for settings management)

## Usage

### GUI Mode

Run the application with GUI (default):
```bash
python PearlEdit.py
```

Or specify an input directory:
```bash
python PearlEdit.py --input /path/to/images
```

You can also use the CLI module directly:
```bash
python -m pearl_edit.cli --input /path/to/images
```

### Headless Mode

Run without GUI for automated processing:
```bash
python PearlEdit.py --headless --input /path/to/images --auto-crop
```

Or with custom output directory:
```bash
python PearlEdit.py --headless --input /path/to/images --output /path/to/output
```

### Command-Line Options

- `--input, -i`: Input directory containing images (default: current directory)
- `--output, -o`: Output directory for processed images (default: input_dir/pass_images)
- `--headless`: Run in headless mode (no GUI)
- `--auto-crop`: Automatically crop all images (headless mode only)
- `--auto-split`: Automatically split all images (headless mode only, requires manual setup)
- `--verbose, -v`: Enable verbose logging

## Settings

Settings are automatically saved to your user configuration directory:
- **Windows**: `%APPDATA%\PearlEdit\settings.json`
- **Linux/Mac**: `~/.config/PearlEdit/settings.json`

Settings include:
- Default threshold for auto-crop
- Default margin for cropping
- Auto-split mode preference
- Rotation step size
- Image quality settings

## Temporary Files

PearlEdit creates temporary directories for processing images. These are automatically cleaned up when the application exits normally. If the application crashes, temporary directories may remain in your system's temp folder (prefixed with `pearledit_`). These can be safely deleted.

## Output

Processed images are saved to the `pass_images` directory within the input directory by default. You can specify a custom output directory using the `--output` option.

## Keyboard Shortcuts

- `Ctrl+V`: Activate vertical split tool
- `Ctrl+H`: Activate horizontal split tool
- `Ctrl+A`: Toggle auto-split mode
- `Ctrl+L`: Straighten image by line
- `Ctrl+Shift+C`: Activate crop tool
- `Ctrl+Shift+A`: Auto crop active image
- `[` / `]`: Rotate cursor angle
- `Ctrl+[` / `Ctrl+]`: Rotate image by 90 degrees
- `←` / `→`: Navigate between images
- `Enter`: Apply crop (when crop tool is active)
- `Escape`: Cancel current operation

## Menu Options

- **File**: Save images, quit application
- **Edit**: Revert images, delete current image
- **View**: Navigate between images
- **Process**: Split images, crop images, rotate images

## Project Structure

```
PearlEdit/
├── pearl_edit/
│   ├── __init__.py
│   ├── config.py              # Settings management
│   ├── paths.py               # Temp directory management
│   ├── image_ops.py           # Pure image processing functions
│   ├── repository.py          # File operations
│   ├── services.py            # Service layer orchestrating operations
│   ├── state.py               # Session state management
│   ├── cli.py                 # Command-line interface
│   ├── logging_config.py      # Logging configuration
│   └── ui/
│       ├── __init__.py
│       └── app.py              # Main GUI application
├── PearlEdit.py               # Entry point (delegates to CLI)
├── requirements.txt
├── README.md
└── .gitignore
```

## Architecture

The application follows a modular architecture with clear separation of concerns:

- **Backend modules** (`image_ops`, `repository`, `services`, `state`): Pure Python logic with no UI dependencies
- **UI module** (`ui/app.py`): Tkinter interface that calls backend services
- **CLI module** (`cli.py`): Command-line interface for headless operation
- **Config module** (`config.py`): Settings persistence using platformdirs

This architecture allows for:
- Easy testing of backend logic
- Multiple UI implementations (GUI, CLI, API)
- Clear separation between business logic and presentation

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
