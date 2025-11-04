# PearlEdit

Transcription Pearl Image Preprocessing Tool v0.9 beta

A Python-based image editing tool for preprocessing transcription images with features for splitting, cropping, rotating, and adjusting images.

## Features

- **Image Splitting**: Split images vertically, horizontally, or at custom angles
- **Auto Cropping**: Automatic cropping to largest white area with threshold adjustment
- **Image Rotation**: Rotate images by 90 degrees or custom angles
- **Batch Processing**: Process multiple images in batch mode
- **Image Navigation**: Navigate through image collections with ease

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

## Requirements

- Python 3.7 or higher
- OpenCV (opencv-python)
- NumPy
- Pillow (PIL)
- Pandas

## Usage

Run the application:
```bash
python main.py [directory]
```

If no directory is specified, it will use the current directory.

### Keyboard Shortcuts

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

### Menu Options

- **File**: Save images, quit application
- **Edit**: Revert images, delete current image
- **View**: Navigate between images
- **Process**: Split images, crop images, rotate images

## Project Structure

```
PearlEdit/
├── pearledit/
│   ├── __init__.py
│   ├── threshold_adjuster.py
│   └── image_splitter.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

