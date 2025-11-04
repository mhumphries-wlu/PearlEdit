"""Command-line interface for PearlEdit."""
import argparse
import sys
import logging
from pathlib import Path

from .config import AppSettings, load_settings, save_settings
from .services import ImageService, UserFacingError
from .paths import TempManager
from .ui.app import PearlEditApp

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_headless(
    input_dir: Path,
    output_dir: Path = None,
    auto_crop: bool = False,
    auto_split: bool = False,
    settings: AppSettings = None
) -> int:
    """
    Run PearlEdit in headless mode.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory (defaults to input_dir/pass_images)
        auto_crop: Whether to auto-crop all images
        auto_split: Whether to auto-split all images
        settings: Application settings
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        settings = settings or load_settings()
        temp_manager = TempManager()
        service = ImageService(settings, temp_manager)
        
        # Load images
        logger.info(f"Loading images from {input_dir}")
        count = service.load_images(input_dir)
        if count == 0:
            logger.error("No images found to process")
            return 1
        
        logger.info(f"Loaded {count} images")
        
        # Process images
        if auto_crop:
            logger.info("Auto-cropping all images...")
            try:
                service.auto_crop_all()
                logger.info("Auto-crop completed")
            except UserFacingError as e:
                logger.error(f"Auto-crop failed: {e}")
                return 1
        
        if auto_split:
            logger.warning("Auto-split in headless mode requires manual positioning - skipping")
        
        # Save images
        logger.info("Saving processed images...")
        try:
            service.save_images(output_dir)
            logger.info(f"Images saved successfully")
        except UserFacingError as e:
            logger.error(f"Save failed: {e}")
            return 1
        
        # Cleanup
        service.cleanup()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in headless mode: {e}", exc_info=True)
        return 1


def run_gui(input_dir: Path = None, settings: AppSettings = None) -> int:
    """
    Run PearlEdit in GUI mode.
    
    Args:
        input_dir: Optional input directory containing images (for backward compatibility)
        settings: Application settings
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        settings = settings or load_settings()
        
        app = PearlEditApp(input_dir, settings)
        app.run()
        
        return 0
    except Exception as e:
        logger.error(f"Error in GUI mode: {e}", exc_info=True)
        return 1


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="PearlEdit - Image preprocessing tool for transcription images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run GUI mode with current directory
  python -m pearl_edit.cli
  
  # Run GUI mode with specific directory
  python -m pearl_edit.cli --input /path/to/images
  
  # Run headless mode with auto-crop
  python -m pearl_edit.cli --headless --input /path/to/images --auto-crop
  
  # Run headless mode with custom output directory
  python -m pearl_edit.cli --headless --input /path/to/images --output /path/to/output
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='.',
        help='Input directory containing images (default: current directory)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for processed images (default: input_dir/pass_images)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode (no GUI)'
    )
    
    parser.add_argument(
        '--auto-crop',
        action='store_true',
        help='Automatically crop all images (headless mode only)'
    )
    
    parser.add_argument(
        '--auto-split',
        action='store_true',
        help='Automatically split all images (headless mode only, requires manual setup)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate input directory (optional for GUI mode)
    input_dir = None
    if args.input and args.input != '.':
        input_dir = Path(args.input).resolve()
        if not input_dir.exists():
            logger.warning(f"Input directory does not exist: {input_dir}. Starting with empty state.")
            input_dir = None
        elif not input_dir.is_dir():
            logger.warning(f"Input path is not a directory: {input_dir}. Starting with empty state.")
            input_dir = None
    
    # Validate output directory if specified
    output_dir = None
    if args.output:
        output_dir = Path(args.output).resolve()
        if output_dir.exists() and not output_dir.is_dir():
            logger.error(f"Output path exists but is not a directory: {output_dir}")
            return 1
    
    # Load settings
    settings = load_settings()
    
    # Run appropriate mode
    if args.headless:
        return run_headless(input_dir, output_dir, args.auto_crop, args.auto_split, settings)
    else:
        return run_gui(input_dir, settings)


if __name__ == '__main__':
    sys.exit(main())

