"""Path and temporary directory management."""
import os
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PathError(Exception):
    """Raised when path operations fail."""
    pass


class TempManager:
    """Manages temporary directories for image processing."""
    
    def __init__(self, prefix: str = "pearledit_"):
        """
        Initialize temp manager.
        
        Args:
            prefix: Prefix for temp directory name
        """
        self.prefix = prefix
        self._temp_dir: Optional[Path] = None
        self._originals_dir: Optional[Path] = None
    
    @property
    def path(self) -> Optional[Path]:
        """Get the current temp directory path."""
        return self._temp_dir
    
    @property
    def originals_path(self) -> Optional[Path]:
        """Get the originals backup directory path."""
        return self._originals_dir
    
    def create(self) -> Path:
        """
        Create a new temporary directory.
        
        Returns:
            Path to the created temp directory
            
        Raises:
            PathError: If directory creation fails
        """
        try:
            # Create main temp directory
            self._temp_dir = Path(tempfile.mkdtemp(prefix=self.prefix))
            logger.info(f"Created temp directory: {self._temp_dir}")
            
            # Create originals backup directory
            self._originals_dir = self._temp_dir / "originals_backup"
            self._originals_dir.mkdir(exist_ok=True)
            
            return self._temp_dir
        except Exception as e:
            logger.error(f"Failed to create temp directory: {e}")
            raise PathError(f"Failed to create temp directory: {str(e)}") from e
    
    def cleanup(self) -> None:
        """Clean up temporary directories."""
        if self._temp_dir and self._temp_dir.exists():
            try:
                # Remove all contents
                for item in self._temp_dir.iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                    except Exception as e:
                        logger.warning(f"Could not remove {item}: {e}")
                
                # Remove the directory itself
                try:
                    shutil.rmtree(self._temp_dir, ignore_errors=True)
                    logger.info(f"Cleaned up temp directory: {self._temp_dir}")
                except Exception as e:
                    logger.warning(f"Could not remove temp directory: {e}")
            except Exception as e:
                logger.error(f"Error during temp cleanup: {e}")
        
        self._temp_dir = None
        self._originals_dir = None
    
    def copy_source_images(self, source_dir: Path) -> int:
        """
        Copy images from source directory to temp directory.
        
        Args:
            source_dir: Source directory containing images
            
        Returns:
            Number of images copied
            
        Raises:
            PathError: If copying fails
        """
        if not self._temp_dir:
            raise PathError("Temp directory not created. Call create() first.")
        
        if not source_dir.exists():
            logger.warning(f"Source directory does not exist: {source_dir}")
            return 0
        
        files_copied = 0
        try:
            for file in source_dir.iterdir():
                if file.suffix.lower() in ('.jpg', '.jpeg'):
                    # Copy to temp directory
                    dst = self._temp_dir / file.name
                    shutil.copy2(file, dst)
                    
                    # Also copy to originals backup
                    if self._originals_dir:
                        orig_dst = self._originals_dir / file.name
                        shutil.copy2(file, orig_dst)
                    
                    files_copied += 1
            
            logger.info(f"Copied {files_copied} images from {source_dir}")
        except Exception as e:
            logger.error(f"Error copying files: {e}")
            raise PathError(f"Error copying files: {str(e)}") from e
        
        return files_copied
    
    def __enter__(self):
        """Context manager entry."""
        self.create()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        self.cleanup()


def pass_images_for(input_dir: Path) -> Path:
    """
    Get or create pass_images directory for output.
    
    Args:
        input_dir: Input directory path
        
    Returns:
        Path to pass_images directory
        
    Raises:
        PathError: If directory creation fails
    """
    try:
        # Default to input_dir/pass_images
        pass_images_dir = input_dir / "pass_images"
        pass_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear existing files
        for item in pass_images_dir.iterdir():
            try:
                if item.is_file():
                    item.unlink()
            except Exception as e:
                logger.warning(f"Error deleting file {item}: {e}")
        
        return pass_images_dir
    except Exception as e:
        logger.error(f"Error ensuring pass_images directory: {e}")
        raise PathError(f"Error ensuring pass_images directory: {str(e)}") from e



