"""Repository for image file operations."""
import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class RepositoryError(Exception):
    """Raised when repository operations fail."""
    pass


def scan_images(directory: Path) -> List[Path]:
    """
    Scan directory for image files.
    
    Args:
        directory: Directory to scan
        
    Returns:
        List of image file paths
        
    Raises:
        RepositoryError: If scanning fails
    """
    try:
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []
        
        images = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg')
        ]
        
        # Sort by filename for consistent ordering
        images.sort(key=lambda p: p.name)
        
        logger.info(f"Found {len(images)} images in {directory}")
        return images
    except Exception as e:
        logger.error(f"Error scanning images: {e}")
        raise RepositoryError(f"Error scanning images: {str(e)}") from e


def load_image(image_path: Path) -> Optional[bytes]:
    """
    Load image file as bytes.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image bytes or None if file doesn't exist
    """
    try:
        if image_path.exists():
            return image_path.read_bytes()
        return None
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def save_image(image_path: Path, image_data: bytes) -> bool:
    """
    Save image data to file.
    
    Args:
        image_path: Destination path
        image_data: Image data as bytes
        
    Returns:
        True if successful
        
    Raises:
        RepositoryError: If save fails
    """
    try:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_data)
        logger.debug(f"Saved image to {image_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving image {image_path}: {e}")
        raise RepositoryError(f"Error saving image: {str(e)}") from e


def remove_image(image_path: Path) -> bool:
    """
    Remove an image file.
    
    Args:
        image_path: Path to image file to remove
        
    Returns:
        True if successful
    """
    try:
        if image_path.exists():
            image_path.unlink()
            logger.debug(f"Removed image {image_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error removing image {image_path}: {e}")
        return False


def copy_image(source: Path, destination: Path) -> bool:
    """
    Copy an image file.
    
    Args:
        source: Source path
        destination: Destination path
        
    Returns:
        True if successful
        
    Raises:
        RepositoryError: If copy fails
    """
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        logger.debug(f"Copied image from {source} to {destination}")
        return True
    except Exception as e:
        logger.error(f"Error copying image: {e}")
        raise RepositoryError(f"Error copying image: {str(e)}") from e


def backup_image(image_path: Path, backup_dir: Path) -> bool:
    """
    Create a backup of an image file.
    
    Args:
        image_path: Path to image to backup
        backup_dir: Directory to store backup
        
    Returns:
        True if successful
        
    Raises:
        RepositoryError: If backup fails
    """
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / image_path.name
        shutil.copy2(image_path, backup_path)
        logger.debug(f"Backed up image {image_path} to {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Error backing up image: {e}")
        raise RepositoryError(f"Error backing up image: {str(e)}") from e


def restore_from_backup(image_path: Path, backup_dir: Path) -> bool:
    """
    Restore an image from backup.
    
    Args:
        image_path: Path where image should be restored
        backup_dir: Directory containing backup
        
    Returns:
        True if successful
        
    Raises:
        RepositoryError: If restore fails
    """
    try:
        # Try direct filename match first
        backup_path = backup_dir / image_path.name
        
        if not backup_path.exists():
            # Try to find similar filename
            base_name = image_path.stem
            for backup_file in backup_dir.iterdir():
                if backup_file.stem.startswith(base_name) or base_name.startswith(backup_file.stem):
                    backup_path = backup_file
                    break
        
        if backup_path.exists():
            shutil.copy2(backup_path, image_path)
            logger.debug(f"Restored image {image_path} from {backup_path}")
            return True
        else:
            logger.warning(f"Backup not found for {image_path}")
            return False
    except Exception as e:
        logger.error(f"Error restoring image: {e}")
        raise RepositoryError(f"Error restoring image: {str(e)}") from e


def generate_split_filename(base_path: Path, split_index: int, is_right: bool = False) -> Path:
    """
    Generate filename for split image.
    
    Args:
        base_path: Original image path
        split_index: Index for split image (fallback if sequence can't be determined)
        is_right: True if this is the right/bottom split
        
    Returns:
        Generated path for split image
    """
    base_name = base_path.stem
    # Handle format like "0001_p001" or just "0001"
    if '_p' in base_name:
        current_seq = int(base_name.split('_p')[0])
    else:
        try:
            current_seq = int(''.join(filter(str.isdigit, base_name)))
        except ValueError:
            current_seq = split_index
    
    if is_right:
        # Calculate next sequence number by scanning existing files
        # This ensures we don't overwrite existing files
        image_dir = base_path.parent
        existing_numbers = [current_seq]
        
        # Scan directory for existing image files
        if image_dir.exists():
            for fname in image_dir.iterdir():
                if fname.is_file() and fname.suffix.lower() in ('.jpg', '.jpeg'):
                    if fname == base_path:
                        continue  # Skip the original file we're splitting
                    try:
                        fname_stem = fname.stem
                        if '_p' in fname_stem:
                            num = int(fname_stem.split('_p')[0])
                        else:
                            num = int(''.join(filter(str.isdigit, fname_stem)))
                        existing_numbers.append(num)
                    except (ValueError, IndexError):
                        continue
        
        # Next sequence is max of all existing + 1
        next_seq = max(existing_numbers) + 1
        name = f"{next_seq:04d}_p{next_seq:03d}.jpg"
    else:
        name = f"{current_seq:04d}_p{current_seq:03d}.jpg"
    
    return base_path.parent / name

