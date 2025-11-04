"""Repository for image file operations."""
import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional
from io import BytesIO

logger = logging.getLogger(__name__)

# Try to import PyMuPDF for PDF support
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    fitz = None
    PDF_AVAILABLE = False


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


def extract_images_from_pdf(pdf_path: Path, output_dir: Path) -> List[Path]:
    """
    Extract images from a PDF file in order, discarding text.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save extracted images
        
    Returns:
        List of paths to extracted image files
        
    Raises:
        RepositoryError: If extraction fails
    """
    if not PDF_AVAILABLE:
        raise RepositoryError("PyMuPDF is not installed. Install it with: pip install PyMuPDF")
    
    try:
        if not pdf_path.exists():
            raise RepositoryError(f"PDF file does not exist: {pdf_path}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_images = []
        pdf_stem = pdf_path.stem
        
        # Open PDF
        doc = fitz.open(str(pdf_path))
        
        image_index = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get images from page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                # Get image data
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Convert to PIL Image for processing
                from PIL import Image
                pil_image = Image.open(BytesIO(image_bytes))
                
                # Resize if needed (600 DPI max, longest edge 2500 max)
                pil_image = resize_image_if_needed(pil_image, max_dpi=600, max_longest_edge=2500)
                
                # Save as JPEG
                image_index += 1
                output_filename = f"{pdf_stem}_page{page_num + 1:03d}_img{image_index:03d}.jpg"
                output_path = output_dir / output_filename
                
                # Handle name collisions
                counter = 1
                while output_path.exists():
                    output_filename = f"{pdf_stem}_page{page_num + 1:03d}_img{image_index:03d}_{counter}.jpg"
                    output_path = output_dir / output_filename
                    counter += 1
                
                # Save as JPEG with quality 95
                pil_image.convert('RGB').save(str(output_path), 'JPEG', quality=95, optimize=True)
                extracted_images.append(output_path)
                logger.debug(f"Extracted image {image_index} from page {page_num + 1} to {output_path}")
        
        doc.close()
        
        logger.info(f"Extracted {len(extracted_images)} images from PDF {pdf_path}")
        return extracted_images
        
    except Exception as e:
        if isinstance(e, RepositoryError):
            raise
        logger.error(f"Error extracting images from PDF {pdf_path}: {e}")
        raise RepositoryError(f"Error extracting images from PDF: {str(e)}") from e


def resize_image_if_needed(image, max_dpi: int = 600, max_longest_edge: int = 2500):
    """
    Resize image if it exceeds DPI or longest edge limits.
    
    Args:
        image: PIL Image object
        max_dpi: Maximum DPI allowed (default: 600)
        max_longest_edge: Maximum pixels for longest edge (default: 2500)
        
    Returns:
        PIL Image (resized if needed, original otherwise)
    """
    from PIL import Image
    
    # Get current dimensions
    width, height = image.size
    longest_edge = max(width, height)
    
    # Get DPI from image info (if available)
    dpi = image.info.get('dpi', (72, 72))
    current_dpi = max(dpi) if isinstance(dpi, tuple) else dpi
    
    # Check if resize is needed
    needs_resize = False
    scale_factor = 1.0
    
    # Check longest edge constraint first (always applicable)
    if longest_edge > max_longest_edge:
        scale_factor = max_longest_edge / longest_edge
        needs_resize = True
    
    # Check DPI constraint (only if DPI is meaningfully high, > 72)
    # If DPI is low or default, we rely on pixel dimension constraints
    if current_dpi > 72 and current_dpi > max_dpi:
        dpi_scale = max_dpi / current_dpi
        # Use the more restrictive scale factor
        if dpi_scale < scale_factor:
            scale_factor = dpi_scale
            needs_resize = True
    
    # Resize if needed
    if needs_resize:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Update DPI info if it was set in original
        if 'dpi' in image.info and current_dpi > 72:
            new_dpi = int(current_dpi * scale_factor)
            resized.info['dpi'] = (new_dpi, new_dpi)
        elif 'dpi' not in image.info:
            # If no DPI was set, estimate based on longest edge constraint
            # This is approximate - assume reasonable default if we had to resize
            if longest_edge > max_longest_edge:
                # Estimate: if we resized based on edge, maintain reasonable DPI
                estimated_dpi = min(600, int(72 * (max_longest_edge / longest_edge)))
                resized.info['dpi'] = (estimated_dpi, estimated_dpi)
        
        logger.debug(f"Resized image from {width}x{height} (DPI: {current_dpi}) to {new_width}x{new_height} (DPI: {resized.info.get('dpi', (72, 72))[0]})")
        return resized
    
    return image

