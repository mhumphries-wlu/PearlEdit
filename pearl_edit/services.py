"""Service layer for orchestrating image operations."""
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Callable
from PIL import Image

from .image_ops import (
    auto_crop, crop_image, rotate_image, straighten_by_line,
    split_image_vertical, split_image_horizontal, split_image_angled,
    ImageProcessingError
)
from .repository import (
    scan_images, copy_image, remove_image, backup_image,
    restore_from_backup, generate_split_filename, RepositoryError
)
from .state import SessionState, ImageRecord
from .paths import TempManager, pass_images_for, PathError
from .config import AppSettings

logger = logging.getLogger(__name__)


class UserFacingError(Exception):
    """User-facing error that should be shown in UI."""
    pass


class ImageService:
    """Service for managing image editing operations."""
    
    def __init__(self, settings: AppSettings, temp_manager: TempManager):
        """
        Initialize image service.
        
        Args:
            settings: Application settings
            temp_manager: Temporary directory manager
        """
        self.settings = settings
        self.temp_manager = temp_manager
        self.state = SessionState()
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """Set callback for progress updates (current, total, message)."""
        self._progress_callback = callback
    
    def _update_progress(self, current: int, total: int, message: str) -> None:
        """Internal method to call progress callback."""
        if self._progress_callback:
            self._progress_callback(current, total, message)
    
    def import_image_files(self, image_files: List[Path]) -> int:
        """
        Import images from a list of file paths.
        
        Args:
            image_files: List of image file paths
            
        Returns:
            Number of images imported
            
        Raises:
            UserFacingError: If importing fails
        """
        try:
            # Ensure temp directory exists
            if not self.temp_manager.path:
                self.temp_manager.create()
            
            temp_dir = self.temp_manager.path
            if not temp_dir:
                raise UserFacingError("Temp directory not available")
            
            # Copy images to temp directory
            imported_count = 0
            for img_file in image_files:
                if not img_file.exists():
                    logger.warning(f"Image file does not exist: {img_file}")
                    continue
                
                if img_file.suffix.lower() not in ('.jpg', '.jpeg'):
                    logger.warning(f"Skipping non-image file: {img_file}")
                    continue
                
                # Copy to temp directory
                dst = temp_dir / img_file.name
                # Handle name collisions
                counter = 1
                while dst.exists():
                    stem = img_file.stem
                    dst = temp_dir / f"{stem}_{counter}{img_file.suffix}"
                    counter += 1
                
                copy_image(img_file, dst)
                
                # Also copy to originals backup
                if self.temp_manager.originals_path:
                    orig_dst = self.temp_manager.originals_path / dst.name
                    copy_image(img_file, orig_dst)
                
                imported_count += 1
            
            if imported_count == 0:
                raise UserFacingError("No valid images to import")
            
            # Reload images from temp directory
            image_paths = scan_images(temp_dir)
            
            # Create image records (only add new ones)
            existing_paths = {img.original_image for img in self.state.images}
            new_count = 0
            for img_path in image_paths:
                if img_path not in existing_paths:
                    record = ImageRecord(
                        image_index=len(self.state.images) + 1,
                        original_image=img_path
                    )
                    self.state.images.append(record)
                    new_count += 1
            
            # Reindex all images
            for i, img in enumerate(self.state.images):
                img.image_index = i + 1
            
            if new_count > 0:
                # Navigate to first new image if none currently selected
                if self.state.current_image_index >= len(self.state.images):
                    self.state.current_image_index = 0
            
            logger.info(f"Imported {new_count} new images (total: {len(self.state.images)})")
            
            return new_count
            
        except (PathError, RepositoryError) as e:
            raise UserFacingError(str(e)) from e
        except Exception as e:
            logger.error(f"Error importing images: {e}")
            raise UserFacingError(f"Failed to import images: {str(e)}") from e
    
    def load_images(self, source_dir: Path) -> int:
        """
        Load images from source directory into session.
        
        Args:
            source_dir: Source directory path
            
        Returns:
            Number of images loaded
            
        Raises:
            UserFacingError: If loading fails
        """
        try:
            # Create temp directory and copy images
            self.temp_manager.create()
            files_copied = self.temp_manager.copy_source_images(source_dir)
            
            if files_copied == 0:
                raise UserFacingError(f"No images found in {source_dir}")
            
            # Scan temp directory for images
            temp_dir = self.temp_manager.path
            if not temp_dir:
                raise UserFacingError("Temp directory not available")
            
            image_paths = scan_images(temp_dir)
            
            # Create image records
            self.state.images.clear()
            for i, img_path in enumerate(image_paths, start=1):
                record = ImageRecord(
                    image_index=i,
                    original_image=img_path
                )
                self.state.images.append(record)
            
            self.state.current_image_index = 0
            logger.info(f"Loaded {len(self.state.images)} images")
            
            return len(self.state.images)
            
        except (PathError, RepositoryError) as e:
            raise UserFacingError(str(e)) from e
        except Exception as e:
            logger.error(f"Error loading images: {e}")
            raise UserFacingError(f"Failed to load images: {str(e)}") from e
    
    def get_current_image(self) -> Optional[ImageRecord]:
        """Get the current image record."""
        return self.state.get_current_image()
    
    def navigate(self, direction: int) -> bool:
        """Navigate to a different image."""
        return self.state.navigate(direction)
    
    def auto_crop_current(self) -> bool:
        """
        Auto-crop the current image.
        
        Returns:
            True if successful
            
        Raises:
            UserFacingError: If operation fails
        """
        current = self.get_current_image()
        if not current:
            raise UserFacingError("No image selected")
        
        try:
            image_path = current.current_image_path
            success = auto_crop(str(image_path), self.settings.threshold, self.settings.margin)
            if success:
                self.state.mark_changed()
            return success
        except ImageProcessingError as e:
            raise UserFacingError(str(e)) from e
    
    def auto_crop_all(self) -> None:
        """
        Auto-crop all images.
        
        Raises:
            UserFacingError: If operation fails
        """
        total = len(self.state.images)
        if total == 0:
            raise UserFacingError("No images to process")
        
        errors = []
        for i, record in enumerate(self.state.images):
            self._update_progress(i + 1, total, f"Processing image {i + 1} of {total}")
            
            try:
                image_path = record.current_image_path
                auto_crop(str(image_path), self.settings.threshold, self.settings.margin)
            except ImageProcessingError as e:
                errors.append(f"Image {record.current_image_path.name}: {str(e)}")
                logger.error(f"Error cropping {image_path}: {e}")
        
        if errors:
            error_msg = f"Errors occurred during auto-crop:\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                error_msg += f"\n...and {len(errors) - 5} more errors"
            raise UserFacingError(error_msg)
        
        self.state.mark_changed()
    
    def crop_current(self, coords: Tuple[int, int, int, int]) -> bool:
        """
        Crop current image to specified coordinates.
        
        Args:
            coords: Tuple of (left, top, right, bottom)
            
        Returns:
            True if successful
            
        Raises:
            UserFacingError: If operation fails
        """
        current = self.get_current_image()
        if not current:
            raise UserFacingError("No image selected")
        
        try:
            image_path = current.current_image_path
            crop_image(str(image_path), coords)
            self.state.mark_changed()
            return True
        except ImageProcessingError as e:
            raise UserFacingError(str(e)) from e
    
    def split_current(
        self,
        orientation: str,
        split_coord: Optional[int] = None,
        line_coords: Optional[Tuple[int, int, int, int]] = None,
        angle: Optional[float] = None
    ) -> bool:
        """
        Split current image.
        
        Args:
            orientation: 'vertical', 'horizontal', or 'angled'
            split_coord: X or Y coordinate for straight splits
            line_coords: Line coordinates for angled splits
            angle: Angle for angled splits
            
        Returns:
            True if successful
            
        Raises:
            UserFacingError: If operation fails
        """
        current = self.get_current_image()
        if not current:
            raise UserFacingError("No image selected")
        
        # Always use the original image for splitting, not any split version
        # This ensures we're splitting the original A, B, C images, not already-split images
        image_path = current.original_image
        
        # Skip if this image is already split
        if current.left_or_right is not None:
            raise UserFacingError("This image is already split. Please select an unsplit image.")
        
        try:
            # Load image
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                width, height = image.size
                
                # Perform split based on orientation
                if orientation == 'vertical':
                    if split_coord is None:
                        raise UserFacingError("Split coordinate required for vertical split")
                    left_img, right_img = split_image_vertical(image, split_coord)
                elif orientation == 'horizontal':
                    if split_coord is None:
                        raise UserFacingError("Split coordinate required for horizontal split")
                    left_img, right_img = split_image_horizontal(image, split_coord)
                elif orientation == 'angled':
                    if line_coords is None:
                        raise UserFacingError("Line coordinates required for angled split")
                    base_orient = 'vertical' if angle and angle < 45 else 'horizontal'
                    left_img, right_img = split_image_angled(image, line_coords, base_orient)
                else:
                    raise UserFacingError(f"Invalid orientation: {orientation}")
                
                # Generate filenames using the original image path
                # This ensures consistent sequence numbering based on the original image
                left_path = generate_split_filename(image_path, current.image_index, False)
                right_path = generate_split_filename(image_path, current.image_index, True)
                
                # Save split images
                left_img.save(str(left_path), 'JPEG', quality=self.settings.default_quality, optimize=True)
                right_img.save(str(right_path), 'JPEG', quality=self.settings.default_quality, optimize=True)
                
                # Update records
                current.split_image = left_path
                current.left_or_right = 'Left'
                
                # Create right image record
                right_record = ImageRecord(
                    image_index=len(self.state.images) + 1,
                    original_image=current.original_image,
                    split_image=right_path,
                    left_or_right='Right'
                )
                
                # Insert right record after current
                current_idx = self.state.current_image_index
                self.state.images.insert(current_idx + 1, right_record)
                
                # Reindex
                for i, img in enumerate(self.state.images):
                    img.image_index = i + 1
                
                self.state.mark_changed()
                return True
                
        except ImageProcessingError as e:
            raise UserFacingError(str(e)) from e
        except Exception as e:
            logger.error(f"Error splitting image: {e}")
            raise UserFacingError(f"Error splitting image: {str(e)}") from e
    
    def rotate_current(self, angle: float) -> bool:
        """
        Rotate current image.
        
        Args:
            angle: Rotation angle in degrees
            
        Returns:
            True if successful
            
        Raises:
            UserFacingError: If operation fails
        """
        current = self.get_current_image()
        if not current:
            raise UserFacingError("No image selected")
        
        try:
            image_path = current.current_image_path
            rotate_image(str(image_path), angle)
            self.state.mark_changed()
            return True
        except ImageProcessingError as e:
            raise UserFacingError(str(e)) from e
    
    def rotate_all(self, angle: float) -> None:
        """
        Rotate all images.
        
        Args:
            angle: Rotation angle in degrees
            
        Raises:
            UserFacingError: If operation fails
        """
        errors = []
        for record in self.state.images:
            try:
                image_path = record.current_image_path
                rotate_image(str(image_path), angle)
            except ImageProcessingError as e:
                errors.append(f"Image {record.current_image_path.name}: {str(e)}")
        
        if errors:
            error_msg = f"Errors occurred during rotation:\n" + "\n".join(errors[:5])
            raise UserFacingError(error_msg)
        
        self.state.mark_changed()
    
    def straighten_current(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """
        Straighten current image by line.
        
        Args:
            start: Start point (x, y)
            end: End point (x, y)
            
        Returns:
            True if successful
            
        Raises:
            UserFacingError: If operation fails
        """
        current = self.get_current_image()
        if not current:
            raise UserFacingError("No image selected")
        
        try:
            image_path = current.current_image_path
            straighten_by_line(str(image_path), start, end)
            self.state.mark_changed()
            return True
        except ImageProcessingError as e:
            raise UserFacingError(str(e)) from e
    
    def delete_current(self) -> bool:
        """
        Delete current image.
        
        Returns:
            True if successful
            
        Raises:
            UserFacingError: If operation fails
        """
        current = self.get_current_image()
        if not current:
            raise UserFacingError("No image selected")
        
        try:
            # Remove split image if it exists
            if current.split_image and current.split_image.exists():
                remove_image(current.split_image)
            
            # Remove original if it's not shared
            original_count = sum(1 for img in self.state.images if img.original_image == current.original_image)
            if original_count == 1:
                remove_image(current.original_image)
            
            # Remove from state
            self.state.remove_image(self.state.current_image_index)
            self.state.mark_changed()
            return True
        except Exception as e:
            logger.error(f"Error deleting image: {e}")
            raise UserFacingError(f"Error deleting image: {str(e)}") from e
    
    def revert_current(self) -> bool:
        """
        Revert current image to original.
        
        Returns:
            True if successful
            
        Raises:
            UserFacingError: If operation fails
        """
        current = self.get_current_image()
        if not current:
            raise UserFacingError("No image selected")
        
        try:
            originals_dir = self.temp_manager.originals_path
            if not originals_dir:
                raise UserFacingError("Originals backup directory not available")
            
            # Restore original
            restored = restore_from_backup(current.original_image, originals_dir)
            if not restored:
                raise UserFacingError("Original backup image not found")
            
            # Remove split image if exists
            if current.split_image and current.split_image.exists():
                remove_image(current.split_image)
            
            # Reset record
            current.split_image = None
            current.left_or_right = None
            
            self.state.mark_changed()
            return True
        except RepositoryError as e:
            raise UserFacingError(str(e)) from e
    
    def revert_all(self) -> None:
        """
        Revert all images to original.
        
        Raises:
            UserFacingError: If operation fails
        """
        originals_dir = self.temp_manager.originals_path
        if not originals_dir:
            raise UserFacingError("Originals backup directory not available")
        
        errors = []
        processed_originals = set()
        
        for record in self.state.images:
            if record.original_image in processed_originals:
                continue
            
            try:
                restored = restore_from_backup(record.original_image, originals_dir)
                if not restored:
                    errors.append(f"Could not find original backup for: {record.original_image.name}")
                    continue
                
                processed_originals.add(record.original_image)
                
                # Remove split images
                if record.split_image and record.split_image.exists():
                    remove_image(record.split_image)
                
                # Reset record
                record.split_image = None
                record.left_or_right = None
                
            except Exception as e:
                errors.append(f"Error reverting {record.original_image.name}: {str(e)}")
        
        # Rebuild state with only originals
        self.state.images = [
            ImageRecord(image_index=i+1, original_image=orig)
            for i, orig in enumerate(sorted(processed_originals))
        ]
        self.state.current_image_index = 0
        self.state.mark_changed()
        
        if errors:
            error_msg = "\n".join(errors[:10])
            if len(errors) > 10:
                error_msg += f"\n...and {len(errors) - 10} more errors"
            raise UserFacingError(f"Some images could not be reverted:\n{error_msg}")
    
    def save_images(self, output_dir: Optional[Path] = None) -> bool:
        """
        Save processed images to output directory.
        
        Args:
            output_dir: Output directory (defaults to pass_images)
            
        Returns:
            True if successful
            
        Raises:
            UserFacingError: If save fails
        """
        if not output_dir:
            # Determine output directory
            if self.temp_manager.path:
                source_dir = self.temp_manager.path.parent
            else:
                raise UserFacingError("Cannot determine output directory")
            output_dir = pass_images_for(source_dir)
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save all current images
            for record in self.state.images:
                image_path = record.current_image_path
                output_path = output_dir / image_path.name
                copy_image(image_path, output_path)
            
            self.state.mark_saved()
            logger.info(f"Saved {len(self.state.images)} images to {output_dir}")
            return True
        except (RepositoryError, PathError) as e:
            raise UserFacingError(str(e)) from e
    
    def cleanup(self) -> None:
        """Clean up temporary resources."""
        self.temp_manager.cleanup()

