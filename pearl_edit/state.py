"""State management for image editing session."""
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class ImageRecord:
    """Represents a single image record in the session."""
    image_index: int
    original_image: Path
    split_image: Optional[Path] = None
    left_or_right: Optional[str] = None  # 'Left' or 'Right'
    
    @property
    def current_image_path(self) -> Path:
        """Get the current active image path (split if exists, otherwise original)."""
        return self.split_image if self.split_image else self.original_image


@dataclass
class SessionState:
    """Manages the current editing session state."""
    current_image_index: int = 0
    images: List[ImageRecord] = field(default_factory=list)
    status: str = "no_changes"  # "no_changes", "saved", "discarded"
    
    def get_current_image(self) -> Optional[ImageRecord]:
        """Get the current image record."""
        if 0 <= self.current_image_index < len(self.images):
            return self.images[self.current_image_index]
        return None
    
    def navigate(self, direction: int) -> bool:
        """
        Navigate to a different image.
        
        Args:
            direction: -2 = first, -1 = previous, 1 = next, 2 = last
            
        Returns:
            True if navigation was successful
        """
        total = len(self.images) - 1
        if total < 0:
            return False
            
        if direction == -2:  # First
            self.current_image_index = 0
        elif direction == -1:  # Previous
            self.current_image_index = max(0, self.current_image_index - 1)
        elif direction == 1:  # Next
            self.current_image_index = min(total, self.current_image_index + 1)
        elif direction == 2:  # Last
            self.current_image_index = total
        else:
            return False
            
        return True
    
    def add_image(self, record: ImageRecord) -> None:
        """Add an image record to the session."""
        self.images.append(record)
    
    def remove_image(self, index: int) -> bool:
        """Remove an image record at the given index."""
        if 0 <= index < len(self.images):
            del self.images[index]
            # Reindex remaining images
            for i, img in enumerate(self.images):
                img.image_index = i + 1
            # Adjust current index if necessary
            if self.current_image_index >= len(self.images):
                self.current_image_index = max(0, len(self.images) - 1)
            return True
        return False
    
    def mark_changed(self) -> None:
        """Mark the session as having unsaved changes."""
        if self.status == "no_changes":
            self.status = "changed"
    
    def mark_saved(self) -> None:
        """Mark the session as saved."""
        self.status = "saved"
    
    def mark_discarded(self) -> None:
        """Mark the session as discarded."""
        self.status = "discarded"




