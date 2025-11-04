"""Pure image processing operations without UI dependencies."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageChops
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """Raised when image processing operations fail."""
    pass


def auto_crop(image_path: str, threshold: int = 127, margin: int = 10) -> bool:
    """
    Crop image to largest white area using threshold detection.
    
    Args:
        image_path: Path to the image file
        threshold: Threshold value for binary conversion (0-255)
        margin: Margin to add around detected area
        
    Returns:
        True if successful, False otherwise
        
    Raises:
        ImageProcessingError: If image cannot be processed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ImageProcessingError(f"Failed to load image: {image_path}")
            
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning(f"No contours found in {image_path}")
            return False
            
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle with margin
        x, y, w, h = cv2.boundingRect(largest_contour)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(width - x, w + 2 * margin)
        h = min(height - y, h + 2 * margin)
        
        # Crop the image
        cropped = image[y:y+h, x:x+w]
        
        # Save the cropped image
        cv2.imwrite(image_path, cropped)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during auto-crop: {e}")
        raise ImageProcessingError(f"Error during auto-crop: {str(e)}") from e


def crop_image(image_path: str, coords: Tuple[int, int, int, int]) -> bool:
    """
    Crop image to specified coordinates.
    
    Args:
        image_path: Path to the image file
        coords: Tuple of (left, top, right, bottom) coordinates
        
    Returns:
        True if successful
        
    Raises:
        ImageProcessingError: If crop operation fails
    """
    try:
        left, top, right, bottom = coords
        with Image.open(image_path) as image:
            cropped_image = image.crop((left, top, right, bottom))
            cropped_image.save(image_path)
        return True
    except Exception as e:
        logger.error(f"Error applying crop: {e}")
        raise ImageProcessingError(f"Error applying crop: {str(e)}") from e


def split_image_vertical(image: Image.Image, split_x: int) -> Tuple[Image.Image, Image.Image]:
    """
    Split image vertically at specified x coordinate.
    
    Args:
        image: PIL Image object
        split_x: X coordinate to split at
        
    Returns:
        Tuple of (left_image, right_image)
    """
    width, height = image.size
    split_x = max(0, min(split_x, width))
    
    left_image = image.crop((0, 0, split_x, height))
    right_image = image.crop((split_x, 0, width, height))
    
    return left_image, right_image


def split_image_horizontal(image: Image.Image, split_y: int) -> Tuple[Image.Image, Image.Image]:
    """
    Split image horizontally at specified y coordinate.
    
    Args:
        image: PIL Image object
        split_y: Y coordinate to split at
        
    Returns:
        Tuple of (top_image, bottom_image)
    """
    width, height = image.size
    split_y = max(0, min(split_y, height))
    
    top_image = image.crop((0, 0, width, split_y))
    bottom_image = image.crop((0, split_y, width, height))
    
    return top_image, bottom_image


def split_image_angled(
    image: Image.Image,
    line_coords: Tuple[int, int, int, int],
    orientation: str = 'vertical'
) -> Tuple[Image.Image, Image.Image]:
    """
    Split image along an angled line.
    
    Args:
        image: PIL Image object
        line_coords: Tuple of (x1, y1, x2, y2) line coordinates
        orientation: 'vertical' or 'horizontal' base orientation
        
    Returns:
        Tuple of (left/top_image, right/bottom_image)
    """
    width, height = image.size
    x1, y1, x2, y2 = line_coords
    
    # Create mask images
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Determine how to fill the mask based on orientation
    if orientation == 'horizontal' or (x2 - x1) == 0:
        # For horizontal-based splits, fill above the line
        points = [(0, 0), (width, 0), (x2, y2), (x1, y1)]
    else:
        # For vertical-based splits, fill left of the line
        points = [(0, 0), (x1, y1), (x2, y2), (0, height)]
        
    # Draw the polygon to create the mask
    draw.polygon(points, fill=255)
    
    # Create left and right images
    left_image = Image.new('RGB', (width, height), (0, 0, 0))
    right_image = Image.new('RGB', (width, height), (0, 0, 0))
    
    # Copy the appropriate parts of the original image
    left_image.paste(image, mask=mask)
    right_image.paste(image, mask=ImageChops.invert(mask))
    
    # Crop images to content
    left_bbox = left_image.convert('L').getbbox()
    right_bbox = right_image.convert('L').getbbox()
    
    if left_bbox:
        left_image = left_image.crop(left_bbox)
    if right_bbox:
        right_image = right_image.crop(right_bbox)
        
    return left_image, right_image


def rotate_image(image_path: str, angle: float) -> bool:
    """
    Rotate image by specified angle.
    
    Args:
        image_path: Path to the image file
        angle: Rotation angle in degrees (positive = counter-clockwise)
        
    Returns:
        True if successful
        
    Raises:
        ImageProcessingError: If rotation fails
    """
    try:
        with Image.open(image_path) as image:
            rotated_image = image.rotate(angle, expand=True, resample=Image.BICUBIC)
            rotated_image.save(image_path, 'JPEG', quality=95, optimize=True)
        return True
    except Exception as e:
        logger.error(f"Error rotating image: {e}")
        raise ImageProcessingError(f"Error rotating image: {str(e)}") from e


def straighten_by_line(
    image_path: str,
    start_point: Tuple[int, int],
    end_point: Tuple[int, int]
) -> bool:
    """
    Straighten image by rotating based on line drawn between two points.
    
    Args:
        image_path: Path to the image file
        start_point: Tuple of (x, y) start point
        end_point: Tuple of (x, y) end point
        
    Returns:
        True if successful
        
    Raises:
        ImageProcessingError: If straightening fails
    """
    try:
        import math
        
        # Calculate angle between points
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        angle = math.degrees(math.atan2(dy, dx))
        
        # Determine if line is more vertical or horizontal
        is_vertical = abs(dx) < abs(dy)
        
        if is_vertical:
            # For vertical lines, normalize angle to be relative to 90°
            target_angle = 90
            current_angle = angle % 180
            if current_angle < 0:
                current_angle += 180
        else:
            # For horizontal lines, normalize angle to be relative to 0°
            target_angle = 0
            current_angle = angle % 180
            if current_angle < 0:
                current_angle += 180
        
        # Calculate the minimum rotation needed
        rotation_needed = target_angle - current_angle
        
        # Normalize rotation to smallest angle
        if abs(rotation_needed) > 90:
            if rotation_needed > 0:
                rotation_needed -= 180
            else:
                rotation_needed += 180
        
        # Rotate the image
        return rotate_image(image_path, rotation_needed)
        
    except Exception as e:
        logger.error(f"Error straightening image: {e}")
        raise ImageProcessingError(f"Error straightening image: {str(e)}") from e


def find_optimal_threshold(image_path: str) -> int:
    """
    Automatically find the optimal threshold for document detection.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Optimal threshold value (0-255)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ImageProcessingError(f"Failed to load image: {image_path}")
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize variables for best threshold
    best_threshold = 0
    max_score = 0
    
    # Try different threshold values
    for threshold in range(0, 255, 5):
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Calculate metrics for this threshold
        white_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        white_ratio = white_pixels / total_pixels
        
        # Prefer thresholds that result in moderate white ratios (not too much or too little)
        # This helps find document boundaries
        score = white_ratio * (1 - abs(white_ratio - 0.5))
        
        if score > max_score:
            max_score = score
            best_threshold = threshold
    
    return best_threshold

