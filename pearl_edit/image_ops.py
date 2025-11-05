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
    
    Left image: pixels above the cursor line (black fill below)
    Right image: pixels below the cursor line (black fill above)
    
    Uses line equation to determine which side of the line each pixel is on.
    
    Args:
        image: PIL Image object
        line_coords: Tuple of (x1, y1, x2, y2) line coordinates
        orientation: 'vertical' or 'horizontal' base orientation (unused, kept for compatibility)
        
    Returns:
        Tuple of (left_image, right_image) where left is above the line, right is below
    """
    width, height = image.size
    x1, y1, x2, y2 = line_coords
    
    # Calculate line direction vector
    dx = x2 - x1
    dy = y2 - y1
    
    # Find where the line intersects the image boundaries
    # This ensures the line extends through the entire image at the correct angle
    # We'll use parametric line equations: x = x1 + t*dx, y = y1 + t*dy
    
    intersections = []
    
    # Check intersection with left edge (x = 0)
    if abs(dx) > 1e-6:
        t = (0 - x1) / dx
        y = y1 + t * dy
        if 0 <= y <= height:
            intersections.append((0, y))
    
    # Check intersection with right edge (x = width)
    if abs(dx) > 1e-6:
        t = (width - x1) / dx
        y = y1 + t * dy
        if 0 <= y <= height:
            intersections.append((width, y))
    
    # Check intersection with top edge (y = 0)
    if abs(dy) > 1e-6:
        t = (0 - y1) / dy
        x = x1 + t * dx
        if 0 <= x <= width:
            intersections.append((x, 0))
    
    # Check intersection with bottom edge (y = height)
    if abs(dy) > 1e-6:
        t = (height - y1) / dy
        x = x1 + t * dx
        if 0 <= x <= width:
            intersections.append((x, height))
    
    # Remove duplicates (with tolerance for floating point)
    unique_intersections = []
    for point in intersections:
        is_duplicate = False
        for existing in unique_intersections:
            if abs(point[0] - existing[0]) < 1e-3 and abs(point[1] - existing[1]) < 1e-3:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_intersections.append(point)
    
    # If we have 2 intersection points, use them
    # Otherwise, use the original points (clamped) as fallback
    if len(unique_intersections) >= 2:
        # Use the two intersection points that are farthest apart
        # This ensures we span the entire image
        max_dist = 0
        best_pair = (unique_intersections[0], unique_intersections[1])
        for i in range(len(unique_intersections)):
            for j in range(i + 1, len(unique_intersections)):
                p1 = unique_intersections[i]
                p2 = unique_intersections[j]
                dist = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (p1, p2)
        x1, y1 = best_pair[0]
        x2, y2 = best_pair[1]
    else:
        # Fallback: clamp to bounds (shouldn't happen, but safety)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
    
    # Recalculate dx, dy after getting intersection points
    dx = x2 - x1
    dy = y2 - y1
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Handle edge case: vertical line
    if abs(dx) < 1e-6:
        # Vertical line: left = pixels to the left of line (x < x1)
        # Above/below doesn't make sense for vertical, so use left/right
        above_mask = (x_coords < x1).astype(np.uint8) * 255
    # Handle edge case: horizontal line
    elif abs(dy) < 1e-6:
        # Horizontal line: left = pixels above the line (y < y1)
        above_mask = (y_coords < y1).astype(np.uint8) * 255
    else:
        # General case: use line equation to determine above/below
        # Line equation: (y - y1)(x2 - x1) - (x - x1)(y2 - y1) = 0
        # For a point (x, y) to be "above" the line, it means:
        # At that x coordinate, the point's y is smaller than the line's y value
        
        line_value = (y_coords - y1) * dx - (x_coords - x1) * dy
        
        # Determine which sign means "above" by testing a point we know is above
        # Use a point with the same x as the start point but smaller y (definitely above)
        test_x = x1
        test_y = max(0, y1 - 10)  # A point 10 pixels above the start point
        test_value = (test_y - y1) * dx - (test_x - x1) * dy
        
        # If this test point gives negative value, then negative = above
        # If positive, then positive = above
        if test_value < 0:
            # Negative means above
            above_mask = (line_value < 0).astype(np.uint8) * 255
        else:
            # Positive means above
            above_mask = (line_value > 0).astype(np.uint8) * 255
    
    # Convert numpy array to PIL Image
    above_mask_img = Image.fromarray(above_mask, mode='L')
    
    # Create left and right images (full size, with black background)
    left_image = Image.new('RGB', (width, height), (0, 0, 0))
    right_image = Image.new('RGB', (width, height), (0, 0, 0))
    
    # Left image: paste pixels that are above the line
    left_image.paste(image, mask=above_mask_img)
    
    # Right image: paste pixels that are below the line (inverse mask)
    below_mask_img = ImageChops.invert(above_mask_img)
    right_image.paste(image, mask=below_mask_img)
    
    # Crop images to content (remove black edges)
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


def auto_straighten(image_path: str, threshold: int = 127) -> bool:
    """
    Automatically straighten image by detecting document edge using threshold.
    
    Uses threshold to find the largest contour, then computes the minimum area
    rectangle to determine the dominant edge angle. Rotates to nearest 0°/90°.
    
    Args:
        image_path: Path to the image file
        threshold: Threshold value for binary conversion (0-255)
        
    Returns:
        True if successful
        
    Raises:
        ImageProcessingError: If straightening fails
    """
    try:
        import math
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ImageProcessingError(f"Failed to load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Optional: Apply slight morphology to stabilize contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ImageProcessingError("No contours found in image")
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if contour is large enough (at least 1% of image area)
        height, width = image.shape[:2]
        min_area = (width * height) * 0.01
        if cv2.contourArea(largest_contour) < min_area:
            raise ImageProcessingError("Document contour too small to detect orientation")
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        # rect format: ((center_x, center_y), (width, height), angle)
        # In OpenCV, width is always >= height, and angle is in [-90, 0) degrees
        # The angle represents the rotation of the width (longest) side relative to horizontal
        _, (w, h), theta = rect
        
        # Theta is in [-90, 0) range (negative = clockwise from horizontal)
        # We want to determine if the longest side is closer to horizontal or vertical
        # and rotate accordingly to 0° (horizontal) or 90° (vertical)
        
        # Convert theta to a positive angle in [0, 90) for easier logic
        # If theta is -30°, the side is rotated 30° clockwise from horizontal
        # We want to rotate it back 30° counter-clockwise to make it horizontal
        abs_theta = abs(theta)  # Angle in [0, 90) range
        
        # Determine target orientation: if angle < 45°, rotate to horizontal (0°)
        # If angle >= 45°, rotate to vertical (90°)
        if abs_theta <= 45:
            # Closer to horizontal, rotate to 0°
            # Rotation needed is counter-clockwise by abs_theta degrees
            rotation_needed = abs_theta
        else:
            # Closer to vertical (45° to 90°), rotate to 90°
            # We need to rotate by (90 - abs_theta) degrees counter-clockwise
            # But we also need to account for the direction
            # If theta is -60°, abs_theta = 60°, we need to rotate 30° counter-clockwise
            rotation_needed = 90 - abs_theta
        
        # The rotation direction: counter-clockwise (positive) to straighten
        # Since theta is negative (clockwise), we rotate positive (counter-clockwise)
        # So rotation_needed is already positive and correct
        
        # Normalize rotation to smallest angle (mirror straighten_by_line logic)
        if abs(rotation_needed) > 90:
            if rotation_needed > 0:
                rotation_needed -= 180
            else:
                rotation_needed += 180
        
        # Only rotate if significant correction is needed (avoid micro-rotations)
        if abs(rotation_needed) < 0.5:
            logger.info(f"Image already straight (detected angle: {abs_theta:.2f}°)")
            return True
        
        # Rotate the image
        return rotate_image(image_path, rotation_needed)
        
    except ImageProcessingError:
        raise
    except Exception as e:
        logger.error(f"Error during auto-straighten: {e}")
        raise ImageProcessingError(f"Error during auto-straighten: {str(e)}") from e