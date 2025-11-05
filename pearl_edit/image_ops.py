"""Pure image processing operations without UI dependencies."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageChops
from typing import Tuple, Optional
import logging
import math

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
    
    Rotates the image so the drawn line aligns to the nearest axis (0° or 90°)
    using the smallest rotation angle.
    
    Args:
        image_path: Path to the image file
        start_point: Tuple of (x, y) start point in image coordinates
        end_point: Tuple of (x, y) end point in image coordinates
        
    Returns:
        True if successful
        
    Raises:
        ImageProcessingError: If straightening fails
    """
    try:
        import math
        
        start_x, start_y = start_point
        end_x, end_y = end_point
        
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Check for very short lines (less than 2 pixels)
        line_length = math.sqrt(dx * dx + dy * dy)
        if line_length < 2.0:
            logger.warning(f"Line too short to determine orientation (length: {line_length:.2f})")
            return False
        
        # Calculate angle in degrees, normalized to [-90°, 90°]
        phi = math.degrees(math.atan2(dy, dx))
        while phi <= -90:
            phi += 180
        while phi > 90:
            phi -= 180
        
        # Calculate two candidate rotations (screen coords, y-down):
        # After rotating CCW by theta, the screen angle becomes phi' = phi - theta.
        # To align to target T, choose theta = phi - T.
        
        # 1. Rotate to horizontal (T = 0°)
        delta_h = phi  # theta = phi - 0
        
        # 2. Rotate to vertical (T = +90° or -90°, whichever closer)
        target_v = 90 if abs(phi - 90) <= abs(phi + 90) else -90
        delta_v = phi - target_v
        
        # Choose the rotation with smaller absolute value
        # On ties (exactly 45°), prefer horizontal (0°)
        if abs(delta_h) <= abs(delta_v):
            rotation_needed = delta_h
        else:
            rotation_needed = delta_v
        
        # Add deadzone to avoid micro-rotations on already-straight images
        if abs(rotation_needed) < 0.25:
            logger.info(f"Image already straight (detected angle: {phi:.2f}°, rotation: {rotation_needed:.2f}°)")
            return True
        
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
        # To straighten, we need to rotate counter-clockwise (positive direction)
        # If |theta| <= 45°, rotate toward horizontal (0°): rotation = -theta
        # If |theta| > 45°, rotate toward vertical (90°): rotation = 90 - |theta|
        
        abs_theta = abs(theta)  # Angle in [0, 90) range
        
        if abs_theta <= 45:
            # Closer to horizontal, rotate to 0°
            # Theta is negative (e.g., -30°), so -theta gives positive rotation (counter-clockwise)
            rotation_needed = -theta
        else:
            # Closer to vertical (45° to 90°), rotate to 90°
            # Rotate counter-clockwise by (90 - abs_theta) degrees
            rotation_needed = 90 - abs_theta
        
        # Add deadzone to avoid micro-rotations on already-straight images
        if abs(rotation_needed) < 0.25:
            logger.info(f"Image already straight (detected angle: {abs_theta:.2f}°)")
            return True
        
        # Rotate the image
        return rotate_image(image_path, rotation_needed)
        
    except ImageProcessingError:
        raise
    except Exception as e:
        logger.error(f"Error during auto-straighten: {e}")
        raise ImageProcessingError(f"Error during auto-straighten: {str(e)}") from e


def detect_gutter_line(
    image: np.ndarray,
    threshold: int,
    angle_max_deg: float = 12.0,
    min_length_ratio: float = 0.6,
    scale_factor: float = 0.5
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """
    Detect the gutter seam line in a book scan image.
    
    Args:
        image: BGR image as numpy array
        threshold: Threshold value for black-hat binary mask (0-255)
        angle_max_deg: Maximum angle deviation from vertical (degrees)
        min_length_ratio: Minimum line length as ratio of image height
        scale_factor: Scale down image for faster processing (0.0-1.0)
        
    Returns:
        Tuple of (line_coords, confidence) where:
        - line_coords: (x1, y1, x2, y2) in original image space, or None if not found
        - confidence: float in [0, 1], higher is better
    """
    try:
        if image is None or image.size == 0:
            return None, 0.0
        
        height, width = image.shape[:2]
        if height < 50 or width < 50:
            logger.warning("Image too small for seam detection")
            return None, 0.0
        
        # Work on scaled-down version for speed
        work_height = int(height * scale_factor)
        work_width = int(width * scale_factor)
        if work_height < 20 or work_width < 20:
            scale_factor = 1.0
            work_height = height
            work_width = width
        
        if scale_factor < 1.0:
            work_image = cv2.resize(image, (work_width, work_height), interpolation=cv2.INTER_AREA)
        else:
            work_image = image.copy()
        
        # Preprocess: grayscale -> CLAHE -> blur
        gray = cv2.cvtColor(work_image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Small blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Multi-scale black-hat morphology to emphasize vertical dark seams
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 17)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 31))
        ]
        
        blackhat = np.zeros_like(gray, dtype=np.float32)
        for kernel in kernels:
            bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            blackhat = blackhat + bh.astype(np.float32)
        
        # Normalize to 0-255
        blackhat = np.clip(blackhat, 0, 255).astype(np.uint8)
        
        # Threshold to binary mask
        _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply Canny edge detection on mask
        edges = cv2.Canny(mask, 50, 150)
        
        # HoughLinesP parameters
        angle_max_rad = math.radians(angle_max_deg)
        min_length = int(work_height * min_length_ratio)
        
        # Find lines with HoughLinesP
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,  # 1 degree resolution
            threshold=max(10, min_length // 10),
            minLineLength=min_length,
            maxLineGap=max(5, min_length // 20)
        )
        
        best_line = None
        best_score = 0.0
        candidates = []
        
        if lines is not None:
            center_x = work_width / 2.0
            center_y = work_height / 2.0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line properties
                dx = x2 - x1
                dy = y2 - y1
                length = math.sqrt(dx * dx + dy * dy)
                
                if length < min_length:
                    continue
                
                # Calculate angle from vertical
                angle = math.atan2(abs(dx), abs(dy))  # angle from vertical
                if angle > angle_max_rad:
                    continue
                
                # Calculate center of line
                line_center_x = (x1 + x2) / 2.0
                line_center_y = (y1 + y2) / 2.0
                
                # Center bias: prefer lines near image center
                center_dist = abs(line_center_x - center_x)
                center_bias = 1.0 - min(center_dist / (work_width * 0.5), 1.0)
                
                # Calculate intensity along line in blackhat image
                # Sample points along the line
                num_samples = max(10, int(length / 5))
                intensities = []
                for i in range(num_samples):
                    t = i / (num_samples - 1) if num_samples > 1 else 0
                    x = int(x1 + t * dx)
                    y = int(y1 + t * dy)
                    if 0 <= x < work_width and 0 <= y < work_height:
                        intensities.append(blackhat[y, x])
                
                if not intensities:
                    continue
                
                avg_intensity = np.mean(intensities) / 255.0
                
                # Coverage: how much of image height the line covers
                y_min = min(y1, y2)
                y_max = max(y1, y2)
                coverage = (y_max - y_min) / work_height
                
                # Angle bias: prefer more vertical lines
                angle_bias = 1.0 - (angle / angle_max_rad)
                
                # Combined score (weights: 0.5 intensity, 0.2 coverage, 0.2 center, 0.1 angle)
                score = (
                    0.5 * avg_intensity +
                    0.2 * coverage +
                    0.2 * center_bias +
                    0.1 * angle_bias
                )
                
                candidates.append({
                    'line': (x1, y1, x2, y2),
                    'score': score,
                    'intensity': avg_intensity,
                    'coverage': coverage,
                    'center_bias': center_bias,
                    'angle_bias': angle_bias
                })
                
                if score > best_score:
                    best_score = score
                    best_line = (x1, y1, x2, y2)
        
        # Fallback: sweep small angles if no good Hough line found
        if best_line is None or best_score < 0.3:
            logger.debug("No good Hough line found, trying angle sweep fallback")
            
            # Try lines at different angles through center
            best_fallback_score = 0.0
            best_fallback_line = None
            
            center_x_int = work_width // 2
            angle_range = math.radians(angle_max_deg * 2)  # Allow ±angle_max_deg
            num_angles = 20
            offset_range = work_width * 0.1  # ±10% width
            
            for angle_idx in range(num_angles):
                angle = -angle_range / 2 + (angle_idx / (num_angles - 1)) * angle_range
                
                for offset_idx in range(11):  # 11 offsets including center
                    offset = -offset_range + (offset_idx / 10.0) * offset_range * 2
                    x_center = center_x_int + offset
                    
                    # Calculate line endpoints
                    dx_per_pixel = math.tan(angle)
                    half_height = work_height / 2
                    
                    x1 = int(x_center - half_height * dx_per_pixel)
                    x2 = int(x_center + half_height * dx_per_pixel)
                    y1 = 0
                    y2 = work_height - 1
                    
                    if x1 < 0 or x1 >= work_width or x2 < 0 or x2 >= work_width:
                        continue
                    
                    # Sample intensity along this line
                    intensities = []
                    num_samples = 50
                    for i in range(num_samples):
                        t = i / (num_samples - 1) if num_samples > 1 else 0
                        x = int(x1 + t * (x2 - x1))
                        y = int(y1 + t * (y2 - y1))
                        if 0 <= x < work_width and 0 <= y < work_height:
                            # Sample a small band around the line
                            band_sum = 0
                            band_count = 0
                            for bx in range(max(0, x - 1), min(work_width, x + 2)):
                                for by in range(max(0, y - 1), min(work_height, y + 2)):
                                    band_sum += blackhat[by, bx]
                                    band_count += 1
                            if band_count > 0:
                                intensities.append(band_sum / band_count)
                    
                    if not intensities:
                        continue
                    
                    avg_intensity = np.mean(intensities) / 255.0
                    
                    if avg_intensity > best_fallback_score:
                        best_fallback_score = avg_intensity
                        best_fallback_line = (x1, y1, x2, y2)
            
            if best_fallback_line and best_fallback_score > 0.15:
                best_line = best_fallback_line
                best_score = best_fallback_score * 0.8  # Slightly lower confidence for fallback
        
        if best_line is None:
            return None, 0.0
        
        # Scale coordinates back to original image size
        if scale_factor < 1.0:
            x1, y1, x2, y2 = best_line
            x1 = int(x1 / scale_factor)
            y1 = int(y1 / scale_factor)
            x2 = int(x2 / scale_factor)
            y2 = int(y2 / scale_factor)
            best_line = (x1, y1, x2, y2)
        
        # Calculate confidence
        # Base confidence from score, but also consider margin over second-best
        confidence = min(1.0, best_score)
        
        if len(candidates) > 1:
            # Sort by score descending
            candidates_sorted = sorted(candidates, key=lambda c: c['score'], reverse=True)
            if candidates_sorted[0]['score'] > 0:
                ratio = candidates_sorted[1]['score'] / candidates_sorted[0]['score'] if len(candidates_sorted) > 1 else 0
                # Higher margin = higher confidence
                margin_factor = 1.0 - (ratio * 0.5)  # Reduce confidence if close second exists
                confidence *= margin_factor
        
        # Ensure confidence is reasonable
        confidence = max(0.0, min(1.0, confidence))
        
        logger.debug(f"Detected gutter line with confidence {confidence:.2f}, score {best_score:.2f}")
        
        return best_line, confidence
        
    except Exception as e:
        logger.error(f"Error detecting gutter line: {e}")
        return None, 0.0


def find_optimal_seam_threshold(
    image: np.ndarray,
    angle_max_deg: float = 12.0,
    min_length_ratio: float = 0.6
) -> int:
    """
    Find the optimal threshold for seam detection by grid search.
    
    Args:
        image: BGR image as numpy array
        angle_max_deg: Maximum angle deviation from vertical (degrees)
        min_length_ratio: Minimum line length as ratio of image height
        
    Returns:
        Optimal threshold value (0-255) that maximizes confidence
    """
    try:
        if image is None or image.size == 0:
            return 140  # Default
        
        best_threshold = 140
        best_confidence = 0.0
        
        # Coarse grid search
        for threshold in range(50, 200, 10):
            line_coords, confidence = detect_gutter_line(
                image, threshold, angle_max_deg, min_length_ratio
            )
            if confidence > best_confidence:
                best_confidence = confidence
                best_threshold = threshold
        
        # Fine refinement around best threshold
        if best_threshold > 50 and best_threshold < 200:
            for threshold in range(max(50, best_threshold - 9), min(200, best_threshold + 10)):
                line_coords, confidence = detect_gutter_line(
                    image, threshold, angle_max_deg, min_length_ratio
                )
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_threshold = threshold
        
        logger.debug(f"Optimal seam threshold: {best_threshold} (confidence: {best_confidence:.2f})")
        return best_threshold
        
    except Exception as e:
        logger.error(f"Error finding optimal seam threshold: {e}")
        return 140  # Default fallback