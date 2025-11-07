"""Pure image processing operations without UI dependencies."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageChops
from typing import Tuple, Optional
import logging
import math
import concurrent.futures

logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """Raised when image processing operations fail."""
    pass


def auto_crop(image_path: str, threshold: int = 127, margin: int = 10) -> bool:
    """
    Crop the image to the detected document region using a robust pipeline.

    This uses the shared `compute_document_crop_rect` helper so behavior matches
    the preview. The helper performs thresholding, light morphology, border
    suppression, contour filtering, and (optionally) unions two side‑by‑side
    page regions.
    
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
        
        # Convert to grayscale once and delegate crop rectangle computation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rect = compute_document_crop_rect(gray, threshold, margin)
        if rect is None:
            logger.warning(f"Auto-crop: no valid document region found for {image_path}")
            return False

        x, y, w, h = rect
        cropped = image[y:y+h, x:x+w]
        cv2.imwrite(image_path, cropped)
        return True
        
    except ImageProcessingError:
        raise
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
        # OpenCV returns angle in [-90, 0). Define orientation of the long edge:
        # if height > width, add 90 so angle reflects long-edge tilt relative to horizontal.
        (cx, cy), (w, h), theta = rect

        # Normalize angle so it represents the long edge orientation in (-90, 90]
        angle = float(theta)
        if w < h:
            angle = angle + 90.0

        # Rotate toward the nearest canonical orientation: -90°, 0°, or 90°
        targets = (-90.0, 0.0, 90.0)
        best_target = min(targets, key=lambda t: abs(angle - t))
        rotation_needed = angle - best_target
        
        # Add deadzone to avoid micro-rotations on already-straight images
        if abs(rotation_needed) < 0.25:
            logger.info(f"Image already straight (detected angle: {abs(angle):.2f}°)")
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


def compute_largest_white_roi(
    gray: np.ndarray,
    roi_threshold: int,
    margin: int = 8
) -> Tuple[int, int, int, int]:
    """
    Compute the bounding rectangle of the largest white mass in the image.
    
    Uses thresholding to find the largest contour, which represents the document
    area, excluding black borders and film edges.
    
    Args:
        gray: Grayscale image as numpy array
        roi_threshold: Threshold value for binary conversion (0-255)
        margin: Margin to add around detected area (default: 8)
        
    Returns:
        Tuple of (x, y, w, h) bounding rectangle, or (0, 0, width, height) if no contour found
    """
    try:
        if gray is None or gray.size == 0:
            return (0, 0, 0, 0)
        
        height, width = gray.shape[:2]
        
        # Apply threshold
        _, binary = cv2.threshold(gray, roi_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No contours found, using full image")
            return (0, 0, width, height)
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if contour is large enough (at least 1% of image area)
        min_area = (width * height) * 0.01
        if cv2.contourArea(largest_contour) < min_area:
            logger.warning("Largest contour too small, using full image")
            return (0, 0, width, height)
        
        # Get bounding rectangle with margin
        x, y, w, h = cv2.boundingRect(largest_contour)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(width - x, w + 2 * margin)
        h = min(height - y, h + 2 * margin)
        
        return (x, y, w, h)
        
    except Exception as e:
        logger.error(f"Error computing largest white ROI: {e}")
        if gray is not None and gray.size > 0:
            height, width = gray.shape[:2]
            return (0, 0, width, height)
        return (0, 0, 0, 0)


def detect_gutter_line_two_pages(
    image: np.ndarray,
    roi_threshold: int,
    angle_max_deg: float = 20.0,
    scale_width: int = 1200,
    sensitivity: Optional[int] = None
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """
    Fast seam detector for already-cropped two-page documents using two components.

    Approach:
    - Downscale to a fixed working width for speed; operate at that scale.
    - Threshold to white-on-black using the provided ROI threshold (Otsu fallback).
    - Morphology: square close to connect text islands; vertical open to break
      thin white bridges in the gutter so two pages remain separate.
    - Keep two largest white components; if side-by-side with strong vertical
      overlap, compute the midline between their facing edges per row and fit a
      straight line x = m*y + b (clamped to ±angle_max_deg from vertical).

    Args:
        image: BGR image as numpy array (full resolution)
        roi_threshold: Binary threshold for page whitening (0-255)
        angle_max_deg: Max allowed angle from vertical for the seam
        scale_width: Working width for downscale (keeps runtime bounded)
        sensitivity: Optional 0-255; controls vertical open kernel length

    Returns:
        (line_coords, confidence) where line_coords=(x1,y1,x2,y2) in original
        image coordinates, or (None, 0.0) if not detected.
    """
    try:
        if image is None or image.size == 0:
            return None, 0.0

        # Downscale for speed
        h0, w0 = image.shape[:2]
        if w0 > scale_width:
            scale = scale_width / float(w0)
            work_w = int(round(w0 * scale))
            work_h = int(round(h0 * scale))
            work_img = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            work_img = image
            work_h, work_w = h0, w0

        # Grayscale + optional CLAHE for microfilm contrast
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass

        # Threshold; Otsu fallback if very few white pixels
        _, binary = cv2.threshold(gray, roi_threshold, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(binary) < 0.02 * binary.size:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphology: close then vertical open
        short_side = min(work_h, work_w)
        k_close = max(3, int(round(short_side * 0.006)))
        if k_close % 2 == 0:
            k_close += 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        # Map sensitivity (0-255) to vertical kernel as ~1%-2% of height
        if sensitivity is None:
            vert_ratio = 0.015
        else:
            vert_ratio = 0.01 + 0.01 * max(0, min(255, sensitivity)) / 255.0
        k_vert = max(7, int(round(work_h * vert_ratio)))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_vert))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)

        # Connected components
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num <= 2:
            return None, 0.0

        # Pick two largest components (exclude background at index 0)
        idx_sorted = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
        ids = idx_sorted[:2]
        if len(ids) < 2:
            return None, 0.0

        area1 = float(stats[ids[0], cv2.CC_STAT_AREA])
        area2 = float(stats[ids[1], cv2.CC_STAT_AREA])

        # Masks for two largest components
        mask1 = (labels == ids[0])
        mask2 = (labels == ids[1])

        # Ensure left/right assignment by mean x position
        xs1 = np.where(mask1)[1]
        xs2 = np.where(mask2)[1]
        if xs1.size == 0 or xs2.size == 0:
            return None, 0.0
        if xs1.mean() > xs2.mean():
            mask1, mask2 = mask2, mask1
            area1, area2 = area2, area1

        # Validate side-by-side via vertical overlap
        rows1 = np.where(mask1.any(axis=1))[0]
        rows2 = np.where(mask2.any(axis=1))[0]
        if rows1.size == 0 or rows2.size == 0:
            return None, 0.0
        y1_min, y1_max = rows1[0], rows1[-1]
        y2_min, y2_max = rows2[0], rows2[-1]
        v_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min) + 1)
        min_h = float(min(y1_max - y1_min + 1, y2_max - y2_min + 1))
        if min_h <= 0:
            return None, 0.0
        v_ratio = v_overlap / min_h
        if v_ratio < 0.5:
            return None, 0.0

        # Row-wise facing edges using vectorized scans
        # Rightmost x of left page per row
        mask1_any = mask1.any(axis=1)
        rev1 = np.flip(mask1, axis=1)
        idx_from_right = np.argmax(rev1, axis=1)
        xL_all = work_w - idx_from_right - 1
        xL_all[~mask1_any] = -1

        # Leftmost x of right page per row
        mask2_any = mask2.any(axis=1)
        idx_from_left = np.argmax(mask2, axis=1)
        # If a row has no True, argmax returns 0; invalidate using mask
        xR_all = idx_from_left
        xR_all[~mask2_any] = -1

        rows = np.arange(work_h)
        valid = (xL_all >= 0) & (xR_all >= 0) & (xR_all > xL_all)
        if not np.any(valid):
            return None, 0.0

        y_samples = rows[valid].astype(np.float32)
        x_gap = (xR_all[valid] - xL_all[valid]).astype(np.float32)
        x_mid = (xR_all[valid] + xL_all[valid]).astype(np.float32) * 0.5

        # Robust least squares for x = m*y + b
        A = np.vstack([y_samples, np.ones_like(y_samples)]).T
        m, b = np.linalg.lstsq(A, x_mid, rcond=None)[0]

        # Clamp angle from vertical
        angle_from_vertical = math.degrees(math.atan(abs(m)))
        if angle_from_vertical > angle_max_deg:
            m = math.tan(math.radians(angle_max_deg)) * (1 if m >= 0 else -1)
            b = float(np.mean(x_mid - m * y_samples))

        # Build endpoints at this scale
        y1 = 0
        y2 = work_h - 1
        x1 = int(round(m * y1 + b))
        x2 = int(round(m * y2 + b))
        x1 = max(0, min(work_w - 1, x1))
        x2 = max(0, min(work_w - 1, x2))

        # Confidence: coverage, area balance, normalized gap
        coverage = float(np.count_nonzero(valid)) / float(work_h)
        area_balance = float(min(area1, area2)) / float(max(area1, area2))
        gap_med = float(np.median(x_gap)) / float(max(1, work_w))
        gap_norm = max(0.0, min(1.0, gap_med / 0.15))  # 15% width -> strong gap
        confidence = (
            0.5 * coverage +
            0.2 * area_balance +
            0.3 * gap_norm
        )
        confidence = max(0.0, min(1.0, confidence))

        # Scale back to original image coordinates if needed
        if scale != 1.0:
            x1 = int(round(x1 / scale))
            y1 = int(round(y1 / scale))
            x2 = int(round(x2 / scale))
            y2 = int(round(y2 / scale))

        return (x1, y1, x2, y2), confidence

    except Exception as e:
        logger.error(f"Error in two-pages seam detection: {e}")
        return None, 0.0


def detect_seam_by_page_border(
    image: np.ndarray,
    roi_threshold: int,
    margin: int = 20,
    scale_width: int = 1200
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """
    Detect seam by finding one page and using its inner border nearest image center.

    Approach:
    - Downscale to fixed working width for speed.
    - Threshold to white-on-black using ROI threshold (Otsu fallback).
    - Morphology: square close (fills text holes), vertical open (breaks narrow
      white bridges in gutter).
    - Find largest white component (one page).
    - Get its bounding rect; choose inner border (left side if page is right of
      center, right side if page is left of center).
    - Compute split_x = border_x ± margin toward image center.
    - Return vertical line at split_x and confidence based on area ratio.

    Args:
        image: BGR image as numpy array (full resolution)
        roi_threshold: Binary threshold for page whitening (0-255)
        margin: Pixels to shift split line toward image center (default: 20)
        scale_width: Working width for downscale (keeps runtime bounded)

    Returns:
        (line_coords, confidence) where line_coords=(x1,y1,x2,y2) is a vertical
        line in original image coordinates, or (None, 0.0) if not detected.
    """
    try:
        if image is None or image.size == 0:
            return None, 0.0

        # Downscale for speed
        h0, w0 = image.shape[:2]
        if w0 > scale_width:
            scale = scale_width / float(w0)
            work_w = int(round(w0 * scale))
            work_h = int(round(h0 * scale))
            work_img = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            work_img = image
            work_h, work_w = h0, w0

        # Grayscale + optional CLAHE for microfilm contrast
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass

        # Threshold; Otsu fallback if very few white pixels
        _, binary = cv2.threshold(gray, roi_threshold, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(binary) < 0.02 * binary.size:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphology: close then vertical open
        short_side = min(work_h, work_w)
        k_close = max(3, int(round(short_side * 0.006)))
        if k_close % 2 == 0:
            k_close += 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        # Vertical open to break thin white bridges in gutter
        vert_ratio = 0.015  # ~1.5% of height
        k_vert = max(7, int(round(work_h * vert_ratio)))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_vert))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)

        # Connected components - find largest (one page)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num <= 1:
            return None, 0.0

        # Get largest component (exclude background at index 0)
        idx_sorted = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
        largest_id = idx_sorted[0]
        area = float(stats[largest_id, cv2.CC_STAT_AREA])

        # Get bounding rectangle
        x = int(stats[largest_id, cv2.CC_STAT_LEFT])
        y = int(stats[largest_id, cv2.CC_STAT_TOP])
        w = int(stats[largest_id, cv2.CC_STAT_WIDTH])
        h = int(stats[largest_id, cv2.CC_STAT_HEIGHT])

        # Determine which border is closer to image center
        center_x = work_w / 2.0
        page_center_x = x + w / 2.0

        # Choose inner border: if page is left of center, use right border; if right, use left border
        if page_center_x < center_x:
            # Page is on left, use right border (x + w)
            border_x = x + w
            # Shift margin pixels toward center (to the right)
            split_x = border_x + margin
        else:
            # Page is on right, use left border (x)
            border_x = x
            # Shift margin pixels toward center (to the left)
            split_x = border_x - margin

        # Clamp to image bounds
        split_x = max(0, min(work_w - 1, int(round(split_x))))

        # Scale back to original image coordinates if needed
        if scale != 1.0:
            split_x = int(round(split_x / scale))

        # Build vertical line coordinates
        x1 = split_x
        y1 = 0
        x2 = split_x
        y2 = h0 - 1

        # Confidence: area ratio and distance from center
        image_area = float(work_w * work_h)
        area_ratio = area / image_area if image_area > 0 else 0.0
        # Distance from center (normalized) - closer to center is better
        center_dist = abs(page_center_x - center_x) / (work_w / 2.0) if work_w > 0 else 1.0
        center_bias = 1.0 - min(center_dist, 1.0)

        confidence = (
            0.7 * min(1.0, area_ratio / 0.3) +  # Area ratio (expect ~30% for one page)
            0.3 * center_bias  # Center bias
        )
        confidence = max(0.0, min(1.0, confidence))

        return (x1, y1, x2, y2), confidence

    except Exception as e:
        logger.error(f"Error in page-border seam detection: {e}")
        return None, 0.0


def detect_vertical_trench_near_center(
    image: np.ndarray,
    roi_threshold: int,
    center_band_ratio: float = 0.10,
    scale_width: int = 1200
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """
    Detect a near-vertical gutter ('trench') within ±center_band_ratio of image center.
    Returns a single vertical seam line and a confidence score in [0, 1].

    Improved pipeline:
    - Robust ROI via compute_document_crop_rect (fallback to largest white ROI)
    - Binary whitening + black-hat with vertical kernel to enhance dark trench
    - Vertical close to connect breaks; light horizontal open to suppress text bridges
    - Column-wise score in central band ∩ ROI using fused maps
    - Refine seam by per-row local maxima near best column; confidence uses
      prominence, center proximity, and continuity of per-row detections
    """
    try:
        if image is None or image.size == 0:
            return None, 0.0

        h0, w0 = image.shape[:2]

        # Downscale to bounded working size
        if w0 > scale_width:
            scale = scale_width / float(w0)
            work_w = int(round(w0 * scale))
            work_h = int(round(h0 * scale))
            work_img = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            work_img = image
            work_h, work_w = h0, w0

        # Grayscale + optional CLAHE
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass

        # Whitening with Otsu fallback
        _, binary = cv2.threshold(gray, roi_threshold, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(binary) < 0.02 * binary.size:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ROI detection (prefer robust crop rect; fallback to largest white ROI)
        roi = compute_document_crop_rect(gray, threshold=roi_threshold, margin=8)
        if roi is None:
            rx, ry, rw, rh = compute_largest_white_roi(gray, roi_threshold, margin=8)
        else:
            rx, ry, rw, rh = roi
        rx = max(0, min(rx, work_w - 1))
        ry = max(0, min(ry, work_h - 1))
        rw = max(1, min(rw, work_w - rx))
        rh = max(1, min(rh, work_h - ry))

        # Central band limits
        cx = work_w * 0.5
        band_half = int(round(work_w * center_band_ratio))
        band_x1 = max(0, int(cx - band_half))
        band_x2 = min(work_w, int(cx + band_half))
        if band_x2 - band_x1 < 3:
            return None, 0.0

        # Enhance dark trench: black-hat with vertical kernel + dark mask
        short_side = min(work_h, work_w)
        k_vert = max(7, int(round(work_h * 0.02)))
        if k_vert % 2 == 0:
            k_vert += 1
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_vert))
        # Black-hat highlights dark lines on bright background
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, vert_kernel)
        blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

        dark = (binary == 0).astype(np.uint8) * 255
        # Connect breaks; suppress horizontal noise
        dark_connected = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, vert_kernel, iterations=1)
        k_h = max(5, int(round(work_w * 0.01)))
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_h, 1))
        dark_clean = cv2.morphologyEx(dark_connected, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

        # Fuse maps: emphasize consistent vertical darkness
        dark_clean_f = dark_clean.astype(np.float32) / 255.0
        blackhat_f = blackhat.astype(np.float32) / 255.0
        fused = 0.6 * dark_clean_f + 0.4 * blackhat_f

        # Score columns within ROI ∩ central band
        x1 = max(rx, band_x1)
        x2 = min(rx + rw, band_x2)
        y1 = ry
        y2 = ry + rh
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None, 0.0
        band = fused[y1:y2, x1:x2]

        profile = band.sum(axis=0)
        # Smooth the 1D profile
        k = max(5, int(round((x2 - x1) * 0.02)))
        if k % 2 == 0:
            k += 1
        smooth = cv2.GaussianBlur(profile.reshape(1, -1), (1, k), 0, borderType=cv2.BORDER_REPLICATE).ravel()

        idx = int(np.argmax(smooth))
        if idx <= 0 or idx >= smooth.size - 1:
            return None, 0.0

        # Refine by per-row local maxima near the best column
        col_abs = x1 + idx
        window = max(3, int(round((x2 - x1) * 0.02)))
        x_lo = max(x1, col_abs - window)
        x_hi = min(x2, col_abs + window)
        row_xs = []
        band_rows = fused[y1:y2, x_lo:x_hi]
        if band_rows.size > 0:
            # Argmax per row
            local_idx = np.argmax(band_rows, axis=1)
            row_xs = (x_lo + local_idx).astype(np.int32)
        
        if len(row_xs) > 0:
            # Use median x for robustness
            refined_x = int(np.median(row_xs))
        else:
            refined_x = col_abs

        # Confidence: peak prominence, center proximity, and continuity
        left_max = float(np.max(smooth[:max(1, idx - k)])) if idx > 0 else 0.0
        right_max = float(np.max(smooth[min(idx + k, smooth.size - 1):])) if idx < smooth.size - 1 else 0.0
        neighbors = max(left_max, right_max, 1e-6)
        peak = float(smooth[idx])
        prominence = max(0.0, (peak - neighbors) / (neighbors + 1e-6))
        center_bias = 1.0 - (abs((x1 + idx) - cx) / max(1.0, (band_x2 - band_x1)))
        if len(row_xs) > 0:
            continuity = float(np.count_nonzero(band_rows.max(axis=1) > 0.25)) / float(band_rows.shape[0])
            jitter = float(np.std(row_xs)) / max(1.0, (x2 - x1))
            continuity_score = max(0.0, min(1.0, 0.8 * continuity + 0.2 * (1.0 - jitter * 4.0)))
        else:
            continuity_score = 0.0

        confidence = (
            0.55 * min(1.0, prominence) +
            0.25 * max(0.0, center_bias) +
            0.20 * continuity_score
        )
        confidence = max(0.0, min(1.0, confidence))

        # Map to original coordinates
        split_x = refined_x
        if scale != 1.0:
            split_x = int(round(split_x / scale))
        x = int(split_x)
        return (x, 0, x, h0 - 1), float(confidence)

    except Exception as e:
        logger.error(f"Error in vertical-trench detection: {e}")
        return None, 0.0


def detect_vertical_trench_near_center_parallel(
    image: np.ndarray,
    roi_threshold: int,
    center_band_ratio: float = 0.10,
    scale_width: int = 1200,
    num_workers: int = 20
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """
    Parallel variant that evaluates candidate columns within the center band using
    multiple workers and returns the line with the highest confidence.

    This function shares the same preprocessing as detect_vertical_trench_near_center,
    then scores many candidate columns in parallel using per-row continuity metrics.
    """
    try:
        if image is None or image.size == 0:
            return None, 0.0

        h0, w0 = image.shape[:2]

        # Downscale to bounded working size
        if w0 > scale_width:
            scale = scale_width / float(w0)
            work_w = int(round(w0 * scale))
            work_h = int(round(h0 * scale))
            work_img = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            work_img = image
            work_h, work_w = h0, w0

        # Grayscale + optional CLAHE
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass

        # Whitening with Otsu fallback
        _, binary = cv2.threshold(gray, roi_threshold, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(binary) < 0.02 * binary.size:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ROI detection (prefer robust crop rect; fallback to largest white ROI)
        roi = compute_document_crop_rect(gray, threshold=roi_threshold, margin=8)
        if roi is None:
            rx, ry, rw, rh = compute_largest_white_roi(gray, roi_threshold, margin=8)
        else:
            rx, ry, rw, rh = roi
        rx = max(0, min(rx, work_w - 1))
        ry = max(0, min(ry, work_h - 1))
        rw = max(1, min(rw, work_w - rx))
        rh = max(1, min(rh, work_h - ry))

        # Central band limits
        cx = work_w * 0.5
        band_half = int(round(work_w * center_band_ratio))
        band_x1 = max(0, int(cx - band_half))
        band_x2 = min(work_w, int(cx + band_half))
        if band_x2 - band_x1 < 3:
            return None, 0.0

        # Enhance dark trench: black-hat with vertical kernel + dark mask
        k_vert = max(7, int(round(work_h * 0.02)))
        if k_vert % 2 == 0:
            k_vert += 1
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_vert))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, vert_kernel)
        blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

        dark = (binary == 0).astype(np.uint8) * 255
        dark_connected = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, vert_kernel, iterations=1)
        k_h = max(5, int(round(work_w * 0.01)))
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_h, 1))
        dark_clean = cv2.morphologyEx(dark_connected, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

        # Fuse maps
        dark_clean_f = dark_clean.astype(np.float32) / 255.0
        blackhat_f = blackhat.astype(np.float32) / 255.0
        fused = 0.6 * dark_clean_f + 0.4 * blackhat_f

        # Candidate band (ROI ∩ center band)
        x1 = max(rx, band_x1)
        x2 = min(rx + rw, band_x2)
        y1 = ry
        y2 = ry + rh
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None, 0.0
        band = fused[y1:y2, x1:x2]

        # Base profile and smoothing for peak strength
        profile = band.sum(axis=0)
        k = max(5, int(round((x2 - x1) * 0.02)))
        if k % 2 == 0:
            k += 1
        smooth = cv2.GaussianBlur(profile.reshape(1, -1), (1, k), 0, borderType=cv2.BORDER_REPLICATE).ravel()
        smooth_max = float(np.max(smooth)) if smooth.size else 1.0
        if smooth_max <= 0.0:
            return None, 0.0

        # Evaluate all candidates (each column) in parallel
        band_width = x2 - x1
        window = max(3, int(round(band_width * 0.02)))

        def score_at_idx(idx: int) -> Tuple[float, int, Optional[np.ndarray]]:
            # Peak normalization
            peak_norm = float(smooth[idx]) / smooth_max
            # Center bias
            center_bias = 1.0 - (abs((x1 + idx) - cx) / max(1.0, (band_x2 - band_x1)))
            # Per-row continuity around this column
            col_abs = x1 + idx
            x_lo = max(x1, col_abs - window)
            x_hi = min(x2, col_abs + window)
            rows = fused[y1:y2, x_lo:x_hi]
            if rows.size == 0:
                return 0.0, col_abs, None
            local_idx = np.argmax(rows, axis=1)
            row_xs = (x_lo + local_idx).astype(np.int32)
            row_max = rows.max(axis=1)
            continuity = float(np.count_nonzero(row_max > 0.25)) / float(rows.shape[0])
            jitter = float(np.std(row_xs)) / max(1.0, band_width)
            continuity_score = max(0.0, min(1.0, 0.8 * continuity + 0.2 * (1.0 - jitter * 4.0)))
            # Combine
            conf = 0.5 * peak_norm + 0.25 * max(0.0, center_bias) + 0.25 * continuity_score
            return conf, col_abs, row_xs

        best_conf = -1.0
        best_x = None
        best_rows = None
        indices = list(range(band_width))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
            for conf, col_abs, row_xs in ex.map(score_at_idx, indices):
                if conf > best_conf:
                    best_conf = conf
                    best_x = col_abs
                    best_rows = row_xs

        if best_x is None or best_conf <= 0.0:
            return None, 0.0

        # Refine X using median of per-row maxima if available
        if isinstance(best_rows, np.ndarray) and best_rows.size > 0:
            refined_x = int(np.median(best_rows))
        else:
            refined_x = int(best_x)

        split_x = refined_x
        if scale != 1.0:
            split_x = int(round(split_x / scale))
        x = int(split_x)
        return (x, 0, x, h0 - 1), float(max(0.0, min(1.0, best_conf)))

    except Exception as e:
        logger.error(f"Error in parallel vertical-trench detection: {e}")
        return None, 0.0
def compute_document_crop_rect(
    gray: np.ndarray,
    threshold: int,
    margin: int,
    *,
    min_area_ratio: float = 0.03,
    border_pct: Tuple[float, float] = (0.03, 0.01),
    union_two: bool = True
) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute a robust document crop rectangle from a grayscale image.

    Pipeline:
    1) Global threshold to white-on-black; if it yields no valid component,
       fall back to Otsu threshold.
    2) Suppress thin outer borders (left/right and top/bottom percentages) to
       remove bright microfilm labels and sprocket holes.
    3) Morphological close with a small kernel to connect text islands.
    4) Find external contours and filter by a minimum image-area ratio.
    5) If two large components are side-by-side with strong vertical overlap,
       return the union bounding box; otherwise return the largest.

    Args:
        gray: Grayscale image array.
        threshold: Threshold in [0, 255] used for the primary binarization.
        margin: Pixels to expand around the detected rectangle.
        min_area_ratio: Minimum contour area as a fraction of image area.
        border_pct: (horizontal_pct, vertical_pct) border to suppress.
        union_two: When True, union two side-by-side page rectangles.

    Returns:
        (x, y, w, h) rectangle in image coordinates, or None if not found.
    """
    try:
        if gray is None or gray.size == 0:
            return None

        img_height, img_width = gray.shape[:2]
        image_area = float(img_width * img_height)

        # Determine morphology kernel size based on image size
        short_side = min(img_width, img_height)
        kernel_size = max(3, int(round(short_side * 0.006)))
        kernel_size = min(9, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        hor_border = max(0, int(img_width * max(0.0, border_pct[0])))
        ver_border = max(0, int(img_height * max(0.0, border_pct[1])))

        def make_binary(thresh_value: int, use_otsu: bool = False) -> np.ndarray:
            if use_otsu:
                _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, bin_img = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)

            # Suppress borders to ignore edge labels/sprocket holes
            if hor_border > 0:
                bin_img[:, :hor_border] = 0
                bin_img[:, img_width - hor_border:img_width] = 0
            if ver_border > 0:
                bin_img[:ver_border, :] = 0
                bin_img[img_height - ver_border:img_height, :] = 0

            # Connect nearby white regions
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
            return bin_img

        def find_candidate_rects(bin_img: np.ndarray) -> list:
            contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            if contours:
                min_area = image_area * min_area_ratio
                for c in contours:
                    area = cv2.contourArea(c)
                    if area < min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    candidates.append((x, y, w, h, area))
            return candidates

        # First pass: provided threshold
        binary = make_binary(threshold, use_otsu=False)
        rects = find_candidate_rects(binary)

        # Fallback: Otsu if nothing valid
        if not rects:
            binary = make_binary(0, use_otsu=True)
            rects = find_candidate_rects(binary)
            if not rects:
                return None

        # Sort by area descending
        rects.sort(key=lambda r: r[4], reverse=True)

        def union_if_two_pages(r1: Tuple[int, int, int, int, float], r2: Tuple[int, int, int, int, float]) -> Optional[Tuple[int, int, int, int]]:
            x1, y1, w1, h1, _ = r1
            x2, y2, w2, h2, _ = r2
            # Vertical overlap fraction relative to smaller height
            top = max(y1, y2)
            bottom = min(y1 + h1, y2 + h2)
            v_overlap = max(0, bottom - top)
            v_overlap_ratio = v_overlap / max(1.0, float(min(h1, h2)))

            # Horizontal overlap fraction
            left = max(x1, x2)
            right = min(x1 + w1, x2 + w2)
            h_overlap = max(0, right - left)
            h_overlap_ratio = h_overlap / max(1.0, float(min(w1, w2)))

            # Consider side-by-side when vertical overlap is strong and horizontal overlap is small
            if v_overlap_ratio >= 0.5 and h_overlap_ratio <= 0.3:
                x = min(x1, x2)
                y = min(y1, y2)
                w = max(x1 + w1, x2 + w2) - x
                h = max(y1 + h1, y2 + h2) - y
                return (x, y, w, h)
            return None

        # Choose rectangle
        x, y, w, h = rects[0][0], rects[0][1], rects[0][2], rects[0][3]
        if union_two and len(rects) >= 2:
            two_union = union_if_two_pages(rects[0], rects[1])
            if two_union is not None:
                x, y, w, h = two_union

        # Expand by margin and clamp
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img_width - x, w + 2 * margin)
        h = min(img_height - y, h + 2 * margin)
        return (x, y, w, h)

    except Exception as e:
        logger.error(f"Error computing document crop rect: {e}")
        return None

def detect_gutter_line_profile(
    image: np.ndarray,
    roi_threshold: int,
    sensitivity: int,
    angle_max_deg: float = 20.0,
    min_length_ratio: float = 0.6,
    scale_factor: float = 0.5
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """
    Detect the gutter seam line using ROI-first profile-based method.
    
    Algorithm:
    1. Find largest white mass (ROI) using roi_threshold
    2. Compute column darkness profile within ROI
    3. Find darkest column (valley) with center weighting
    4. Refine angle in narrow band around detected x using Hough
    5. Return line coordinates and confidence
    
    Args:
        image: BGR image as numpy array
        roi_threshold: Threshold for finding largest white mass (0-255)
        sensitivity: Sensitivity threshold for black-hat detection (0-255)
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
        
        # Convert to grayscale
        gray = cv2.cvtColor(work_image, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Find ROI (largest white mass)
        roi_x, roi_y, roi_w, roi_h = compute_largest_white_roi(gray, roi_threshold, margin=8)
        
        # Validate ROI
        if roi_w < 50 or roi_h < 50 or roi_w == 0 or roi_h == 0:
            logger.warning("ROI too small or invalid, using full image")
            roi_x, roi_y, roi_w, roi_h = 0, 0, work_width, work_height
        
        # Extract ROI
        roi_gray = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # Step 2: Build black-hat image for vertical dark bands
        # Multi-scale black-hat morphology to emphasize vertical dark seams
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 17)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 31))
        ]
        
        blackhat = np.zeros_like(roi_gray, dtype=np.float32)
        for kernel in kernels:
            bh = cv2.morphologyEx(roi_gray, cv2.MORPH_BLACKHAT, kernel)
            blackhat = blackhat + bh.astype(np.float32)
        
        # Normalize to 0-255
        blackhat = np.clip(blackhat, 0, 255).astype(np.uint8)
        
        # Step 3: Compute column darkness profile
        # Mean darkness per column (averaging vertically)
        col_profile = np.mean(blackhat, axis=0)
        
        # Smooth the profile with Gaussian blur
        col_profile = cv2.GaussianBlur(col_profile.reshape(1, -1), (1, 21), 0).flatten()
        
        # Step 4: Center-weighted scoring
        # Create Gaussian center weight (strongest at center, weaker at edges)
        center_x = roi_w / 2.0
        x_coords = np.arange(roi_w)
        center_weight = np.exp(-((x_coords - center_x)**2) / (2 * (0.2 * roi_w)**2))
        
        # Normalize profile to [0, 1] range
        if col_profile.max() > col_profile.min():
            col_normalized = (col_profile - col_profile.min()) / (col_profile.max() - col_profile.min())
        else:
            col_normalized = np.ones_like(col_profile) * 0.5
        
        # Combined score: darkness * center_weight
        score = col_normalized * center_weight
        
        # Find peak (darkest column)
        x_star_local = int(np.argmax(score))
        x_star = x_star_local + roi_x  # Convert to full image coordinates
        
        # Step 5: Refine angle in narrow band around x_star
        # Extract narrow vertical band for angle detection
        band_width = max(50, int(roi_w * 0.15))  # 15% of ROI width, at least 50px
        band = blackhat[:, max(0, x_star_local - band_width // 2):min(roi_w, x_star_local + band_width // 2)]
        
        # Apply sensitivity threshold
        _, band_mask = cv2.threshold(band, sensitivity, 255, cv2.THRESH_BINARY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(band_mask, 50, 150)
        
        # HoughLinesP with angle restriction
        angle_max_rad = math.radians(angle_max_deg)
        min_length = int(roi_h * min_length_ratio)
        
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
        
        if lines is not None:
            # Adjust line coordinates to account for band offset
            band_offset = max(0, x_star_local - band_width // 2)
            
            for line in lines:
                x1_local, y1, x2_local, y2 = line[0]
                
                # Convert back to ROI coordinates
                x1_roi = x1_local + band_offset
                x2_roi = x2_local + band_offset
                
                # Calculate line properties
                dx = x2_roi - x1_roi
                dy = y2 - y1
                length = math.sqrt(dx * dx + dy * dy)
                
                if length < min_length:
                    continue
                
                # Calculate angle from vertical
                angle = math.atan2(abs(dx), abs(dy))  # angle from vertical
                if angle > angle_max_rad:
                    continue
                
                # Sample intensity along line
                num_samples = max(10, int(length / 5))
                intensities = []
                for i in range(num_samples):
                    t = i / (num_samples - 1) if num_samples > 1 else 0
                    x = int(x1_roi + t * dx)
                    y = int(y1 + t * dy)
                    if 0 <= x < roi_w and 0 <= y < roi_h:
                        intensities.append(blackhat[y, x])
                
                if not intensities:
                    continue
                
                avg_intensity = np.mean(intensities) / 255.0
                
                # Coverage: how much of ROI height the line covers
                y_min = min(y1, y2)
                y_max = max(y1, y2)
                coverage = (y_max - y_min) / roi_h
                
                # Angle bias: prefer more vertical lines
                angle_bias = 1.0 - (angle / angle_max_rad)
                
                # Combined score
                line_score = 0.5 * avg_intensity + 0.3 * coverage + 0.2 * angle_bias
                
                if line_score > best_score:
                    best_score = line_score
                    # Convert to full image coordinates
                    x1_full = x1_roi + roi_x
                    x2_full = x2_roi + roi_x
                    y1_full = y1 + roi_y
                    y2_full = y2 + roi_y
                    best_line = (x1_full, y1_full, x2_full, y2_full)
        
        # Fallback: use straight vertical line if no angled line found
        if best_line is None:
            # Use straight vertical line at x_star
            y1_full = roi_y
            y2_full = roi_y + roi_h - 1
            best_line = (x_star, y1_full, x_star, y2_full)
            best_score = col_normalized[x_star_local] * 0.7  # Lower confidence for fallback
        
        # Scale coordinates back to original image size
        if scale_factor < 1.0:
            x1, y1, x2, y2 = best_line
            x1 = int(x1 / scale_factor)
            y1 = int(y1 / scale_factor)
            x2 = int(x2 / scale_factor)
            y2 = int(y2 / scale_factor)
            best_line = (x1, y1, x2, y2)
        
        # Calculate confidence
        # Base confidence from score, also consider valley depth
        valley_depth = col_normalized[x_star_local]
        
        # Distance from center (normalized)
        center_dist = abs(x_star_local - center_x) / (roi_w / 2.0)
        center_bias = 1.0 - min(center_dist, 1.0)
        
        # Coverage: how much of image height the line covers
        y1_final, y2_final = best_line[1], best_line[3]
        coverage = abs(y2_final - y1_final) / height
        
        # Combined confidence
        confidence = (
            0.4 * min(1.0, best_score) +
            0.3 * valley_depth +
            0.2 * center_bias +
            0.1 * coverage
        )
        
        # Ensure confidence is reasonable
        confidence = max(0.0, min(1.0, confidence))
        
        logger.debug(f"Detected gutter line (profile method) with confidence {confidence:.2f}, score {best_score:.2f}")
        
        return best_line, confidence
        
    except Exception as e:
        logger.error(f"Error detecting gutter line (profile method): {e}")
        return None, 0.0


def find_optimal_seam_params(
    image: np.ndarray,
    angle_max_deg: float = 20.0,
    min_length_ratio: float = 0.6
) -> Tuple[int, int]:
    """
    Find optimal ROI threshold and seam sensitivity by grid search.
    
    Args:
        image: BGR image as numpy array
        angle_max_deg: Maximum angle deviation from vertical (degrees)
        min_length_ratio: Minimum line length as ratio of image height
        
    Returns:
        Tuple of (roi_threshold, sensitivity) that maximizes confidence
    """
    try:
        if image is None or image.size == 0:
            return 170, 140  # Defaults
        
        best_roi_threshold = 170
        best_sensitivity = 140
        best_confidence = 0.0
        
        # Coarse grid search
        # ROI threshold: typical document threshold range
        for roi_threshold in range(120, 230, 20):
            # Sensitivity: typical black-hat threshold range
            for sensitivity in range(50, 200, 20):
                line_coords, confidence = detect_gutter_line_profile(
                    image, roi_threshold, sensitivity, angle_max_deg, min_length_ratio
                )
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_roi_threshold = roi_threshold
                    best_sensitivity = sensitivity
        
        # Fine refinement around best values
        if best_roi_threshold > 120 and best_roi_threshold < 230:
            for roi_threshold in range(max(120, best_roi_threshold - 19), min(230, best_roi_threshold + 20), 5):
                for sensitivity in range(max(50, best_sensitivity - 19), min(200, best_sensitivity + 20), 5):
                    line_coords, confidence = detect_gutter_line_profile(
                        image, roi_threshold, sensitivity, angle_max_deg, min_length_ratio
                    )
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_roi_threshold = roi_threshold
                        best_sensitivity = sensitivity
        
        logger.debug(f"Optimal seam params: roi_threshold={best_roi_threshold}, sensitivity={best_sensitivity} (confidence: {best_confidence:.2f})")
        return best_roi_threshold, best_sensitivity
        
    except Exception as e:
        logger.error(f"Error finding optimal seam params: {e}")
        return 170, 140  # Default fallback


def make_darkness_map(
    gray: np.ndarray,
    sensitivity: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a darkness map combining inverted grayscale and black-hat morphology.
    
    Args:
        gray: Grayscale image as numpy array
        sensitivity: Sensitivity parameter (0-255) controlling black-hat emphasis
        
    Returns:
        Tuple of (map_norm, dark_mask) where:
        - map_norm: Normalized darkness map [0, 1]
        - dark_mask: Binary mask of dark regions (255 = dark, 0 = light)
    """
    try:
        # Normalize grayscale to [0, 1]
        gray_norm = gray.astype(np.float32) / 255.0
        
        # Inverted grayscale component (darker = higher value)
        inverted_gray = 1.0 - gray_norm
        
        # Black-hat morphology to emphasize vertical dark features
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 17)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 31))
        ]
        
        blackhat_sum = np.zeros_like(gray, dtype=np.float32)
        for kernel in kernels:
            bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            blackhat_sum = blackhat_sum + bh.astype(np.float32)
        
        # Normalize black-hat to [0, 1]
        blackhat_max = blackhat_sum.max()
        if blackhat_max > 0:
            blackhat_norm = blackhat_sum / blackhat_max
        else:
            blackhat_norm = np.zeros_like(gray_norm, dtype=np.float32)
        
        # Combine: M = α·inverted_gray + β·blackhat_norm
        # sensitivity controls β (0-255 maps to 0-1.0)
        alpha = 0.6  # Base inverted grayscale weight
        beta = (sensitivity / 255.0) * 0.4  # Black-hat weight (max 0.4)
        
        darkness_map = alpha * inverted_gray + beta * blackhat_norm
        
        # Normalize to [0, 1]
        if darkness_map.max() > darkness_map.min():
            map_norm = (darkness_map - darkness_map.min()) / (darkness_map.max() - darkness_map.min())
        else:
            map_norm = darkness_map
        
        # Adaptive thresholding to create dark mask
        # Use Gaussian adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(
            (map_norm * 255).astype(np.uint8),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # C constant
        )
        
        # Morphological operations to bridge gaps and suppress ruled lines
        # Close with vertical kernel to bridge gaps in seam
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        dark_mask = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, vertical_kernel)
        
        # Open with horizontal kernel to suppress ruled lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, horizontal_kernel)
        
        return map_norm, dark_mask
        
    except Exception as e:
        logger.error(f"Error making darkness map: {e}")
        # Fallback: simple inverted grayscale
        gray_norm = gray.astype(np.float32) / 255.0
        map_norm = 1.0 - gray_norm
        _, dark_mask = cv2.threshold((map_norm * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        return map_norm, dark_mask


def detect_gutter_line_dark_path(
    image: np.ndarray,
    roi_threshold: int,
    sensitivity: int,
    angle_max_deg: float = 20.0,
    min_length_ratio: float = 0.6,
    scale_factor: float = 0.5
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """
    Detect the gutter seam line using dark-ridge path finding with dynamic programming.
    
    Algorithm:
    1. Find ROI (largest white mass)
    2. Build darkness map (inverted gray + black-hat)
    3. Compute distance transform for thickness
    4. Find max-thickness path using dynamic programming
    5. Fit line to path points
    6. Return line coordinates and confidence
    
    Args:
        image: BGR image as numpy array
        roi_threshold: Threshold for finding largest white mass (0-255)
        sensitivity: Sensitivity parameter controlling dark emphasis (0-255)
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
        
        # Convert to grayscale
        gray = cv2.cvtColor(work_image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Small blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Step 1: Find ROI (largest white mass)
        roi_x, roi_y, roi_w, roi_h = compute_largest_white_roi(gray, roi_threshold, margin=8)
        
        # Validate ROI
        if roi_w < 50 or roi_h < 50 or roi_w == 0 or roi_h == 0:
            logger.warning("ROI too small or invalid, using full image")
            roi_x, roi_y, roi_w, roi_h = 0, 0, work_width, work_height
        
        # Extract ROI
        roi_gray = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # Step 2: Build darkness map
        map_norm, dark_mask = make_darkness_map(roi_gray, sensitivity)
        
        # Step 3: Compute distance transform for thickness
        # Distance transform gives distance to nearest non-dark pixel
        # Invert dark_mask so dark pixels are 0 and we compute distance from dark regions
        dark_mask_inv = cv2.bitwise_not(dark_mask)
        dist_transform = cv2.distanceTransform(dark_mask_inv, cv2.DIST_L2, 5)
        
        # Normalize distance transform to [0, 1]
        if dist_transform.max() > 0:
            thickness_map = dist_transform / dist_transform.max()
        else:
            thickness_map = np.zeros_like(dist_transform, dtype=np.float32)
        
        # Step 4: Dynamic programming to find max-thickness path
        # Search within center band (±30% of ROI width)
        center_x = roi_w / 2.0
        band_width_ratio = 0.3
        x_min = int(max(0, center_x - roi_w * band_width_ratio))
        x_max = int(min(roi_w, center_x + roi_w * band_width_ratio))
        
        # DP table: dp[y][x] = max thickness path ending at (y, x)
        dp = np.full((roi_h, roi_w), -np.inf, dtype=np.float32)
        backpointers = np.zeros((roi_h, roi_w), dtype=np.int32)  # Store previous x for path reconstruction
        
        # Initialize first row
        for x in range(x_min, x_max):
            dp[0, x] = thickness_map[0, x]
        
        # Fill DP table
        # Constraint: |Δx| ≤ 1 per row (allows up to ~20° angle)
        lambda_s = 0.1  # Slope penalty
        lambda_c = 0.05  # Center bias penalty
        
        for y in range(1, roi_h):
            for x in range(x_min, x_max):
                # Consider transitions from previous row
                best_prev_score = -np.inf
                best_prev_x = x
                
                for dx in [-1, 0, 1]:  # Allow ±1 pixel horizontal movement
                    prev_x = x + dx
                    if x_min <= prev_x < x_max:
                        center_offset = abs(prev_x - center_x) / (roi_w / 2.0)
                        score = dp[y-1, prev_x] + thickness_map[y, x] - lambda_s * abs(dx) - lambda_c * center_offset
                        if score > best_prev_score:
                            best_prev_score = score
                            best_prev_x = prev_x
                
                if best_prev_score > -np.inf:
                    dp[y, x] = best_prev_score
                    backpointers[y, x] = best_prev_x
        
        # Find best path ending
        best_end_score = -np.inf
        best_end_x = x_min
        for x in range(x_min, x_max):
            if dp[roi_h-1, x] > best_end_score:
                best_end_score = dp[roi_h-1, x]
                best_end_x = x
        
        # Reconstruct path
        path_points = []
        x = best_end_x
        for y in range(roi_h-1, -1, -1):
            path_points.append((x + roi_x, y + roi_y))  # Convert to full image coords
            if y > 0:
                x = backpointers[y, x]
        
        path_points.reverse()  # Now ordered from top to bottom
        
        # Check if path is valid (covers enough height)
        if len(path_points) < roi_h * min_length_ratio:
            logger.warning("Path too short, using fallback")
            # Fallback: vertical line at center
            y1_full = roi_y
            y2_full = roi_y + roi_h - 1
            x_center = int(roi_x + center_x)
            best_line = (x_center, y1_full, x_center, y2_full)
            confidence = 0.3  # Low confidence for fallback
        else:
            # Step 5: Fit line to path points
            path_x = np.array([p[0] for p in path_points], dtype=np.float32)
            path_y = np.array([p[1] for p in path_points], dtype=np.float32)
            
            # Robust line fitting using least squares
            # Fit line: y = mx + b, but we want x as function of y for vertical seams
            # Use: x = m*y + b
            A = np.vstack([path_y, np.ones(len(path_y))]).T
            m, b = np.linalg.lstsq(A, path_x, rcond=None)[0]
            
            # Calculate angle from vertical
            angle_rad = math.atan(abs(m))
            angle_deg = math.degrees(angle_rad)
            
            # Clamp angle to max allowed
            if angle_deg > angle_max_deg:
                # Recalculate with constrained angle
                target_angle_rad = math.radians(angle_max_deg)
                m = math.tan(target_angle_rad) * (1 if m >= 0 else -1)
                # Recalculate intercept
                b = np.mean(path_x - m * path_y)
            
            # Generate line endpoints
            y1_full = path_y[0]
            y2_full = path_y[-1]
            x1_full = int(m * y1_full + b)
            x2_full = int(m * y2_full + b)
            
            best_line = (x1_full, y1_full, x2_full, y2_full)
            
            # Step 6: Calculate confidence
            # Mean thickness along path
            path_thicknesses = []
            for x, y in path_points:
                x_local = x - roi_x
                y_local = y - roi_y
                if 0 <= x_local < roi_w and 0 <= y_local < roi_h:
                    path_thicknesses.append(thickness_map[y_local, x_local])
            
            mean_thickness = np.mean(path_thicknesses) if path_thicknesses else 0.0
            
            # Margin over neighbors (compare to columns ±5% width away)
            neighbor_x_left = int(max(0, center_x - roi_w * 0.05))
            neighbor_x_right = int(min(roi_w, center_x + roi_w * 0.05))
            neighbor_thicknesses = []
            for x in [neighbor_x_left, neighbor_x_right]:
                if x != int(center_x):
                    col_thickness = np.mean(thickness_map[:, x])
                    neighbor_thicknesses.append(col_thickness)
            
            margin = mean_thickness - np.mean(neighbor_thicknesses) if neighbor_thicknesses else 0.0
            margin_norm = max(0.0, min(1.0, (margin + 0.1) / 0.2))  # Normalize to [0, 1]
            
            # Distance to center bias
            path_center_x = np.mean(path_x)
            center_dist = abs(path_center_x - (roi_x + center_x)) / (roi_w / 2.0)
            center_bias = 1.0 - min(center_dist, 1.0)
            
            # Vertical coverage
            coverage = len(path_points) / roi_h
            
            # Combined confidence
            confidence = (
                0.4 * mean_thickness +
                0.3 * margin_norm +
                0.2 * center_bias +
                0.1 * coverage
            )
            confidence = max(0.0, min(1.0, confidence))
        
        # Scale coordinates back to original image size
        if scale_factor < 1.0:
            x1, y1, x2, y2 = best_line
            x1 = int(x1 / scale_factor)
            y1 = int(y1 / scale_factor)
            x2 = int(x2 / scale_factor)
            y2 = int(y2 / scale_factor)
            best_line = (x1, y1, x2, y2)
        
        logger.debug(f"Detected gutter line (dark-path method) with confidence {confidence:.2f}")
        
        return best_line, confidence
        
    except Exception as e:
        logger.error(f"Error detecting gutter line (dark-path method): {e}")
        return None, 0.0


def find_optimal_seam_params_dark(
    image: np.ndarray,
    angle_max_deg: float = 20.0,
    min_length_ratio: float = 0.6
) -> Tuple[int, int]:
    """
    Find optimal ROI threshold and seam sensitivity for dark-path method by grid search.
    
    Args:
        image: BGR image as numpy array
        angle_max_deg: Maximum angle deviation from vertical (degrees)
        min_length_ratio: Minimum line length as ratio of image height
        
    Returns:
        Tuple of (roi_threshold, sensitivity) that maximizes confidence
    """
    try:
        if image is None or image.size == 0:
            return 170, 140  # Defaults
        
        best_roi_threshold = 170
        best_sensitivity = 140
        best_confidence = 0.0
        
        # Coarse grid search
        # ROI threshold: typical document threshold range
        for roi_threshold in range(120, 230, 25):
            # Sensitivity: controls dark emphasis
            for sensitivity in range(50, 200, 25):
                line_coords, confidence = detect_gutter_line_dark_path(
                    image, roi_threshold, sensitivity, angle_max_deg, min_length_ratio
                )
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_roi_threshold = roi_threshold
                    best_sensitivity = sensitivity
        
        # Fine refinement around best values
        if best_roi_threshold > 120 and best_roi_threshold < 230:
            for roi_threshold in range(max(120, best_roi_threshold - 24), min(230, best_roi_threshold + 25), 5):
                for sensitivity in range(max(50, best_sensitivity - 24), min(200, best_sensitivity + 25), 5):
                    line_coords, confidence = detect_gutter_line_dark_path(
                        image, roi_threshold, sensitivity, angle_max_deg, min_length_ratio
                    )
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_roi_threshold = roi_threshold
                        best_sensitivity = sensitivity
        
        logger.debug(f"Optimal dark-path params: roi_threshold={best_roi_threshold}, sensitivity={best_sensitivity} (confidence: {best_confidence:.2f})")
        return best_roi_threshold, best_sensitivity
        
    except Exception as e:
        logger.error(f"Error finding optimal dark-path seam params: {e}")
        return 170, 140  # Default fallback