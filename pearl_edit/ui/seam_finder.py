"""Seam finder dialog for auto-split detection."""
import cv2
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from PIL import Image, ImageTk
import numpy as np
from typing import Optional, Tuple, Callable
import logging

from ..image_ops import (
    detect_gutter_line, 
    detect_gutter_line_profile,
    detect_gutter_line_dark_path,
    find_optimal_seam_threshold,
    find_optimal_seam_params,
    find_optimal_seam_params_dark,
    compute_largest_white_roi,
    ImageProcessingError
)
from ..config import AppSettings

logger = logging.getLogger(__name__)


class SeamFinderDialog(tk.Toplevel):
    """Dialog for finding and adjusting the gutter seam for auto-split."""
    
    def __init__(self, parent, image_path: Path, service, on_apply: Optional[Callable] = None, on_skip: Optional[Callable] = None):
        """
        Initialize seam finder dialog.
        
        Args:
            parent: Parent window
            image_path: Path to image to process
            service: ImageService instance
            on_apply: Callback when apply is clicked (line_coords, threshold)
            on_skip: Callback when skip is clicked
        """
        super().__init__(parent)
        self.title("Auto Split - Seam Finder")
        self.parent = parent
        self.image_path = image_path
        self.service = service
        self.on_apply = on_apply
        self.on_skip = on_skip
        
        # Initialize variables
        settings: AppSettings = service.settings
        # Get ROI threshold from settings (with backward compatibility)
        roi_threshold = getattr(settings, 'seam_roi_threshold', 170)
        self.roi_threshold_var = tk.IntVar(value=roi_threshold)
        self.sensitivity_var = tk.IntVar(value=settings.seam_threshold)
        self.angle_max_deg = settings.seam_angle_max_deg
        self.min_length_ratio = settings.seam_min_length_ratio
        
        # Load the image
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            messagebox.showerror(
                "Error",
                f"Failed to load image: {image_path.name}\nPlease ensure the file exists and is a valid image.",
                parent=self
            )
            self.destroy()
            return
        
        self.height, self.width = self.original_image.shape[:2]
        
        # Calculate preview scale
        screen_height = self.winfo_screenheight()
        max_preview_height = screen_height * 0.6
        self.preview_scale = min(0.5, max_preview_height / self.height)
        
        # Calculate preview dimensions
        self.preview_width = int(self.width * self.preview_scale)
        self.preview_height = int(self.height * self.preview_scale)
        
        # Find optimal parameters on initialization using dark-path method
        try:
            optimal_roi_threshold, optimal_sensitivity = find_optimal_seam_params_dark(
                self.original_image,
                self.angle_max_deg,
                self.min_length_ratio
            )
            self.roi_threshold_var.set(optimal_roi_threshold)
            self.sensitivity_var.set(optimal_sensitivity)
            settings.seam_roi_threshold = optimal_roi_threshold
            settings.seam_threshold = optimal_sensitivity
            logger.info(f"Found optimal dark-path seam params: ROI={optimal_roi_threshold}, Sensitivity={optimal_sensitivity} for {image_path.name}")
        except ImageProcessingError as e:
            logger.warning(f"Could not find optimal params for {image_path.name}: {e}")
            # Use defaults from settings
        except Exception as e:
            logger.error(f"Unexpected error finding optimal params for {image_path.name}: {e}")
            # Use defaults from settings
        
        # Current detection result
        self.current_line_coords = None
        self.current_confidence = 0.0
        self.current_roi = None  # Store ROI rectangle for overlay
        
        # Create the GUI
        self.create_widgets()
        self.update_preview()
        
        # Make window modal
        self.transient(parent)
        self.grab_set()
    
    def create_widgets(self):
        """Create UI widgets."""
        # Main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Preview canvas
        self.preview_canvas = tk.Canvas(
            main_frame,
            width=self.preview_width,
            height=self.preview_height
        )
        self.preview_canvas.grid(row=0, column=0, columnspan=2, pady=5)
        
        # ROI Threshold control
        ttk.Label(main_frame, text="ROI Threshold:").grid(row=1, column=0, sticky=tk.W, pady=5)
        roi_threshold_scale = ttk.Scale(
            main_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.roi_threshold_var,
            command=lambda _: self.update_preview()
        )
        roi_threshold_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Seam Sensitivity control
        ttk.Label(main_frame, text="Seam Sensitivity:").grid(row=2, column=0, sticky=tk.W, pady=5)
        sensitivity_scale = ttk.Scale(
            main_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.sensitivity_var,
            command=lambda _: self.update_preview()
        )
        sensitivity_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Confidence label
        self.confidence_label = ttk.Label(
            main_frame,
            text="Confidence: --",
            font=("Arial", 9)
        )
        self.confidence_label.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Info label
        info_label = ttk.Label(
            main_frame,
            text="Adjust ROI Threshold to find the largest white mass (document area).\n"
                 "Adjust Seam Sensitivity to detect the dark center seam (black/grey gutter).\n"
                 "Low confidence pages will prompt for review during batch processing.",
            font=("Arial", 8),
            foreground="gray",
            justify=tk.LEFT
        )
        info_label.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Apply Split", command=self.apply_split).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Skip", command=self.skip).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
        # Configure column weights
        main_frame.columnconfigure(1, weight=1)
    
    def update_preview(self):
        """Update preview image with current thresholds and detected seam."""
        line_coords = None
        confidence = 0.0
        roi_rect = None
        
        try:
            # Convert to RGB for display
            preview = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2RGB)
            
            # Detect seam line using dark-path method
            roi_threshold = self.roi_threshold_var.get()
            sensitivity = self.sensitivity_var.get()
            
            # Compute ROI for overlay
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            roi_x, roi_y, roi_w, roi_h = compute_largest_white_roi(gray, roi_threshold, margin=8)
            roi_rect = (roi_x, roi_y, roi_w, roi_h)
            self.current_roi = roi_rect
            
            # Detect seam using dark-path method
            line_coords, confidence = detect_gutter_line_dark_path(
                self.original_image,
                roi_threshold,
                sensitivity,
                self.angle_max_deg,
                self.min_length_ratio
            )
            
            self.current_line_coords = line_coords
            self.current_confidence = confidence
        except ImageProcessingError as e:
            logger.error(f"Error detecting gutter line in preview: {e}")
            self.current_line_coords = None
            self.current_confidence = 0.0
            self.current_roi = None
            preview = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2RGB)
            line_coords = None
            confidence = 0.0
            roi_rect = None
        except Exception as e:
            logger.error(f"Unexpected error in update_preview: {e}")
            self.current_line_coords = None
            self.current_confidence = 0.0
            self.current_roi = None
            preview = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2RGB)
            line_coords = None
            confidence = 0.0
            roi_rect = None
        
        # Draw ROI rectangle (green outline)
        if roi_rect:
            roi_x, roi_y, roi_w, roi_h = roi_rect
            cv2.rectangle(preview, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        
        # Draw detected line on preview
        if line_coords:
            x1, y1, x2, y2 = line_coords
            # Color based on confidence: green (high), orange (medium), red (low)
            if confidence >= 0.55:
                line_color = (0, 255, 0)  # Green
            elif confidence >= 0.3:
                line_color = (255, 165, 0)  # Orange
            else:
                line_color = (255, 0, 0)  # Red
            
            # Draw line
            cv2.line(preview, (x1, y1), (x2, y2), line_color, 3)
            # Draw endpoints
            cv2.circle(preview, (x1, y1), 5, line_color, -1)
            cv2.circle(preview, (x2, y2), 5, line_color, -1)
        
        # Resize for preview
        preview = cv2.resize(preview, (self.preview_width, self.preview_height))
        
        # Convert to PhotoImage
        preview = Image.fromarray(preview)
        self.preview_photo = ImageTk.PhotoImage(preview)
        
        # Update canvas - check if window still exists
        try:
            if not self.winfo_exists():
                return  # Window was destroyed, don't try to update
        except tk.TclError:
            return  # Window path is invalid, don't try to update
        
        try:
            # Update canvas
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.preview_photo)
            
            # Update confidence label
            if line_coords:
                confidence_pct = confidence * 100
                color = "green" if confidence >= 0.55 else "orange" if confidence >= 0.3 else "red"
                self.confidence_label.config(
                    text=f"Confidence: {confidence_pct:.1f}%",
                    foreground=color
                )
            else:
                self.confidence_label.config(
                    text="Confidence: No seam detected",
                    foreground="red"
                )
        except tk.TclError as e:
            # Window or widgets were destroyed during update
            logger.debug(f"Window destroyed during preview update: {e}")
            return
    
    def apply_split(self):
        """Apply split with current settings."""
        # Check if window still exists
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        
        if self.current_line_coords is None:
            try:
                messagebox.showwarning(
                    "No Seam Detected",
                    "No seam line was detected. Please adjust the threshold or skip this image.",
                    parent=self
                )
            except tk.TclError:
                pass  # Window was destroyed
            logger.warning(f"No seam detected for {self.image_path.name} at ROI threshold {self.roi_threshold_var.get()}, sensitivity {self.sensitivity_var.get()}")
            return
        
        roi_threshold = self.roi_threshold_var.get()
        sensitivity = self.sensitivity_var.get()
        
        try:
            # Update settings
            self.service.settings.seam_roi_threshold = roi_threshold
            self.service.settings.seam_threshold = sensitivity
            
            logger.info(f"Applying split for {self.image_path.name} with ROI threshold {roi_threshold}, sensitivity {sensitivity}, confidence {self.current_confidence:.2f}")
            
            # Call callback if provided
            if self.on_apply:
                self.on_apply(self.current_line_coords, sensitivity)
            
            # Close the window
            try:
                self.destroy()
            except tk.TclError:
                pass  # Already destroyed
        except tk.TclError:
            # Window was destroyed during operation
            logger.debug("Window destroyed during apply_split")
            return
        except Exception as e:
            logger.error(f"Error applying split for {self.image_path.name}: {e}")
            try:
                messagebox.showerror(
                    "Error",
                    f"Failed to apply split: {str(e)}",
                    parent=self
                )
            except tk.TclError:
                pass  # Window was destroyed
    
    def skip(self):
        """Skip this image."""
        # Check if window still exists
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        
        if self.on_skip:
            self.on_skip()
        
        # Close the window
        try:
            self.destroy()
        except tk.TclError:
            pass  # Already destroyed



