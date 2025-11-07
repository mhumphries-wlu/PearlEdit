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
    detect_vertical_trench_near_center,
    detect_vertical_trench_near_center_parallel,
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
        
        # Confidence label
        self.confidence_label = ttk.Label(
            main_frame,
            text="Confidence: --",
            font=("Arial", 9)
        )
        self.confidence_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Info label
        info_label = ttk.Label(
            main_frame,
            text="Adjust ROI Threshold to find the largest white mass (document area).\n"
                 "The detection finds the vertical trench near the image center.\n"
                 "Low confidence pages will prompt for review during batch processing.",
            font=("Arial", 8),
            foreground="gray",
            justify=tk.LEFT
        )
        info_label.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
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

            roi_threshold = self.roi_threshold_var.get()

            # Compute ROI for overlay
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            roi_x, roi_y, roi_w, roi_h = compute_largest_white_roi(gray, roi_threshold, margin=8)
            roi_rect = (roi_x, roi_y, roi_w, roi_h)
            self.current_roi = roi_rect

            # Detect seam using vertical trench detection near center
            line_coords, confidence = detect_vertical_trench_near_center(
                self.original_image,
                roi_threshold,
                center_band_ratio=0.05
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
            # Always red per requirement
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
            logger.warning(f"No seam detected for {self.image_path.name} at ROI threshold {self.roi_threshold_var.get()}")
            return
        
        roi_threshold = self.roi_threshold_var.get()
        
        try:
            # Update settings
            self.service.settings.seam_roi_threshold = roi_threshold
            
            # Recompute using parallel detector for highest-confidence final line
            best_coords, best_conf = detect_vertical_trench_near_center_parallel(
                self.original_image,
                roi_threshold,
                center_band_ratio=0.05,
                num_workers=20
            )
            final_coords = best_coords if best_coords else self.current_line_coords
            final_conf = best_conf if best_coords else self.current_confidence
            
            logger.info(f"Applying split for {self.image_path.name} with ROI threshold {roi_threshold}, confidence {final_conf:.2f}")
            
            # Call callback if provided
            if self.on_apply:
                self.on_apply(final_coords, 0)  # margin no longer used, pass 0 for compatibility
            
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



