"""Threshold adjustment dialog for auto-crop."""
import cv2
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from PIL import Image, ImageTk


class ThresholdAdjuster(tk.Toplevel):
    """Dialog for adjusting threshold and margin for auto-crop."""
    
    def __init__(self, parent, image_path: Path, service, on_apply=None, mode='crop'):
        """
        Initialize threshold adjuster.
        
        Args:
            parent: Parent window
            image_path: Path to image to adjust
            service: ImageService instance
            on_apply: Callback when apply is clicked (threshold, margin) for crop mode,
                      or (threshold,) for straighten mode
            mode: 'crop' or 'straighten' - determines which controls to show
        """
        super().__init__(parent)
        self.mode = mode
        if mode == 'straighten':
            self.title("Threshold Adjustment - Auto Straighten")
        else:
            self.title("Threshold Adjustment")
        self.parent = parent
        self.image_path = image_path
        self.service = service
        self.on_apply = on_apply
        
        # Initialize variables
        settings = service.settings
        self.threshold_var = tk.IntVar(value=settings.threshold)
        self.margin_var = tk.IntVar(value=settings.margin)
        
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
        
        # Threshold control
        ttk.Label(main_frame, text="Threshold:").grid(row=1, column=0, sticky=tk.W, pady=5)
        threshold_scale = ttk.Scale(
            main_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=lambda _: self.update_preview()
        )
        threshold_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Margin control (only show for crop mode)
        if self.mode == 'crop':
            ttk.Label(main_frame, text="Margin:").grid(row=2, column=0, sticky=tk.W, pady=5)
            margin_scale = ttk.Scale(
                main_frame,
                from_=0,
                to=100,
                orient=tk.HORIZONTAL,
                variable=self.margin_var,
                command=lambda _: self.update_preview()
            )
            margin_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
            button_row = 3
        else:
            button_row = 2
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=button_row, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Apply", command=self.apply_crop).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
        # Configure column weights
        main_frame.columnconfigure(1, weight=1)
    
    def update_preview(self):
        """Update preview image with current threshold and margin."""
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, self.threshold_var.get(), 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create preview image
        preview = self.original_image.copy()
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Draw the contour (green outline) - always shown
            cv2.drawContours(preview, [largest_contour], -1, (0, 255, 0), 2)
            
            # Draw crop rectangle (red) - only for crop mode
            if self.mode == 'crop':
                # Get bounding rectangle with margin
                x, y, w, h = cv2.boundingRect(largest_contour)
                margin = self.margin_var.get()
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(self.width - x, w + 2 * margin)
                h = min(self.height - y, h + 2 * margin)
                
                # Draw crop rectangle
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Resize for preview
        preview = cv2.resize(preview, (self.preview_width, self.preview_height))
        
        # Convert to PhotoImage
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        preview = Image.fromarray(preview)
        self.preview_photo = ImageTk.PhotoImage(preview)
        
        # Update canvas
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.preview_photo)
    
    def apply_crop(self):
        """Apply with current settings."""
        threshold = self.threshold_var.get()
        
        # Update settings
        self.service.settings.threshold = threshold
        if self.mode == 'crop':
            margin = self.margin_var.get()
            self.service.settings.margin = margin
        
        # Call callback if provided
        if self.on_apply:
            if self.mode == 'crop':
                self.on_apply(threshold, margin)
            else:
                # For straighten mode, callback only gets threshold
                self.on_apply(threshold)
        
        # Close the window
        self.destroy()

