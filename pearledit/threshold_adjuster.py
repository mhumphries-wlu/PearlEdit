import os, cv2, shutil, threading
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import math
from PIL import Image, ImageTk, ImageDraw, ImageChops
import pandas as pd
import concurrent.futures
class ThresholdAdjuster(tk.Toplevel):
    def __init__(self, parent, image_path):
        super().__init__(parent)
        self.title("Threshold Adjustment")
        self.parent = parent
        self.image_path = image_path
        
        # Initialize variables
        self.threshold_var = tk.IntVar(value=127)
        self.margin_var = tk.IntVar(value=10)
        
        # Load the image
        self.original_image = cv2.imread(image_path)
        # --- ADDED: Check if image loaded successfully ---
        if self.original_image is None:
            messagebox.showerror("Error", f"Failed to load image: {os.path.basename(image_path)}\\nPlease ensure the file exists and is a valid image.", parent=self)
            self.destroy() # Close the adjuster window
            return # Stop initialization
        # --- END ADDED ---
        self.height, self.width = self.original_image.shape[:2]
        
        # Calculate preview scale to ensure window isn't too tall
        screen_height = self.winfo_screenheight()
        max_preview_height = screen_height * 0.6  # Use 60% of screen height as maximum
        self.preview_scale = min(0.5, max_preview_height / self.height)  # Use 0.5 or smaller if needed
        
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
        
        # Margin control
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
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Apply", command=self.apply_crop).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)

        # Configure column weights
        main_frame.columnconfigure(1, weight=1)

    def update_preview(self):
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
            
            # Draw the contour and crop rectangle
            cv2.drawContours(preview, [largest_contour], -1, (0, 255, 0), 2)
            
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
        # Store the values in parent
        self.parent.threshold_value = self.threshold_var.get()
        self.parent.margin_value = self.margin_var.get()
        
        # Apply the crop
        self.parent.crop_to_largest_white_area(
            self.image_path,
            threshold=self.threshold_var.get(),
            margin=self.margin_var.get()
        )
        
        # Close the window
        self.destroy()

