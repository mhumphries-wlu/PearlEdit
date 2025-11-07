"""Main Tkinter application for PearlEdit."""
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from pathlib import Path
from typing import Tuple
import logging

from PIL import Image, ImageTk

# Try to import tkinterdnd2 for drag and drop support
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except ImportError:
    TkinterDnD = None
    DND_FILES = None
    DND_AVAILABLE = False

from ..services import ImageService, UserFacingError
from ..config import AppSettings, load_settings
from ..paths import TempManager
from .threshold_adjuster import ThresholdAdjuster
from .seam_finder import SeamFinderDialog

logger = logging.getLogger(__name__)


# Use TkinterDnD.Tk if available, otherwise fall back to tk.Tk
BaseTk = TkinterDnD.Tk if DND_AVAILABLE else tk.Tk


class ToolTip:
    """Tooltip widget for showing hints on hover."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.after_id = None
        self.widget.bind('<Enter>', self.on_enter)
        self.widget.bind('<Leave>', self.on_leave)
    
    def on_enter(self, event=None):
        """Show tooltip when mouse enters widget."""
        # Schedule tooltip to appear after a short delay
        self.after_id = self.widget.after(500, self.show_tooltip)
    
    def on_leave(self, event=None):
        """Hide tooltip when mouse leaves widget."""
        # Cancel scheduled tooltip if mouse leaves before delay
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        self.hide_tooltip()
    
    def show_tooltip(self):
        """Display the tooltip."""
        if self.tip_window or not self.text:
            return
        
        # Get widget position
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 20
        
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Arial", 9)
        )
        label.pack(ipadx=5, ipady=2)
    
    def hide_tooltip(self):
        """Hide the tooltip."""
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class PearlEditApp(BaseTk):
    """Main application window."""
    
    def __init__(self, source_directory: Path = None, settings: AppSettings = None):
        """
        Initialize the application.
        
        Args:
            source_directory: Optional directory containing images to edit (for backward compatibility)
            settings: Application settings (optional)
        """
        super().__init__()
        
        self.settings = settings or load_settings()
        self.temp_manager = TempManager()
        self.service = ImageService(self.settings, self.temp_manager)
        
        # Store source directory for optional auto-load
        self._source_directory = source_directory
        
        # Track current save directory
        self._current_save_directory = None
        # Track opened folder (for "Open Folder" functionality)
        self._opened_folder = None
        
        # Icon cache to prevent garbage collection
        self._icon_cache = []
        
        # Store tooltip instances
        self._tooltips = []
        
        # UI state
        self.current_scale = 1.0  # Fit-to-window scale
        self.zoom_level = 1.0  # Current zoom level (1.0 = fit to window)
        self.pan_x = 0  # Pan offset in x direction
        self.pan_y = 0  # Pan offset in y direction
        self.panning = False  # Whether panning mode is active
        self.pan_start_x = 0  # Starting x position for pan
        self.pan_start_y = 0  # Starting y position for pan
        self.original_image = None
        self._cached_photo = None  # Cached PhotoImage for smooth updates
        self._cached_zoom = 1.0  # Zoom level of cached image
        self._image_item = None  # Canvas item ID for the image
        self._zoom_update_pending = False  # Throttle zoom updates
        self.image_origin_x = 0  # X position of image on canvas (top-left)
        self.image_origin_y = 0  # Y position of image on canvas (top-left)
        self.special_cursor_active = False
        self.cursor_orientation = 'vertical'
        self.cursor_angle = 0
        self.cursor_line = None
        self.vertical_line = None
        self.horizontal_line = None
        self.cropping = False
        self.crop_start = None
        self.crop_end = None
        self.crop_rect = None
        self.straightening_mode = False
        self.straighten_start = None
        self.straighten_line = None
        self.guide_line = None
        self.active_tool_button = None  # Track which tool button is currently depressed
        
        # Initialize UI
        self.setup_window()
        self.create_widgets()
        self.create_menus()
        self.create_key_bindings()
        
        # Setup drag and drop support (after widgets are created)
        self.after(100, self.setup_drag_drop)
        
        # Show empty state after window is fully initialized
        self.update_idletasks()  # Ensure window is rendered
        
        # Add placeholder text after canvas is rendered
        def add_placeholder():
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                self._placeholder_text = self.image_canvas.create_text(
                    canvas_width // 2,
                    canvas_height // 2,
                    text="No images loaded.\n\nDrag and drop image files here,\nor use File > Import Images",
                    font=("Arial", 14),
                    fill="gray",
                    justify=tk.CENTER
                )
            else:
                self.after(50, add_placeholder)
        
        self.after(100, add_placeholder)
        
        # Initialize counter
        self.update_counter()
        
        # Optional: Auto-load if source directory provided (for backward compatibility)
        if source_directory:
            self.after(300, lambda: self.load_images_from_directory(source_directory))
    
    def setup_window(self):
        """Set up the main window."""
        self.title("PearlEdit 1.0")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Set application icon
        try:
            project_root = Path(__file__).parent.parent.parent
            icon_path = project_root / "util" / "icons" / "PearlEdit.png"
            if icon_path.exists():
                # Load icon at a reasonable size for window icon
                app_icon = self.load_icon(icon_path, size=(64, 64))
                self.iconphoto(True, app_icon)
            else:
                logger.warning(f"Application icon not found: {icon_path}")
        except Exception as e:
            logger.warning(f"Failed to set application icon: {e}")
        
        # Set zoomed state after window is mapped
        self.after_idle(lambda: self.state('zoomed'))
    
    def load_icon(self, icon_path: Path, size: tuple = (24, 24)) -> ImageTk.PhotoImage:
        """
        Load and prepare an icon image for use in buttons.
        
        Args:
            icon_path: Path to icon image file
            size: Target size (width, height) in pixels
            
        Returns:
            ImageTk.PhotoImage object ready for button use
        """
        try:
            if not icon_path.exists():
                logger.warning(f"Icon file not found: {icon_path}")
                # Return a blank placeholder
                blank = Image.new('RGBA', size, (0, 0, 0, 0))
                photo = ImageTk.PhotoImage(blank)
                self._icon_cache.append(photo)
                return photo
            
            # Load the image
            img = Image.open(icon_path)
            
            # Convert to RGBA if not already (for transparency support)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Resize if needed
            if img.size != size:
                img = img.resize(size, Image.LANCZOS)
            
            # Convert to PhotoImage for Tkinter
            photo = ImageTk.PhotoImage(img)
            
            # Store reference to prevent garbage collection
            self._icon_cache.append(photo)
            
            return photo
        except Exception as e:
            logger.error(f"Failed to load icon {icon_path}: {e}")
            # Return a blank placeholder
            blank = Image.new('RGBA', size, (0, 0, 0, 0))
            photo = ImageTk.PhotoImage(blank)
            self._icon_cache.append(photo)
            return photo
    
    def create_widgets(self):
        """Create UI widgets."""
        self.geometry("800x600")
        
        # Get icons directory path (relative to project root)
        project_root = Path(__file__).parent.parent.parent
        icons_dir = project_root / "util" / "icons"
        
        # Load all icons (note: filenames are lowercase)
        self.open_icon = self.load_icon(icons_dir / "open.png", size=(24, 24))
        self.save_icon = self.load_icon(icons_dir / "save.png", size=(24, 24))
        self.reset_icon = self.load_icon(icons_dir / "new.png", size=(24, 24))
        self.split_icon = self.load_icon(icons_dir / "split.png", size=(24, 24))
        self.crop_icon = self.load_icon(icons_dir / "crop.png", size=(24, 24))
        self.crop_on_icon = self.load_icon(icons_dir / "crop-on.png", size=(24, 24))
        self.autocrop_icon = self.load_icon(icons_dir / "autocrop.png", size=(24, 24))
        self.autocrop_on_icon = self.load_icon(icons_dir / "autocrop-on.png", size=(24, 24))
        self.straighten_icon = self.load_icon(icons_dir / "straighten.png", size=(24, 24))
        self.straighten_on_icon = self.load_icon(icons_dir / "straighten-on.png", size=(24, 24))
        self.autostraighten_icon = self.load_icon(icons_dir / "autostraighten.png", size=(24, 24))
        self.autostraighten_on_icon = self.load_icon(icons_dir / "autostraighten-on.png", size=(24, 24))
        
        # Navigation icons
        self.start_icon = self.load_icon(icons_dir / "start.png", size=(24, 24))
        self.back_icon = self.load_icon(icons_dir / "back.png", size=(24, 24))
        self.forward_icon = self.load_icon(icons_dir / "forward.png", size=(24, 24))
        self.end_icon = self.load_icon(icons_dir / "end.png", size=(24, 24))
        
        # Rotation icons
        self.rotate_left_icon = self.load_icon(icons_dir / "rotate-left.png", size=(24, 24))
        self.rotate_right_icon = self.load_icon(icons_dir / "rotate-right.png", size=(24, 24))
        
        # Batch process icons
        self.batch_process_off_icon = self.load_icon(icons_dir / "batch-process-off.png", size=(24, 24))
        self.batch_process_on_icon = self.load_icon(icons_dir / "batch-process-on.png", size=(24, 24))
        
        # Apply to all images toggle icons
        self.current_page_icon = self.load_icon(icons_dir / "current-page.png", size=(24, 24))
        self.all_pages_icon = self.load_icon(icons_dir / "all-pages.png", size=(24, 24))
        
        # Delete icons
        self.delete_icon = self.load_icon(icons_dir / "delete.png", size=(24, 24))
        self.delete_multi_icon = self.load_icon(icons_dir / "delete-multi.png", size=(24, 24))
        
        # Toolbar frame at the top
        self.toolbar_frame = tk.Frame(self, relief=tk.RAISED, borderwidth=1)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        
        # Create section containers with labels
        # File Operations section
        file_ops_section = tk.Frame(self.toolbar_frame)
        file_ops_section.pack(side=tk.LEFT, padx=2)
        
        file_ops_buttons = tk.Frame(file_ops_section)
        file_ops_buttons.pack(side=tk.TOP, pady=2)
        
        # Left side: New button
        def new_wrapper():
            self.update_status_display("New: Clears all loaded images and resets application to initial state | All unsaved changes will be lost")
            self.reset_program()
        self.new_button = self.create_button_with_hint(
            file_ops_buttons,
            self.reset_icon,
            new_wrapper,
            "New\nClear all loaded images",
            "New: Clears all loaded images and resets application to initial state | All unsaved changes will be lost"
        )
        self.new_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Left side: Open button
        def open_wrapper():
            self.update_status_display("Import Tool: Select image files (.jpg, .jpeg) to import | Shortcut: Ctrl+I | Or drag and drop images onto canvas")
            self.import_images()
        self.open_button = self.create_button_with_hint(
            file_ops_buttons,
            self.open_icon,
            open_wrapper,
            "Import Images\nSelect image files to import\nShortcut: Ctrl+I",
            "Import Tool: Select image files (.jpg, .jpeg) to import | Shortcut: Ctrl+I | Or drag and drop images onto canvas"
        )
        self.open_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Left side: Save button
        def save_wrapper():
            self.update_status_display("Save Tool: Saves all processed images to current directory | Shortcut: Ctrl+S | Use Save As (Ctrl+Shift+S) to choose location")
            self.save_images()
        self.save_button = self.create_button_with_hint(
            file_ops_buttons,
            self.save_icon,
            save_wrapper,
            "Save Images\nSave processed images to current directory\nShortcut: Ctrl+S",
            "Save Tool: Saves all processed images to current directory | Shortcut: Ctrl+S | Use Save As (Ctrl+Shift+S) to choose location"
        )
        self.save_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Label for File Operations
        file_ops_label = tk.Label(
            file_ops_section,
            text="File Operations",
            font=("Arial", 7),
            fg="gray"
        )
        file_ops_label.pack(side=tk.BOTTOM, pady=(0, 2))
        
        # Divider between File Operations and Delete buttons
        separator_file_ops = tk.Frame(self.toolbar_frame, width=2, relief=tk.SUNKEN, borderwidth=1)
        separator_file_ops.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        
        # Delete section
        delete_section = tk.Frame(self.toolbar_frame)
        delete_section.pack(side=tk.LEFT, padx=2)
        
        delete_buttons = tk.Frame(delete_section)
        delete_buttons.pack(side=tk.TOP, pady=2)
        
        # Delete button
        def delete_wrapper():
            self.update_status_display("Delete Tool: Deletes the current image | Use Delete Range to delete multiple images")
            self.delete_current()
        self.delete_button = self.create_button_with_hint(
            delete_buttons,
            self.delete_icon,
            delete_wrapper,
            "Delete Image\nDelete the current image",
            "Delete Tool: Deletes the current image | Use Delete Range to delete multiple images"
        )
        self.delete_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Delete Range button
        def delete_range_wrapper():
            self.update_status_display("Delete Range Tool: Opens dialog to delete multiple images | Select range and deletion mode")
            self.delete_range_dialog()
        self.delete_range_button = self.create_button_with_hint(
            delete_buttons,
            self.delete_multi_icon,
            delete_range_wrapper,
            "Delete Range\nDelete multiple images\nOpens selection dialog",
            "Delete Range Tool: Opens dialog to delete multiple images | Select range and deletion mode"
        )
        self.delete_range_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Label for Delete section
        delete_label = tk.Label(
            delete_section,
            text="Delete Page",
            font=("Arial", 7),
            fg="gray"
        )
        delete_label.pack(side=tk.BOTTOM, pady=(0, 2))
        
        # Divider after Delete section
        separator = tk.Frame(self.toolbar_frame, width=2, relief=tk.SUNKEN, borderwidth=1)
        separator.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        
        # Processing section
        processing_section = tk.Frame(self.toolbar_frame)
        processing_section.pack(side=tk.LEFT, padx=2)
        
        processing_buttons = tk.Frame(processing_section)
        processing_buttons.pack(side=tk.TOP, pady=2)
        
        # Batch Process toggle button
        self.batch_process = tk.BooleanVar(value=self.settings.batch_process)
        initial_relief = tk.SUNKEN if self.settings.batch_process else tk.RAISED
        initial_icon = self.batch_process_on_icon if self.settings.batch_process else self.batch_process_off_icon
        
        def batch_process_wrapper():
            # Will update status display in toggle_batch_process
            self.toggle_batch_process()
        
        self.batch_process_button = tk.Button(
            processing_buttons,
            image=initial_icon,
            command=batch_process_wrapper,
            width=30,
            height=30,
            relief=initial_relief
        )
        self.batch_process_button.pack(side=tk.LEFT, padx=5, pady=2)
        self._tooltips.append(ToolTip(self.batch_process_button, "Batch Process\nToggle automatic advancement to next image\nafter operations"))
        # Update icon based on initial state (ensures consistency)
        self.update_batch_process_icon()
        
        # Apply to All Images toggle button
        self.apply_to_all = tk.BooleanVar(value=False)  # Default to current image only
        
        def apply_to_all_wrapper():
            # Will update status display in toggle_apply_to_all
            self.toggle_apply_to_all()
        
        self.apply_to_all_button = tk.Button(
            processing_buttons,
            image=self.current_page_icon,
            command=apply_to_all_wrapper,
            width=30,
            height=30,
            relief=tk.RAISED
        )
        self.apply_to_all_button.pack(side=tk.LEFT, padx=5, pady=2)
        self._tooltips.append(ToolTip(self.apply_to_all_button, "Apply to All\nToggle whether operations apply to\ncurrent image or all images"))
        
        # Label for Processing
        processing_label = tk.Label(
            processing_section,
            text="Processing Type",
            font=("Arial", 7),
            fg="gray"
        )
        processing_label.pack(side=tk.BOTTOM, pady=(0, 2))
        
        # Divider after Processing (NEW - between Processing and Image Editing)
        separator_processing = tk.Frame(self.toolbar_frame, width=2, relief=tk.SUNKEN, borderwidth=1)
        separator_processing.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        
        # Split section
        split_section = tk.Frame(self.toolbar_frame)
        split_section.pack(side=tk.LEFT, padx=2)
        
        split_buttons = tk.Frame(split_section)
        split_buttons.pack(side=tk.TOP, pady=2)
        
        def split_wrapper():
            self.clear_button_depressed()
            self.update_status_display("Split Tool: To change between Horizontal and Vertical Cursor use Ctrl+H and Ctrl+V | To rotate cursor use [ and ] | To split image, click mouse")
            self.switch_to_vertical()
        self.split_button = self.create_button_with_hint(
            split_buttons,
            self.split_icon,
            split_wrapper,
            "Split Image\nActivate vertical split cursor\nClick to split at cursor position\nShortcuts: Ctrl+V (vertical), Ctrl+H (horizontal)",
            "Split Tool: To change between Horizontal and Vertical Cursor use Ctrl+H and Ctrl+V | To rotate cursor use [ and ] | To split image, click mouse"
        )
        self.split_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        def autosplit_finder_wrapper():
            self.clear_button_depressed()
            self.update_status_display("Auto Split Finder: Adjust threshold slider to find the book seam | Click Apply Split to split along detected line")
            self.auto_split_current_finder()
        self.autosplit_finder_button = self.create_button_with_hint(
            split_buttons,
            self.split_icon,  # Reuse split icon for now
            autosplit_finder_wrapper,
            "Auto Split (Finder)\nAutomatically detect book seam\nAdjust threshold in dialog\nClick Apply Split to split",
            "Auto Split Finder: Adjust threshold slider to find the book seam | Click Apply Split to split along detected line"
        )
        self.autosplit_finder_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Label for Split
        split_label = tk.Label(
            split_section,
            text="Split",
            font=("Arial", 7),
            fg="gray"
        )
        split_label.pack(side=tk.BOTTOM, pady=(0, 2))
        
        # Divider after Split
        separator_split = tk.Frame(self.toolbar_frame, width=2, relief=tk.SUNKEN, borderwidth=1)
        separator_split.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        
        # Crop section
        crop_section = tk.Frame(self.toolbar_frame)
        crop_section.pack(side=tk.LEFT, padx=2)
        
        crop_buttons = tk.Frame(crop_section)
        crop_buttons.pack(side=tk.TOP, pady=2)
        
        self.crop_button = self.create_button_with_hint(
            crop_buttons,
            self.crop_icon,
            lambda: self._crop_button_clicked(),
            "Crop Tool\nActivate crop mode\nDrag to select area\nEnter to apply, Escape to cancel\nShortcut: Ctrl+Shift+C",
            "Crop Tool: Drag mouse to select area | To apply crop, press Enter | To cancel, press Escape"
        )
        self.crop_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.autocrop_button = self.create_button_with_hint(
            crop_buttons,
            self.autocrop_icon,
            lambda: self._autocrop_button_clicked(),
            "Auto Crop\nAutomatically crop image using edge detection\nAdjust threshold in dialog\nShortcut: Ctrl+Shift+A",
            "Auto Crop Tool: Adjust threshold and margin sliders in dialog | Click Apply to crop image | Click Cancel to abort"
        )
        self.autocrop_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Label for Crop
        crop_label = tk.Label(
            crop_section,
            text="Crop",
            font=("Arial", 7),
            fg="gray"
        )
        crop_label.pack(side=tk.BOTTOM, pady=(0, 2))
        
        # Divider after Crop
        separator_crop = tk.Frame(self.toolbar_frame, width=2, relief=tk.SUNKEN, borderwidth=1)
        separator_crop.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        
        # Straighten section
        straighten_section = tk.Frame(self.toolbar_frame)
        straighten_section.pack(side=tk.LEFT, padx=2)
        
        straighten_buttons = tk.Frame(straighten_section)
        straighten_buttons.pack(side=tk.TOP, pady=2)
        
        self.straighten_button = self.create_button_with_hint(
            straighten_buttons,
            self.straighten_icon,
            lambda: self._straighten_button_clicked(),
            "Straighten Image\nDraw a line to straighten image\nClick start point, then end point\nShortcut: Ctrl+L",
            "Straighten Tool: Click first point to start line | Click second point to end line | Image will rotate to align line"
        )
        self.straighten_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.autostraighten_button = self.create_button_with_hint(
            straighten_buttons,
            self.autostraighten_icon,
            lambda: self._autostraighten_button_clicked(),
            "Auto Straighten\nAutomatically straighten image using edge detection\nAdjust threshold in dialog\nShortcut: Ctrl+Shift+L",
            "Auto Straighten Tool: Adjust threshold slider in dialog | Click Apply to straighten image | Click Cancel to abort"
        )
        self.autostraighten_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Label for Straighten
        straighten_label = tk.Label(
            straighten_section,
            text="Straighten",
            font=("Arial", 7),
            fg="gray"
        )
        straighten_label.pack(side=tk.BOTTOM, pady=(0, 2))
        
        # Divider after Straighten
        separator_straighten = tk.Frame(self.toolbar_frame, width=2, relief=tk.SUNKEN, borderwidth=1)
        separator_straighten.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        
        # Rotation section (left side, after Image Editing)
        rotation_section = tk.Frame(self.toolbar_frame)
        rotation_section.pack(side=tk.LEFT, padx=2)
        
        rotation_buttons = tk.Frame(rotation_section)
        rotation_buttons.pack(side=tk.TOP, pady=2)
        
        # Rotate buttons
        def rotate_right_wrapper():
            self.clear_button_depressed()
            self.update_status_display("Rotate Tool: Rotates image 90 degrees clockwise | Shortcut: Ctrl+] | Use Apply to All to rotate all images")
            self.rotate_image(-90)
        self.rotate_right_button = self.create_button_with_hint(
            rotation_buttons,
            self.rotate_right_icon,
            rotate_right_wrapper,
            "Rotate Clockwise\nRotate image 90° clockwise\nShortcut: Ctrl+]",
            "Rotate Tool: Rotates image 90 degrees clockwise | Shortcut: Ctrl+] | Use Apply to All to rotate all images"
        )
        self.rotate_right_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        def rotate_left_wrapper():
            self.clear_button_depressed()
            self.update_status_display("Rotate Tool: Rotates image 90 degrees counter-clockwise | Shortcut: Ctrl+[ | Use Apply to All to rotate all images")
            self.rotate_image(90)
        self.rotate_left_button = self.create_button_with_hint(
            rotation_buttons,
            self.rotate_left_icon,
            rotate_left_wrapper,
            "Rotate Counter-Clockwise\nRotate image 90° counter-clockwise\nShortcut: Ctrl+[",
            "Rotate Tool: Rotates image 90 degrees counter-clockwise | Shortcut: Ctrl+[ | Use Apply to All to rotate all images"
        )
        self.rotate_left_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Label for Rotation
        rotation_label = tk.Label(
            rotation_section,
            text="Rotation",
            font=("Arial", 7),
            fg="gray"
        )
        rotation_label.pack(side=tk.BOTTOM, pady=(0, 2))
        
        # Spacer to push navigation to the right
        tk.Frame(self.toolbar_frame, width=20).pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # Navigation section (right side, pressed against the right edge)
        navigation_section = tk.Frame(self.toolbar_frame)
        navigation_section.pack(side=tk.RIGHT, padx=(2, 2))
        
        navigation_buttons = tk.Frame(navigation_section)
        navigation_buttons.pack(side=tk.TOP, pady=2)
        
        # Right side: Navigation buttons (order: start, back, counter, forward, end)
        def start_wrapper():
            self.update_status_display("Navigation Tool: Jump to first image in collection | Use arrow buttons or Left/Right arrow keys to navigate")
            self.navigate_images(-2)
        self.start_button = self.create_button_with_hint(
            navigation_buttons,
            self.start_icon,
            start_wrapper,
            "First Image\nGo to the first image",
            "Navigation Tool: Jump to first image in collection | Use arrow buttons or Left/Right arrow keys to navigate"
        )
        self.start_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        def back_wrapper():
            self.update_status_display("Navigation Tool: Move to previous image | Shortcut: Left Arrow | Counter shows current position (e.g., 3 / 10)")
            self.navigate_images(-1)
        self.back_button = self.create_button_with_hint(
            navigation_buttons,
            self.back_icon,
            back_wrapper,
            "Previous Image\nNavigate to previous image\nShortcut: Left Arrow",
            "Navigation Tool: Move to previous image | Shortcut: Left Arrow | Counter shows current position (e.g., 3 / 10)"
        )
        self.back_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Counter between back and forward
        self.counter_label = ttk.Label(
            navigation_buttons,
            text="0 / 0",
            font=("Arial", 10),
            padding=(5, 0)
        )
        self.counter_label.pack(side=tk.LEFT, padx=10)
        
        def forward_wrapper():
            self.update_status_display("Navigation Tool: Move to next image | Shortcut: Right Arrow | Use First/Last buttons to jump to ends")
            self.navigate_images(1)
        self.forward_button = self.create_button_with_hint(
            navigation_buttons,
            self.forward_icon,
            forward_wrapper,
            "Next Image\nNavigate to next image\nShortcut: Right Arrow",
            "Navigation Tool: Move to next image | Shortcut: Right Arrow | Use First/Last buttons to jump to ends"
        )
        self.forward_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        def end_wrapper():
            self.update_status_display("Navigation Tool: Jump to last image in collection | Use arrow buttons or Left/Right arrow keys to navigate")
            self.navigate_images(2)
        self.end_button = self.create_button_with_hint(
            navigation_buttons,
            self.end_icon,
            end_wrapper,
            "Last Image\nGo to the last image",
            "Navigation Tool: Jump to last image in collection | Use arrow buttons or Left/Right arrow keys to navigate"
        )
        self.end_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Label for Navigation
        navigation_label = tk.Label(
            navigation_section,
            text="Navigation",
            font=("Arial", 7),
            fg="gray"
        )
        navigation_label.pack(side=tk.BOTTOM, pady=(0, 2))
        
        # Main frame for canvas
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Bottom frame for status display (pack first to ensure it's at bottom)
        self.bottom_frame = tk.Frame(main_frame, bg='#e0e0e0', relief=tk.RAISED, borderwidth=2)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=0, padx=0)
        
        # Status label for showing button instructions
        self.status_label = tk.Label(
            self.bottom_frame,
            text="Ready | Zoom: CTRL + Mouse Wheel | Pan: Hold SPACEBAR + Drag Mouse",
            anchor=tk.W,
            justify=tk.LEFT,
            font=("Arial", 10),
            bg='#e0e0e0',
            fg='#000000',
            padx=15,
            pady=8,
            wraplength=1500,
            relief=tk.FLAT
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas with minimum size (pack after bottom frame)
        self.image_canvas = tk.Canvas(main_frame, highlightthickness=0, bg='lightgray')
        self.image_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add a placeholder message (will be positioned after window is rendered)
        self._placeholder_text = None
        
        # Bind mouse events
        self.image_canvas.bind("<Motion>", self.update_cursor_line)
    
    def update_status_display(self, description: str):
        """
        Update the bottom status display with button information.
        
        Args:
            description: Full description with instructions and shortcuts (pipe-separated format)
        """
        try:
            self.status_label.config(text=description)
            # Force update to ensure visibility
            self.status_label.update_idletasks()
            self.bottom_frame.update_idletasks()
        except Exception as e:
            logger.error(f"Error updating status display: {e}")
    
    def create_button_with_hint(self, parent, image, command, tooltip_text, description, shortcuts=None):
        """
        Create a button with tooltip and status display update.
        
        Args:
            parent: Parent widget
            image: Button icon
            command: Command function to execute (should already call update_status_display)
            tooltip_text: Text for tooltip
            description: Description for status display (unused, kept for compatibility)
            shortcuts: List of keyboard shortcuts (unused, kept for compatibility)
        
        Returns:
            The created button widget
        """
        # Note: command already calls update_status_display, so we don't need to call it again
        button = tk.Button(
            parent,
            image=image,
            command=command,
            width=30,
            height=30
        )
        
        # Add tooltip
        tooltip = ToolTip(button, tooltip_text)
        self._tooltips.append(tooltip)
        
        return button
    
    def set_button_depressed(self, button, on_icon):
        """Set a button to depressed state with on icon."""
        # If clicking the same button that's already depressed, toggle it off
        if self.active_tool_button == button:
            self.clear_button_depressed()
            return False  # Return False to indicate button was toggled off
        
        # Clear any previously depressed button
        self.clear_button_depressed()
        
        # Set new button to depressed state
        button.config(image=on_icon, relief=tk.SUNKEN)
        self.active_tool_button = button
        return True  # Return True to indicate button was set to depressed
    
    def clear_button_depressed(self):
        """Clear the depressed state of the active button."""
        if self.active_tool_button:
            # Restore normal icon and relief based on which button it is
            if self.active_tool_button == self.crop_button:
                self.active_tool_button.config(image=self.crop_icon, relief=tk.RAISED)
            elif self.active_tool_button == self.autocrop_button:
                self.active_tool_button.config(image=self.autocrop_icon, relief=tk.RAISED)
            elif self.active_tool_button == self.straighten_button:
                self.active_tool_button.config(image=self.straighten_icon, relief=tk.RAISED)
            elif self.active_tool_button == self.autostraighten_button:
                self.active_tool_button.config(image=self.autostraighten_icon, relief=tk.RAISED)
            self.active_tool_button = None
    
    def _crop_button_clicked(self):
        """Handle crop button click."""
        # Check if button was toggled off (user clicked same button again)
        was_set = self.set_button_depressed(self.crop_button, self.crop_on_icon)
        if not was_set:
            # Button was toggled off, clear modes and return to neutral state
            self.clear_all_modes()
            return
        
        self.update_status_display("Crop Tool: Drag mouse to select area | To apply crop, press Enter | To cancel, press Escape")
        self.activate_crop_tool()
        # activate_crop_tool uses _clear_modes_preserve_button which preserves the button state,
        # but if it returned early (warning canceled), we need to clear the button
        if not self.cropping:
            # Tool wasn't activated (warning was canceled), clear button
            self.clear_button_depressed()
    
    def _straighten_button_clicked(self):
        """Handle straighten button click."""
        # Check if button was toggled off (user clicked same button again)
        was_set = self.set_button_depressed(self.straighten_button, self.straighten_on_icon)
        if not was_set:
            # Button was toggled off, clear modes and return to neutral state
            self.clear_all_modes()
            return
        
        self.update_status_display("Straighten Tool: Click first point to start line | Click second point to end line | Image will rotate to align line")
        self.manual_straighten()
    
    def _autocrop_button_clicked(self):
        """Handle autocrop button click."""
        # Check if button was toggled off (user clicked same button again)
        was_set = self.set_button_depressed(self.autocrop_button, self.autocrop_on_icon)
        if not was_set:
            # Button was toggled off, clear modes and return to neutral state
            self.clear_all_modes()
            return
        
        self.update_status_display("Auto Crop Tool: Adjust threshold and margin sliders in dialog | Click Apply to crop image | Click Cancel to abort")
        self.auto_crop_current()
    
    def _autostraighten_button_clicked(self):
        """Handle autostraighten button click."""
        # Check if button was toggled off (user clicked same button again)
        was_set = self.set_button_depressed(self.autostraighten_button, self.autostraighten_on_icon)
        if not was_set:
            # Button was toggled off, clear modes and return to neutral state
            self.clear_all_modes()
            return
        
        self.update_status_display("Auto Straighten Tool: Adjust threshold slider in dialog | Click Apply to straighten image | Click Cancel to abort")
        self.auto_straighten_current()
    
    def create_menus(self):
        """Create menu bar."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New", command=self.reset_program)
        file_menu.add_separator()
        file_menu.add_command(label="Open Folder...", command=self.open_folder, accelerator="Ctrl+O")
        file_menu.add_command(label="Import Images...", command=self.import_images, accelerator="Ctrl+I")
        file_menu.add_command(label="Import PDF...", command=self.import_pdf)
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self.save_images, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_images_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", command=self.undo_operation, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo_operation, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Revert Current Image", command=self.revert_current)
        edit_menu.add_command(label="Revert All Images", command=self.revert_all)
        edit_menu.add_separator()
        edit_menu.add_command(label="Delete Image", command=self.delete_current)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="First Image", command=lambda: self.navigate_images(-2))
        view_menu.add_command(label="Back Image", command=lambda: self.navigate_images(-1))
        view_menu.add_command(label="Forward Image", command=lambda: self.navigate_images(1))
        view_menu.add_command(label="Last Image", command=lambda: self.navigate_images(2))
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Process menu
        process_menu = tk.Menu(menubar, tearoff=0)
        
        # Processing Mode submenu
        processing_mode_menu = tk.Menu(process_menu, tearoff=0)
        self.processing_mode_var = tk.StringVar(value="Sequential" if not self.batch_process.get() else "Batch Processing")
        
        def set_processing_mode(mode):
            """Set processing mode and sync with button."""
            self.processing_mode_var.set(mode)
            is_batch_processing = (mode == "Batch Processing")
            if self.batch_process.get() != is_batch_processing:
                self.batch_process.set(is_batch_processing)
                self.settings.batch_process = is_batch_processing
                
                # If turning batch processing ON, turn OFF apply to all
                if is_batch_processing and self.apply_to_all.get():
                    self.apply_to_all.set(False)
                    self.update_apply_to_all_icon()
                    # Sync with menu
                    if hasattr(self, 'apply_to_var'):
                        self.apply_to_var.set("Current Image")
                
                self.update_batch_process_icon()
                # Update status display
                if is_batch_processing:
                    self.update_status_display("Batch Process: When enabled, automatically advances to next image after operations | Toggle on/off to control batch processing")
                else:
                    self.update_status_display("Sequential Process: Operations complete on current image only | No automatic advancement | Toggle Batch Processing to enable auto-advance")
        
        processing_mode_menu.add_radiobutton(
            label="Sequential",
            variable=self.processing_mode_var,
            value="Sequential",
            command=lambda: set_processing_mode("Sequential")
        )
        processing_mode_menu.add_radiobutton(
            label="Batch Processing",
            variable=self.processing_mode_var,
            value="Batch Processing",
            command=lambda: set_processing_mode("Batch Processing")
        )
        process_menu.add_cascade(label="Processing Mode", menu=processing_mode_menu)
        
        # Apply to... submenu
        apply_to_menu = tk.Menu(process_menu, tearoff=0)
        self.apply_to_var = tk.StringVar(value="All Images" if self.apply_to_all.get() else "Current Image")
        
        def set_apply_to(mode):
            """Set apply to mode and sync with button."""
            self.apply_to_var.set(mode)
            is_all = (mode == "All Images")
            if self.apply_to_all.get() != is_all:
                self.apply_to_all.set(is_all)
                
                # If turning apply to all ON, turn OFF batch processing
                if is_all and self.batch_process.get():
                    self.batch_process.set(False)
                    self.settings.batch_process = False
                    self.update_batch_process_icon()
                    # Sync with menu
                    if hasattr(self, 'processing_mode_var'):
                        self.processing_mode_var.set("Sequential")
                
                self.update_apply_to_all_icon()
                # Update status display
                if is_all:
                    self.update_status_display("Apply to All: When enabled, operations apply to all images | When disabled, operations apply only to current image | Toggle on/off as needed")
                else:
                    self.update_status_display("Apply to Current: Operations apply only to current image | Enable Apply to All to process all images at once | Toggle on/off as needed")
        
        apply_to_menu.add_radiobutton(
            label="Current Image",
            variable=self.apply_to_var,
            value="Current Image",
            command=lambda: set_apply_to("Current Image")
        )
        apply_to_menu.add_radiobutton(
            label="All Images",
            variable=self.apply_to_var,
            value="All Images",
            command=lambda: set_apply_to("All Images")
        )
        process_menu.add_cascade(label="Apply to...", menu=apply_to_menu)
        
        process_menu.add_separator()
        
        # Process menu items in toolbar order
        process_menu.add_command(label="Split Image", command=lambda: self.switch_to_vertical())
        process_menu.add_command(label="Crop Image", command=self.activate_crop_tool)
        process_menu.add_command(label="Auto Crop", command=self.auto_crop_current)
        process_menu.add_command(label="Straighten Image", command=self.manual_straighten)
        process_menu.add_command(label="Auto Straighten", command=self.auto_straighten_current)
        process_menu.add_command(label="Rotate Clockwise", command=lambda: self.rotate_image(-90))
        process_menu.add_command(label="Rotate Counter-Clockwise", command=lambda: self.rotate_image(90))
        
        menubar.add_cascade(label="Process", menu=process_menu)
    
    def create_key_bindings(self):
        """Create keyboard bindings."""
        # Cursor bindings
        self.bind("<Control-h>", lambda e: self.switch_to_horizontal())
        self.bind("<Control-v>", lambda e: self.switch_to_vertical())
        # Autosplit keyboard shortcut removed per user request
        # self.bind("<Control-a>", lambda e: self.toggle_auto_split())
        
        # Straighten image
        self.bind("<Control-l>", lambda e: self.manual_straighten())
        
        # Crop image
        self.bind("<Control-Shift-c>", lambda e: self.activate_crop_tool())
        
        # Auto crop
        self.bind("<Control-Shift-a>", lambda e: self.auto_crop_current())
        
        # Auto straighten
        self.bind("<Control-Shift-l>", lambda e: self.auto_straighten_current())
        
        # Cursor rotation
        self.bind("<bracketright>", lambda e: self.rotate_cursor(-1))
        self.bind("<bracketleft>", lambda e: self.rotate_cursor(1))
        
        # Mouse and navigation
        self.image_canvas.bind("<Button-1>", self.handle_mouse_click)
        self.image_canvas.bind("<ButtonRelease-1>", self.handle_mouse_release)
        self.image_canvas.bind("<B1-Motion>", self.handle_mouse_drag)
        self.bind("<Left>", lambda e: self.navigate_images(-1))
        self.bind("<Right>", lambda e: self.navigate_images(1))
        
        # Zoom and pan
        # Mousewheel binding works differently on different platforms
        # On Windows, use <MouseWheel>
        # On Linux/Mac, use <Button-4> and <Button-5>
        self.image_canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.image_canvas.bind("<Button-4>", self.on_mousewheel)  # Linux/Mac scroll up
        self.image_canvas.bind("<Button-5>", self.on_mousewheel)  # Linux/Mac scroll down
        self.bind("<KeyPress-space>", self.start_pan)
        self.bind("<KeyRelease-space>", self.stop_pan)
        # Focus on canvas for key bindings
        self.image_canvas.focus_set()
        
        # Image rotation
        self.bind("<Control-bracketright>", lambda e: self.rotate_image(-90))
        self.bind("<Control-bracketleft>", lambda e: self.rotate_image(90))
        self.bind("<Control-Shift-bracketright>", lambda e: self.rotate_all_images(-90))
        self.bind("<Control-Shift-bracketleft>", lambda e: self.rotate_all_images(90))
        
        # File operations
        self.bind("<Control-o>", lambda e: self.open_folder())
        self.bind("<Control-i>", lambda e: self.import_images())
        self.bind("<Control-s>", lambda e: self.save_images())
        self.bind("<Control-Shift-s>", lambda e: self.save_images_as())
        
        # Undo/Redo
        self.bind("<Control-z>", lambda e: self.undo_operation())
        self.bind("<Control-y>", lambda e: self.redo_operation())
    
    def setup_drag_drop(self):
        """Setup drag and drop support for image files."""
        if DND_AVAILABLE:
            try:
                # Register the main window as drop target
                self.drop_target_register(DND_FILES)
                self.dnd_bind('<<Drop>>', self.on_drop)
                logger.info("Main window registered for drag and drop")
                
                # Also register canvas for better UX
                self.image_canvas.drop_target_register(DND_FILES)
                self.image_canvas.dnd_bind('<<Drop>>', self.on_drop)
                logger.info("Canvas registered for drag and drop")
                logger.info("Drag and drop support fully enabled")
            except AttributeError as e:
                # This might happen if the window/widget doesn't support DnD
                logger.warning(f"Widget does not support drag and drop: {e}")
            except Exception as e:
                logger.warning(f"Error setting up drag and drop: {e}", exc_info=True)
        else:
            logger.info("tkinterdnd2 not available - drag and drop disabled. Install with: pip install tkinterdnd2")
            # Show hint in placeholder text
            if self._placeholder_text:
                try:
                    self.image_canvas.delete(self._placeholder_text)
                except:
                    pass
                canvas_width = self.image_canvas.winfo_width()
                canvas_height = self.image_canvas.winfo_height()
                if canvas_width > 1 and canvas_height > 1:
                    self._placeholder_text = self.image_canvas.create_text(
                        canvas_width // 2,
                        canvas_height // 2,
                        text="No images loaded.\n\nInstall tkinterdnd2 for drag and drop:\npip install tkinterdnd2\n\nOr use File > Import Images",
                        font=("Arial", 12),
                        fill="gray",
                        justify=tk.CENTER
                    )
    
    def _show_drag_drop_hint(self):
        """Show hint about drag and drop if not available."""
        if not hasattr(self, '_drag_drop_hint_shown'):
            try:
                import tkinterdnd2
            except ImportError:
                messagebox.showinfo(
                    "Drag and Drop",
                    "To enable drag and drop support, install tkinterdnd2:\n\npip install tkinterdnd2\n\n"
                    "You can still import images using File > Import Images"
                )
                self._drag_drop_hint_shown = True
    
    def on_drop(self, event):
        """Handle dropped files."""
        if not DND_AVAILABLE:
            return
        
        try:
            # Get the dropped files - event.data contains the file paths
            # On Windows, paths are wrapped in braces and separated by spaces
            data = event.data
            
            # Parse the dropped files
            # tkinterdnd2 returns paths wrapped in braces on Windows: {file1} {file2}
            # We need to split them properly
            files = []
            if data.startswith('{') and data.endswith('}'):
                # Single file or multiple files wrapped in braces
                # Remove outer braces and split
                inner = data[1:-1]  # Remove outer braces
                # Split by '} {' to handle multiple files
                parts = inner.split('} {')
                for part in parts:
                    # Remove any remaining braces
                    clean = part.strip('{}')
                    if clean:
                        files.append(clean)
            else:
                # Try splitting by space (might work on some systems)
                files = data.split()
            
            # If that didn't work, try using tk.splitlist
            if not files:
                try:
                    files = self.tk.splitlist(data)
                except:
                    # Last resort: treat as single file
                    files = [data]
            
            image_files = []
            for f in files:
                # Remove any remaining braces and whitespace
                f_clean = f.strip('{}').strip()
                if not f_clean:
                    continue
                    
                path = Path(f_clean)
                
                # Check if file exists
                if not path.exists():
                    logger.warning(f"Dropped path does not exist: {path}")
                    continue
                
                if path.suffix.lower() in ('.jpg', '.jpeg'):
                    image_files.append(path)
                elif path.suffix.lower() == '.pdf':
                    # PDF files are handled by import_image_files
                    image_files.append(path)
                elif path.is_dir():
                    # If a directory is dropped, scan it for images
                    from ..repository import scan_images
                    try:
                        dir_images = scan_images(path)
                        image_files.extend(dir_images)
                    except Exception as e:
                        logger.error(f"Error scanning directory {path}: {e}")
            
            if not image_files:
                messagebox.showwarning("No Images", "No valid image files (.jpg, .jpeg) or PDF files (.pdf) were found in the dropped items.")
                return
            
            self.import_image_files(image_files)
            # Counter will be updated in import_image_files
        except Exception as e:
            logger.error(f"Error handling dropped files: {e}", exc_info=True)
            messagebox.showerror("Error", f"Error importing dropped files: {str(e)}")
    
    def open_folder(self):
        """Open a folder and load all images from it, setting it as the save directory."""
        from tkinter import filedialog
        
        # Open folder selection dialog
        folder = filedialog.askdirectory(
            title="Select Folder to Open",
            initialdir=str(self._current_save_directory) if self._current_save_directory else None
        )
        
        if not folder:
            return  # User cancelled
        
        folder_path = Path(folder)
        
        # Check if folder exists and has images
        if not folder_path.exists():
            messagebox.showerror("Error", f"Folder does not exist: {folder_path}")
            return
        
        if not folder_path.is_dir():
            messagebox.showerror("Error", f"Selected path is not a directory: {folder_path}")
            return
        
        # Load images from folder
        try:
            # Clear placeholder
            if self._placeholder_text:
                self.image_canvas.delete(self._placeholder_text)
                self._placeholder_text = None
            
            # Load images using the service
            count = self.service.load_images(folder_path)
            
            if count == 0:
                messagebox.showwarning("No Images", f"No images found in:\n{folder_path}")
                return
            
            # Set this folder as both the opened folder and save directory
            # When opening a folder, we save directly to it (not a subdirectory)
            self._opened_folder = folder_path
            self._current_save_directory = folder_path
            
            # Show success and display first image
            messagebox.showinfo("Folder Opened", f"Loaded {count} image(s) from:\n{folder_path}\n\nThis folder will be used for saving.")
            self.after(100, lambda: (self.show_current_image(), self.update_counter()))
            
        except UserFacingError as e:
            messagebox.showerror("Error", str(e))
            self.update_counter()
        except Exception as e:
            logger.error(f"Error opening folder: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to open folder: {str(e)}")
            self.update_counter()
    
    def import_images(self):
        """Open file dialog to import images."""
        from tkinter import filedialog
        
        files = filedialog.askopenfilenames(
            title="Select Images to Import",
            filetypes=[
                ("Image files", "*.jpg *.jpeg"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if files:
            image_files = [Path(f) for f in files]
            self.import_image_files(image_files)
    
    def import_pdf(self):
        """Open file dialog to import PDF and extract images."""
        from tkinter import filedialog
        
        file = filedialog.askopenfilename(
            title="Select PDF to Import",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        
        if file:
            pdf_file = Path(file)
            self.import_image_files([pdf_file])
    
    def import_image_files(self, image_files: list):
        """Import images from a list of file paths."""
        try:
            # Clear placeholder
            if self._placeholder_text:
                self.image_canvas.delete(self._placeholder_text)
                self._placeholder_text = None
            
            count = self.service.import_image_files(image_files)
            if count == 0:
                messagebox.showinfo("Import", "No new images were imported. They may already be loaded.")
                return
            
            messagebox.showinfo("Import", f"Successfully imported {count} image(s).")
            
            # Show the first/new image and update counter
            self.after(100, lambda: (self.show_current_image(), self.update_counter()))
        except UserFacingError as e:
            messagebox.showerror("Error", str(e))
            self.update_counter()  # Update counter even on error
        except Exception as e:
            logger.error(f"Error importing images: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to import images: {str(e)}")
            self.update_counter()  # Update counter even on error
    
    def load_images_from_directory(self, source_dir: Path):
        """Load images from source directory (for backward compatibility)."""
        try:
            # Clear placeholder
            if self._placeholder_text:
                self.image_canvas.delete(self._placeholder_text)
                self._placeholder_text = None
            
            count = self.service.load_images(source_dir)
            if count == 0:
                # Show message on canvas
                self.image_canvas.create_text(
                    self.image_canvas.winfo_width() // 2 if self.image_canvas.winfo_width() > 1 else 400,
                    self.image_canvas.winfo_height() // 2 if self.image_canvas.winfo_height() > 1 else 300,
                    text="No images found in:\n" + str(source_dir) + "\n\nUse File > Import Images to add images",
                    font=("Arial", 12),
                    fill="gray",
                    justify=tk.CENTER
                )
                messagebox.showwarning("No Images", f"No images found in:\n{source_dir}")
                return
            
            # Small delay to ensure window is fully rendered
            self.after(100, lambda: (self.show_current_image(), self.update_counter()))
        except UserFacingError as e:
            # Show error on canvas
            self.image_canvas.delete("all")
            self.image_canvas.create_text(
                self.image_canvas.winfo_width() // 2 if self.image_canvas.winfo_width() > 1 else 400,
                self.image_canvas.winfo_height() // 2 if self.image_canvas.winfo_height() > 1 else 300,
                text=f"Error:\n{str(e)}\n\nUse File > Import Images to add images",
                font=("Arial", 12),
                fill="red",
                justify=tk.CENTER
            )
            messagebox.showerror("Error", str(e))
        except Exception as e:
            logger.error(f"Error loading images: {e}", exc_info=True)
            # Show error on canvas
            self.image_canvas.delete("all")
            self.image_canvas.create_text(
                self.image_canvas.winfo_width() // 2 if self.image_canvas.winfo_width() > 1 else 400,
                self.image_canvas.winfo_height() // 2 if self.image_canvas.winfo_height() > 1 else 300,
                text=f"Error loading images:\n{str(e)}\n\nUse File > Import Images to add images",
                font=("Arial", 12),
                fill="red",
                justify=tk.CENTER
            )
            messagebox.showerror("Error", f"Failed to load images: {str(e)}")
    
    def update_counter(self):
        """Update the image counter display."""
        try:
            total = len(self.service.state.images)
            if total == 0:
                self.counter_label.config(text="0 / 0")
                return
            
            # Get current index (0-based) and convert to 1-based
            current_index = self.service.state.current_image_index
            current_num = current_index + 1  # 1-based
            
            # Ensure current_num is within valid range
            if current_num < 1:
                current_num = 1
            elif current_num > total:
                current_num = total
            
            self.counter_label.config(text=f"{current_num} / {total}")
        except Exception as e:
            logger.error(f"Error updating counter: {e}")
            # Gracefully handle error - show default
            try:
                self.counter_label.config(text="0 / 0")
            except:
                pass  # If even this fails, just continue
    
    def show_current_image(self):
        """Display the current image."""
        current = self.service.get_current_image()
        if not current:
            self.update_counter()
            return
        
        image_path = current.current_image_path
        if not image_path.exists():
            messagebox.showerror("Error", "Image file not found.")
            return
        
        # Reset zoom and pan when showing a new image (if image changed)
        if not hasattr(self, '_last_image_path') or self._last_image_path != image_path:
            self.zoom_level = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self._last_image_path = image_path
            self._cached_photo = None
            self._cached_zoom = 1.0
            self._image_item = None
        
        try:
            with Image.open(image_path) as image:
                self.original_image = image.copy()
                
                # Calculate scaling
                image_width, image_height = self.original_image.size
                canvas_width = self.image_canvas.winfo_width()
                canvas_height = self.image_canvas.winfo_height()
                
                if canvas_width <= 1 or canvas_height <= 1:
                    # Canvas not ready yet, try again
                    self.after(100, self.show_current_image)
                    return
                
                # Calculate fit-to-window scale
                scale_x = canvas_width / image_width
                scale_y = canvas_height / image_height
                self.current_scale = min(scale_x, scale_y)
                
                # Apply zoom level
                actual_scale = self.current_scale * self.zoom_level
                
                # Resize for display
                new_width = int(image_width * actual_scale)
                new_height = int(image_height * actual_scale)
                resized_image = self.original_image.resize((new_width, new_height), Image.LANCZOS)
                
                # Display with pan offset
                photo = ImageTk.PhotoImage(resized_image)
                self.image_canvas.delete("all")
                self.image_canvas.image = photo
                
                # Calculate position with pan offset
                x_pos = self.pan_x
                y_pos = self.pan_y
                
                # Center image if it's smaller than canvas
                if new_width < canvas_width:
                    x_pos = (canvas_width - new_width) // 2 + self.pan_x
                if new_height < canvas_height:
                    y_pos = (canvas_height - new_height) // 2 + self.pan_y
                
                # Store image item ID and cached photo
                self._image_item = self.image_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=photo)
                self._cached_photo = photo
                self._cached_zoom = self.zoom_level
                # Store image origin for coordinate conversion
                self.image_origin_x = x_pos
                self.image_origin_y = y_pos
                self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))
                
                # Redraw cursor if active
                if self.special_cursor_active:
                    self.update_cursor_line_display()
                
                # Update counter
                self.update_counter()
        except Exception as e:
            logger.error(f"Error showing image: {e}")
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
            self.update_counter()  # Update counter even on error
    
    def canvas_to_image_coords(self, canvas_x: float, canvas_y: float) -> Tuple[int, int]:
        """
        Convert canvas coordinates to image pixel coordinates.
        
        Args:
            canvas_x: X coordinate on canvas
            canvas_y: Y coordinate on canvas
            
        Returns:
            Tuple of (image_x, image_y) in image pixel coordinates, clamped to image bounds
        """
        if not self.original_image:
            return (0, 0)
        
        # Calculate actual scale (accounting for zoom level)
        actual_scale = self.current_scale * self.zoom_level
        
        # Get the image's position on the canvas (try bbox first, then fallback)
        image_x = self.image_origin_x
        image_y = self.image_origin_y
        
        if self._image_item:
            image_bbox = self.image_canvas.bbox(self._image_item)
            if image_bbox:
                image_x = image_bbox[0]
                image_y = image_bbox[1]
        
        # Convert canvas coordinates to image-relative coordinates
        rel_x = canvas_x - image_x
        rel_y = canvas_y - image_y
        
        # Convert to image coordinates (accounting for actual scale)
        img_x = int(rel_x / actual_scale)
        img_y = int(rel_y / actual_scale)
        
        # Clamp coordinates to image bounds
        image_width, image_height = self.original_image.size
        img_x = max(0, min(img_x, image_width - 1))
        img_y = max(0, min(img_y, image_height - 1))
        
        return (img_x, img_y)
    
    def navigate_images(self, direction: int):
        """Navigate to a different image."""
        if self.service.navigate(direction):
            self.show_current_image()
            self.ensure_cursor_bindings()
            # Counter is updated in show_current_image, but update explicitly to be sure
            self.update_counter()
    
    def update_cursor_line(self, event):
        """Update cursor line display."""
        if not self.special_cursor_active:
            return
        
        self.clear_cursor_lines()
        
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        x = self.image_canvas.canvasx(event.x)
        y = self.image_canvas.canvasy(event.y)
        
        if self.cursor_orientation == 'angled':
            import math
            line_length = (canvas_width**2 + canvas_height**2)**0.5 * 2
            angle_rad = math.radians(self.cursor_angle)
            dx = line_length * math.cos(angle_rad)
            dy = line_length * math.sin(angle_rad)
            x1 = x - dx/2
            y1 = y - dy/2
            x2 = x + dx/2
            y2 = y + dy/2
            self.cursor_line = self.image_canvas.create_line(x1, y1, x2, y2, fill="red", width=2)
        elif self.cursor_orientation == 'vertical':
            self.vertical_line = self.image_canvas.create_line(x, 0, x, canvas_height, fill="red", width=2)
        else:  # horizontal
            self.horizontal_line = self.image_canvas.create_line(0, y, canvas_width, y, fill="red", width=2)
    
    def update_cursor_line_display(self):
        """Update cursor line with current mouse position."""
        if self.special_cursor_active:
            x = self.image_canvas.winfo_pointerx() - self.image_canvas.winfo_rootx()
            y = self.image_canvas.winfo_pointery() - self.image_canvas.winfo_rooty()
            mock_event = type('MockEvent', (), {'x': x, 'y': y})
            self.update_cursor_line(mock_event)
    
    def clear_cursor_lines(self):
        """Clear all cursor lines."""
        if self.vertical_line:
            self.image_canvas.delete(self.vertical_line)
            self.vertical_line = None
        if self.horizontal_line:
            self.image_canvas.delete(self.horizontal_line)
            self.horizontal_line = None
        if self.cursor_line:
            self.image_canvas.delete(self.cursor_line)
            self.cursor_line = None
    
    def switch_to_vertical(self, event=None):
        """Switch to vertical cursor mode."""
        # Show warning if apply to all is enabled
        if self.apply_to_all.get():
            if not self.show_all_images_warning("Split"):
                return
        
        self.clear_all_modes()
        self.cursor_orientation = 'vertical'
        self.special_cursor_active = True
        self.image_canvas.config(cursor="none")
        self.image_canvas.bind("<Motion>", self.update_cursor_line)
        self.image_canvas.bind("<Button-1>", self.handle_mouse_click)
        self.update_cursor_line_display()
    
    def switch_to_horizontal(self, event=None):
        """Switch to horizontal cursor mode."""
        # Show warning if apply to all is enabled
        if self.apply_to_all.get():
            if not self.show_all_images_warning("Split"):
                return
        
        self.clear_all_modes()
        self.cursor_orientation = 'horizontal'
        self.special_cursor_active = True
        self.image_canvas.config(cursor="none")
        self.image_canvas.bind("<Motion>", self.update_cursor_line)
        self.image_canvas.bind("<Button-1>", self.handle_mouse_click)
        self.update_cursor_line_display()
    
    def rotate_cursor(self, direction: int):
        """Rotate cursor angle."""
        if not self.special_cursor_active:
            return
        
        if self.cursor_orientation in ['vertical', 'horizontal']:
            self.cursor_angle = 90 if self.cursor_orientation == 'vertical' else 0
            self.cursor_orientation = 'angled'
        
        self.cursor_angle = (self.cursor_angle + direction) % 360
        self.update_cursor_line_display()
    
    def toggle_auto_split(self):
        """Toggle auto-split mode."""
        self.settings.auto_split = not self.settings.auto_split
        # Note: auto_split_var was removed from menu, but setting is still saved
    
    def toggle_batch_process(self):
        """Toggle batch process mode."""
        new_value = not self.batch_process.get()
        self.batch_process.set(new_value)
        self.settings.batch_process = new_value
        
        # If turning batch processing ON, turn OFF apply to all
        if new_value and self.apply_to_all.get():
            self.apply_to_all.set(False)
            self.update_apply_to_all_icon()
            # Sync with menu
            if hasattr(self, 'apply_to_var'):
                self.apply_to_var.set("Current Image")
        
        self.update_batch_process_icon()
        # Sync with menu
        if hasattr(self, 'processing_mode_var'):
            self.processing_mode_var.set("Batch Processing" if self.batch_process.get() else "Sequential")
        # Update status display
        if self.batch_process.get():
            self.update_status_display("Batch Process: When enabled, automatically advances to next image after operations | Toggle on/off to control batch processing")
        else:
            self.update_status_display("Sequential Process: Operations complete on current image only | No automatic advancement | Toggle Batch Processing to enable auto-advance")
    
    def update_batch_process_icon(self):
        """Update the batch process button icon based on current state."""
        if self.batch_process.get():
            self.batch_process_button.config(image=self.batch_process_on_icon, relief=tk.SUNKEN)
        else:
            self.batch_process_button.config(image=self.batch_process_off_icon, relief=tk.RAISED)
    
    def toggle_apply_to_all(self):
        """Toggle apply to all images mode."""
        new_value = not self.apply_to_all.get()
        self.apply_to_all.set(new_value)
        
        # If turning apply to all ON, turn OFF batch processing
        if new_value and self.batch_process.get():
            self.batch_process.set(False)
            self.settings.batch_process = False
            self.update_batch_process_icon()
            # Sync with menu
            if hasattr(self, 'processing_mode_var'):
                self.processing_mode_var.set("Sequential")
        
        self.update_apply_to_all_icon()
        # Sync with menu
        if hasattr(self, 'apply_to_var'):
            self.apply_to_var.set("All Images" if self.apply_to_all.get() else "Current Image")
        # Update status display
        if self.apply_to_all.get():
            self.update_status_display("Apply to All: When enabled, operations apply to all images | When disabled, operations apply only to current image | Toggle on/off as needed")
        else:
            self.update_status_display("Apply to Current: Operations apply only to current image | Enable Apply to All to process all images at once | Toggle on/off as needed")
    
    def update_apply_to_all_icon(self):
        """Update the apply to all button icon based on current state."""
        if self.apply_to_all.get():
            self.apply_to_all_button.config(image=self.all_pages_icon, relief=tk.SUNKEN)
        else:
            self.apply_to_all_button.config(image=self.current_page_icon, relief=tk.RAISED)
    
    def show_all_images_warning(self, operation_name: str) -> bool:
        """
        Show warning dialog when applying operation to all images.
        
        Args:
            operation_name: Name of the operation being performed
            
        Returns:
            True if user wants to proceed, False if cancelled
        """
        if self.settings.suppress_all_images_warning:
            return True
        
        # Create warning dialog
        warning_window = tk.Toplevel(self)
        warning_window.title("Apply to All Images")
        warning_window.transient(self)
        warning_window.grab_set()
        
        # Prevent window resizing
        warning_window.resizable(False, False)
        
        result = {'proceed': False, 'suppress': False}
        
        # Main container frame with padding
        main_frame = tk.Frame(warning_window, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Warning message
        message = f"This will apply '{operation_name}' to all {len(self.service.state.images)} images.\n\nAre you sure you want to continue?"
        label = tk.Label(
            main_frame,
            text=message,
            wraplength=400,
            justify=tk.LEFT,
            font=("Arial", 10)
        )
        label.pack(anchor=tk.W, pady=(0, 15))
        
        # Suppress checkbox
        suppress_var = tk.BooleanVar(value=False)
        suppress_check = tk.Checkbutton(
            main_frame,
            text="Don't show these warnings in future",
            variable=suppress_var,
            font=("Arial", 9)
        )
        suppress_check.pack(anchor=tk.W, pady=(0, 20))
        
        # Buttons frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        def proceed():
            result['proceed'] = True
            result['suppress'] = suppress_var.get()
            warning_window.destroy()
        
        def cancel():
            result['proceed'] = False
            result['suppress'] = suppress_var.get()
            warning_window.destroy()
        
        # Buttons with proper sizing
        proceed_button = tk.Button(
            button_frame,
            text="Yes, Apply to All",
            command=proceed,
            width=18,
            height=2
        )
        proceed_button.pack(side=tk.RIGHT, padx=(10, 0))
        
        cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            command=cancel,
            width=18,
            height=2
        )
        cancel_button.pack(side=tk.RIGHT)
        
        # Update window to get proper size, then center
        warning_window.update_idletasks()
        width = warning_window.winfo_reqwidth()
        height = warning_window.winfo_reqheight()
        x = (warning_window.winfo_screenwidth() // 2) - (width // 2)
        y = (warning_window.winfo_screenheight() // 2) - (height // 2)
        warning_window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Make Enter key proceed and Escape cancel
        warning_window.bind('<Return>', lambda e: proceed())
        warning_window.bind('<Escape>', lambda e: cancel())
        proceed_button.focus_set()
        
        # Wait for dialog to close
        warning_window.wait_window()
        
        # Update settings if user wants to suppress
        if result['suppress']:
            self.settings.suppress_all_images_warning = True
            from ..config import save_settings
            try:
                save_settings(self.settings)
            except Exception as e:
                logger.warning(f"Failed to save settings: {e}")
        
        return result['proceed']
    
    def _clear_modes_preserve_button(self):
        """Clear all active modes but preserve button state."""
        # Reset status to default zoom/pan info when nothing is selected
        self.update_status_display("Ready | Zoom: CTRL + Mouse Wheel | Pan: Hold SPACEBAR + Drag Mouse")
        self.cropping = False
        self.crop_start = None
        self.crop_end = None
        if self.crop_rect:
            self.image_canvas.delete(self.crop_rect)
            self.crop_rect = None
        self.straightening_mode = False
        self.straighten_start = None
        if self.guide_line:
            self.image_canvas.delete(self.guide_line)
            self.guide_line = None
        self.clear_cursor_lines()
        self.image_canvas.config(cursor="")
        self.image_canvas.unbind("<ButtonPress-1>")
        self.image_canvas.unbind("<B1-Motion>")
        self.image_canvas.unbind("<ButtonRelease-1>")
        self.image_canvas.unbind("<Motion>")
        self.unbind("<Return>")
        self.unbind("<Escape>")
        self.special_cursor_active = False
        self.cursor_angle = 0
        self.cursor_orientation = 'vertical'
        self.bind("<Left>", lambda e: self.navigate_images(-1))
        self.bind("<Right>", lambda e: self.navigate_images(1))
        self.bind("<Control-h>", lambda e: self.switch_to_horizontal())
        self.bind("<Control-v>", lambda e: self.switch_to_vertical())
    
    def clear_all_modes(self):
        """Clear all active modes and reset to default state."""
        # Clear depressed button state
        self.clear_button_depressed()
        
        # Reset status to default zoom/pan info when nothing is selected
        self.update_status_display("Ready | Zoom: CTRL + Mouse Wheel | Pan: Hold SPACEBAR + Drag Mouse")
        self.cropping = False
        self.crop_start = None
        self.crop_end = None
        if self.crop_rect:
            self.image_canvas.delete(self.crop_rect)
            self.crop_rect = None
        self.clear_cursor_lines()
        self.image_canvas.config(cursor="")
        self.image_canvas.unbind("<ButtonPress-1>")
        self.image_canvas.unbind("<B1-Motion>")
        self.image_canvas.unbind("<ButtonRelease-1>")
        self.image_canvas.unbind("<Motion>")
        self.unbind("<Return>")
        self.unbind("<Escape>")
        self.special_cursor_active = False
        self.cursor_angle = 0
        self.cursor_orientation = 'vertical'
        self.bind("<Left>", lambda e: self.navigate_images(-1))
        self.bind("<Right>", lambda e: self.navigate_images(1))
        self.bind("<Control-h>", lambda e: self.switch_to_horizontal())
        self.bind("<Control-v>", lambda e: self.switch_to_vertical())
    
    def ensure_cursor_bindings(self):
        """Ensure cursor bindings are active if cursor is active."""
        if self.special_cursor_active:
            self.image_canvas.bind("<Motion>", self.update_cursor_line)
            self.image_canvas.bind("<Button-1>", self.handle_mouse_click)
    
    def handle_mouse_click(self, event):
        """Handle mouse click events."""
        if self.special_cursor_active and self.original_image:
            try:
                # Calculate actual scale (accounting for zoom level)
                actual_scale = self.current_scale * self.zoom_level
                
                # Get the image's position on the canvas
                image_x = None
                image_y = None
                if self._image_item:
                    image_bbox = self.image_canvas.bbox(self._image_item)
                    if image_bbox:
                        image_x = image_bbox[0]
                        image_y = image_bbox[1]
                
                # Fallback: calculate position manually if bbox not available
                if image_x is None or image_y is None:
                    canvas_width = self.image_canvas.winfo_width()
                    canvas_height = self.image_canvas.winfo_height()
                    image_width = int(self.original_image.size[0] * actual_scale)
                    image_height = int(self.original_image.size[1] * actual_scale)
                    image_x = self.pan_x
                    image_y = self.pan_y
                    if image_width < canvas_width:
                        image_x = (canvas_width - image_width) // 2 + self.pan_x
                    if image_height < canvas_height:
                        image_y = (canvas_height - image_height) // 2 + self.pan_y
                
                # Split image based on cursor position
                if self.cursor_orientation == 'vertical' and self.vertical_line:
                    coords = self.image_canvas.coords(self.vertical_line)
                    if coords:
                        # Convert canvas coordinate to image-relative, then to image coordinate
                        canvas_x = coords[0]
                        rel_x = canvas_x - image_x
                        split_x = int(rel_x / actual_scale)
                        # Clamp to image bounds
                        image_width, image_height = self.original_image.size
                        split_x = max(0, min(split_x, image_width))
                        self.split_image('vertical', split_coord=split_x)
                elif self.cursor_orientation == 'horizontal' and self.horizontal_line:
                    coords = self.image_canvas.coords(self.horizontal_line)
                    if coords:
                        # Convert canvas coordinate to image-relative, then to image coordinate
                        canvas_y = coords[1]
                        rel_y = canvas_y - image_y
                        split_y = int(rel_y / actual_scale)
                        # Clamp to image bounds
                        image_width, image_height = self.original_image.size
                        split_y = max(0, min(split_y, image_height))
                        self.split_image('horizontal', split_coord=split_y)
                elif self.cursor_orientation == 'angled' and self.cursor_line:
                    coords = self.image_canvas.coords(self.cursor_line)
                    if coords:
                        # Convert canvas coordinates to image-relative, then to image coordinates
                        canvas_x1, canvas_y1, canvas_x2, canvas_y2 = coords
                        rel_x1 = canvas_x1 - image_x
                        rel_y1 = canvas_y1 - image_y
                        rel_x2 = canvas_x2 - image_x
                        rel_y2 = canvas_y2 - image_y
                        x1 = rel_x1 / actual_scale
                        y1 = rel_y1 / actual_scale
                        x2 = rel_x2 / actual_scale
                        y2 = rel_y2 / actual_scale
                        # Don't clamp here - let split function find proper intersections with image boundaries
                        # This preserves the exact angle of the cursor line
                        self.split_image('angled', line_coords=(x1, y1, x2, y2), angle=self.cursor_angle)
                
                self.clear_cursor_lines()
                
                if self.batch_process.get():
                    self.after(100, lambda: self.navigate_images(1))
                    self.after(200, lambda: self.navigate_images(1))
            except Exception as e:
                logger.error(f"Error in handle_mouse_click: {e}")
                messagebox.showerror("Error", f"Error splitting image: {str(e)}")
    
    def handle_mouse_release(self, event):
        """Handle mouse release events."""
        if self.cropping:
            self.crop_end = (self.image_canvas.canvasx(event.x), self.image_canvas.canvasy(event.y))
            if self.batch_process.get():
                self.apply_crop()
        elif self.panning:
            self.panning = False
    
    def handle_mouse_drag(self, event):
        """Handle mouse drag events."""
        if self.panning:
            # Calculate pan delta (mouse movement direction = image movement direction)
            # Mouse right → image right (positive dx)
            # Mouse down → image down
            dx = event.x - self.pan_start_x
            dy_raw = event.y - self.pan_start_y  # Raw mouse movement
            
            # Update pan offset (pan_y increases = image moves down in canvas)
            self.pan_x += dx
            self.pan_y += dy_raw  # pan_y tracks actual position
            
            # Update start position for next drag
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            
            # Smooth pan: move the image item (canvas.move uses screen coords)
            if self._image_item:
                self.image_canvas.move(self._image_item, dx, dy_raw)
            else:
                # Fallback to full redraw if item not found
                self.show_current_image()
    
    def on_mousewheel(self, event):
        """Handle mouse wheel zoom with CTRL modifier."""
        # Check if CTRL is pressed (0x4 = Control key)
        # On some systems, we need to check event.state, on others we check the binding
        ctrl_pressed = (event.state & 0x4) or (hasattr(event, 'state') and event.state & 0x0004)
        
        # For Linux/Mac, button numbers indicate scroll direction
        if event.num == 4:  # Scroll up
            delta = 1
        elif event.num == 5:  # Scroll down
            delta = -1
        else:
            # Windows mousewheel
            delta = event.delta
        
        # Only zoom if CTRL is pressed
        if not ctrl_pressed:
            return
        
        if not self.original_image:
            return
        
        # Determine zoom direction
        if delta > 0:
            zoom_factor = 1.1  # Zoom in
        else:
            zoom_factor = 0.9  # Zoom out
        
        # Calculate zoom around mouse position
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        # Get mouse position relative to canvas
        mouse_x = event.x
        mouse_y = event.y
        
        # Get current image position
        image_width = int(self.original_image.size[0] * self.current_scale * self.zoom_level)
        image_height = int(self.original_image.size[1] * self.current_scale * self.zoom_level)
        
        # Calculate image position (accounting for centering)
        if image_width < canvas_width:
            image_x = (canvas_width - image_width) // 2 + self.pan_x
        else:
            image_x = self.pan_x
        
        if image_height < canvas_height:
            image_y = (canvas_height - image_height) // 2 + self.pan_y
        else:
            image_y = self.pan_y
        
        # Calculate mouse position relative to image
        mouse_rel_x = mouse_x - image_x
        mouse_rel_y = mouse_y - image_y
        
        # Calculate mouse position in original image coordinates
        if image_width > 0 and image_height > 0:
            mouse_image_x = mouse_rel_x / (self.current_scale * self.zoom_level)
            mouse_image_y = mouse_rel_y / (self.current_scale * self.zoom_level)
        else:
            mouse_image_x = 0
            mouse_image_y = 0
        
        # Update zoom level
        old_zoom = self.zoom_level
        self.zoom_level *= zoom_factor
        
        # Limit zoom range
        self.zoom_level = max(0.1, min(10.0, self.zoom_level))
        
        # If zoom changed significantly, adjust pan to keep mouse position stable
        if abs(old_zoom - self.zoom_level) > 0.001:
            # Calculate new image dimensions
            new_image_width = int(self.original_image.size[0] * self.current_scale * self.zoom_level)
            new_image_height = int(self.original_image.size[1] * self.current_scale * self.zoom_level)
            
            # Calculate new image position with centering
            if new_image_width < canvas_width:
                new_image_x = (canvas_width - new_image_width) // 2
            else:
                new_image_x = 0
            
            if new_image_height < canvas_height:
                new_image_y = (canvas_height - new_image_height) // 2
            else:
                new_image_y = 0
            
            # Calculate new mouse position in image coordinates
            new_mouse_image_x = mouse_image_x * self.current_scale * self.zoom_level
            new_mouse_image_y = mouse_image_y * self.current_scale * self.zoom_level
            
            # Adjust pan to keep mouse position stable
            self.pan_x = mouse_x - new_mouse_image_x - new_image_x
            self.pan_y = mouse_y - new_mouse_image_y - new_image_y
        
        # Update display for zoom - regenerate image if needed
        # Only regenerate if zoom changed significantly (1% threshold), otherwise just update position
        if abs(self._cached_zoom - self.zoom_level) > 0.01 or self._cached_photo is None:
            # Need to regenerate image at new zoom level
            self.show_current_image()
        else:
            # Just update position for smooth incremental zoom
            self._update_image_position()
    
    def _update_image_position(self):
        """Update image position without redrawing."""
        if not self._image_item or not self.original_image:
            return
        
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        image_width = int(self.original_image.size[0] * self.current_scale * self.zoom_level)
        image_height = int(self.original_image.size[1] * self.current_scale * self.zoom_level)
        
        # Calculate position with pan offset
        x_pos = self.pan_x
        y_pos = self.pan_y
        
        # Center image if it's smaller than canvas
        if image_width < canvas_width:
            x_pos = (canvas_width - image_width) // 2 + self.pan_x
        if image_height < canvas_height:
            y_pos = (canvas_height - image_height) // 2 + self.pan_y
        
        # Get current position and update smoothly
        current_pos = self.image_canvas.coords(self._image_item)
        if current_pos:
            # Move to new position
            self.image_canvas.coords(self._image_item, x_pos, y_pos)
    
    def start_pan(self, event):
        """Start panning mode when spacebar is pressed."""
        if not self.original_image:
            return
        self.panning = True
        self.image_canvas.config(cursor="hand2")
        # Get initial mouse position relative to canvas
        try:
            self.pan_start_x = self.image_canvas.winfo_pointerx() - self.image_canvas.winfo_rootx()
            self.pan_start_y = self.image_canvas.winfo_pointery() - self.image_canvas.winfo_rooty()
        except:
            # Fallback if pointer position unavailable
            self.pan_start_x = 0
            self.pan_start_y = 0
    
    def stop_pan(self, event):
        """Stop panning mode when spacebar is released."""
        self.panning = False
        # Restore cursor based on current mode
        if self.special_cursor_active:
            self.image_canvas.config(cursor="none")
        elif self.cropping:
            self.image_canvas.config(cursor="crosshair")
        else:
            self.image_canvas.config(cursor="")
    
    def split_image(self, orientation: str, split_coord: int = None, line_coords: tuple = None, angle: float = None):
        """Split current image or all images based on toggle."""
        if self.apply_to_all.get():
            # Show warning if not suppressed
            if not self.show_all_images_warning("Split"):
                return
            # Use batch split API
            try:
                initial_index = self.service.state.current_image_index
                self.service.split_all(orientation, split_coord, line_coords, angle)
                
                # Find the left split of the originally selected image
                restored_index = None
                for i, img in enumerate(self.service.state.images):
                    if img.left_or_right == 'Left':
                        if restored_index is None:
                            restored_index = i
                        # Try to find one that matches the original index range
                        if initial_index <= i <= initial_index + 1:
                            restored_index = i
                            break
                
                if restored_index is None:
                    restored_index = 0
                
                restored_index = min(restored_index, len(self.service.state.images) - 1)
                self.service.state.current_image_index = restored_index
                self.show_current_image()
                self.update_counter()
            except UserFacingError as e:
                messagebox.showerror("Error", str(e))
        else:
            # Split current image only
            try:
                self.service.split_current(orientation, split_coord, line_coords, angle)
                self.show_current_image()
                # Counter updated in show_current_image, but update explicitly since split adds images
                self.update_counter()
            except UserFacingError as e:
                messagebox.showerror("Error", str(e))
    
    def auto_crop_current(self):
        """Auto-crop current image or all images with threshold adjustment."""
        current = self.service.get_current_image()
        if not current:
            return
        
        image_path = current.current_image_path
        
        # Check if apply to all is enabled
        if self.apply_to_all.get():
            # Show warning if not suppressed
            if not self.show_all_images_warning("Auto Crop"):
                return
            # Use auto_crop_all function
            self.auto_crop_all()
            return
        
        # Track whether apply was called to avoid clearing button when dialog closes after apply
        apply_called = {'value': False}
        
        def apply_crop(threshold, margin):
            apply_called['value'] = True
            try:
                # Update settings first
                self.service.settings.threshold = threshold
                self.service.settings.margin = margin
                # Use service method which has history tracking
                self.service.auto_crop_current()
                self.show_current_image()
                # Clear button state after operation completes
                if not self.batch_process.get():
                    self.clear_button_depressed()
                # Advance to next image if batch processing is enabled
                if self.batch_process.get():
                    # Check if we're not at the last image before navigating
                    current_index = self.service.state.current_image_index
                    total_images = len(self.service.state.images)
                    if current_index < total_images - 1:
                        # Navigate to next image and continue processing
                        self.after(100, lambda: self.navigate_images(1))
                        self.after(200, self.auto_crop_current)
                    else:
                        # Last image processed, clear button state
                        self.clear_button_depressed()
            except UserFacingError as e:
                messagebox.showerror("Error", str(e))
                # Clear button state on error
                self.clear_button_depressed()
            except Exception as e:
                messagebox.showerror("Error", f"Error during auto-crop: {str(e)}")
                # Clear button state on error
                self.clear_button_depressed()
        
        dialog = ThresholdAdjuster(self, image_path, self.service, apply_crop)
        # Handle dialog close (cancel) - clear button state if dialog is closed without applying
        original_destroy = dialog.destroy
        def on_dialog_close():
            if not apply_called['value'] and self.active_tool_button == self.autocrop_button:
                self.clear_button_depressed()
            original_destroy()
        dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)
    
    def auto_crop_all(self):
        """Auto-crop all images with threshold adjustment."""
        current = self.service.get_current_image()
        if not current:
            return
        
        image_path = current.current_image_path
        
        def start_crop_all(threshold, margin):
            """Start auto-crop all with specified settings."""
            # Update settings
            self.service.settings.threshold = threshold
            self.service.settings.margin = margin
            
            # Show progress window
            progress_window = tk.Toplevel(self)
            progress_window.title("Auto-cropping Progress")
            progress_window.geometry("300x150")
            progress_window.transient(self)
            
            progress_label = ttk.Label(progress_window, text="Processing images...", padding=10)
            progress_label.pack()
            
            progress_bar = ttk.Progressbar(progress_window, length=200, mode='determinate')
            progress_bar.pack(pady=20)
            
            total = len(self.service.state.images)
            progress_bar['maximum'] = total
            
            def update_progress(current, total, message):
                progress_bar['value'] = current
                progress_label.config(text=message)
                progress_window.update()
            
            self.service.set_progress_callback(update_progress)
            
            def process():
                try:
                    self.service.auto_crop_all()
                    progress_label.config(text="Auto-cropping completed!")
                    self.after(1000, lambda: (progress_window.destroy(), self.show_current_image()))
                except UserFacingError as e:
                    messagebox.showerror("Error", str(e))
                    progress_window.destroy()
            
            import threading
            threading.Thread(target=process, daemon=True).start()
        
        ThresholdAdjuster(self, image_path, self.service, start_crop_all)
    
    def auto_straighten_current(self):
        """Auto-straighten current image or all images with threshold adjustment."""
        current = self.service.get_current_image()
        if not current:
            return
        
        # Clear any leftover guide lines from manual straighten tool
        self.clear_all_modes()
        
        image_path = current.current_image_path
        
        # Check if apply to all is enabled
        if self.apply_to_all.get():
            # Show warning if not suppressed
            if not self.show_all_images_warning("Auto Straighten"):
                return
            # Use auto_straighten_all function
            self.auto_straighten_all()
            return
        
        def apply_straighten(threshold):
            try:
                # Update settings first
                self.service.settings.threshold = threshold
                # Use service method which has history tracking
                self.service.auto_straighten_current()
                self.show_current_image()
                # Advance to next image if batch processing is enabled
                if self.batch_process.get():
                    # Check if we're not at the last image before navigating
                    current_index = self.service.state.current_image_index
                    total_images = len(self.service.state.images)
                    if current_index < total_images - 1:
                        # Navigate to next image and continue processing
                        self.after(100, lambda: self.navigate_images(1))
                        self.after(200, self.auto_straighten_current)
            except UserFacingError as e:
                messagebox.showerror("Error", str(e))
            except Exception as e:
                messagebox.showerror("Error", f"Error during auto-straighten: {str(e)}")
        
        ThresholdAdjuster(self, image_path, self.service, apply_straighten, mode='straighten')
    
    def auto_straighten_all(self):
        """Auto-straighten all images with threshold adjustment."""
        current = self.service.get_current_image()
        if not current:
            return
        
        image_path = current.current_image_path
        
        def start_straighten_all(threshold):
            """Start auto-straighten all with specified settings."""
            # Update settings
            self.service.settings.threshold = threshold
            
            # Show progress window
            progress_window = tk.Toplevel(self)
            progress_window.title("Auto-straightening Progress")
            progress_window.geometry("300x150")
            progress_window.transient(self)
            
            progress_label = ttk.Label(progress_window, text="Processing images...", padding=10)
            progress_label.pack()
            
            progress_bar = ttk.Progressbar(progress_window, length=200, mode='determinate')
            progress_bar.pack(pady=20)
            
            total = len(self.service.state.images)
            progress_bar['maximum'] = total
            
            def update_progress(current, total, message):
                progress_bar['value'] = current
                progress_label.config(text=message)
                progress_window.update()
            
            self.service.set_progress_callback(update_progress)
            
            def process():
                try:
                    self.service.auto_straighten_all()
                    progress_label.config(text="Auto-straightening completed!")
                    self.after(1000, lambda: (progress_window.destroy(), self.show_current_image()))
                except UserFacingError as e:
                    messagebox.showerror("Error", str(e))
                    progress_window.destroy()
            
            import threading
            threading.Thread(target=process, daemon=True).start()
        
        ThresholdAdjuster(self, image_path, self.service, start_straighten_all, mode='straighten')
    
    def auto_split_current_finder(self):
        """Auto-split current image using seam finder dialog."""
        current = self.service.get_current_image()
        if not current:
            return
        
        # Check if apply to all is enabled
        if self.apply_to_all.get():
            # Show warning if not suppressed
            if not self.show_all_images_warning("Auto Split"):
                return
            # Use auto_split_all_finder function
            self.auto_split_all_finder()
            return
        
        image_path = current.current_image_path
        
        def apply_split(line_coords, sensitivity):
            try:
                # Settings are already updated by SeamFinderDialog (both ROI threshold and sensitivity)
                # Use service method which has history tracking
                # Apply 1% inner margin for auto-split
                self.service.split_current('angled', line_coords=line_coords, inner_margin_ratio=0.01)
                self.show_current_image()
                # Advance to next image if batch processing is enabled
                if self.batch_process.get():
                    # Check if we're not at the last image before navigating
                    current_index = self.service.state.current_image_index
                    total_images = len(self.service.state.images)
                    if current_index < total_images - 1:
                        # Navigate to next image and continue processing
                        self.after(100, lambda: self.navigate_images(1))
                        self.after(200, self.auto_split_current_finder)
            except UserFacingError as e:
                messagebox.showerror("Error", str(e))
            except Exception as e:
                messagebox.showerror("Error", f"Error during auto-split: {str(e)}")
        
        def skip_split():
            # Just skip, no action needed
            pass
        
        SeamFinderDialog(self, image_path, self.service, apply_split, skip_split)
    
    def auto_split_selection_dialog(self):
        """Open dialog to select images for auto-split with filmstrip interface."""
        if not self.service.state.images:
            messagebox.showwarning("No Images", "No images to split.")
            return
        
        # Filter out already-split images
        unsplit_images = [(i, record) for i, record in enumerate(self.service.state.images) 
                         if record.left_or_right is None]
        
        if not unsplit_images:
            messagebox.showinfo("No Images", "All images have already been split.")
            return
        
        # Create dialog window
        dialog = tk.Toplevel(self)
        dialog.title("Auto Split Images")
        dialog.transient(self)
        dialog.grab_set()
        
        # Variables
        selected_indices = set()
        
        # Thumbnail size - smaller for filmstrip
        thumb_width = 100
        thumb_height = 120
        thumb_padding = 3
        
        # Main container with tighter padding
        main_frame = tk.Frame(dialog, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Filmstrip section
        filmstrip_container = tk.LabelFrame(main_frame, text="Image Selection", relief=tk.GROOVE, bg='#f0f0f0')
        filmstrip_container.pack(fill=tk.X, pady=(0, 10))
        
        # Canvas for thumbnails with proper height
        filmstrip_canvas = tk.Canvas(
            filmstrip_container, 
            bg="white", 
            highlightthickness=1, 
            highlightbackground="#cccccc",
            height=thumb_height + 30
        )
        filmstrip_canvas.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(5, 0))
        
        # Horizontal scrollbar positioned below canvas
        scrollbar = ttk.Scrollbar(filmstrip_container, orient=tk.HORIZONTAL, command=filmstrip_canvas.xview)
        scrollbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))
        filmstrip_canvas.configure(xscrollcommand=scrollbar.set)
        
        # Inner frame for thumbnails
        thumbnails_frame = tk.Frame(filmstrip_canvas, bg="white")
        filmstrip_canvas_window = filmstrip_canvas.create_window((0, 0), window=thumbnails_frame, anchor=tk.NW)
        
        def configure_scroll_region(event=None):
            filmstrip_canvas.update_idletasks()
            dialog.update_idletasks()
            
            thumbnails_width = thumbnails_frame.winfo_reqwidth()
            thumbnails_height = thumbnails_frame.winfo_reqheight()
            
            if thumbnails_width <= 1 and len(thumbnail_widgets) > 0:
                last_container = thumbnails_frame.winfo_children()[-1] if thumbnails_frame.winfo_children() else None
                if last_container:
                    thumbnails_width = last_container.winfo_x() + last_container.winfo_reqwidth() + thumb_padding
            
            canvas_width = filmstrip_canvas.winfo_width()
            canvas_height = filmstrip_canvas.winfo_height()
            
            if thumbnails_width <= 1:
                thumbnails_width = len(thumbnail_widgets) * (thumb_width + thumb_padding * 2 + 10)
            
            scroll_width = max(thumbnails_width, canvas_width, 1)
            scroll_height = max(thumbnails_height, canvas_height, 1)
            
            filmstrip_canvas.configure(scrollregion=(0, 0, scroll_width, scroll_height))
            
            if thumbnails_width > 1:
                filmstrip_canvas.itemconfig(filmstrip_canvas_window, width=thumbnails_width)
            
            filmstrip_canvas.update_idletasks()
        
        def on_canvas_configure(event):
            configure_scroll_region()
        
        thumbnails_frame.bind('<Configure>', configure_scroll_region)
        filmstrip_canvas.bind('<Configure>', on_canvas_configure)
        
        # Add mouse wheel support for scrolling
        def on_mousewheel(event):
            if event.delta:
                filmstrip_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                if event.num == 4:
                    filmstrip_canvas.xview_scroll(-1, "units")
                elif event.num == 5:
                    filmstrip_canvas.xview_scroll(1, "units")
        
        filmstrip_canvas.bind("<MouseWheel>", on_mousewheel)
        filmstrip_canvas.bind("<Button-4>", on_mousewheel)
        filmstrip_canvas.bind("<Button-5>", on_mousewheel)
        
        # Store thumbnail references
        thumbnail_widgets = []
        thumbnail_photos = []
        
        # Create count label early so it can be referenced
        count_label = None  # Will be created later, but declare here for scope
        
        def update_count_label():
            """Update the count label."""
            nonlocal count_label
            if count_label:
                count = len(selected_indices)
                count_label.config(text=f"{count} image(s) selected for splitting")
        
        def update_selection_display():
            """Update visual selection display."""
            for i, (orig_idx, record) in enumerate(unsplit_images):
                if orig_idx in selected_indices:
                    thumbnail_widgets[i].config(bg="#4a90e2", relief=tk.SUNKEN, borderwidth=3)
                    for child in thumbnail_widgets[i].winfo_children():
                        if isinstance(child, tk.Label) and hasattr(child, 'cget'):
                            try:
                                if child.cget('text').startswith('Page'):
                                    child.config(fg="white", bg="#4a90e2")
                            except:
                                pass
                    thumbnail_widgets[i].selected = True
                else:
                    thumbnail_widgets[i].config(bg="white", relief=tk.RAISED, borderwidth=1)
                    for child in thumbnail_widgets[i].winfo_children():
                        if isinstance(child, tk.Label) and hasattr(child, 'cget'):
                            try:
                                if child.cget('text').startswith('Page'):
                                    child.config(fg="black", bg="white")
                            except:
                                pass
                    thumbnail_widgets[i].selected = False
            update_count_label()
        
        def load_thumbnails():
            """Load all thumbnails for unsplit images."""
            nonlocal thumbnail_widgets, thumbnail_photos
            
            # Clear existing
            for widget in thumbnail_widgets:
                widget.destroy()
            thumbnail_widgets.clear()
            thumbnail_photos.clear()
            
            # Load thumbnails with proper spacing
            for i, (orig_idx, record) in enumerate(unsplit_images):
                # Container frame for each thumbnail
                container = tk.Frame(thumbnails_frame, bg="white", padx=2, pady=5)
                container.grid(row=0, column=i, padx=thumb_padding, sticky=tk.N)
                
                # Thumbnail frame with border
                frame = tk.Frame(container, relief=tk.RAISED, borderwidth=1, bg="white")
                frame.pack()
                
                # Load and resize image
                try:
                    image_path = record.current_image_path
                    if not image_path.exists():
                        raise FileNotFoundError(f"Image not found: {image_path}")
                    
                    with Image.open(image_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        max_thumb_width = thumb_width
                        max_thumb_height = thumb_height - 25
                        
                        img_ratio = img.width / img.height
                        if img_ratio > max_thumb_width / max_thumb_height:
                            new_width = max_thumb_width
                            new_height = int(max_thumb_width / img_ratio)
                        else:
                            new_height = max_thumb_height
                            new_width = int(max_thumb_height * img_ratio)
                        
                        img_thumb = img.resize((new_width, new_height), Image.LANCZOS)
                        
                        padded_thumb = Image.new('RGB', (max_thumb_width, max_thumb_height), 'white')
                        x_offset = (max_thumb_width - new_width) // 2
                        y_offset = (max_thumb_height - new_height) // 2
                        padded_thumb.paste(img_thumb, (x_offset, y_offset))
                        
                        photo = ImageTk.PhotoImage(padded_thumb)
                        thumbnail_photos.append(photo)
                        
                        label = tk.Label(frame, image=photo, bg="white", borderwidth=0)
                        label.image = photo
                        label.pack(pady=2)
                        
                        page_label = tk.Label(
                            frame, 
                            text=f"Page {orig_idx+1}", 
                            bg="white", 
                            font=("Arial", 8, "bold"),
                            pady=2
                        )
                        page_label.pack()
                        
                        frame.image_index = orig_idx
                        frame.selected = False
                        thumbnail_widgets.append(frame)
                        
                        # Bind click events
                        def make_click_handler(idx):
                            def handler(event):
                                if event.state & 0x0004:  # CTRL
                                    if idx in selected_indices:
                                        selected_indices.discard(idx)
                                    else:
                                        selected_indices.add(idx)
                                elif event.state & 0x0001:  # SHIFT
                                    if selected_indices:
                                        start_idx = min(selected_indices)
                                        end_idx = max(selected_indices)
                                        if idx < start_idx:
                                            selected_indices.update(range(idx, start_idx + 1))
                                        elif idx > end_idx:
                                            selected_indices.update(range(end_idx, idx + 1))
                                    else:
                                        selected_indices.add(idx)
                                else:
                                    selected_indices.clear()
                                    selected_indices.add(idx)
                                update_selection_display()
                            return handler
                        
                        frame.bind("<Button-1>", make_click_handler(orig_idx))
                        label.bind("<Button-1>", make_click_handler(orig_idx))
                        page_label.bind("<Button-1>", make_click_handler(orig_idx))
                        container.bind("<Button-1>", make_click_handler(orig_idx))
                        
                except Exception as e:
                    logger.error(f"Error loading thumbnail {orig_idx}: {e}")
                    placeholder = tk.Label(
                        frame, 
                        text=f"Page {orig_idx+1}\n\n⚠\nError", 
                        bg="#e0e0e0", 
                        width=thumb_width//10, 
                        height=thumb_height//20,
                        font=("Arial", 9)
                    )
                    placeholder.pack(fill=tk.BOTH, expand=True)
                    frame.image_index = orig_idx
                    frame.selected = False
                    thumbnail_widgets.append(frame)
        
        # Selection info label
        info_frame = tk.Frame(main_frame, bg='#f0f0f0')
        info_frame.pack(fill=tk.X, pady=(5, 5))
        
        selection_label = tk.Label(
            info_frame,
            text="Tip: Use Ctrl+Click to toggle selection, Shift+Click to select range",
            bg='#f0f0f0',
            font=("Arial", 9),
            fg="#666666"
        )
        selection_label.pack(side=tk.LEFT)
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=(10, 5))
        
        # Create count label before initial load
        count_label = tk.Label(
            button_frame,
            text="0 image(s) selected for splitting",
            bg='#f0f0f0',
            font=("Arial", 9),
            fg="#333333"
        )
        count_label.pack(side=tk.LEFT)
        
        # Initial load
        load_thumbnails()
        update_selection_display()
        
        dialog.update_idletasks()
        dialog.update()
        
        def delayed_scroll_update():
            configure_scroll_region()
            dialog.after(50, configure_scroll_region)
        
        dialog.after(100, delayed_scroll_update)
        
        def split_all():
            """Split all unsplit images."""
            indices = [orig_idx for orig_idx, _ in unsplit_images]
            if not indices:
                messagebox.showwarning("No Images", "No images to split.", parent=dialog)
                return
            
            dialog.destroy()
            self._perform_auto_split(indices)
        
        def split_selected():
            """Split selected images."""
            if not selected_indices:
                messagebox.showwarning("No Selection", "No images selected for splitting.", parent=dialog)
                return
            
            indices = list(selected_indices)
            dialog.destroy()
            self._perform_auto_split(indices)
        
        def cancel():
            dialog.destroy()
        
        # Buttons
        split_all_btn = tk.Button(
            button_frame, 
            text="Split All", 
            command=split_all,
            bg="#5cb85c",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2"
        )
        split_all_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        split_selected_btn = tk.Button(
            button_frame, 
            text="Split Selected", 
            command=split_selected,
            bg="#5cb85c",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2"
        )
        split_selected_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            command=cancel,
            bg="#6c757d",
            fg="white",
            font=("Arial", 10),
            padx=15,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2"
        )
        cancel_btn.pack(side=tk.RIGHT)
        
        # Set initial size and position - ensure buttons are visible
        # Wait for all content to load first
        dialog.update_idletasks()
        dialog.update()
        
        # Calculate required height after all content is loaded
        required_height = (
            filmstrip_container.winfo_reqheight() +
            info_frame.winfo_reqheight() +
            button_frame.winfo_reqheight() +
            60  # padding for margins
        )
        
        # Ensure minimum height
        final_height = max(520, required_height)
        
        # Set geometry
        dialog.geometry(f"900x{final_height}")
        dialog.update_idletasks()
        
        # Center dialog
        x = (dialog.winfo_screenwidth() // 2) - 450
        y = (dialog.winfo_screenheight() // 2) - (final_height // 2)
        dialog.geometry(f"900x{final_height}+{x}+{y}")
        
        # Make dialog resizable
        dialog.resizable(True, True)
        
        # Set minimum size to ensure buttons are always visible
        dialog.minsize(700, 520)
    
    def _perform_auto_split(self, indices):
        """Perform auto-split on the specified image indices."""
        if not indices:
            return
        
        # Show initial progress window
        progress_window = tk.Toplevel(self)
        progress_window.title("Auto-splitting Progress")
        progress_window.geometry("300x150")
        progress_window.transient(self)
        
        progress_label = ttk.Label(progress_window, text="Processing images...", padding=10)
        progress_label.pack()
        
        progress_bar = ttk.Progressbar(progress_window, length=200, mode='determinate')
        progress_bar.pack(pady=20)
        
        total = len(indices)
        progress_bar['maximum'] = total
        
        def update_progress(current_num, total_num, message):
            progress_bar['value'] = current_num
            progress_label.config(text=message)
            progress_window.update()
        
        self.service.set_progress_callback(update_progress)
        
        def process_all():
            import cv2
            from ..image_ops import detect_vertical_trench_near_center_parallel
            
            try:
                processed_count = 0
                skipped_count = 0
                
                # Track images by original_image path (stable identifier) instead of indices
                # This avoids index invalidation when splits insert new records
                target_originals = {}
                for i in indices:
                    if i < len(self.service.state.images):
                        record = self.service.state.images[i]
                        if record.left_or_right is None:  # Only track unsplit images
                            target_originals[record.original_image] = record.current_image_path
                
                # Process each target image by finding its current unsplit record
                for idx_num, (original_path, initial_image_path) in enumerate(target_originals.items()):
                    # Re-find the current unsplit record for this original image
                    # (indices may have shifted due to previous splits)
                    current_idx = None
                    for j, rec in enumerate(self.service.state.images):
                        if rec.original_image == original_path and rec.left_or_right is None:
                            current_idx = j
                            break
                    
                    if current_idx is None:
                        # Image was already split or removed
                        logger.info(f"Skipping already-split image: {initial_image_path.name}")
                        skipped_count += 1
                        continue
                    
                    self.service.state.current_image_index = current_idx
                    record = self.service.state.images[current_idx]
                    image_path = record.current_image_path
                    
                    update_progress(idx_num + 1, len(target_originals), f"Processing image {idx_num + 1} of {len(target_originals)}")
                    
                    # Load image
                    image = cv2.imread(str(image_path))
                    if image is None:
                        logger.warning(f"Could not load image: {image_path}")
                        skipped_count += 1
                        continue
                    
                    # Detect seam: try multiple ROI thresholds to find best for this specific image
                    settings = self.service.settings
                    base_threshold = getattr(settings, 'seam_roi_threshold', 170)
                    
                    # Try a range of thresholds around the base to find best for this image
                    threshold_candidates = [
                        base_threshold - 20,
                        base_threshold - 10,
                        base_threshold,
                        base_threshold + 10,
                        base_threshold + 20
                    ]
                    threshold_candidates = [max(0, min(255, t)) for t in threshold_candidates]
                    threshold_candidates = sorted(set(threshold_candidates))
                    
                    best_line_coords = None
                    best_confidence = -1.0
                    best_threshold = base_threshold
                    
                    # Evaluate each threshold and keep the best result
                    for test_threshold in threshold_candidates:
                        line_coords, confidence = detect_vertical_trench_near_center_parallel(
                            image,
                            test_threshold,
                            center_band_ratio=0.05,
                            num_workers=20
                        )
                        
                        if line_coords and confidence > best_confidence:
                            best_line_coords = line_coords
                            best_confidence = confidence
                            best_threshold = test_threshold
                    
                    # Use the best result found
                    line_coords = best_line_coords
                    confidence = best_confidence
                    
                    # Apply automatically if any line was detected (use highest confidence found)
                    if line_coords:
                        try:
                            # Pass inner_margin_ratio to trim 1% from inner edges of both pages
                            self.service.split_current('angled', line_coords=line_coords, inner_margin_ratio=0.01)
                            processed_count += 1
                            logger.info(f"Auto-split image ({image_path.name}) with confidence {confidence:.2f} using threshold {best_threshold}")
                        except UserFacingError as e:
                            logger.error(f"Error splitting image: {e}")
                            skipped_count += 1
                    else:
                        # No line detected - skip this image
                        logger.warning(f"No seam detected for image: {image_path.name}")
                        skipped_count += 1
                
                # Close progress window
                progress_window.destroy()
                
                # Show completion message
                message = f"Auto-splitting completed!\n\nProcessed: {processed_count}\nSkipped: {skipped_count}"
                messagebox.showinfo("Auto-split Complete", message)
                self.show_current_image()
                
            except Exception as e:
                logger.error(f"Error in auto_split: {e}")
                messagebox.showerror("Error", f"An error occurred during auto-split: {str(e)}")
                progress_window.destroy()
        
        import threading
        threading.Thread(target=process_all, daemon=True).start()
    
    def auto_split_all_finder(self):
        """Auto-split all images automatically using highest confidence line, no preview."""
        # Open selection dialog instead of direct processing
        self.auto_split_selection_dialog()
    
    def activate_crop_tool(self, event=None):
        """Activate crop tool."""
        # Show warning if apply to all is enabled
        if self.apply_to_all.get():
            if not self.show_all_images_warning("Crop"):
                return
        
        # Set button to depressed if not already set (e.g., when called from keyboard shortcut)
        if self.active_tool_button != self.crop_button:
            self.set_button_depressed(self.crop_button, self.crop_on_icon)
        
        # Clear modes but preserve button state for crop tool
        self._clear_modes_preserve_button()
        self.cropping = True
        self.image_canvas.config(cursor="crosshair")
        self.image_canvas.bind("<ButtonPress-1>", self.start_crop)
        self.image_canvas.bind("<B1-Motion>", self.draw_crop)
        self.image_canvas.bind("<ButtonRelease-1>", self.handle_mouse_release)
        self.bind("<Return>", lambda e: self.apply_crop())
        self.bind("<Escape>", lambda e: self.cancel_crop())
    
    def start_crop(self, event):
        """Start crop selection."""
        self.crop_start = (self.image_canvas.canvasx(event.x), self.image_canvas.canvasy(event.y))
        if self.crop_rect:
            self.image_canvas.delete(self.crop_rect)
    
    def draw_crop(self, event):
        """Draw crop rectangle."""
        if self.crop_start:
            x, y = self.crop_start
            if self.crop_rect:
                self.image_canvas.delete(self.crop_rect)
            self.crop_end = (self.image_canvas.canvasx(event.x), self.image_canvas.canvasy(event.y))
            self.crop_rect = self.image_canvas.create_rectangle(x, y, *self.crop_end, outline="red")
    
    def apply_crop(self, event=None):
        """Apply crop to current image."""
        if self.crop_start and self.crop_end:
            try:
                x1, y1 = self.crop_start
                x2, y2 = self.crop_end
                
                # Calculate actual scale (accounting for zoom level)
                actual_scale = self.current_scale * self.zoom_level
                
                # Get the image's position on the canvas
                image_x = None
                image_y = None
                if self._image_item:
                    image_bbox = self.image_canvas.bbox(self._image_item)
                    if image_bbox:
                        image_x = image_bbox[0]
                        image_y = image_bbox[1]
                
                # Fallback: calculate position manually if bbox not available
                if image_x is None or image_y is None:
                    canvas_width = self.image_canvas.winfo_width()
                    canvas_height = self.image_canvas.winfo_height()
                    image_width = int(self.original_image.size[0] * actual_scale)
                    image_height = int(self.original_image.size[1] * actual_scale)
                    image_x = self.pan_x
                    image_y = self.pan_y
                    if image_width < canvas_width:
                        image_x = (canvas_width - image_width) // 2 + self.pan_x
                    if image_height < canvas_height:
                        image_y = (canvas_height - image_height) // 2 + self.pan_y
                
                # Convert canvas coordinates to image-relative coordinates
                rel_x1 = x1 - image_x
                rel_y1 = y1 - image_y
                rel_x2 = x2 - image_x
                rel_y2 = y2 - image_y
                
                # Convert to image coordinates (accounting for actual scale)
                left = int(min(rel_x1, rel_x2) / actual_scale)
                top = int(min(rel_y1, rel_y2) / actual_scale)
                right = int(max(rel_x1, rel_x2) / actual_scale)
                bottom = int(max(rel_y1, rel_y2) / actual_scale)
                
                # Clamp coordinates to image bounds
                image_width, image_height = self.original_image.size
                left = max(0, min(left, image_width))
                top = max(0, min(top, image_height))
                right = max(0, min(right, image_width))
                bottom = max(0, min(bottom, image_height))
                
                crop_coords = (left, top, right, bottom)
                
                if self.apply_to_all.get():
                    # Apply crop to all images using batch API
                    try:
                        original_index = self.service.state.current_image_index
                        self.service.crop_all(crop_coords)
                        # Restore original position
                        self.service.state.current_image_index = original_index
                        self.show_current_image()
                    except UserFacingError as e:
                        messagebox.showerror("Error", str(e))
                        self.clear_all_modes()
                        return
                    except Exception as e:
                        messagebox.showerror("Error", f"Error cropping images: {str(e)}")
                        self.clear_all_modes()
                        return
                else:
                    # Crop current image only
                    self.service.crop_current(crop_coords)
                    self.show_current_image()
                
                if self.crop_rect:
                    self.image_canvas.delete(self.crop_rect)
                self.crop_rect = None
                self.crop_start = None
                self.crop_end = None
                
                if self.batch_process.get():
                    self.after(100, lambda: self.navigate_images(1))
                    self.after(200, self.activate_crop_tool)
                else:
                    self.clear_all_modes()
            except UserFacingError as e:
                messagebox.showerror("Error", str(e))
                self.clear_all_modes()
    
    def cancel_crop(self, event=None):
        """Cancel crop operation."""
        self.clear_all_modes()
    
    def manual_straighten(self):
        """Activate manual straighten mode."""
        current = self.service.get_current_image()
        if not current:
            return
        
        # Show warning if apply to all is enabled
        if self.apply_to_all.get():
            if not self.show_all_images_warning("Straighten"):
                return
        
        # Clear modes but preserve button state for straighten tool
        self._clear_modes_preserve_button()
        self.straightening_mode = True
        self.straighten_start = None
        
        def on_click(event):
            # Use canvasx/canvasy to handle scrolling
            canvas_x = self.image_canvas.canvasx(event.x)
            canvas_y = self.image_canvas.canvasy(event.y)
            
            if not self.straighten_start:
                self.straighten_start = (canvas_x, canvas_y)
                self.image_canvas.bind('<Motion>', update_guide_line)
            else:
                end_point = (canvas_x, canvas_y)
                calculate_and_rotate(self.straighten_start, end_point)
                cleanup()
                if self.batch_process.get():
                    self.after(100, lambda: self.navigate_images(1))
                    self.after(200, self.manual_straighten)
        
        def update_guide_line(event):
            if self.straighten_start:
                # Use canvasx/canvasy for guide line
                canvas_x = self.image_canvas.canvasx(event.x)
                canvas_y = self.image_canvas.canvasy(event.y)
                
                if self.guide_line:
                    self.image_canvas.delete(self.guide_line)
                self.guide_line = self.image_canvas.create_line(
                    self.straighten_start[0], self.straighten_start[1],
                    canvas_x, canvas_y,
                    fill='blue', width=2
                )
        
        def calculate_and_rotate(start, end):
            # Convert canvas coordinates to image coordinates using robust helper
            start_img = self.canvas_to_image_coords(start[0], start[1])
            end_img = self.canvas_to_image_coords(end[0], end[1])
            
            if self.apply_to_all.get():
                # Apply straighten to all images using batch API
                try:
                    original_index = self.service.state.current_image_index
                    self.service.straighten_all(start_img, end_img)
                    # Restore original position
                    self.service.state.current_image_index = original_index
                    self.show_current_image()
                except UserFacingError as e:
                    messagebox.showerror("Error", str(e))
                    cleanup()
                    return
                except Exception as e:
                    messagebox.showerror("Error", f"Error straightening images: {str(e)}")
                    cleanup()
                    return
            else:
                # Straighten current image only
                self.service.straighten_current(start_img, end_img)
                self.show_current_image()
        
        def cleanup():
            if not self.batch_process.get():
                self.image_canvas.unbind('<Button-1>')
                self.image_canvas.unbind('<Motion>')
                if self.guide_line:
                    self.image_canvas.delete(self.guide_line)
                self.straightening_mode = False
                self.straighten_start = None
                self.guide_line = None
                self.image_canvas.config(cursor="")
                self.clear_button_depressed()
            else:
                if self.guide_line:
                    self.image_canvas.delete(self.guide_line)
                self.straighten_start = None
                self.guide_line = None
        
        self.image_canvas.config(cursor="crosshair")
        self.image_canvas.bind('<Button-1>', on_click)
    
    def rotate_image(self, angle: float):
        """Rotate current image or all images based on toggle."""
        if self.apply_to_all.get():
            # Show warning if not suppressed
            if not self.show_all_images_warning("Rotate"):
                return
            # Rotate all images
            try:
                self.service.rotate_all(angle)
                self.show_current_image()
            except UserFacingError as e:
                messagebox.showerror("Error", str(e))
        else:
            # Rotate current image only
            try:
                self.service.rotate_current(angle)
                self.show_current_image()
            except UserFacingError as e:
                messagebox.showerror("Error", str(e))
    
    def rotate_all_images(self, angle: float):
        """Rotate all images."""
        try:
            self.service.rotate_all(angle)
            self.show_current_image()
        except UserFacingError as e:
            messagebox.showerror("Error", str(e))
    
    def incremental_rotate(self):
        """Show dialog for incremental rotation."""
        angle = simpledialog.askfloat(
            "Rotate Image",
            "Enter rotation angle (positive for counter-clockwise, negative for clockwise):",
            initialvalue=0
        )
        if angle is not None:
            self.rotate_image(angle)
    
    def undo_operation(self):
        """Undo the last operation."""
        try:
            success = self.service.undo()
            if success:
                self.show_current_image()
                self.update_counter()
            else:
                messagebox.showinfo("Undo", "Nothing to undo.")
        except Exception as e:
            logger.error(f"Error during undo: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to undo: {str(e)}")
    
    def redo_operation(self):
        """Redo the last undone operation."""
        try:
            success = self.service.redo()
            if success:
                self.show_current_image()
                self.update_counter()
            else:
                messagebox.showinfo("Redo", "Nothing to redo.")
        except Exception as e:
            logger.error(f"Error during redo: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to redo: {str(e)}")
    
    def delete_current(self):
        """Delete current image."""
        confirm = messagebox.askyesno("Delete Image", "Are you sure you want to delete the current image?")
        if confirm:
            try:
                self.service.delete_current()
                self.show_current_image()
                self.update_counter()  # Update counter after deletion
            except UserFacingError as e:
                messagebox.showerror("Error", str(e))
                self.update_counter()  # Update counter even on error
    
    def delete_range_dialog(self):
        """Open dialog to delete multiple images with range selection."""
        if not self.service.state.images:
            messagebox.showwarning("No Images", "No images to delete.")
            return
        
        # Create dialog window
        dialog = tk.Toplevel(self)
        dialog.title("Delete Range")
        dialog.transient(self)
        dialog.grab_set()
        
        # Variables
        deletion_mode = tk.StringVar(value="selected")
        range_start = tk.StringVar(value="1")
        range_end = tk.StringVar(value=str(len(self.service.state.images)))
        selected_indices = set()
        
        # Thumbnail size - smaller for filmstrip
        thumb_width = 100
        thumb_height = 120
        thumb_padding = 3
        
        # Main container with tighter padding
        main_frame = tk.Frame(dialog, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top controls section
        controls_frame = tk.Frame(main_frame, bg='#f0f0f0')
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left side: Deletion mode radio buttons
        mode_frame = tk.LabelFrame(controls_frame, text="Deletion Mode", relief=tk.GROOVE, bg='#f0f0f0', padx=10, pady=5)
        mode_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        def update_selection_display():
            """Update visual selection display."""
            # Update based on mode
            mode = deletion_mode.get()
            try:
                start = max(1, int(range_start.get()))
                end = min(len(self.service.state.images), int(range_end.get()))
            except ValueError:
                start = 1
                end = len(self.service.state.images)
            
            # Determine which indices should be selected based on mode
            if mode == "all":
                indices_to_show = set(range(start - 1, end))
            elif mode == "even":
                indices_to_show = set(i for i in range(start - 1, end) if (i + 1) % 2 == 0)
            elif mode == "odd":
                indices_to_show = set(i for i in range(start - 1, end) if (i + 1) % 2 == 1)
            else:  # selected
                indices_to_show = selected_indices.copy()
            
            # Update frame colors
            for i, frame in enumerate(thumbnail_widgets):
                if i in indices_to_show:
                    frame.config(bg="#4a90e2", relief=tk.SUNKEN, borderwidth=3)
                    for child in frame.winfo_children():
                        if isinstance(child, tk.Label) and hasattr(child, 'cget'):
                            try:
                                if child.cget('text').startswith('Page'):
                                    child.config(fg="white", bg="#4a90e2")
                            except:
                                pass
                    frame.selected = True
                else:
                    frame.config(bg="white", relief=tk.RAISED, borderwidth=1)
                    for child in frame.winfo_children():
                        if isinstance(child, tk.Label) and hasattr(child, 'cget'):
                            try:
                                if child.cget('text').startswith('Page'):
                                    child.config(fg="black", bg="white")
                            except:
                                pass
                    frame.selected = False
        
        tk.Radiobutton(
            mode_frame,
            text="All images in range",
            variable=deletion_mode,
            value="all",
            command=update_selection_display,
            bg='#f0f0f0'
        ).pack(anchor=tk.W, pady=2)
        
        tk.Radiobutton(
            mode_frame,
            text="Even Images",
            variable=deletion_mode,
            value="even",
            command=update_selection_display,
            bg='#f0f0f0'
        ).pack(anchor=tk.W, pady=2)
        
        tk.Radiobutton(
            mode_frame,
            text="Odd Images",
            variable=deletion_mode,
            value="odd",
            command=update_selection_display,
            bg='#f0f0f0'
        ).pack(anchor=tk.W, pady=2)
        
        tk.Radiobutton(
            mode_frame,
            text="Selected Images Only",
            variable=deletion_mode,
            value="selected",
            command=update_selection_display,
            bg='#f0f0f0'
        ).pack(anchor=tk.W, pady=2)
        
        # Right side: Range input
        range_frame = tk.LabelFrame(controls_frame, text="Page Range", relief=tk.GROOVE, bg='#f0f0f0', padx=10, pady=5)
        range_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        range_input_frame = tk.Frame(range_frame, bg='#f0f0f0')
        range_input_frame.pack(pady=5)
        
        tk.Label(range_input_frame, text="From:", bg='#f0f0f0').grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        tk.Entry(range_input_frame, textvariable=range_start, width=8).grid(row=0, column=1, padx=2)
        tk.Label(range_input_frame, text="To:", bg='#f0f0f0').grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        tk.Entry(range_input_frame, textvariable=range_end, width=8).grid(row=1, column=1, padx=2, pady=(5, 0))
        
        def update_range():
            try:
                start = int(range_start.get())
                end = int(range_end.get())
                if start < 1:
                    range_start.set("1")
                if end > len(self.service.state.images):
                    range_end.set(str(len(self.service.state.images)))
                update_selection_display()
            except ValueError:
                pass
        
        range_start.trace('w', lambda *args: update_range())
        range_end.trace('w', lambda *args: update_range())
        
        # Filmstrip section with better layout
        filmstrip_container = tk.LabelFrame(main_frame, text="Image Selection", relief=tk.GROOVE, bg='#f0f0f0')
        filmstrip_container.pack(fill=tk.X, pady=(0, 10))  # Changed from BOTH/expand to X only
        
        # Canvas for thumbnails with proper height
        filmstrip_canvas = tk.Canvas(
            filmstrip_container, 
            bg="white", 
            highlightthickness=1, 
            highlightbackground="#cccccc",
            height=thumb_height + 30  # Fixed height for thumbnails
        )
        filmstrip_canvas.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(5, 0))
        
        # Horizontal scrollbar positioned below canvas
        scrollbar = ttk.Scrollbar(filmstrip_container, orient=tk.HORIZONTAL, command=filmstrip_canvas.xview)
        scrollbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))
        filmstrip_canvas.configure(xscrollcommand=scrollbar.set)
        
        # Inner frame for thumbnails
        thumbnails_frame = tk.Frame(filmstrip_canvas, bg="white")
        filmstrip_canvas_window = filmstrip_canvas.create_window((0, 0), window=thumbnails_frame, anchor=tk.NW)
        
        def configure_scroll_region(event=None):
            # Update scroll region when content changes
            filmstrip_canvas.update_idletasks()
            dialog.update_idletasks()
            
            # Get the actual width of the thumbnails frame (which contains all thumbnails)
            thumbnails_width = thumbnails_frame.winfo_reqwidth()
            thumbnails_height = thumbnails_frame.winfo_reqheight()
            
            # If reqwidth is 1 (not yet rendered), calculate from children
            if thumbnails_width <= 1 and len(thumbnail_widgets) > 0:
                # Calculate width from last thumbnail position
                last_container = thumbnails_frame.winfo_children()[-1] if thumbnails_frame.winfo_children() else None
                if last_container:
                    thumbnails_width = last_container.winfo_x() + last_container.winfo_reqwidth() + thumb_padding
            
            canvas_width = filmstrip_canvas.winfo_width()
            canvas_height = filmstrip_canvas.winfo_height()
            
            # Ensure we have a valid width
            if thumbnails_width <= 1:
                # Calculate approximate width based on thumbnail count
                thumbnails_width = len(thumbnail_widgets) * (thumb_width + thumb_padding * 2 + 10)
            
            # Set scroll region to encompass all thumbnails
            # Use the greater of thumbnails width or canvas width
            scroll_width = max(thumbnails_width, canvas_width, 1)
            scroll_height = max(thumbnails_height, canvas_height, 1)
            
            filmstrip_canvas.configure(scrollregion=(0, 0, scroll_width, scroll_height))
            
            # Don't constrain the canvas window width - let it expand to fit all thumbnails
            # This allows horizontal scrolling
            if thumbnails_width > 1:
                filmstrip_canvas.itemconfig(filmstrip_canvas_window, width=thumbnails_width)
            
            # Update scrollbar
            filmstrip_canvas.update_idletasks()
        
        def on_canvas_configure(event):
            # When canvas resizes, update scroll region but don't constrain thumbnails width
            configure_scroll_region()
        
        thumbnails_frame.bind('<Configure>', configure_scroll_region)
        filmstrip_canvas.bind('<Configure>', on_canvas_configure)
        
        # Add mouse wheel support for scrolling
        def on_mousewheel(event):
            # Scroll horizontally with mouse wheel
            if event.delta:
                filmstrip_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                # For Linux
                if event.num == 4:
                    filmstrip_canvas.xview_scroll(-1, "units")
                elif event.num == 5:
                    filmstrip_canvas.xview_scroll(1, "units")
        
        filmstrip_canvas.bind("<MouseWheel>", on_mousewheel)
        filmstrip_canvas.bind("<Button-4>", on_mousewheel)
        filmstrip_canvas.bind("<Button-5>", on_mousewheel)
        
        # Store thumbnail references
        thumbnail_widgets = []
        thumbnail_photos = []
        
        def load_thumbnails():
            """Load all thumbnails."""
            nonlocal thumbnail_widgets, thumbnail_photos
            
            # Clear existing
            for widget in thumbnail_widgets:
                widget.destroy()
            thumbnail_widgets.clear()
            thumbnail_photos.clear()
            
            # Load thumbnails with proper spacing
            for i, record in enumerate(self.service.state.images):
                # Container frame for each thumbnail
                container = tk.Frame(thumbnails_frame, bg="white", padx=2, pady=5)
                container.grid(row=0, column=i, padx=thumb_padding, sticky=tk.N)
                
                # Thumbnail frame with border
                frame = tk.Frame(container, relief=tk.RAISED, borderwidth=1, bg="white")
                frame.pack()
                
                # Load and resize image
                try:
                    image_path = record.current_image_path
                    if not image_path.exists():
                        raise FileNotFoundError(f"Image not found: {image_path}")
                    
                    with Image.open(image_path) as img:
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Calculate aspect ratio preserving resize for thumbnail
                        # Use smaller size for thumbnails
                        max_thumb_width = thumb_width
                        max_thumb_height = thumb_height - 25  # Leave space for label
                        
                        img_ratio = img.width / img.height
                        if img_ratio > max_thumb_width / max_thumb_height:
                            # Image is wider than tall
                            new_width = max_thumb_width
                            new_height = int(max_thumb_width / img_ratio)
                        else:
                            # Image is taller than wide
                            new_height = max_thumb_height
                            new_width = int(max_thumb_height * img_ratio)
                        
                        # Resize the image
                        img_thumb = img.resize((new_width, new_height), Image.LANCZOS)
                        
                        # Create padded thumbnail with white background
                        padded_thumb = Image.new('RGB', (max_thumb_width, max_thumb_height), 'white')
                        x_offset = (max_thumb_width - new_width) // 2
                        y_offset = (max_thumb_height - new_height) // 2
                        padded_thumb.paste(img_thumb, (x_offset, y_offset))
                        
                        # Convert to PhotoImage
                        photo = ImageTk.PhotoImage(padded_thumb)
                        thumbnail_photos.append(photo)  # Keep reference to prevent garbage collection
                        
                        # Create image label
                        label = tk.Label(frame, image=photo, bg="white", borderwidth=0)
                        label.image = photo  # Keep reference
                        label.pack(pady=2)
                        
                        # Page number label
                        page_label = tk.Label(
                            frame, 
                            text=f"Page {i+1}", 
                            bg="white", 
                            font=("Arial", 8, "bold"),
                            pady=2
                        )
                        page_label.pack()
                        
                        # Store frame and index
                        frame.image_index = i
                        frame.selected = False
                        thumbnail_widgets.append(frame)
                        
                        # Bind click events
                        def make_click_handler(idx):
                            def handler(event):
                                # Handle CTRL and SHIFT
                                if event.state & 0x0004:  # CTRL
                                    if idx in selected_indices:
                                        selected_indices.discard(idx)
                                    else:
                                        selected_indices.add(idx)
                                elif event.state & 0x0001:  # SHIFT
                                    if selected_indices:
                                        start_idx = min(selected_indices)
                                        end_idx = max(selected_indices)
                                        if idx < start_idx:
                                            selected_indices.update(range(idx, start_idx + 1))
                                        elif idx > end_idx:
                                            selected_indices.update(range(end_idx, idx + 1))
                                    else:
                                        selected_indices.add(idx)
                                else:
                                    selected_indices.clear()
                                    selected_indices.add(idx)
                                update_selection_display()
                            return handler
                        
                        frame.bind("<Button-1>", make_click_handler(i))
                        label.bind("<Button-1>", make_click_handler(i))
                        page_label.bind("<Button-1>", make_click_handler(i))
                        container.bind("<Button-1>", make_click_handler(i))
                        
                except Exception as e:
                    logger.error(f"Error loading thumbnail {i}: {e}")
                    # Create placeholder
                    placeholder = tk.Label(
                        frame, 
                        text=f"Page {i+1}\n\n⚠\nError", 
                        bg="#e0e0e0", 
                        width=thumb_width//10, 
                        height=thumb_height//20,
                        font=("Arial", 9)
                    )
                    placeholder.pack(fill=tk.BOTH, expand=True)
                    frame.image_index = i
                    frame.selected = False
                    thumbnail_widgets.append(frame)
        
        # Selection info label
        info_frame = tk.Frame(main_frame, bg='#f0f0f0')
        info_frame.pack(fill=tk.X, pady=(5, 5))
        
        selection_label = tk.Label(
            info_frame,
            text="Tip: Use Ctrl+Click to toggle selection, Shift+Click to select range",
            bg='#f0f0f0',
            font=("Arial", 9),
            fg="#666666"
        )
        selection_label.pack(side=tk.LEFT)
        
        # Buttons frame - pack normally in sequence
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=(10, 5))
        
        # Initial load - after all UI elements are created
        load_thumbnails()
        update_selection_display()
        
        # Force update of scroll region after loading all thumbnails
        # Use multiple updates to ensure all widgets are rendered
        dialog.update_idletasks()
        dialog.update()
        
        # Schedule scroll region update after a short delay to ensure all thumbnails are rendered
        def delayed_scroll_update():
            configure_scroll_region()
            # Double-check after another short delay
            dialog.after(50, configure_scroll_region)
        
        dialog.after(100, delayed_scroll_update)
        
        def delete_selected():
            """Delete selected images."""
            mode = deletion_mode.get()
            try:
                start = max(1, int(range_start.get()))
                end = min(len(self.service.state.images), int(range_end.get()))
            except ValueError:
                messagebox.showerror("Error", "Invalid page range.", parent=dialog)
                return
            
            # Determine indices to delete
            if mode == "all":
                indices = list(range(start - 1, end))
            elif mode == "even":
                indices = [i for i in range(start - 1, end) if (i + 1) % 2 == 0]
            elif mode == "odd":
                indices = [i for i in range(start - 1, end) if (i + 1) % 2 == 1]
            else:  # selected
                indices = list(selected_indices)
            
            if not indices:
                messagebox.showwarning("No Selection", "No images selected for deletion.", parent=dialog)
                return
            
            # Confirm deletion
            count = len(indices)
            pages_str = "pages" if count > 1 else "page"
            if not messagebox.askyesno(
                "Confirm Deletion", 
                f"Are you sure you want to delete {count} {pages_str}?\n\nThis action cannot be undone.", 
                parent=dialog
            ):
                return
            
            # Delete
            try:
                deleted_count = self.service.delete_range(indices)
                dialog.destroy()
                self.show_current_image()
                self.update_counter()
                messagebox.showinfo("Success", f"Successfully deleted {deleted_count} image(s).", parent=self)
            except UserFacingError as e:
                messagebox.showerror("Error", str(e), parent=dialog)
        
        def cancel():
            dialog.destroy()
        
        # Style buttons to match main UI
        delete_btn = tk.Button(
            button_frame, 
            text="Delete Selected", 
            command=delete_selected,
            bg="#d9534f",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2"
        )
        delete_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            command=cancel,
            bg="#6c757d",
            fg="white",
            font=("Arial", 10),
            padx=15,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2"
        )
        cancel_btn.pack(side=tk.RIGHT)
        
        # Count label
        def update_count_label():
            mode = deletion_mode.get()
            try:
                start = max(1, int(range_start.get()))
                end = min(len(self.service.state.images), int(range_end.get()))
            except ValueError:
                start = 1
                end = len(self.service.state.images)
            
            if mode == "all":
                count = end - start + 1
            elif mode == "even":
                count = len([i for i in range(start - 1, end) if (i + 1) % 2 == 0])
            elif mode == "odd":
                count = len([i for i in range(start - 1, end) if (i + 1) % 2 == 1])
            else:
                count = len(selected_indices)
            
            count_label.config(text=f"{count} image(s) selected for deletion")
        
        count_label = tk.Label(
            button_frame,
            text="0 image(s) selected for deletion",
            bg='#f0f0f0',
            font=("Arial", 9, "bold"),
            fg="#333333"
        )
        count_label.pack(side=tk.LEFT)
        
        # Update count when selection changes
        original_update = update_selection_display
        def new_update():
            original_update()
            update_count_label()
        update_selection_display = new_update
        
        # Initial count
        update_count_label()
        
        # Set initial size and position - ensure buttons are visible
        # Wait for all content to load first
        dialog.update_idletasks()
        dialog.update()
        
        # Calculate required height after all content is loaded
        required_height = (
            controls_frame.winfo_reqheight() +
            filmstrip_container.winfo_reqheight() +
            info_frame.winfo_reqheight() +
            button_frame.winfo_reqheight() +
            60  # padding for margins
        )
        
        # Ensure minimum height
        final_height = max(520, required_height)
        
        # Set geometry
        dialog.geometry(f"900x{final_height}")
        dialog.update_idletasks()
        
        # Center dialog
        x = (dialog.winfo_screenwidth() // 2) - 450
        y = (dialog.winfo_screenheight() // 2) - (final_height // 2)
        dialog.geometry(f"900x{final_height}+{x}+{y}")
        
        # Make dialog resizable
        dialog.resizable(True, True)
        
        # Set minimum size to ensure buttons are always visible
        dialog.minsize(700, 520)
    
    def revert_current(self):
        """Revert current image to original."""
        try:
            self.service.revert_current()
            self.show_current_image()
            # Counter might not change, but update to be safe
            self.update_counter()
        except UserFacingError as e:
            messagebox.showerror("Error", str(e))
            self.update_counter()
    
    def revert_all(self):
        """Revert all images to original."""
        try:
            # Show progress
            progress_window = tk.Toplevel(self)
            progress_window.title("Reverting Images")
            progress_window.geometry("300x100")
            progress_window.transient(self)
            
            progress_label = ttk.Label(progress_window, text="Reverting images...", padding=10)
            progress_label.pack()
            
            def process():
                try:
                    self.service.revert_all()
                    progress_label.config(text="All images reverted!")
                    self.after(1000, lambda: (progress_window.destroy(), self.show_current_image(), self.update_counter()))
                except UserFacingError as e:
                    messagebox.showwarning("Revert Warnings", str(e))
                    progress_window.destroy()
                    self.show_current_image()
                    self.update_counter()
            
            import threading
            threading.Thread(target=process, daemon=True).start()
        except Exception as e:
            logger.error(f"Error in revert_all: {e}")
            messagebox.showerror("Error", f"An error occurred while reverting images: {str(e)}")
    
    def save_images(self):
        """Save processed images to current save directory."""
        # If no save directory is set, default to Save As...
        if self._current_save_directory is None:
            self.save_images_as()
            return
        
        # Check if there are images to save
        if len(self.service.state.images) == 0:
            messagebox.showwarning("No Images", "No images to save.")
            return
        
        # Determine save directory - use opened folder if available, otherwise use current save directory
        save_dir = self._opened_folder if self._opened_folder else self._current_save_directory
        
        confirm = messagebox.askyesno(
            "Save Images",
            f"Save {len(self.service.state.images)} image(s) to:\n{save_dir}\n\n"
            "This will finalize the images and cannot be undone."
        )
        if confirm:
            try:
                self.service.save_images(save_dir)
                messagebox.showinfo("Success", f"Images saved successfully to:\n{save_dir}")
                self.service.state.mark_saved()
            except UserFacingError as e:
                messagebox.showerror("Error", str(e))
    
    def save_images_as(self):
        """Save processed images to a selected directory."""
        from tkinter import filedialog
        
        # Check if there are images to save
        if len(self.service.state.images) == 0:
            messagebox.showwarning("No Images", "No images to save.")
            return
        
        # Open folder selection dialog
        # Use initialdir if we have a previous save directory
        initialdir = str(self._current_save_directory) if self._current_save_directory else None
        
        folder = filedialog.askdirectory(
            title="Select Folder to Save Images",
            initialdir=initialdir
        )
        
        if not folder:  # User cancelled
            return
        
        save_dir = Path(folder)
        
        confirm = messagebox.askyesno(
            "Save Images",
            f"Save {len(self.service.state.images)} image(s) to:\n{save_dir}\n\n"
            "This will finalize the images and cannot be undone."
        )
        
        if confirm:
            try:
                self.service.save_images(save_dir)
                # Store the save directory for future saves
                self._current_save_directory = save_dir
                # Also update opened folder if we're saving to a new location
                if not self._opened_folder:
                    self._opened_folder = save_dir
                messagebox.showinfo("Success", f"Images saved successfully to:\n{save_dir}")
                self.service.state.mark_saved()
            except UserFacingError as e:
                messagebox.showerror("Error", str(e))
    
    def reset_program(self):
        """Reset the program to its initial empty state."""
        # Ask for confirmation
        if self.service.state.images:
            response = messagebox.askyesno(
                "New",
                "Are you sure you want to start a new session?\n\n"
                "This will clear all loaded images and reset the application to its initial state.\n"
                "Unsaved changes will be lost."
            )
            if not response:
                return
        
        try:
            # Clear the service state
            self.service.state.images = []
            self.service.state.current_image_index = 0
            self.service.state.status = "no_changes"
            
            # Cleanup temporary files
            self.service.cleanup()
            
            # Reset save directory
            self._current_save_directory = None
            self._opened_folder = None
            
            # Clear canvas
            self.image_canvas.delete("all")
            self.image_canvas.image = None
            self.original_image = None
            
            # Reset UI state variables
            self.current_scale = 1.0
            self.special_cursor_active = False
            self.cursor_orientation = 'vertical'
            self.cursor_angle = 0
            self.cursor_line = None
            self.vertical_line = None
            self.horizontal_line = None
            self.cropping = False
            self.crop_start = None
            self.crop_end = None
            if self.crop_rect:
                self.image_canvas.delete(self.crop_rect)
            self.crop_rect = None
            self.straightening_mode = False
            self.straighten_start = None
            if self.straighten_line:
                self.image_canvas.delete(self.straighten_line)
            self.straighten_line = None
            if self.guide_line:
                self.image_canvas.delete(self.guide_line)
            self.guide_line = None
            
            # Clear cursor bindings
            self.clear_all_modes()
            
            # Show placeholder text
            def add_placeholder():
                canvas_width = self.image_canvas.winfo_width()
                canvas_height = self.image_canvas.winfo_height()
                if canvas_width > 1 and canvas_height > 1:
                    if self._placeholder_text:
                        self.image_canvas.delete(self._placeholder_text)
                    self._placeholder_text = self.image_canvas.create_text(
                        canvas_width // 2,
                        canvas_height // 2,
                        text="No images loaded.\n\nDrag and drop image files here,\nor use File > Import Images",
                        font=("Arial", 14),
                        fill="gray",
                        justify=tk.CENTER
                    )
                else:
                    self.after(50, add_placeholder)
            
            self.after(100, add_placeholder)
            
            # Update counter
            self.update_counter()
            
            logger.info("Program reset to initial state")
            
        except Exception as e:
            logger.error(f"Error during program reset: {e}")
            messagebox.showerror("Error", f"Failed to start new session: {str(e)}")
    
    def on_closing(self):
        """Handle window closing."""
        if self.service.state.status != "no_changes":
            response = messagebox.askyesnocancel(
                "Save Changes",
                "Do you want to save your changes before closing?"
            )
            if response is None:  # Cancel
                return
            elif response:  # Yes
                # If no save directory set, use Save As...
                if self._current_save_directory is None:
                    self.save_images_as()
                    # If user cancelled Save As, don't close
                    if self._current_save_directory is None:
                        return  # Stay open
                    # Otherwise save was successful, continue with close
                else:
                    # Use opened folder if available, otherwise use current save directory
                    save_dir = self._opened_folder if self._opened_folder else self._current_save_directory
                    try:
                        self.service.save_images(save_dir)
                        self.service.state.mark_saved()
                    except UserFacingError as e:
                        if messagebox.askyesno("Save Failed", f"Failed to save changes. Try again?\n{str(e)}"):
                            return  # Stay open for user to try again
                        self.service.state.mark_discarded()
            else:  # No
                self.service.state.mark_discarded()
        
        # Cleanup
        self.service.cleanup()
        self.destroy()
    
    def run(self):
        """Start the application main loop."""
        self.mainloop()

