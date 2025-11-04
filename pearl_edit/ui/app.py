"""Main Tkinter application for PearlEdit."""
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from pathlib import Path
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
        self.title("Transcription Pearl Image Preprocessing Tool 0.9 beta")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
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
        self.autocrop_icon = self.load_icon(icons_dir / "autocrop.png", size=(24, 24))
        self.straighten_icon = self.load_icon(icons_dir / "straighten.png", size=(24, 24))
        self.autostraighten_icon = self.load_icon(icons_dir / "autostraighten.png", size=(24, 24))
        
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
        
        # Divider after File Operations
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
        
        # Image Editing section
        image_editing_section = tk.Frame(self.toolbar_frame)
        image_editing_section.pack(side=tk.LEFT, padx=2)
        
        image_editing_buttons = tk.Frame(image_editing_section)
        image_editing_buttons.pack(side=tk.TOP, pady=2)
        
        # Left side: Split button with icon
        def split_wrapper():
            self.update_status_display("Split Tool: To change between Horizontal and Vertical Cursor use Ctrl+H and Ctrl+V | To rotate cursor use [ and ] | To split image, click mouse")
            self.switch_to_vertical()
        self.split_button = self.create_button_with_hint(
            image_editing_buttons,
            self.split_icon,
            split_wrapper,
            "Split Image\nActivate vertical split cursor\nClick to split at cursor position\nShortcuts: Ctrl+V (vertical), Ctrl+H (horizontal)",
            "Split Tool: To change between Horizontal and Vertical Cursor use Ctrl+H and Ctrl+V | To rotate cursor use [ and ] | To split image, click mouse"
        )
        self.split_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Left side: Crop button
        def crop_wrapper():
            self.update_status_display("Crop Tool: Drag mouse to select area | To apply crop, press Enter | To cancel, press Escape")
            self.activate_crop_tool()
        self.crop_button = self.create_button_with_hint(
            image_editing_buttons,
            self.crop_icon,
            crop_wrapper,
            "Crop Tool\nActivate crop mode\nDrag to select area\nEnter to apply, Escape to cancel\nShortcut: Ctrl+Shift+C",
            "Crop Tool: Drag mouse to select area | To apply crop, press Enter | To cancel, press Escape"
        )
        self.crop_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Left side: Auto-crop button
        def autocrop_wrapper():
            self.update_status_display("Auto Crop Tool: Adjust threshold and margin sliders in dialog | Click Apply to crop image | Click Cancel to abort")
            self.auto_crop_current()
        self.autocrop_button = self.create_button_with_hint(
            image_editing_buttons,
            self.autocrop_icon,
            autocrop_wrapper,
            "Auto Crop\nAutomatically crop image using edge detection\nAdjust threshold in dialog\nShortcut: Ctrl+Shift+A",
            "Auto Crop Tool: Adjust threshold and margin sliders in dialog | Click Apply to crop image | Click Cancel to abort"
        )
        self.autocrop_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Left side: Straighten button
        def straighten_wrapper():
            self.update_status_display("Straighten Tool: Click first point to start line | Click second point to end line | Image will rotate to align line")
            self.manual_straighten()
        self.straighten_button = self.create_button_with_hint(
            image_editing_buttons,
            self.straighten_icon,
            straighten_wrapper,
            "Straighten Image\nDraw a line to straighten image\nClick start point, then end point\nShortcut: Ctrl+L",
            "Straighten Tool: Click first point to start line | Click second point to end line | Image will rotate to align line"
        )
        self.straighten_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Left side: Auto-straighten button
        def autostraighten_wrapper():
            self.update_status_display("Auto Straighten Tool: Adjust threshold slider in dialog | Click Apply to straighten image | Click Cancel to abort")
            self.auto_straighten_current()
        self.autostraighten_button = self.create_button_with_hint(
            image_editing_buttons,
            self.autostraighten_icon,
            autostraighten_wrapper,
            "Auto Straighten\nAutomatically straighten image using edge detection\nAdjust threshold in dialog\nShortcut: Ctrl+Shift+L",
            "Auto Straighten Tool: Adjust threshold slider in dialog | Click Apply to straighten image | Click Cancel to abort"
        )
        self.autostraighten_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Label for Image Editing
        image_editing_label = tk.Label(
            image_editing_section,
            text="Image Editing",
            font=("Arial", 7),
            fg="gray"
        )
        image_editing_label.pack(side=tk.BOTTOM, pady=(0, 2))
        
        # Divider after Image Editing
        separator_image_editing = tk.Frame(self.toolbar_frame, width=2, relief=tk.SUNKEN, borderwidth=1)
        separator_image_editing.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        
        # Rotation section (left side, after Image Editing)
        rotation_section = tk.Frame(self.toolbar_frame)
        rotation_section.pack(side=tk.LEFT, padx=2)
        
        rotation_buttons = tk.Frame(rotation_section)
        rotation_buttons.pack(side=tk.TOP, pady=2)
        
        # Rotate buttons
        def rotate_right_wrapper():
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
        edit_menu.add_command(label="Delete Current Image", command=self.delete_current)
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
        self.bind("<Control-a>", lambda e: self.toggle_auto_split())
        
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
        self.batch_process.set(not self.batch_process.get())
        self.settings.batch_process = self.batch_process.get()
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
        self.apply_to_all.set(not self.apply_to_all.get())
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
    
    def clear_all_modes(self):
        """Clear all active modes and reset to default state."""
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
                # Split image based on cursor position
                if self.cursor_orientation == 'vertical' and self.vertical_line:
                    coords = self.image_canvas.coords(self.vertical_line)
                    if coords:
                        split_x = int(coords[0] / self.current_scale)
                        self.split_image('vertical', split_coord=split_x)
                elif self.cursor_orientation == 'horizontal' and self.horizontal_line:
                    coords = self.image_canvas.coords(self.horizontal_line)
                    if coords:
                        split_y = int(coords[1] / self.current_scale)
                        self.split_image('horizontal', split_coord=split_y)
                elif self.cursor_orientation == 'angled' and self.cursor_line:
                    coords = self.image_canvas.coords(self.cursor_line)
                    if coords:
                        x1, y1, x2, y2 = [int(c / self.current_scale) for c in coords]
                        self.split_image('angled', line_coords=(x1, y1, x2, y2))
                
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
            # Mouse up → image up
            dx = event.x - self.pan_start_x
            dy_raw = event.y - self.pan_start_y  # Raw mouse movement
            
            # Update pan offset (pan_y increases = image moves down in canvas)
            self.pan_x += dx
            self.pan_y += dy_raw  # pan_y tracks actual position
            
            # Update start position for next drag
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            
            # Smooth pan: move the image item (canvas.move uses screen coords)
            # For canvas.move: positive dy moves DOWN, but we want mouse up = image up
            # So we need to invert dy for canvas.move()
            if self._image_item:
                self.image_canvas.move(self._image_item, dx, -dy_raw)  # Invert Y for canvas.move
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
    
    def split_image(self, orientation: str, split_coord: int = None, line_coords: tuple = None):
        """Split current image or all images based on toggle."""
        if self.apply_to_all.get():
            # Show warning if not suppressed
            if not self.show_all_images_warning("Split"):
                return
            # Use batch split API
            try:
                initial_index = self.service.state.current_image_index
                self.service.split_all(orientation, split_coord, line_coords)
                
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
                self.service.split_current(orientation, split_coord, line_coords)
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
        
        def apply_crop(threshold, margin):
            try:
                # Update settings first
                self.service.settings.threshold = threshold
                self.service.settings.margin = margin
                # Use service method which has history tracking
                self.service.auto_crop_current()
                self.show_current_image()
                # Advance to next image if batch processing is enabled
                if self.batch_process.get():
                    # Check if we're not at the last image before navigating
                    current_index = self.service.state.current_image_index
                    total_images = len(self.service.state.images)
                    if current_index < total_images - 1:
                        # Navigate to next image and continue processing
                        self.after(100, lambda: self.navigate_images(1))
                        self.after(200, self.auto_crop_current)
            except UserFacingError as e:
                messagebox.showerror("Error", str(e))
            except Exception as e:
                messagebox.showerror("Error", f"Error during auto-crop: {str(e)}")
        
        ThresholdAdjuster(self, image_path, self.service, apply_crop)
    
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
    
    def activate_crop_tool(self, event=None):
        """Activate crop tool."""
        # Show warning if apply to all is enabled
        if self.apply_to_all.get():
            if not self.show_all_images_warning("Crop"):
                return
        
        self.clear_all_modes()
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
                
                # Convert to image coordinates
                left = int(min(x1, x2) / self.current_scale)
                top = int(min(y1, y2) / self.current_scale)
                right = int(max(x1, x2) / self.current_scale)
                bottom = int(max(y1, y2) / self.current_scale)
                
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
        
        self.clear_all_modes()
        self.straightening_mode = True
        self.straighten_start = None
        
        def on_click(event):
            if not self.straighten_start:
                self.straighten_start = (event.x, event.y)
                self.image_canvas.bind('<Motion>', update_guide_line)
            else:
                end_point = (event.x, event.y)
                calculate_and_rotate(self.straighten_start, end_point)
                cleanup()
                if self.batch_process.get():
                    self.after(100, lambda: self.navigate_images(1))
                    self.after(200, self.manual_straighten)
        
        def update_guide_line(event):
            if self.straighten_start:
                if self.guide_line:
                    self.image_canvas.delete(self.guide_line)
                self.guide_line = self.image_canvas.create_line(
                    self.straighten_start[0], self.straighten_start[1],
                    event.x, event.y,
                    fill='blue', width=2
                )
        
        def calculate_and_rotate(start, end):
            # Convert to image coordinates
            start_img = (int(start[0] / self.current_scale), int(start[1] / self.current_scale))
            end_img = (int(end[0] / self.current_scale), int(end[1] / self.current_scale))
            
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

