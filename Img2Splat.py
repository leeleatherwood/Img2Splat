#!/usr/bin/env python3
"""Unity Splatmap Generator from Terrain Textures.

This tool generates Unity-format splatmaps from terrain textures by matching
colors to a user-defined palette using perceptual color matching (CIELAB).

Features:
    * GUI mode with live preview
    * Command-line mode for batch processing
    * Load terrain texture and palette
    * Generate splatmap_0 and splatmap_1 (Unity RGBA format)
    * Preview generated splatmaps (click to expand)
    * Export with automatic naming

Example:
    GUI Mode::

        $ python img2splat.py

    Command-Line Mode::

        $ python img2splat.py texture.png --palette palette.json

Note:
    Requires Python 3.7+ with Pillow, NumPy, scikit-image, and tkinter.

Shazbot! 🔥
"""

__license__ = "MIT"

# ===== DISCLAIMER =====
# This script was vibe coded using GitHub Copilot (Claude 4.5 Sonnet) by an
# author who freely admits to knowing absolutely nothing about Python.
# ======================

# Standard library imports
import argparse
import json
import logging
import multiprocessing
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, Any

# Third-party imports
import numpy as np
from PIL import Image
from skimage import color

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Optional tkinter for GUI
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, Canvas, Scrollbar
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# Constants
PREVIEW_SIZE = 512
LAYER_PREVIEW_SIZE = 1024
DEFAULT_SAMPLE_SIZE = 16
DEFAULT_SPLATMAP_SIZE = 1024
NUM_LAYERS = 8
VALID_SAMPLE_SIZES = [1, 2, 4, 8, 16, 32, 64]
VALID_SPLATMAP_SIZES = [256, 512, 1024, 2048, 4096, 8192]
WORKER_MULTIPLIER = 2  # Multiply CPU count by this for thread pool

# UI Messages
MSG_NO_TEXTURE = "Please load a texture first"
MSG_NO_PALETTE = "Please select or load a palette"
MSG_NO_SPLATMAPS = "Generate splatmaps first"
MSG_SUCCESS_TITLE = "Success"
MSG_ERROR_TITLE = "Error"
MSG_GENERATING = "🗺️  Generating splatmaps from {}"
MSG_SUCCESS_GENERATION = "✓ Splatmaps generated successfully"
MSG_SUCCESS_SAVED = "✓ Saved: {} ({}×{})"


def check_and_install_dependencies() -> None:
    """
    Check for required packages and install them if missing.
    
    This function is called explicitly rather than at import time
    to give users control over when dependencies are installed.
    """
    required_packages = {
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'skimage': 'scikit-image'
    }
    
    missing_packages = []
    
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        logger.info("Missing dependencies detected: %s", ', '.join(missing_packages))
        logger.info("Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            logger.info("Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            logger.error("Error installing dependencies: %s", e)
            sys.exit(1)


def configure_pil_for_large_images() -> None:
    """
    Configure PIL to handle large terrain textures.
    
    Disables the decompression bomb limit which would otherwise
    prevent loading of large terrain texture files.
    """
    Image.MAX_IMAGE_PIXELS = None


# ===== Core Functions (used by both CLI and GUI) =====

def load_palette(palette_path: Path) -> List[Dict[str, Any]]:
    """
    Load color palette from JSON file.
    
    Expected format:
    {
        "layers": [
            {"name": "grass", "rgb": [126, 200, 12]},
            {"name": "rock", "rgb": [20, 16, 120]},
            ...
        ]
    }
    
    Args:
        palette_path: Path to the JSON palette file
    
    Returns:
        List of dicts with 'name' and 'rgb' keys
    
    Raises:
        ValueError: If palette format is invalid
        FileNotFoundError: If palette file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    try:
        with open(palette_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Palette file not found: {palette_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in palette file: {e}")
    
    if 'layers' not in data:
        raise ValueError("Palette JSON must contain 'layers' array")
    
    layers = data['layers']
    
    if len(layers) != NUM_LAYERS:
        raise ValueError(f"Palette must define exactly {NUM_LAYERS} layers, got {len(layers)}")
    
    # Validate each layer
    for i, layer in enumerate(layers):
        if 'name' not in layer or 'rgb' not in layer:
            raise ValueError(f"Layer {i} must have 'name' and 'rgb' fields")
        
        rgb = layer['rgb']
        if len(rgb) != 3:
            raise ValueError(f"Layer {i} RGB must have 3 values, got {len(rgb)}")
        
        if not all(isinstance(c, (int, float)) and 0 <= c <= 255 for c in rgb):
            raise ValueError(f"Layer {i} RGB values must be numbers between 0-255")
    
    return layers


def find_closest_color(sample_rgb: Tuple[int, int, int], palette: List[Dict[str, Any]]) -> int:
    """
    Find the closest matching color in the palette using perceptual color distance (CIELAB).
    
    Uses the CIELAB color space (L*a*b*) which is designed to be perceptually uniform,
    meaning equal distances in LAB space represent equal perceived color differences.
    This provides more accurate color matching than RGB Euclidean distance.
    
    Args:
        sample_rgb: (R, G, B) tuple of the sampled color (0-255)
        palette: List of palette layer dicts with 'rgb' keys
    
    Returns:
        Index of the closest matching layer (0-7)
    """
    # Convert sample RGB to LAB (needs to be in 0-1 range first)
    sample_rgb_normalized = np.array(sample_rgb, dtype=np.float32) / 255.0
    # Reshape to (1, 1, 3) as required by rgb2lab
    sample_rgb_reshaped = sample_rgb_normalized.reshape(1, 1, 3)
    sample_lab = color.rgb2lab(sample_rgb_reshaped)[0, 0]
    
    min_distance = float('inf')
    closest_idx = 0
    
    for i, layer in enumerate(palette):
        # Convert palette color RGB to LAB
        palette_rgb_normalized = np.array(layer['rgb'], dtype=np.float32) / 255.0
        palette_rgb_reshaped = palette_rgb_normalized.reshape(1, 1, 3)
        palette_lab = color.rgb2lab(palette_rgb_reshaped)[0, 0]
        
        # Calculate Delta E (CIE76) - perceptual color difference
        # deltaE = sqrt((L1-L2)² + (a1-a2)² + (b1-b2)²)
        distance = np.sqrt(np.sum((sample_lab - palette_lab) ** 2))
        
        if distance < min_distance:
            min_distance = distance
            closest_idx = i
    
    return closest_idx


def process_row(row_data: Tuple[int, np.ndarray, int, int, int, int, int, List[Dict[str, Any]]]) -> Tuple[int, Dict[int, List[Tuple[int, int, int, int]]], List[int]]:
    """
    Process a single row of tiles. This function is called in parallel.
    
    Args:
        row_data: Tuple containing (tile_y, texture_array, tiles_x, tile_width, 
                  tile_height, height, width, palette)
    
    Returns:
        Tuple of (tile_y, row_assignments, row_layer_counts)
    """
    tile_y, texture_array, tiles_x, tile_width, tile_height, height, width, palette = row_data
    
    # Row results: dict mapping layer_idx to list of (y_start, y_end, x_start, x_end) tuples
    row_assignments = {i: [] for i in range(NUM_LAYERS)}
    row_layer_counts = [0] * NUM_LAYERS
    
    for tile_x in range(tiles_x):
        # Extract tile region
        y_start = tile_y * tile_height
        y_end = min(y_start + tile_height, height)
        x_start = tile_x * tile_width
        x_end = min(x_start + tile_width, width)
        
        # Get tile as PIL Image and downsample to 1×1 to get average color
        tile_region = texture_array[y_start:y_end, x_start:x_end, :3]
        tile_img = Image.fromarray(tile_region.astype('uint8'))
        
        # Downsample to 1×1 pixel using BOX for fast averaging
        avg_img = tile_img.resize((1, 1), Image.Resampling.BOX)
        rgb = tuple(avg_img.getpixel((0, 0)))
        
        # Find closest palette color
        layer_idx = find_closest_color(rgb, palette)
        row_layer_counts[layer_idx] += 1
        
        # Store assignment for this tile
        row_assignments[layer_idx].append((y_start, y_end, x_start, x_end))
    
    return tile_y, row_assignments, row_layer_counts


def generate_splatmaps(
    texture_img: Image.Image,
    palette: List[Dict[str, Any]],
    sample_size: int,
    splatmap_size: Optional[int] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    row_callback: Optional[Callable[[List[np.ndarray], int, int, List[int], int, int], None]] = None,
    use_multithread: bool = True
) -> Tuple[Image.Image, Image.Image, List[Image.Image], Dict[str, Any]]:
    """
    Generate Unity-format splatmaps from terrain texture.
    
    Each tile is assigned to exactly ONE layer based on the tile's average color.
    The average is computed by downsampling each tile to 1×1 pixel using BOX.
    No blending - pure layer assignment for hard-edged terrain.
    
    Args:
        texture_img: PIL Image of the terrain texture
        palette: List of 8 palette layer dicts
        sample_size: Size of sampling tiles in pixels (e.g., 16)
        splatmap_size: [UNUSED] Kept for backward compatibility - resizing is done during save
        progress_callback: Optional function(percent, message) for progress updates
        row_callback: Optional function(weight_maps, row_num, total_rows, layer_counts, tiles_processed, total_tiles)
        use_multithread: Use parallel processing (default: True)
    
    Returns:
        Tuple of (splatmap_0, splatmap_1, layer_images, stats) where:
        - splatmap_0: PIL Image (RGBA) for layers 0-3
        - splatmap_1: PIL Image (RGBA) for layers 4-7
        - layer_images: List of 8 PIL Images (grayscale) for individual layers
        - stats: Dict with layer_counts, coverage, and total_tiles
    """
    texture_array = np.array(texture_img)
    height, width = texture_array.shape[:2]
    
    # Calculate sample tile size in pixels
    tile_width = sample_size
    tile_height = sample_size
    
    # Calculate number of tiles
    tiles_x = width // tile_width
    tiles_y = height // tile_height
    
    # Create 8 weight maps (one per layer)
    # Output resolution matches texture resolution
    weight_maps = [np.zeros((height, width), dtype=np.uint8) for _ in range(NUM_LAYERS)]
    
    # Process each ROW
    total_tiles = tiles_x * tiles_y
    layer_counts = [0] * NUM_LAYERS  # Track how many tiles assigned to each layer
    
    if use_multithread:
        # Prepare row data for parallel processing
        row_data_list = [
            (tile_y, texture_array, tiles_x, tile_width, tile_height, height, width, palette)
            for tile_y in range(tiles_y)
        ]
        
        # Use ThreadPoolExecutor for parallel processing
        # Use 2x CPU cores for better throughput with image processing operations
        num_workers = min(multiprocessing.cpu_count() * WORKER_MULTIPLIER, tiles_y)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all rows for processing
            futures = [executor.submit(process_row, row_data) for row_data in row_data_list]
            
            # Collect results as they complete
            for future_idx, future in enumerate(futures):
                tile_y, row_assignments, row_layer_counts = future.result()
                
                # Apply assignments to weight maps
                for layer_idx, assignments in row_assignments.items():
                    for y_start, y_end, x_start, x_end in assignments:
                        weight_maps[layer_idx][y_start:y_end, x_start:x_end] = 255
                
                # Update layer counts
                for i in range(NUM_LAYERS):
                    layer_counts[i] += row_layer_counts[i]
                
                # Progress callback
                if progress_callback:
                    progress = ((future_idx + 1) / tiles_y) * 100
                    tiles_processed = (future_idx + 1) * tiles_x
                    progress_callback(progress, f"Processing tiles: {tiles_processed}/{total_tiles}")
                
                # Row callback (only call every 10 rows or last row)
                if row_callback and (tile_y % 10 == 0 or tile_y == tiles_y - 1):
                    tiles_processed_so_far = (tile_y + 1) * tiles_x
                    row_callback(weight_maps, tile_y, tiles_y, layer_counts, tiles_processed_so_far, total_tiles)
    else:
        # Single-threaded fallback
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                # Extract tile region
                y_start = tile_y * tile_height
                y_end = min(y_start + tile_height, height)
                x_start = tile_x * tile_width
                x_end = min(x_start + tile_width, width)
                
                # Get tile as PIL Image and downsample to 1×1 to get average color
                tile_region = texture_array[y_start:y_end, x_start:x_end, :3]
                tile_img = Image.fromarray(tile_region.astype('uint8'))
                
                # Downsample to 1×1 pixel using BOX for fast averaging
                avg_img = tile_img.resize((1, 1), Image.Resampling.BOX)
                rgb = tuple(avg_img.getpixel((0, 0)))
                
                # Find closest palette color
                layer_idx = find_closest_color(rgb, palette)
                layer_counts[layer_idx] += 1
                
                # Assign this tile to ONE layer only (255 = full weight)
                weight_maps[layer_idx][y_start:y_end, x_start:x_end] = 255
            
            # Call row callback after each row is complete
            if row_callback:
                tiles_processed_so_far = (tile_y + 1) * tiles_x
                row_callback(weight_maps, tile_y, tiles_y, layer_counts, tiles_processed_so_far, total_tiles)
    
    # Pack into Unity RGBA format
    # Splatmap 0: RGBA = layers 0, 1, 2, 3
    # Splatmap 1: RGBA = layers 4, 5, 6, 7
    
    # Always create full-resolution splatmaps
    splatmap_0_array = np.stack([weight_maps[i] for i in range(4)], axis=2)
    splatmap_1_array = np.stack([weight_maps[i] for i in range(4, 8)], axis=2)
    
    splatmap_0 = Image.fromarray(splatmap_0_array.astype('uint8'))
    splatmap_1 = Image.fromarray(splatmap_1_array.astype('uint8'))
    
    # Calculate statistics
    coverage = []
    for i in range(NUM_LAYERS):
        total_weight = np.sum(weight_maps[i])
        coverage_percent = (total_weight / (255 * height * width)) * 100
        coverage.append(coverage_percent)
    
    stats = {
        'layer_counts': layer_counts,
        'coverage': coverage,
        'total_tiles': total_tiles
    }
    
    # Convert weight maps to PIL Images for individual layer viewing
    layer_images = [Image.fromarray(weight_maps[i]) for i in range(NUM_LAYERS)]
    
    return splatmap_0, splatmap_1, layer_images, stats


# ===== GUI Classes =====

if TKINTER_AVAILABLE:
    from tkinter import PhotoImage as TkPhotoImage
    
    class FullSizeViewer(tk.Toplevel):
        """Full-size image viewer window with scrollbars.
        
        A modal dialog that displays an image at full resolution with
        vertical and horizontal scrollbars for navigation.
        
        Args:
            parent: Parent tkinter window
            image: PIL Image to display
            title: Window title (default: "Full Size View")
        """
        
        def __init__(
            self,
            parent: tk.Tk,
            image: Image.Image,
            title: str = "Full Size View"
        ) -> None:
            """Initialize the full-size viewer window."""
            super().__init__(parent)
            self.title(title)
            self.image = image
            
            # Set window size (max 80% of screen)
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            window_width = min(int(screen_width * 0.8), image.width + 40)
            window_height = min(int(screen_height * 0.8), image.height + 40)
            
            self.geometry(f"{window_width}x{window_height}")
            
            # Create scrollable canvas
            frame = ttk.Frame(self)
            frame.pack(fill=tk.BOTH, expand=True)
            
            # Scrollbars
            v_scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
            h_scrollbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
            
            # Canvas
            self.canvas = Canvas(frame, 
                                xscrollcommand=h_scrollbar.set,
                                yscrollcommand=v_scrollbar.set)
            
            v_scrollbar.config(command=self.canvas.yview)
            h_scrollbar.config(command=self.canvas.xview)
            
            # Layout
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Display image
            from PIL import ImageTk
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            
            # Close button
            btn_frame = ttk.Frame(self)
            btn_frame.pack(side=tk.BOTTOM, pady=5)
            ttk.Button(btn_frame, text="Close", command=self.destroy).pack()
    
    
    class Img2SplatApp:
        """Main GUI application for splatmap generation.
        
        Provides an interactive interface for loading terrain textures,
        selecting color palettes, generating splatmaps, and previewing results.
        
        Attributes:
            root: Main tkinter window
            source_texture: Loaded terrain texture image
            palette: Color palette for layer matching
            splatmap_0: Generated first splatmap (layers 0-3)
            splatmap_1: Generated second splatmap (layers 4-7)
            layer_maps: Individual layer weight maps
        """
        
        def __init__(self, root: tk.Tk) -> None:
            """Initialize the application GUI.
            
            Args:
                root: Main tkinter window
            """
            self.root = root
            self.root.title("Img2Splat - Splatmap Generator")
            self.root.geometry("660x950")
            
            # Data
            self.source_texture = None          # Full-resolution PIL Image
            self.source_path = None             # Path to source file
            self.palette = None                 # Loaded palette
            self.palette_path = None            # Path to palette file
            self.splatmap_0 = None              # Generated splatmap 0
            self.splatmap_1 = None              # Generated splatmap 1
            self.layer_maps = [None] * NUM_LAYERS        # Individual layer weight maps (0-7)
            self.generation_stats = None        # Generation statistics
            
            # Preview images
            self.preview_size = PREVIEW_SIZE
            self.layer_preview_size = LAYER_PREVIEW_SIZE      # Preview for each layer
            self.source_preview = None
            self.layer_previews = [None] * NUM_LAYERS    # Preview images for each layer
            
            # PhotoImages for canvas
            self.source_photo = None
            self.layer_photos = [None] * NUM_LAYERS      # PhotoImage for each layer canvas
            
            # UI Variables
            self.sample_size_var = tk.IntVar(value=DEFAULT_SAMPLE_SIZE)
            self.splatmap_size_var = tk.IntVar(value=DEFAULT_SPLATMAP_SIZE)
            
            # Available palettes (scan palettes folder)
            self.available_palettes = self.scan_palettes()
            
            # Setup UI
            self.setup_ui()
            self.update_button_states()
            
        def scan_palettes(self) -> List[str]:
            """Scan palettes folder for available palette files.
            
            Returns:
                List of palette names (without .json extension)
            """
            palettes_dir = Path(__file__).parent / 'palettes'
            if not palettes_dir.exists():
                return []
            
            palettes = []
            for file in palettes_dir.glob('*.json'):
                palettes.append(file.stem)  # Filename without extension
            
            return sorted(palettes)
        
        def setup_ui(self) -> None:
            """Create and layout the user interface."""
            from PIL import ImageTk
            
            # Main container
            main_container = ttk.Frame(self.root, padding="10")
            main_container.pack(fill=tk.BOTH, expand=True)
            
            # Top section: Source texture and controls
            top_frame = ttk.Frame(main_container)
            top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False, pady=(0, 10))
            
            # Left: Source texture preview
            source_frame = ttk.LabelFrame(top_frame, text="Source Texture", padding="5")
            source_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
            
            self.source_canvas = Canvas(source_frame, bg='#2b2b2b', cursor='hand2', width=400, height=300)
            self.source_canvas.pack(fill=tk.BOTH, expand=True)
            self.source_canvas.bind('<Button-1>', lambda e: self.show_full_size(self.source_texture, "Source Texture"))
            
            ttk.Button(source_frame, text="📁 Load Texture...", 
                      command=self.load_texture).pack(pady=(5, 0))
            
            # Right: Settings
            settings_frame = ttk.LabelFrame(top_frame, text="Generation Settings", padding="5")
            settings_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
            settings_frame.config(width=320)
            settings_frame.pack_propagate(False)
            
            row = 0
            ttk.Label(settings_frame, text="Palette:").grid(row=row, column=0, sticky=tk.W, pady=5)
            self.palette_combo = ttk.Combobox(settings_frame,
                                              values=self.available_palettes,
                                              state='readonly', width=18)
            self.palette_combo.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=5, padx=(5, 0))
            if self.available_palettes:
                self.palette_combo.current(0)
            
            row += 1
            ttk.Button(settings_frame, text="📂 Load Custom Palette...",
                      command=self.load_custom_palette, width=25).grid(row=row, column=0, columnspan=3, pady=5)
            
            row += 1
            ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, 
                                                                      columnspan=3, sticky=tk.EW, pady=10)
            
            row += 1
            ttk.Label(settings_frame, text="Sample Size:").grid(row=row, column=0, sticky=tk.W, pady=5)
            sample_combo = ttk.Combobox(settings_frame, textvariable=self.sample_size_var,
                                      values=VALID_SAMPLE_SIZES,
                                      state='readonly', width=10)
            sample_combo.grid(row=row, column=1, sticky=tk.W, pady=5, padx=(5, 0))
            ttk.Label(settings_frame, text="pixels").grid(row=row, column=2, sticky=tk.W, pady=5, padx=(5, 0))
            
            row += 1
            ttk.Label(settings_frame, text="Output Size:").grid(row=row, column=0, sticky=tk.W, pady=5)
            size_combo = ttk.Combobox(settings_frame, textvariable=self.splatmap_size_var,
                                      values=VALID_SPLATMAP_SIZES,
                                      state='readonly', width=10)
            size_combo.grid(row=row, column=1, sticky=tk.W, pady=5, padx=(5, 0))
            ttk.Label(settings_frame, text="pixels").grid(row=row, column=2, sticky=tk.W, pady=5, padx=(5, 0))
            
            row += 1
            ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, 
                                                                      columnspan=3, sticky=tk.EW, pady=10)
            
            row += 1
            self.generate_btn = ttk.Button(settings_frame, text="🎨 Generate Splatmaps", 
                                           command=self.generate_splatmaps, width=25)
            self.generate_btn.grid(row=row, column=0, columnspan=3, pady=5)
            
            row += 1
            self.save_btn = ttk.Button(settings_frame, text="💾 Save Splatmaps", 
                                       command=self.save_splatmaps, width=25)
            self.save_btn.grid(row=row, column=0, columnspan=3, pady=5)
            
            # Middle section: Palette display
            palette_frame = ttk.LabelFrame(main_container, text="Palette Layers", padding="5")
            palette_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
            
            self.palette_text = tk.Text(palette_frame, height=6, wrap=tk.NONE, font=('Courier', 9))
            self.palette_text.pack(fill=tk.X)
            self.palette_text.config(state=tk.DISABLED)
            
            # Status bar (pack from bottom first so it doesn't get hidden)
            self.status_label = ttk.Label(main_container, text="Ready - Load a texture and palette to begin", 
                                          relief=tk.SUNKEN, anchor=tk.W)
            self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
            
            # Bottom section: Individual layer previews
            preview_frame = ttk.LabelFrame(main_container, text="Splatmap Layers (Click to view full size)", padding="5")
            preview_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            preview_grid = ttk.Frame(preview_frame)
            preview_grid.pack(fill=tk.BOTH, expand=True)
            
            # Configure grid for 8 columns
            for col in range(8):
                preview_grid.grid_columnconfigure(col, weight=1)
            preview_grid.grid_rowconfigure(0, weight=0)
            preview_grid.grid_rowconfigure(1, weight=1)
            
            # Create 8 layer previews
            self.layer_canvases = []
            self.layer_labels = []
            channel_letters = ['R', 'G', 'B', 'A', 'R', 'G', 'B', 'A']
            
            for i in range(NUM_LAYERS):
                # Label (will be updated with palette layer name)
                label_text = f"Layer {i}"
                if self.palette:
                    layer_name = self.palette[i]['name'].capitalize()
                    label_text = f"{layer_name} ({channel_letters[i]})"
                
                label = ttk.Label(preview_grid, text=label_text, font=('TkDefaultFont', 8, 'bold'))
                label.grid(row=0, column=i, pady=(0, 5))
                self.layer_labels.append(label)
                
                # Canvas - stretch to fill available space
                canvas = Canvas(preview_grid, width=self.layer_preview_size, 
                               height=self.layer_preview_size, bg='#2b2b2b', cursor='hand2')
                canvas.grid(row=1, column=i, padx=2, sticky='nsew')  # nsew makes it stretch
                canvas.bind('<Button-1>', lambda e, idx=i: self.show_layer_full_size(idx))
                self.layer_canvases.append(canvas)
        
        def load_texture(self) -> None:
            """Load source texture from file dialog."""
            filetypes = [
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tga"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
            
            filename = filedialog.askopenfilename(
                title="Select Terrain Texture",
                filetypes=filetypes
            )
            
            if not filename:
                return
            
            try:
                self.source_path = Path(filename)
                self.source_texture = Image.open(self.source_path)
                
                if self.source_texture.mode != 'RGB' and self.source_texture.mode != 'RGBA':
                    self.source_texture = self.source_texture.convert('RGB')
                
                logger.info("✓ Loaded texture: %s", self.source_path.name)
                logger.info("  Size: %d×%d", self.source_texture.size[0], self.source_texture.size[1])
                
                # Update source preview
                self.update_source_preview()
                self.update_status()
                self.update_button_states()
                
            except (IOError, OSError, ValueError) as e:
                messagebox.showerror("Error Loading Texture", 
                    f"Failed to load {filename}:\n{str(e)}")
        
        def load_custom_palette(self) -> None:
            """Load custom palette from file dialog."""
            filetypes = [
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
            
            filename = filedialog.askopenfilename(
                title="Select Palette JSON",
                filetypes=filetypes
            )
            
            if not filename:
                return
            
            try:
                self.palette_path = Path(filename)
                self.palette = load_palette(self.palette_path)
                
                logger.info("✓ Loaded custom palette: %s", self.palette_path.name)
                self.update_palette_display()
                self.update_layer_labels()  # Update layer preview labels
                self.update_status()
                self.update_button_states()
                
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
                messagebox.showerror("Error Loading Palette", 
                    f"Failed to load {filename}:\n{str(e)}")
        
        def update_source_preview(self) -> None:
            """Update source texture preview canvas."""
            if self.source_texture is None:
                return
            
            from PIL import ImageTk
            
            # Get canvas size
            self.source_canvas.update_idletasks()
            canvas_width = self.source_canvas.winfo_width()
            canvas_height = self.source_canvas.winfo_height()
            
            if canvas_width <= 1:
                canvas_width = 400
            if canvas_height <= 1:
                canvas_height = 300
            
            # Resize for preview
            img_width, img_height = self.source_texture.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            preview_width = int(img_width * scale)
            preview_height = int(img_height * scale)
            
            self.source_preview = self.source_texture.resize(
                (preview_width, preview_height),
                Image.Resampling.LANCZOS
            )
            
            self.source_photo = ImageTk.PhotoImage(self.source_preview)
            
            # Clear and display
            self.source_canvas.delete('all')
            x = (canvas_width - preview_width) // 2
            y = (canvas_height - preview_height) // 2
            self.source_canvas.create_image(x, y, anchor=tk.NW, image=self.source_photo)
        
        def update_palette_display(self, stats: Optional[Dict[str, Any]] = None) -> None:
            """Update palette display text with optional statistics."""
            if self.palette is None:
                return
            
            self.palette_text.config(state=tk.NORMAL)
            self.palette_text.delete('1.0', tk.END)
            
            if stats:
                # Show with statistics
                total_tiles = stats['total_tiles']
                for i, layer in enumerate(self.palette):
                    count = stats['layer_counts'][i]
                    percent = (count / total_tiles * 100) if total_tiles > 0 else 0
                    text = f"Layer {i} ({layer['name']:<12}): {count} tiles ({percent:.1f}%)\n"
                    self.palette_text.insert(tk.END, text)
            else:
                # Show just palette info
                for i, layer in enumerate(self.palette):
                    r, g, b = layer['rgb']
                    text = f"Layer {i}: {layer['name']:<12} RGB({r:3d}, {g:3d}, {b:3d})\n"
                    self.palette_text.insert(tk.END, text)
            
            self.palette_text.config(state=tk.DISABLED)
        
        def show_full_size(self, image: Optional[Image.Image], title: str) -> None:
            """Open full-size viewer window."""
            if image is None:
                return
            
            FullSizeViewer(self.root, image, title)
        
        def show_layer_full_size(self, layer_idx: int) -> None:
            """Show full-size view of an individual layer."""
            if self.layer_maps[layer_idx] is None:
                return
            
            layer_name = f"Layer {layer_idx}"
            if self.palette:
                layer_name = f"{self.palette[layer_idx]['name'].capitalize()}"
                channel_letters = ['R', 'G', 'B', 'A', 'R', 'G', 'B', 'A']
                layer_name += f" ({channel_letters[layer_idx]})"
            
            FullSizeViewer(self.root, self.layer_maps[layer_idx], layer_name)
        
        def update_layer_labels(self) -> None:
            """Update layer preview labels with palette names."""
            if not self.palette:
                return
            
            channel_letters = ['R', 'G', 'B', 'A', 'R', 'G', 'B', 'A']
            for i in range(NUM_LAYERS):
                layer_name = self.palette[i]['name'].capitalize()
                self.layer_labels[i].config(text=f"{layer_name} ({channel_letters[i]})")
        
        def generate_splatmaps(self) -> None:
            """Generate splatmaps from source texture."""
            if self.source_texture is None:
                messagebox.showwarning("No Texture", MSG_NO_TEXTURE)
                return
            
            # Load palette if not already loaded
            if self.palette is None:
                selected_palette = self.palette_combo.get()
                if not selected_palette:
                    messagebox.showwarning("No Palette", MSG_NO_PALETTE)
                    return
                
                palettes_dir = Path(__file__).parent / 'palettes'
                self.palette_path = palettes_dir / f"{selected_palette}.json"
                
                try:
                    self.palette = load_palette(self.palette_path)
                    self.update_palette_display()
                    self.update_layer_labels()  # Update layer preview labels
                except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
                    messagebox.showerror("Error Loading Palette", str(e))
                    return
            
            # Validate dimensions
            width, height = self.source_texture.size
            sample_size = self.sample_size_var.get()
            
            if width % sample_size != 0 or height % sample_size != 0:
                valid_sizes = [s for s in VALID_SAMPLE_SIZES if width % s == 0 and height % s == 0]
                msg = f"Texture size {width}×{height} is not evenly divisible by sample size {sample_size}.\n\n"
                if valid_sizes:
                    msg += f"Suggested sample sizes: {', '.join(map(str, valid_sizes))}"
                else:
                    msg += "No standard sample sizes work with this texture."
                messagebox.showerror("Invalid Sample Size", msg)
                return
            
            # Disable buttons during generation
            self.generate_btn.config(state='disabled')
            self.save_btn.config(state='disabled')
            
            logger.info("\n%s", MSG_GENERATING.format(self.source_path.name))
            
            try:
                # Get settings
                splatmap_size = self.splatmap_size_var.get()
                if splatmap_size == 0:
                    splatmap_size = None
                
                # Generate
                def progress_callback(percent, message):
                    self.status_label.config(text=f"{message} ({percent:.1f}%)")
                    self.root.update_idletasks()
                
                # Row callback to update previews after each row
                def row_callback(weight_maps, row_num, total_rows, layer_counts, tiles_processed, total_tiles):
                    # Update previews every 10 rows or on last row
                    if row_num % 10 == 0 or row_num == total_rows - 1:
                        # Convert current weight maps to PIL images and update previews
                        for i in range(NUM_LAYERS):
                            temp_img = Image.fromarray(weight_maps[i])
                            self.layer_maps[i] = temp_img
                            self.update_layer_preview(i)
                        
                        # Update palette display with current statistics (cumulative so far)
                        stats = {
                            'layer_counts': layer_counts,
                            'total_tiles': tiles_processed  # Use tiles processed so far for accurate %
                        }
                        self.update_palette_display(stats)
                        
                        # Update status bar with row progress
                        progress = ((row_num + 1) / total_rows) * 100
                        self.status_label.config(text=f"Processing rows: {row_num + 1}/{total_rows} ({progress:.1f}%) - {tiles_processed}/{total_tiles} tiles")
                        
                        # Process events to keep UI responsive
                        self.root.update()  # Use update() instead of update_idletasks() for better responsiveness
                
                # Generate at full resolution (resizing happens during save)
                self.splatmap_0, self.splatmap_1, self.layer_maps, self.generation_stats = generate_splatmaps(
                    self.source_texture, 
                    self.palette, 
                    sample_size,
                    None,  # Don't resize during generation
                    progress_callback,
                    row_callback
                )
                
                # Layer maps are already updated from the last row callback
                # Just update palette display with final statistics
                self.update_palette_display(self.generation_stats)
                self.status_label.config(text="Splatmaps generated successfully!")
                self.root.update_idletasks()
                
                logger.info(MSG_SUCCESS_GENERATION)
                
                # Print statistics
                logger.info("\nLayer assignment:")
                for i in range(NUM_LAYERS):
                    count = self.generation_stats['layer_counts'][i]
                    percent = (count / self.generation_stats['total_tiles']) * 100
                    logger.info("  Layer %d (%s): %d tiles (%.1f%%)", 
                              i, self.palette[i]['name'], count, percent)
                
                self.update_status()
                self.update_button_states()
                
            except (ValueError, RuntimeError, MemoryError) as e:
                logger.error("✗ Error generating splatmaps: %s", e)
                messagebox.showerror(MSG_ERROR_TITLE, f"Failed to generate splatmaps:\n{str(e)}")
                self.update_button_states()
        
        def update_layer_preview(self, layer_idx: int) -> None:
            """Update an individual layer preview with scaled-down splatmap."""
            if self.layer_maps[layer_idx] is None:
                return
            
            from PIL import ImageTk
            
            image = self.layer_maps[layer_idx]
            img_width, img_height = image.size
            
            # Scale down the entire layer map to preview size
            preview = image.resize(
                (self.layer_preview_size, self.layer_preview_size),
                Image.Resampling.NEAREST
            )
            
            # Convert grayscale to RGB with channel color visualization
            # Get the grayscale data
            preview_array = np.array(preview)
            
            # Create RGB visualization based on channel
            if layer_idx in [3, 7]:  # Alpha channels - keep as grayscale
                colored_preview = Image.fromarray(preview_array).convert('RGB')
            else:
                # For RGB channels, show the intensity in that channel only
                rgb_array = np.zeros((preview_array.shape[0], preview_array.shape[1], 3), dtype=np.uint8)
                
                # Map layer to RGB channel (0=R, 1=G, 2=B for both splatmaps)
                channel_map = [0, 1, 2, 0, 0, 1, 2, 0]  # Layer index -> RGB channel
                channel_idx = channel_map[layer_idx]
                
                # Apply the grayscale values to the appropriate channel
                rgb_array[:, :, channel_idx] = preview_array
                
                colored_preview = Image.fromarray(rgb_array)
            
            photo = ImageTk.PhotoImage(colored_preview)
            self.layer_photos[layer_idx] = photo  # Store reference
            
            # Clear and display
            canvas = self.layer_canvases[layer_idx]
            canvas.delete('all')
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        
        def save_splatmaps(self) -> None:
            """Save generated splatmaps to selected directory."""
            if self.splatmap_0 is None or self.splatmap_1 is None:
                messagebox.showwarning("No Splatmaps", MSG_NO_SPLATMAPS)
                return
            
            directory = filedialog.askdirectory(
                title="Select Directory to Save Splatmaps"
            )
            
            if not directory:
                return
            
            try:
                output_dir = Path(directory)
                base_name = self.source_path.stem
                
                # Remove "_terrain_texture" suffix if present
                if base_name.endswith('_terrain_texture'):
                    base_name = base_name[:-16]
                
                # Get the output size setting
                splatmap_size = self.splatmap_size_var.get()
                
                # Determine if we need to resize
                current_width = self.splatmap_0.size[0]
                
                if splatmap_size != current_width:
                    # Resize splatmaps for export using NEAREST
                    logger.info("  Resizing splatmaps from %d×%d to %d×%d...",
                              current_width, current_width, splatmap_size, splatmap_size)
                    
                    splatmap_0_resized = self.splatmap_0.resize(
                        (splatmap_size, splatmap_size), 
                        Image.Resampling.NEAREST
                    )
                    splatmap_1_resized = self.splatmap_1.resize(
                        (splatmap_size, splatmap_size), 
                        Image.Resampling.NEAREST
                    )
                else:
                    # No resize needed
                    splatmap_0_resized = self.splatmap_0
                    splatmap_1_resized = self.splatmap_1
                
                # Save splatmaps
                splatmap_0_path = output_dir / f"{base_name}_splatmap0.png"
                splatmap_1_path = output_dir / f"{base_name}_splatmap1.png"
                
                splatmap_0_resized.save(str(splatmap_0_path), 'PNG')
                splatmap_1_resized.save(str(splatmap_1_path), 'PNG')
                
                logger.info(MSG_SUCCESS_SAVED, splatmap_0_path.name, 
                          splatmap_0_resized.size[0], splatmap_0_resized.size[1])
                logger.info(MSG_SUCCESS_SAVED, splatmap_1_path.name,
                          splatmap_1_resized.size[0], splatmap_1_resized.size[1])
                
                messagebox.showinfo(MSG_SUCCESS_TITLE, f"Splatmaps saved to:\n{output_dir}")
                
            except (IOError, OSError, PermissionError) as e:
                logger.error("✗ Error saving splatmaps: %s", e)
                messagebox.showerror("Error Saving", f"Failed to save splatmaps:\n{str(e)}")
        
        def update_button_states(self) -> None:
            """Enable/disable buttons based on application state."""
            has_source = self.source_texture is not None
            has_splatmaps = self.splatmap_0 is not None
            
            state_generate = 'normal' if has_source else 'disabled'
            state_save = 'normal' if has_splatmaps else 'disabled'
            
            self.generate_btn.config(state=state_generate)
            self.save_btn.config(state=state_save)
        
        def update_status(self) -> None:
            """Update status bar with current state."""
            if self.source_texture:
                width, height = self.source_texture.size
                status = f"Loaded: {self.source_path.name} ({width}×{height})"
                if self.splatmap_0:
                    status += f" | Splatmaps generated"
            else:
                status = "Ready - Load a texture and palette to begin"
            
            self.status_label.config(text=status)


# ===== Command-Line Interface =====

def main_cli() -> None:
    """Command-line interface for splatmap generation.
    
    Handles argument parsing, file loading, splatmap generation,
    and saving results via command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Generate Unity splatmaps from terrain texture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GUI Mode
  python img2splat.py
  
  # CLI Mode - Basic usage
  python img2splat.py terrain.png --palette lush.json
  
  # Generate with custom sample size
  python img2splat.py terrain.png --palette lush.json --sample-size 64
  
  # Generate with specific output size
  python img2splat.py terrain.png --palette lush.json --splatmap-size 1024
  
  # Custom output directory
  python img2splat.py terrain.png --palette lush.json --output ./my_splatmaps
        """
    )
    
    parser.add_argument('texture', nargs='?', help='Path to terrain texture PNG file')
    parser.add_argument('--palette', '-p', help='Path to color palette JSON file')
    parser.add_argument('--sample-size', '-t', type=int, default=DEFAULT_SAMPLE_SIZE, 
                        help=f'Sampling tile size in pixels (default: {DEFAULT_SAMPLE_SIZE})')
    parser.add_argument('--splatmap-size', '-s', type=int, default=None,
                        help='Output splatmap resolution (e.g., 1024, 2048). If not specified, matches texture size')
    parser.add_argument('--output', '-o', help='Output directory (default: <texture_name>_splatmaps)')
    
    args = parser.parse_args()
    
    # If no arguments, launch GUI
    if args.texture is None:
        if not TKINTER_AVAILABLE:
            logger.error("ERROR: tkinter is not available. Cannot launch GUI mode.")
            logger.error("Please provide command-line arguments for CLI mode.")
            sys.exit(1)
        
        logger.info("Launching GUI mode...")
        root = tk.Tk()
        app = Img2SplatApp(root)
        root.mainloop()
        return
    
    # CLI mode - validate required arguments
    if not args.palette:
        logger.error("ERROR: --palette is required for CLI mode")
        sys.exit(1)
    
    # Validate input texture exists
    texture_path = Path(args.texture)
    if not texture_path.exists():
        logger.error("ERROR: Texture file not found: %s", args.texture)
        sys.exit(1)
    
    # Load palette
    logger.info("\n📋 Loading palette from %s...", args.palette)
    try:
        palette = load_palette(args.palette)
        logger.info("✓ Loaded palette with %d layers:", len(palette))
        for i, layer in enumerate(palette):
            r, g, b = layer['rgb']
            logger.info("  Layer %d: %-12s RGB(%3d, %3d, %3d)", i, layer['name'], r, g, b)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        logger.error("ERROR: Failed to load palette: %s", e)
        sys.exit(1)
    
    # Load terrain texture
    logger.info("\n🖼️  Loading terrain texture from %s...", args.texture)
    try:
        texture_img = Image.open(texture_path)
        if texture_img.mode != 'RGB' and texture_img.mode != 'RGBA':
            texture_img = texture_img.convert('RGB')
        logger.info("✓ Loaded texture: %d×%d pixels", texture_img.size[0], texture_img.size[1])
    except (FileNotFoundError, IOError, OSError) as e:
        logger.error("ERROR: Failed to load texture: %s", e)
        sys.exit(1)
    
    # Validate texture dimensions
    width, height = texture_img.size
    if width % args.sample_size != 0 or height % args.sample_size != 0:
        logger.error("ERROR: Texture size %d×%d is not evenly divisible by sample size %d",
                    width, height, args.sample_size)
        logger.error("       Edge pixels would not be sampled correctly.")
        logger.error("       Please use a texture size that is a multiple of %d, or choose a different sample size.",
                    args.sample_size)
        
        valid_sizes = []
        for size in VALID_SAMPLE_SIZES:
            if width % size == 0 and height % size == 0:
                valid_sizes.append(size)
        
        if valid_sizes:
            logger.error("       Suggested sample sizes for %d×%d: %s",
                       width, height, ", ".join(map(str, valid_sizes)))
        else:
            logger.error("       (use --sample-size with a common divisor of both dimensions)")
        
        sys.exit(1)
    
    # Generate splatmaps
    logger.info("\n🎨 Generating splatmaps...")
    logger.info("  Texture size: %d×%d", width, height)
    logger.info("  Sample size: %d×%d pixels", args.sample_size, args.sample_size)
    logger.info("  Color matching: CIELAB perceptual color space")
    
    def cli_progress(percent: float, message: str) -> None:
        """Progress callback for CLI mode."""
        print(f"  {message} ({percent:.1f}%)", end='\r')
    
    try:
        splatmap_0, splatmap_1, layer_images, stats = generate_splatmaps(
            texture_img, palette, args.sample_size, args.splatmap_size, cli_progress
        )
        logger.info("  Processing complete!%s", " " * 40)
        
        # Print statistics
        logger.info("\n  Layer assignment:")
        for i in range(NUM_LAYERS):
            count = stats['layer_counts'][i]
            percent = (count / stats['total_tiles']) * 100
            coverage = stats['coverage'][i]
            logger.info("    Layer %d (%-12s): %d tiles (%.1f%%), %.2f%% coverage",
                       i, palette[i]['name'], count, percent, coverage)
        
    except (ValueError, RuntimeError, MemoryError) as e:
        logger.error("\n✗ ERROR: Failed to generate splatmaps: %s", e)
        sys.exit(1)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        terrain_name = texture_path.stem
        if terrain_name.endswith('_terrain_texture'):
            terrain_name = terrain_name[:-16]
        output_dir = Path(f"{terrain_name}_splatmaps")
    
    # Create output directory
    output_dir = output_dir.resolve()
    output_dir.mkdir(exist_ok=True)
    logger.info("\n💾 Saving splatmaps to %s\\", output_dir)
    
    # Save splatmaps
    splatmap_0_path = output_dir / 'splatmap_0.png'
    splatmap_1_path = output_dir / 'splatmap_1.png'
    
    splatmap_0.save(str(splatmap_0_path), 'PNG')
    logger.info("  ✓ Saved splatmap_0.png (layers 0-3: %s)",
               ', '.join(p['name'] for p in palette[:4]))
    
    splatmap_1.save(str(splatmap_1_path), 'PNG')
    logger.info("  ✓ Saved splatmap_1.png (layers 4-7: %s)",
               ', '.join(p['name'] for p in palette[4:]))
    
    logger.info("\n✅ Done! Splatmaps generated successfully.")
    logger.info("\nUnity Import Settings:")
    logger.info("  - Set texture type to 'Default'")
    logger.info("  - Disable sRGB color space")
    logger.info("  - Set compression to 'High Quality'")
    logger.info("  - Assign to terrain: splatmap_0 first, then splatmap_1")
    logger.info("\nShazbot! 🔥")


def main() -> int:
    """Main entry point for the application.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Check and install dependencies at startup
        check_and_install_dependencies()
        
        # Configure PIL for large images
        configure_pil_for_large_images()
        
        # Run the CLI
        main_cli()
        return 0
    except KeyboardInterrupt:
        logger.info("\n\nOperation cancelled by user.")
        return 1
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
