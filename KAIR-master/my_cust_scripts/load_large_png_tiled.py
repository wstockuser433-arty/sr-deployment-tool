import os
from pathlib import Path
from PIL import Image, ImageFile
import numpy as np
import matplotlib.pyplot as plt

image_path = r"/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/Niagara Falls, Canada, very high resolution satellite image.jpg"
scale_factor = 8  # Downsampling factor (e.g., 8 = 1/8 size)

def load_large_png_tiled(image_path: str, scale_factor: int = 8, tile_size: int = 512):
    """
    Memory-efficient PNG loader using tiling + downsampling.
    
    Parameters
    ----------
    image_path : str
        Path to PNG file
    scale_factor : int
        Downsampling factor (e.g., 8 = 1/8 size)
    tile_size : int
        Tile size for reading blocks (prevent OOM)
    
    Returns
    -------
    rgb : np.ndarray (H, W, 3) uint8
        Downsampled RGB image
    """
    # Allow valid large satellite PNGs by disabling PIL size limits for this loader only
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(image_path)
    
    # Get original dimensions
    orig_width, orig_height = img.size
    downsampled_width = orig_width // scale_factor
    downsampled_height = orig_height // scale_factor
    
    print(f"[PNG Tiled Loader]")
    print(f"  Original size: {orig_width} × {orig_height}")
    print(f"  Scale factor: {scale_factor}×")
    print(f"  Output size: {downsampled_width} × {downsampled_height}")
    print(f"  Tile size: {tile_size}")
    
    # Create output array
    rgb_out = np.zeros((downsampled_height, downsampled_width, 3), dtype=np.uint8)
    
    # Process image in tiles
    num_tiles_h = (orig_height + tile_size - 1) // tile_size
    num_tiles_w = (orig_width + tile_size - 1) // tile_size
    total_tiles = num_tiles_h * num_tiles_w
    
    tile_count = 0
    for tile_y in range(0, orig_height, tile_size):
        for tile_x in range(0, orig_width, tile_size):
            tile_count += 1
            if tile_count % max(1, total_tiles // 10) == 0 or tile_count == total_tiles:
                print(f"  Progress: {tile_count}/{total_tiles} tiles")
            if tile_count % max(1, total_tiles // 10) == 0 or tile_count == total_tiles:
                print(f"  Progress: {tile_count}/{total_tiles} tiles")
            
            # Read tile region
            x1, y1 = tile_x, tile_y
            x2 = min(tile_x + tile_size, orig_width)
            y2 = min(tile_y + tile_size, orig_height)
            
            tile = img.crop((x1, y1, x2, y2))
            tile_array = np.array(tile, dtype=np.uint8)
            
            # Handle grayscale → RGB
            if tile_array.ndim == 2:
                tile_array = np.stack([tile_array] * 3, axis=2)
            elif tile_array.shape[2] == 4:  # RGBA → RGB
                tile_array = tile_array[:, :, :3]
            
            # Downsample tile
            h, w = tile_array.shape[:2]
            h_down = max(1, h // scale_factor)
            w_down = max(1, w // scale_factor)
            tile_down = np.array(
                Image.fromarray(tile_array).resize((w_down, h_down), Image.Resampling.LANCZOS)
            )
            
            # Write to output
            out_y1 = (tile_y // scale_factor)
            out_x1 = (tile_x // scale_factor)
            out_y2 = out_y1 + h_down
            out_x2 = out_x1 + w_down
            rgb_out[out_y1:out_y2, out_x1:out_x2] = tile_down
    
    print(f"  ✓ Loaded successfully")
    return rgb_out


def auto_load_image(image_path: str, scale_factor: int = 8):
    """
    Auto-detect format and load image efficiently.
    
    - PNG files: Use tiled PIL loader
    - JP2/GeoTIFF: Use rasterio (existing code)
    """
    ext = Path(image_path).suffix.lower()
    
    if ext in [".png", ".jpg", ".jpeg"]:
        print(f"[Auto-Detect] PNG detected → using tiled PIL loader")
        return load_large_png_tiled(image_path, scale_factor), "png"
    elif ext in [".jp2", ".tif", ".tiff"]:
        print(f"[Auto-Detect] {ext.upper()} detected → using rasterio")
        return None, ext  # Signal to use rasterio path
    else:
        raise ValueError(f"Unsupported format: {ext}")


result, fmt = auto_load_image(image_path, scale_factor)

rgb = result
print(f"\nLoaded PNG: shape={rgb.shape}, dtype={rgb.dtype}")

# --------------------------------------------------
# NORMALIZE & DISPLAY
# --------------------------------------------------
rgb_display = rgb.astype(np.float32)

low = np.percentile(rgb_display, 2)
high = np.percentile(rgb_display, 98)

rgb_display = (rgb_display - low) / (high - low)
rgb_display = np.clip(rgb_display, 0, 1)

print(f"Contrast stretch: P2={low:.1f}, P98={high:.1f}")

# Display
plt.figure(figsize=(14, 14))
plt.imshow(rgb_display)
plt.title(f"PNG Preview (downsampled {scale_factor}×, global 2-98% stretch)")
plt.axis("off")
plt.tight_layout()
plt.show()

# Histograms
plt.figure(figsize=(12, 5))
for i, label in enumerate(["Red", "Green", "Blue"]):
    plt.hist(rgb[:, :, i].ravel(), bins=256, alpha=0.4, label=label)
plt.title("RGB Histograms")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# Statistics
print("\n" + "=" * 80)
print("CHANNEL STATISTICS (after downsampling)")
print("=" * 80)
for i, label in enumerate(["Red", "Green", "Blue"]):
    ch = rgb[:, :, i]
    print(f"{label:5s}: min={ch.min():3d}  max={ch.max():3d}  mean={ch.mean():7.1f}  std={ch.std():7.1f}")