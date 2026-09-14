import os
from pathlib import Path
from PIL import Image, ImageFile
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import warnings
warnings.filterwarnings('ignore')

print("All libraries loaded successfully")


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


def display_png_overview(image_path: str, scale_factor: int = 8):
    """Display PNG image with histograms and statistics."""
    rgb = load_large_png_tiled(image_path, scale_factor)
    rgb_float = rgb.astype(np.float32)
    
    low = np.percentile(rgb_float, 2)
    high = np.percentile(rgb_float, 98)
    rgb_display = (rgb_float - low) / (high - low)
    rgb_display = np.clip(rgb_display, 0, 1)
    
    # Display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.imshow(rgb_display)
    ax1.set_title(f"Image Preview (downsampled {scale_factor}×, 2-98% stretch)")
    ax1.axis("off")
    
    # Histograms
    for i, label in enumerate(["Red", "Green", "Blue"]):
        ax2.hist(rgb[:, :, i].ravel(), bins=256, alpha=0.4, label=label)
    ax2.set_title("RGB Histograms")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print(f"\n{'='*70}")
    print("IMAGE STATISTICS")
    print(f"{'='*70}")
    print(f"Shape (downsampled): {rgb.shape}")
    for i, label in enumerate(["Red", "Green", "Blue"]):
        ch = rgb[:, :, i]
        print(f"{label:5s}: min={ch.min():3d}  max={ch.max():3d}  mean={ch.mean():7.1f}  std={ch.std():7.1f}")


import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# CHANGE ONLY THIS
# ==================================================
# image_path = r"/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/maxar-dataset-pak/la-wildfire-59cm/103001010C487900-visual.tif"
image_path = r"/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/dc_int_070726/2024032807000002_29187_377939/PRSS-1_PAN_0114_0126_20231207_L3R_0328050129456.tif"
# ==================================================
# This PRSS-1 PAN product is a SINGLE-BAND panchromatic image at
# 79221 x 77364 px (UInt16) -> a full-resolution band read is ~11.4 GiB
# by itself. Reading full-res bands (as the original stats loop did) is
# almost certainly what was OOM-killing the load. Everything below reads
# through the .ovr overviews via `out_shape` instead, so nothing here
# ever pulls the full-resolution array into memory.
# ==================================================
scale_factor = 16  # used for BOTH the stats sampling and the preview/display read

# --------------------------------------------------
# OPEN IMAGE
# --------------------------------------------------

with rasterio.open(image_path) as src:

    print("=" * 80)
    print("IMAGE INFORMATION")
    print("=" * 80)

    print(f"File            : {image_path}")
    print(f"Width           : {src.width}")
    print(f"Height          : {src.height}")
    print(f"Bands           : {src.count}")
    print(f"Data Type       : {src.dtypes}")
    print(f"CRS             : {src.crs}")
    print(f"NoData          : {src.nodata}")
    print(f"ColorInterp     : {src.colorinterp}")

    print("\nAffine Transform")
    print(src.transform)

    print("\n" + "=" * 80)
    print("BAND INFORMATION")
    print("=" * 80)

    print(f"Total Bands     : {src.count}")

    for band_idx in range(1, src.count + 1):
        tags = src.tags(band_idx)

        band_name = (
            src.descriptions[band_idx - 1]
            or tags.get("BAND_NAME")
            or tags.get("band_name")
            or tags.get("NAME")
            or tags.get("name")
            or f"Band {band_idx}"
        )

        band_description = (
            tags.get("DESCRIPTION")
            or tags.get("description")
            or tags.get("DESC")
            or tags.get("desc")
            or "No description available"
        )

        print(f"Band {band_idx}: name='{band_name}', description='{band_description}'")



    print("\n" + "=" * 80)
    print("BAND STATISTICS")
    print("=" * 80)
    print(f"(stats computed on a {scale_factor}x overview-decimated read, not full-res, to avoid OOM)")

    stats_out_h = max(1, src.height // scale_factor)
    stats_out_w = max(1, src.width // scale_factor)

    for band_idx in range(1, src.count + 1):

        # Decimated read: GDAL pulls from the .ovr overviews whenever the
        # requested out_shape is smaller than the full raster, so this
        # stays cheap in memory no matter how large the source file is.
        # (The original `src.read(band_idx)` here read the FULL band --
        # ~11.4 GiB for this single-band 79221x77364 UInt16 PAN image --
        # which is what was almost certainly OOM-killing the load.)
        band = src.read(
            band_idx,
            out_shape=(stats_out_h, stats_out_w),
            resampling=Resampling.average,
        )

        print(
            f"Band {band_idx}: "
            f"min: {band.min():>8} "
            f"max: {band.max():>8} "
            f"mean: {band.mean():>10.2f} "
            f"std: {band.std():>10.2f}"
        )

        print(
            f"P1: {np.percentile(band, 1):.0f}  "
            f"P5: {np.percentile(band, 5):.0f}  "
            f"P95: {np.percentile(band, 95):.0f}  "
            f"P99: {np.percentile(band, 99):.0f}  "
            f"P99.5: {np.percentile(band, 99.5):.0f}  "
            f"P99.9: {np.percentile(band, 99.9):.0f}"
        )


    # --------------------------------------------------
    # Read preview bands (band-count aware)
    # --------------------------------------------------
    # The original code hardcoded `src.read([1, 2, 3])`, which raises an
    # IndexError on this file: PRSS-1_PAN_... is a SINGLE-BAND (Gray)
    # panchromatic product, it has no bands 2 or 3. Pick the band
    # indices based on what's actually in the file instead.
    n_bands = src.count
    is_panchromatic = n_bands == 1
    band_indices = [1, 2, 3] if n_bands >= 3 else [1]

    preview = src.read(
        band_indices,
        out_shape=(
            len(band_indices),
            src.height // scale_factor,
            src.width // scale_factor
        ),
        resampling=Resampling.average,
    )

# --------------------------------------------------
# PREPARE IMAGE FOR DISPLAY
# --------------------------------------------------

preview = np.transpose(preview, (1, 2, 0)).astype(np.float32)  # (H, W, n_bands)

# --------------------------------------------------
# GLOBAL PERCENTILE STRETCH
# (Preserves color balance across bands)
# --------------------------------------------------

low = np.percentile(preview, 2)
high = np.percentile(preview, 98)

preview_display = (preview - low) / (high - low)
preview_display = np.clip(preview_display, 0, 1)

print("\n" + "=" * 80)
print("DISPLAY STRETCH")
print("=" * 80)

print(f"Bands used : {band_indices} ({'panchromatic (grayscale)' if is_panchromatic else 'RGB'})")
print(f"Global P2  : {low:.2f}")
print(f"Global P98 : {high:.2f}")

# --------------------------------------------------
# DISPLAY IMAGE
# --------------------------------------------------

plt.figure(figsize=(12, 12))
if is_panchromatic:
    plt.imshow(preview_display[:, :, 0], cmap="gray")
    plt.title("Panchromatic Preview (Global 2-98% Stretch)")
else:
    plt.imshow(preview_display)
    plt.title("RGB Preview (Global 2-98% Stretch)")
plt.axis("off")
plt.show()

# --------------------------------------------------
# HISTOGRAM(S)
# --------------------------------------------------

plt.figure(figsize=(12, 6))

if is_panchromatic:
    plt.hist(preview[:, :, 0].ravel(), bins=256, alpha=0.7, label="Pan", color="gray")
    plt.title("Panchromatic Histogram")
else:
    for i, label in enumerate(["Red", "Green", "Blue"]):
        plt.hist(preview[:, :, i].ravel(), bins=256, alpha=0.4, label=label)
    plt.title("RGB Histograms")

plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# --------------------------------------------------
# CHANNEL MEANS SUMMARY
# --------------------------------------------------

print("\n" + "=" * 80)
print("CHANNEL MEANS")
print("=" * 80)

if is_panchromatic:
    print(f"Pan Mean   : {preview[:, :, 0].mean():.2f}")
else:
    for i, label in enumerate(["Red", "Green", "Blue"]):
        print(f"{label} Mean{' ' * (3 - len(label))}: {preview[:, :, i].mean():.2f}")