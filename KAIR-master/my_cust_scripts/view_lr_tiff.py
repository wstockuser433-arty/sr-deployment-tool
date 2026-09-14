import rasterio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

tiff_path = "/home/hassan/Documents/Hassan/SR Enhancement Work/Datasets/WorldStrat/lr_d/Amnesty POI-1-1-1/L2A/Amnesty POI-1-1-1-1-L2A_data.tiff"  # <-- UPDATE THIS PATH

with rasterio.open(tiff_path) as src:
    print("=== Full Metadata ===")
    print(src.profile)
    print(f"Bands: {src.count}, Shape: {src.height}x{src.width}, dtype: {src.dtypes[0]}")
    
    # Read ALL bands for diagnostics
    data = src.read()  # shape: (12, 157, 159)
    print("\n=== Per-band Statistics (important!) ===")
    for i in range(src.count):
        band = data[i]
        print(f"Band {i+1}: min={band.min():.4f}, max={band.max():.4f}, mean={band.mean():.4f}, "
              f"nodata count={np.sum(band == 0)}")

    # Try common RGB combinations for Sentinel-2 L2A (bands are usually ordered B01 to B12, skipping some)
    # Natural color: typically B04 (Red), B03 (Green), B02 (Blue)
    red = data[3]   # 0-based index → Band 4
    green = data[2] # Band 3
    blue = data[1]  # Band 2
    
    # Stack and normalize properly for float32 (0-1 range)
    rgb = np.dstack((red, green, blue))
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)  # Assuming 0-1 reflectance

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title("RGB Composite (B4-B3-B2)")
    plt.axis('off')
    
    # Optional: Show one single band (e.g., NIR if available)
    plt.subplot(1, 2, 2)
    plt.imshow(data[7] if data.shape[0] > 7 else data[0], cmap='gray')  # Try NIR around band 8
    plt.title("Single Band (e.g. NIR)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # Save for use in SR pipeline
    Image.fromarray(rgb).save("lr_rgb_visualized.png")
    print("\nSaved 'lr_rgb_visualized.png' — check this file!")