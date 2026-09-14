import rasterio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# UPDATE THIS PATH to your HR file (prefer -ps.tiff or -rgb.png)
hr_path = "/home/hassan/Documents/Hassan/SR Enhancement Work/Datasets/WorldStrat/hr_dataset_p/Amnesty POI-2-3-2/Amnesty POI-2-3-2_ps.tiff"   # or .png if using the ready RGB

print(f"Loading: {hr_path}")

if hr_path.lower().endswith('.png'):
    # Simple handling for the provided RGB PNG
    img = np.array(Image.open(hr_path))
    print(f"PNG Shape: {img.shape}, dtype: {img.dtype}")
    rgb = img
    title = "HR RGB PNG Preview"
else:
    # For TIFF files (ps.tiff or pan.tiff)
    with rasterio.open(hr_path) as src:
        print("=== Full Metadata ===")
        print(src.profile)
        print(f"Bands: {src.count}, Shape: {src.height}x{src.width}, dtype: {src.dtypes[0]}")
        
        data = src.read()  # (bands, height, width)
        print("\n=== Per-band Statistics ===")
        for i in range(src.count):
            band = data[i]
            print(f"Band {i+1}: min={band.min():.2f}, max={band.max():.2f}, "
                  f"mean={band.mean():.2f}, nodata count={np.sum(band == 0)}")
        
        # RGB composite (for 4-band ps.tiff: usually R,G,B,NIR)
        # Improved normalization for better brightness/contrast
        # Correct band ordering for SPOT 6/7 ps.tiff: Blue=1, Green=2, Red=3, NIR=4
        if src.count >= 3:
            blue  = data[0].astype(np.float32)   # Band 1
            green = data[1].astype(np.float32)   # Band 2
            red   = data[2].astype(np.float32)   # Band 3
            
            # Strong percentile stretch (best for satellite visuals)
            def stretch(band):
                p2, p98 = np.percentile(band[band > 0], (2, 98))  # ignore zeros
                return np.clip((band - p2) / (p98 - p2 + 1e-6) * 255, 0, 255)
            
            rgb = np.dstack((
                stretch(red),
                stretch(green),
                stretch(blue)
            )).astype(np.uint8)
        else:
            # Single band (pan)
            rgb = np.dstack((data[0], data[0], data[0]))  # grayscale to RGB

    # Normalize assuming common 0-255 or 0-1 / 12-bit range
    if rgb.max() > 255:
        rgb = np.clip(rgb / rgb.max() * 255, 0, 255)
    rgb = rgb.astype(np.uint8)

# Display
plt.figure(figsize=(12, 6))
plt.imshow(rgb)
plt.title(f"HR Image - {os.path.basename(hr_path)}")
plt.axis('off')
plt.show()

# Save visualized version
save_path = "hr_rgb_visualized_spot67_band_arrangement.png"
Image.fromarray(rgb).save(save_path)
print(f"\nSaved '{save_path}' — check this file!")