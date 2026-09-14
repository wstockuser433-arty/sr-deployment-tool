from pathlib import Path 
import numpy as np
from PIL import Image, ImageEnhance

# Define your input and output root paths here
INPUT_DIR = Path("/home/hassan/Documents/Hassan/SR Enhancement Work/Datasets/WorldStrat/hr_dataset_p/Amnesty POI-2-3-2")
OUTPUT_DIR = Path("/home/hassan/Documents/Hassan/AI/fast_superresolution/KAIR/my_cust_scripts")

def enhance_image(input_path, output_path):
    # Ensure paths are strings or path-like objects for PIL
    img = Image.open(input_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    
    # Percentile stretch
    p2, p98 = np.percentile(img_array, (2, 98))
    stretched = np.clip((img_array - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
    
    enhanced = Image.fromarray(stretched)
    
    # # Optional: boost brightness/contrast
    # enhancer = ImageEnhance.Brightness(enhanced)
    # enhanced = enhancer.enhance(1.2)
    # enhancer = ImageEnhance.Contrast(enhanced)
    # enhanced = enhancer.enhance(1.3)
    
    # Optional: Automatically create output folder if it doesn't exist yet
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    enhanced.save(output_path)
    print(f"Enhanced saved to: {output_path}")

# --- Clean Usage ---
# The / operator combines the root path with the file name cleanly.
in_file = INPUT_DIR / "Amnesty POI-2-3-2_rgb.png"
out_file = OUTPUT_DIR / "Amnesty POI-2-3-2_rgb_enhanced.png"

enhance_image(in_file, out_file)
