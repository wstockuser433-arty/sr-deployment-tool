import numpy as np
from pathlib import Path
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils_image as util
from PIL import Image, ImageDraw, ImageFont

# ===== IMP: CHANGE THIS METRICS IF SR MODEL UPSCALE PARAMETER
# WAS DIFFERENT FOR SR-MODEL FROM WHICH SR OUTPUT WAS GENERATED
UPSCALE = 2

def calculate_metrics(sr: np.ndarray, hr: np.ndarray, border: int):
    """Calculates all 8 evaluation metrics between SR and HR numpy arrays."""
    return {
        "psnr": util.calculate_psnr(sr, hr, border=border),
        "ssim": util.calculate_ssim(sr, hr, border=border),
        "it_ssim": util.calculate_it_ssim(sr, hr, border=border),
        "sam": util.calculate_sam(sr, hr, border=border),
        "uiqi": util.calculate_uiqi(sr, hr, border=border),
        "rmse": util.calculate_rmse(sr, hr, border=border),
        "fsim": util.calculate_fsim(sr, hr, border=border),
        "srer": util.calculate_srer(sr, hr, border=border),
    }


def create_comparison_images(lr_folder, sr_folder, hr_folder, output_folder):
    lr_dir = Path(lr_folder)
    sr_dir = Path(sr_folder)
    hr_dir = Path(hr_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    border_metric = UPSCALE

    # 1. Map LR and HR files by their direct file stems
    lr_files = {f.stem: f for f in lr_dir.iterdir() if f.suffix.lower() in valid_extensions}
    hr_files = {f.stem: f for f in hr_dir.iterdir() if f.suffix.lower() in valid_extensions}
    
    # 2. Map SR files, but strip out the '_SwinIR' suffix from the lookup key
    sr_files = {}
    for f in sr_dir.iterdir():
        if f.suffix.lower() in valid_extensions:
            # Strip '_SwinIR' if present so it matches the LR/HR base stem
            clean_stem = f.stem.replace("_SwinIR", "")
            sr_files[clean_stem] = f
    # print(lr_files.keys())
    # print(hr_files.keys())
    # print(sr_files.keys())

    # Find common stems across all three mapped sets
    common_stems = set(lr_files.keys()) & set(sr_files.keys()) & set(hr_files.keys())

    if not common_stems:
        print("No matching filenames found across all three folders!")
        return
    
    # print(common_stems)
    print(f"Found {len(common_stems)} matching image sets. Processing...\n")

    for idx, stem in enumerate(common_stems, start=1):
        img_lr = Image.open(lr_files[stem]).convert("RGB")
        img_sr = Image.open(sr_files[stem]).convert("RGB")
        img_hr = Image.open(hr_files[stem]).convert("RGB")
        sr_arr = np.array(img_sr)
        hr_arr = np.array(img_hr)
        metrics = calculate_metrics(sr_arr, hr_arr, border=border_metric)

        print(f"Image #{idx}: {stem}")
        print("==========================================================================")
        print(f"PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f} | IT-SSIM: {metrics['it_ssim']:.4f} | SAM: {metrics['sam']:.4f}")
        print(f"UIQI: {metrics['uiqi']:.4f} | RMSE: {metrics['rmse']:.4f} | FSIM: {metrics['fsim']:.4f} | SRER: {metrics['srer']:.2f}")
        print("==========================================================================")

        # Base dimension on HR image
        target_width, target_height = img_hr.size

        # Resize LR and SR to match HR dimensions if needed
        if img_lr.size != (target_width, target_height):
            img_lr = img_lr.resize((target_width, target_height), Image.Resampling.LANCZOS)
        if img_sr.size != (target_width, target_height):
            img_sr = img_sr.resize((target_width, target_height), Image.Resampling.LANCZOS)

        images = [img_lr, img_sr, img_hr]
        labels = ["LR", "SR", "HR"]

        # Add visual label box in the corner
        for img, label in zip(images, labels):
            draw = ImageDraw.Draw(img)
            box_w = int(target_width * 0.10)
            box_h = int(target_height * 0.08)
            
            draw.rectangle([0, 0, box_w, box_h], fill="black")
            
            try:
                font = ImageFont.truetype("arialbd.ttf", int(box_h * 0.8))
            except IOError:
                font = ImageFont.load_default()

            draw.text((int(box_w * 0.15), int(box_h * 0.15)), label, fill="white", font=font)

        # Concatenate side-by-side --- BORDER CONFIGURATION ---
        border_width = 10  # Width of the white divider in pixels
        combined_width = (target_width * 3) + (border_width * 2)
        combined_image = Image.new("RGB", (combined_width, target_height), color="white")

        # 1. LR goes at the very left edge (x = 0)
        combined_image.paste(img_lr, (0, 0))
        
        # 2. SR is shifted by the width of 1 image + 1 border
        sr_x_offset = target_width + border_width
        combined_image.paste(img_sr, (sr_x_offset, 0))
        
        # 3. HR is shifted by the width of 2 images + 2 borders
        hr_x_offset = (target_width * 2) + (border_width * 2)
        combined_image.paste(img_hr, (hr_x_offset, 0))

        output_path = out_dir / f"{stem}_comparison.png"
        combined_image.save(output_path)
        print(f"Saved: {output_path.name}\n")
        
    print("All comparisons generated successfully!")

if __name__ == "__main__":
    LR_PATH = "/home/hassan/Documents/Hassan/AI/fast_superresolution/Evaluations_done/09-comparison-images-created/LR-images"
    SR_PATH = "/home/hassan/Documents/Hassan/AI/fast_superresolution/Evaluations_done/09-comparison-images-created/SR-images"
    HR_PATH = "/home/hassan/Documents/Hassan/AI/fast_superresolution/Evaluations_done/09-comparison-images-created/HR-images"
    OUTPUT_PATH = "/home/hassan/Documents/Hassan/AI/fast_superresolution/Evaluations_done/09-comparison-images-created/Outputs"

    create_comparison_images(LR_PATH, SR_PATH, HR_PATH, OUTPUT_PATH)
