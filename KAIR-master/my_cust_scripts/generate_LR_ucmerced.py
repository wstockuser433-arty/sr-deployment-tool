import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ==========================
# CONFIGURATION
# ==========================

# HR_DIR = "/home/hassan/Documents/Hassan/AI/fast_superresolution/KAIR/testsets/my_images/HR"
HR_DIR = "/home/hassan/Documents/Hassan/SR Enhancement Work/Datasets/uc-merced-land-use-dataset/UCMerced_LandUse/Images"
# HR_DIR = "/home/hassan/Documents/Hassan/SR Enhancement Work/Datasets/nwpu_resisc45/Dataset/test/test"
# LR_DIR = "/home/hassan/Documents/Hassan/AI/fast_superresolution/KAIR/testsets/my_images/LR_x4_realistic"

SCALE = 2

# Realistic degradation options
USE_GAUSSIAN_BLUR = False
USE_NOISE = False
USE_JPEG_ARTIFACTS = False

JPEG_QUALITY = 50
NOISE_STD = 5

if USE_GAUSSIAN_BLUR and USE_NOISE and USE_JPEG_ARTIFACTS:
    LR_DIR = "/home/hassan/Documents/Hassan/AI/fast_superresolution/KAIR/testsets/my_images/LR_x2_realistic/uc-merced"
else:
    LR_DIR = "/home/hassan/Documents/Hassan/AI/fast_superresolution/KAIR/testsets/my_images/LR_x2_bicubic/uc-merced"

os.makedirs(LR_DIR, exist_ok=True)


# ==========================
# DEGRADATION FUNCTION
# ==========================

def degrade_image(hr_img):

    h, w = hr_img.shape[:2]

    # Step 1: Gaussian Blur
    if USE_GAUSSIAN_BLUR:
        hr_img = cv2.GaussianBlur(
            hr_img,
            (3, 3),
            sigmaX=1.0
        )

    # Step 2: Bicubic Downsample
    lr = cv2.resize(
        hr_img,
        (w // SCALE, h // SCALE),
        interpolation=cv2.INTER_CUBIC
    )

    # Step 3: Add Sensor Noise
    if USE_NOISE:
        noise = np.random.normal(
            0,
            NOISE_STD,
            lr.shape
        ).astype(np.float32)

        lr = lr.astype(np.float32) + noise
        lr = np.clip(lr, 0, 255).astype(np.uint8)

    # Step 4: JPEG Compression Artifacts
    if USE_JPEG_ARTIFACTS:

        encode_param = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            JPEG_QUALITY
        ]

        success, enc = cv2.imencode(
            ".jpg",
            lr,
            encode_param
        )

        if success:
            lr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return lr


# ==========================
# PROCESS DATASET
# ==========================

hr_files = []

for ext in ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"]:
    hr_files.extend(Path(HR_DIR).rglob(ext))

# hr_files = list(Path(HR_DIR).rglob("*.tif"))

print(f"Found {len(hr_files)} TIFF files")

for hr_path in tqdm(hr_files):

    hr = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)

    if hr is None:
        print(f"Could not read {hr_path}")
        continue

    lr = degrade_image(hr)

    # preserve directory structure
    relative_path = hr_path.relative_to(HR_DIR)

    out_path = Path(LR_DIR) / relative_path

    # create parent directories if needed
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(str(out_path), lr)

print("Done.")