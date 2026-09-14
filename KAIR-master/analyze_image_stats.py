import os
import cv2
import numpy as np

def analyze_folder(folder_path):
    print(f"Analyzing folder: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # recursivley find all image files
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                image_files.append(os.path.join(root, file))
                if len(image_files) >= 5: # check first 5 images found
                    break
        if len(image_files) >= 5:
            break

    if not image_files:
        print("No image files found.")
        return

    for file_path in image_files:
        try:
            # Try with cv2 first (it handles 16-bit well)
            img_cv = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img_cv is None:
                 print(f"Skipping {file_path}: cv2 read failed.")
                 continue

            print(f"File: {os.path.basename(file_path)}")
            print(f"  Shape: {img_cv.shape}")
            print(f"  Dtype: {img_cv.dtype}")
            print(f"  Min: {img_cv.min()}")
            print(f"  Max: {img_cv.max()}")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    print("-" * 30)

base_path = 'trainsets/Sen2Venus'
hr_path = os.path.join(base_path, 'HR')
lr_path = os.path.join(base_path, 'LR')

analyze_folder(hr_path)
analyze_folder(lr_path)

