import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_DIR = Path(
    "/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/img-patch-tiles/HR/tiles_512/hurricane-fiona-48cm"
)
OUTPUT_DIR = Path(
    "/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/img-patch-tiles/LR/tiles_256/hurricane-fiona-48cm"
)
SCALE_FACTOR = 2  # Downsample 512 -> 256
BATCH_SIZE = 32  # Adjust based on GPU memory
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Gather all image files
    img_paths = [
        p for p in INPUT_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    print(
        f"Found {len(img_paths)} images. Processing on device: {device.type.upper()}"
    )

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Process in batches
    for i in tqdm(range(0, len(img_paths), BATCH_SIZE), desc="Generating LR Tiles"):
        batch_paths = img_paths[i : i + BATCH_SIZE]
        tensors = []

        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(to_tensor(img))

        # Stack into [B, C, H, W] tensor and push to device
        batch_tensor = torch.stack(tensors).to(device)

        # Apply Bicubic Downsampling
        with torch.no_grad():
            lr_batch = F.interpolate(
                batch_tensor,
                scale_factor=1.0 / SCALE_FACTOR,
                mode="bicubic",
                align_corners=False,
            )

        # Clamp values to valid [0, 1] range to avoid floating-point artifacts
        lr_batch = torch.clamp(lr_batch, 0.0, 1.0)

        # Save generated LR tiles
        for j, p in enumerate(batch_paths):
            lr_img = to_pil(lr_batch[j].cpu())
            lr_img.save(OUTPUT_DIR / p.name)

    print(f"\nSuccessfully generated {len(img_paths)} LR patches at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()