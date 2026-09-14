"""
apply_esrgan_degradation.py

Generate synthetic LR images from HR inputs using learned satellite ESRGAN
degradation parameters saved in best_degradation.json.

Usage
-----
  1) Edit CONFIG below.
  2) Run:  python apply_esrgan_degradation.py

Notes
-----
  - Reuses degrade_satellite() and load_config() from esrgan_mapping_optuna.py.
  - Input images are read as uint8 RGB, degraded in float32 [0,1], and saved
    as uint8 outputs.
"""

import os
import sys
import random
import logging
from typing import List

import cv2

# Ensure KAIR root is importable even when running from pleaides_preprocessing/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.utils_image import imread_uint, uint2single, single2uint

try:
    from pleaides_preprocessing.esrgan_mapping_optuna import degrade_satellite, load_config
except ImportError:
    from esrgan_mapping_optuna import degrade_satellite, load_config


# ===========================================================================
# CONFIG  -  edit this block before running
# ===========================================================================
CONFIG = {
    # Input/Output
    "hr_dir": "E:/FAST/SUPARCO/KAIR/testsets/Set14/hr",
    "lr_dir": "E:/FAST/SUPARCO/KAIR/testsets/Set14/lr",
    "params_path": "best_degradation.json",

    # Processing
    "sf": 2,                     # If omitted in JSON, this value is used
    "seed": 42,                  # For deterministic degradation randomness
    "overwrite": False,          # Skip existing files when False
    "recursive": False,          # Traverse subfolders of hr_dir when True

    # Output control
    "save_ext": None,            # None keeps original extension, e.g. ".png"
}

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _list_images(root: str, recursive: bool) -> List[str]:
    """Return sorted image paths under root."""
    paths: List[str] = []
    if recursive:
        for base, _, files in os.walk(root):
            for name in files:
                if name.lower().endswith(VALID_EXTS):
                    paths.append(os.path.join(base, name))
    else:
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isfile(p) and name.lower().endswith(VALID_EXTS):
                paths.append(p)
    return sorted(paths)


def generate_lr_dataset(
    hr_dir: str,
    lr_dir: str,
    params_path: str,
    sf: int = 2,
    seed: int = 42,
    overwrite: bool = False,
    recursive: bool = False,
    save_ext: str = None,
) -> None:
    """
    Degrade all HR images in hr_dir and save synthetic LR images to lr_dir.

    Parameters
    ----------
    hr_dir : str
        Directory containing HR images.
    lr_dir : str
        Output directory for synthetic LR images.
    params_path : str
        Path to JSON with learned degradation parameters.
    sf : int
        Fallback scale factor if not present in JSON.
    seed : int
        Seed for deterministic stochastic degradations.
    overwrite : bool
        If False, existing outputs are skipped.
    recursive : bool
        If True, traverse hr_dir recursively and preserve folder structure.
    save_ext : str
        Output extension (e.g. ".png"). None keeps input extension.
    """
    if not os.path.isdir(hr_dir):
        raise FileNotFoundError(f"HR directory not found: {hr_dir}")

    params = load_config(params_path)
    params.setdefault("sf", sf)

    os.makedirs(lr_dir, exist_ok=True)
    image_paths = _list_images(hr_dir, recursive=recursive)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {hr_dir}")

    logger.info(f"Found {len(image_paths)} HR image(s) in {hr_dir}")
    logger.info(f"Using params from: {params_path}")

    random.seed(seed)

    processed = 0
    skipped = 0
    failed = 0

    for idx, in_path in enumerate(image_paths, start=1):
        rel = os.path.relpath(in_path, hr_dir)
        rel_dir = os.path.dirname(rel)
        stem, ext = os.path.splitext(os.path.basename(rel))
        out_ext = save_ext if save_ext else ext
        out_dir = os.path.join(lr_dir, rel_dir) if recursive else lr_dir
        out_path = os.path.join(out_dir, f"{stem}{out_ext}")

        if (not overwrite) and os.path.exists(out_path):
            skipped += 1
            continue

        try:
            os.makedirs(out_dir, exist_ok=True)

            hr_u8 = imread_uint(in_path, n_channels=3)
            hr_f32 = uint2single(hr_u8)
            lr_f32, _ = degrade_satellite(hr_f32, **params)
            lr_u8 = single2uint(lr_f32)

            ok = bool(cv2.imwrite(out_path, lr_u8[:, :, ::-1]))
            if not ok:
                raise IOError(f"cv2.imwrite returned False for: {out_path}")

            processed += 1
            if idx % 50 == 0 or idx == len(image_paths):
                logger.info(f"Processed {idx}/{len(image_paths)}")
        except Exception as e:
            failed += 1
            logger.error(f"Failed on {in_path}: {e}")

    logger.info("-" * 60)
    logger.info(f"Done. processed={processed}, skipped={skipped}, failed={failed}")
    logger.info(f"Output directory: {lr_dir}")
    logger.info("-" * 60)


if __name__ == "__main__":
    generate_lr_dataset(
        hr_dir=CONFIG["hr_dir"],
        lr_dir=CONFIG["lr_dir"],
        params_path=CONFIG["params_path"],
        sf=CONFIG["sf"],
        seed=CONFIG["seed"],
        overwrite=CONFIG["overwrite"],
        recursive=CONFIG["recursive"],
        save_ext=CONFIG["save_ext"],
    )
