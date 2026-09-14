"""
run_pipeline.py
===============
KAIR-style preprocessing pipeline for generating LR/HR image pairs.

Two modes are supported:

1. **HR-only mode** (pipeline_mode="hr_only"):
   Reads HR images, optionally applies preprocessing (normalization, cloud masking),
   then degrades them via one of four degradation models to generate LR images.
   Saves both processed HR and degraded LR to separate directories.

2. **HR/LR pair mode** (pipeline_mode="hr_lr_pair"):
   Reads existing HR/LR image pairs, applies the same preprocessing (normalization,
   cloud masking) to BOTH HR and LR images (no degradation). Useful for preprocessing
   pre-existing training pairs. HR and LR images must have matching filenames.

Usage
-----
    python preprocessing_pipeline/run_pipeline.py --config preprocessing_pipeline/config.json

All parameters are controlled from the JSON config file (KAIR comment-style
``//`` comments are supported).
"""

import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from preprocessing_pipeline.degradation_utils import (
    degrade_bsrgan,
    degrade_bsrgan_plus,
    degrade_real_esrgan,
    degrade_satellite,
    imread_uint,
    imsave,
    single2uint,
    uint2single,
)
from preprocessing_pipeline.other_utils import (
    apply_s2cloudless_mask,
    satellite_pre_norm,
)


# ===================================================================
# Config helpers  (KAIR-style JSON with ``//`` comments)
# ===================================================================

def _parse_json(path: str) -> OrderedDict:
    """Read a KAIR-style JSON file, stripping ``//`` comments."""
    json_str = ""
    with open(path, "r") as f:
        for line in f:
            line = line.split("//")[0] + "\n"
            json_str += line
    return json.loads(json_str, object_pairs_hook=OrderedDict)


def _resolve_path(path: str, root: str) -> str:
    """If *path* is relative, resolve it against *root*."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root, path))


# ===================================================================
# Image discovery
# ===================================================================

def _discover_images(input_dir: str, extensions: list) -> list:
    """Return a sorted list of absolute image paths.

    Handles three cases automatically:
    1. input_dir is a single image file  → returns [input_dir]
    2. input_dir is a flat directory     → returns all matching files in that dir
    3. input_dir has class subfolders    → flat mode finds nothing; caller (backend)
       splits per-class before calling this function, so each call is case 2.
    """
    extensions = {e.lower() for e in extensions}
    input_path = os.path.abspath(input_dir)

    # Case 1: single file
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        return [input_path] if ext in extensions else []

    if not os.path.isdir(input_path):
        return []

    # Case 2: flat directory
    paths = []
    for fname in sorted(os.listdir(input_path)):
        fpath = os.path.join(input_path, fname)
        if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in extensions:
            paths.append(fpath)
    return paths


# ===================================================================
# Preprocessing helper
# ===================================================================

def _apply_preprocessing(img: np.ndarray, cfg: dict, fname: str) -> np.ndarray:
    """Apply cloud masking and normalization preprocessing to an image.

    Parameters
    ----------
    img : np.ndarray
        Input image (uint8 or uint16).
    cfg : dict
        Configuration dictionary.
    fname : str
        Filename for logging purposes.

    Returns
    -------
    np.ndarray
        Preprocessed uint8 image.
    """
    # ---- Cloud masking (Sentinel-2 only) ----
    if cfg.get("cloud_mask_enabled", False):
        # apply_s2cloudless_mask expects (1, H, W, 10)
        if img.ndim == 3 and img.shape[2] == 10:
            img_4d = img[np.newaxis, ...]
            img_masked = apply_s2cloudless_mask(
                img_4d,
                nodata=cfg.get("cloud_mask_nodata", 0.0),
                auto_scale=cfg.get("cloud_mask_auto_scale", True),
                threshold=cfg.get("cloud_mask_threshold", 0.4),
                average_over=cfg.get("cloud_mask_average_over", 4),
                dilation_size=cfg.get("cloud_mask_dilation_size", 2),
            )
            # img_masked is float32 (H, W, 10) – convert back to uint8
            img = single2uint(img_masked)
        else:
            print(f"  [WARN] Cloud masking enabled but image {fname} does not "
                  f"have 10 bands (shape={img.shape}). Skipping cloud mask.")

    # ---- Percentile normalization ----
    if cfg.get("normalize_enabled", False):
        img = satellite_pre_norm(
            img,
            low_percentile=cfg.get("normalize_low_percentile", 2),
            high_percentile=cfg.get("normalize_high_percentile", 98),
        )

    return img


# ===================================================================
# Single-image worker
# ===================================================================

def _process_one(args: tuple) -> str:
    """Process a single image (or HR/LR pair).  Designed for use with ``Pool.map``.

    Parameters
    ----------
    args : tuple
        (hr_path, lr_path_or_none, config_dict, worker_seed)

    Returns
    -------
    str
        Status message.
    """
    hr_path, lr_path, cfg, worker_seed = args

    # Per-worker reproducibility
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))

    fname = os.path.basename(hr_path)
    name, _ = os.path.splitext(fname)
    save_ext = "." + cfg["save_format"]

    sf = cfg["scale"]
    n_channels = cfg["n_channels"]
    pipeline_mode = cfg.get("pipeline_mode", "hr_only")

    # ---- Read HR image ----
    img_hr = imread_uint(hr_path, n_channels)

    # ---- Apply preprocessing to HR ----
    img_hr = _apply_preprocessing(img_hr, cfg, fname)

    if pipeline_mode == "hr_lr_pair":
        # ===================================================================
        # HR/LR Pair Mode: preprocess both HR and LR, no degradation
        # ===================================================================
        if lr_path is None:
            raise ValueError(
                f"[{fname}] hr_lr_pair mode requires input_lr_dir to be set, "
                "but no matching LR image was found."
            )

        # Read LR image
        img_lr = imread_uint(lr_path, n_channels)

        # Apply preprocessing to LR
        lr_fname = os.path.basename(lr_path)
        img_lr = _apply_preprocessing(img_lr, cfg, lr_fname)

        # Convert to float for consistency
        img_hr_f = uint2single(img_hr)
        img_lr_f = uint2single(img_lr)

        # Save both (no degradation, no mod-crop)
        hr_out_path = os.path.join(cfg["_abs_output_hr_dir"], name + save_ext)
        lr_out_path = os.path.join(cfg["_abs_output_lr_dir"], name + save_ext)

        imsave(img_hr, hr_out_path)
        imsave(img_lr, lr_out_path)

        return f"  [{fname}] PAIR | HR {img_hr.shape[:2]} | LR {img_lr.shape[:2]}"

    else:
        # ===================================================================
        # HR-only Mode: preprocess HR, then degrade to generate LR
        # ===================================================================

        # Convert to float32 [0, 1]
        img_f = uint2single(img_hr)

        # ---- Select and call degradation ----
        deg_type = cfg["degradation_type"]

        if deg_type == "bsrgan":
            deg_cfg = cfg.get("bsrgan", {})
            img_lq, img_hq = degrade_bsrgan(
                img_f,
                sf=sf,
                jpeg_prob=deg_cfg.get("jpeg_prob", 0.9),
                scale2_prob=deg_cfg.get("scale2_prob", 0.25),
                isp_prob=deg_cfg.get("isp_prob", 0.25),
                noise_level1=deg_cfg.get("noise_level1", 2),
                noise_level2=deg_cfg.get("noise_level2", 25),
            )

        elif deg_type == "real_esrgan":
            deg_cfg = cfg.get("real_esrgan", {})
            img_lq, img_hq = degrade_real_esrgan(
                img_f,
                sf=sf,
                blur_prob_1=deg_cfg.get("blur_prob_1", 1.0),
                resize_prob_1=deg_cfg.get("resize_prob_1", 1.0),
                gaussian_noise_prob_1=deg_cfg.get("gaussian_noise_prob_1", 0.5),
                poisson_noise_prob_1=deg_cfg.get("poisson_noise_prob_1", 0.1),
                speckle_noise_prob_1=deg_cfg.get("speckle_noise_prob_1", 0.1),
                jpeg_prob_1=deg_cfg.get("jpeg_prob_1", 0.9),
                noise_level1_s1=deg_cfg.get("noise_level1_s1", 2),
                noise_level2_s1=deg_cfg.get("noise_level2_s1", 25),
                blur_prob_2=deg_cfg.get("blur_prob_2", 0.8),
                resize_prob_2=deg_cfg.get("resize_prob_2", 1.0),
                gaussian_noise_prob_2=deg_cfg.get("gaussian_noise_prob_2", 0.5),
                poisson_noise_prob_2=deg_cfg.get("poisson_noise_prob_2", 0.1),
                speckle_noise_prob_2=deg_cfg.get("speckle_noise_prob_2", 0.1),
                jpeg_prob_2=deg_cfg.get("jpeg_prob_2", 0.8),
                noise_level1_s2=deg_cfg.get("noise_level1_s2", 2),
                noise_level2_s2=deg_cfg.get("noise_level2_s2", 15),
                final_jpeg_prob=deg_cfg.get("final_jpeg_prob", 0.5),
                resize_back_prob=deg_cfg.get("resize_back_prob", 0.5),
                isp_prob=deg_cfg.get("isp_prob", 0.1),
            )

        elif deg_type == "bsrgan_plus":
            deg_cfg = cfg.get("bsrgan_plus", {})
            img_lq, img_hq = degrade_bsrgan_plus(
                img_f,
                sf=sf,
                shuffle_prob=deg_cfg.get("shuffle_prob", 0.5),
                use_sharp=deg_cfg.get("use_sharp", False),
                sharpening_weight=deg_cfg.get("sharpening_weight", 0.5),
                sharpening_radius=deg_cfg.get("sharpening_radius", 50),
                sharpening_threshold=deg_cfg.get("sharpening_threshold", 10),
                poisson_prob=deg_cfg.get("poisson_prob", 0.1),
                speckle_prob=deg_cfg.get("speckle_prob", 0.1),
                isp_prob=deg_cfg.get("isp_prob", 0.1),
                noise_level1=deg_cfg.get("noise_level1", 2),
                noise_level2=deg_cfg.get("noise_level2", 25),
            )

        elif deg_type == "satellite":
            deg_cfg = cfg.get("satellite", {})
            img_lq, img_hq = degrade_satellite(
                img_f,
                sf=sf,
                blur_prob_1=deg_cfg.get("blur_prob_1", 1.0),
                blur_type_1=deg_cfg.get("blur_type_1", "mtf"),
                resize_prob_1=deg_cfg.get("resize_prob_1", 0.75),
                poisson_prob_1=deg_cfg.get("poisson_prob_1", 0.75),
                read_noise_prob_1=deg_cfg.get("read_noise_prob_1", 0.55),
                haze_prob_1=deg_cfg.get("haze_prob_1", 0.45),
                jpeg_prob_1=deg_cfg.get("jpeg_prob_1", 0.12),
                blur_prob_2=deg_cfg.get("blur_prob_2", 0.92),
                blur_type_2=deg_cfg.get("blur_type_2", "mtf"),
                resize_prob_2=deg_cfg.get("resize_prob_2", 0.70),
                poisson_prob_2=deg_cfg.get("poisson_prob_2", 0.60),
                read_noise_prob_2=deg_cfg.get("read_noise_prob_2", 0.45),
                haze_prob_2=deg_cfg.get("haze_prob_2", 0.35),
                jpeg_prob_2=deg_cfg.get("jpeg_prob_2", 0.08),
                final_jpeg_prob=deg_cfg.get("final_jpeg_prob", 0.10),
                resize_back_prob=deg_cfg.get("resize_back_prob", 0.35),
                isp_prob=deg_cfg.get("isp_prob", 0.08),
                noise_level1=deg_cfg.get("noise_level1", 0.8),
                noise_level2=deg_cfg.get("noise_level2", 10.0),
                mtf_sigma_optics_range=tuple(deg_cfg.get("mtf_sigma_optics_range", [0.8, 2.8])),
                mtf_detector_width_range=tuple(deg_cfg.get("mtf_detector_width_range", [0.7, 1.8])),
                mtf_atm_sigma_range=tuple(deg_cfg.get("mtf_atm_sigma_range", [0.4, 1.8])),
            )

        else:
            raise ValueError(
                f"Unknown degradation_type '{deg_type}'. "
                "Choose from: bsrgan | real_esrgan | bsrgan_plus | satellite"
            )

        # ---- Save outputs ----
        lr_path = os.path.join(cfg["_abs_output_lr_dir"], name + save_ext)
        imsave(single2uint(img_lq), lr_path)

        if cfg.get("save_hr_copy", True):
            hr_path = os.path.join(cfg["_abs_output_hr_dir"], name + save_ext)
            imsave(single2uint(img_hq), hr_path)

        return f"  [{fname}] LR {img_lq.shape[:2]} | HR {img_hq.shape[:2]}"


# ===================================================================
# Main entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KAIR preprocessing pipeline – generate LR/HR pairs from HR images."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "config.json"),
        help="Path to the pipeline config JSON file.",
    )
    args = parser.parse_args()

    # ---- Load config ----
    cfg = _parse_json(args.config)
    project_root = _PROJECT_ROOT

    print("=" * 65)
    print(f"  KAIR Preprocessing Pipeline  –  {cfg.get('task', 'unnamed')}")
    print("=" * 65)

    # ---- Resolve paths relative to project root ----
    pipeline_mode = cfg.get("pipeline_mode", "hr_only")
    input_hr_dir = _resolve_path(cfg["input_hr_dir"], project_root)
    output_hr_dir = _resolve_path(cfg["output_hr_dir"], project_root)
    output_lr_dir = _resolve_path(cfg["output_lr_dir"], project_root)

    os.makedirs(output_hr_dir, exist_ok=True)
    os.makedirs(output_lr_dir, exist_ok=True)

    # Store resolved absolute paths for workers
    cfg["_abs_output_hr_dir"] = output_hr_dir
    cfg["_abs_output_lr_dir"] = output_lr_dir

    # ---- Seed ----
    seed = cfg.get("seed", None)
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed % (2**32))

    # ---- Discover images ----
    extensions = cfg.get("supported_extensions", [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"])
    hr_image_paths = _discover_images(input_hr_dir, extensions)

    if not hr_image_paths:
        print(f"[ERROR] No HR images found in '{input_hr_dir}' with extensions {extensions}")
        sys.exit(1)

    # ---- Mode-specific setup ----
    if pipeline_mode == "hr_lr_pair":
        # HR/LR pair mode: discover LR images and match by filename
        input_lr_dir = cfg.get("input_lr_dir") or None
        if not input_lr_dir:
            print(f"[ERROR] hr_lr_pair mode requires 'input_lr_dir' to be set in config.")
            sys.exit(1)

        input_lr_dir = _resolve_path(input_lr_dir, project_root)
        lr_image_paths = _discover_images(input_lr_dir, extensions)

        # Build LR lookup by basename (without extension)
        lr_lookup = {}
        for lr_path in lr_image_paths:
            lr_basename = os.path.splitext(os.path.basename(lr_path))[0]
            lr_lookup[lr_basename] = lr_path

        # Match HR images with LR images
        image_pairs = []
        unmatched = []
        for hr_path in hr_image_paths:
            hr_basename = os.path.splitext(os.path.basename(hr_path))[0]
            lr_path = lr_lookup.get(hr_basename)
            if lr_path:
                image_pairs.append((hr_path, lr_path))
            else:
                unmatched.append(hr_basename)

        if unmatched:
            print(f"[WARN] {len(unmatched)} HR images have no matching LR image:")
            for name in unmatched[:5]:
                print(f"  - {name}")
            if len(unmatched) > 5:
                print(f"  ... and {len(unmatched) - 5} more")

        if not image_pairs:
            print(f"[ERROR] No matching HR/LR pairs found.")
            sys.exit(1)

        print(f"  Mode             : hr_lr_pair (preprocess both HR and LR)")
        print(f"  Input HR dir     : {input_hr_dir}")
        print(f"  Input LR dir     : {input_lr_dir}")
        print(f"  Matched pairs    : {len(image_pairs)}")

    else:
        # HR-only mode: degrade HR to generate LR
        image_pairs = [(hr_path, None) for hr_path in hr_image_paths]

        print(f"  Mode             : hr_only (preprocess HR, then degrade->LR)")
        print(f"  Degradation type : {cfg['degradation_type']}")
        print(f"  Scale factor     : x{cfg['scale']}")
        print(f"  Input HR dir     : {input_hr_dir}")

    print(f"  Channels         : {cfg['n_channels']}")
    print(f"  Normalize        : {cfg.get('normalize_enabled', False)}")
    print(f"  Cloud masking    : {cfg.get('cloud_mask_enabled', False)}")
    print(f"  Output HR dir    : {output_hr_dir}")
    print(f"  Output LR dir    : {output_lr_dir}")
    print(f"  Images to process: {len(image_pairs)}")
    print(f"  Seed             : {seed}")
    print("-" * 65)

    # ---- Prepare worker arguments ----
    # Each worker gets a unique seed derived from the base seed
    worker_args = []
    for idx, (hr_path, lr_path) in enumerate(image_pairs):
        worker_seed = seed + idx
        worker_args.append((hr_path, lr_path, cfg, worker_seed))

    # ---- Process ----
    num_workers = cfg.get("num_workers", 1)
    t0 = time.time()

    if num_workers <= 1:
        # Sequential
        for idx, wa in enumerate(worker_args, 1):
            msg = _process_one(wa)
            print(f"  [{idx}/{len(worker_args)}] {msg}")
    else:
        # Parallel
        with Pool(processes=num_workers) as pool:
            results = pool.imap(_process_one, worker_args)
            for idx, msg in enumerate(results, 1):
                print(f"  [{idx}/{len(worker_args)}] {msg}")

    elapsed = time.time() - t0
    print("-" * 65)
    print(f"  Done – {len(image_pairs)} images processed in {elapsed:.1f}s")
    print(f"  LR saved to: {output_lr_dir}")
    if cfg.get("save_hr_copy", True):
        print(f"  HR saved to: {output_hr_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()

