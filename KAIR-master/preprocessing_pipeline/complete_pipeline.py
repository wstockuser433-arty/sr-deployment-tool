"""
complete_pipeline.py
====================
KAIR-style preprocessing pipeline for Level-2 Paired Satellite Imagery.

Processing Steps (each independently configurable):

1. **Cloud / Shadow / Snow Masking**
   - QA-band based (Sentinel-2 SCL classes) or s2cloudless ML detector.
   - Produces a binary mask (1 = valid, 0 = invalid) carried through
     all subsequent steps.

2. **Relative Normalization**
   - Adjusts the radiometric distribution of one image to match the other.
   - Methods: histogram matching (CDF transfer) or mean/std transfer.
   - Mask-aware: statistics computed only on valid pixels.

3. **Absolute Normalization (Percentile Clipping)**
   - Clips at configurable percentiles and stretches to [0, 255].
   - Mask-aware: percentiles from valid pixels only.

4. **Spatial Co-registration (ECC)**
   - Aligns LR to HR via OpenCV Enhanced Correlation Coefficient.
   - Supports translation / euclidean / affine / homography warp modes.
   - Optionally warps the mask with the same transform.

5. **Degradation (Optional)**
   - Applies realistic image degradations to HR to generate synthetic LR.
   - Supports 4 degradation models: bsrgan, real_esrgan, bsrgan_plus, satellite.
   - Each model has fully configurable parameters (blur, noise, compression, etc.).
   - If enabled, replaces the input LR with the degraded version.

6. **Mask-Aware Tiling & Filtering**
   - Sliding-window crop; tiles exceeding a configurable invalid-pixel
     ratio are discarded.
   - Saves matching HR/LR tile pairs with identical filenames.

Usage
-----
    python preprocessing_pipeline/complete_pipeline.py \\
        --config preprocessing_pipeline/config_l2.json
"""

import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict
from multiprocessing import Pool

import cv2
import numpy as np

# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from preprocessing_pipeline.degradation_utils import (
    imread_uint,
    imsave,
    single2uint,
    uint2single,
    degrade_bsrgan,
    degrade_bsrgan_plus,
    degrade_real_esrgan,
    degrade_satellite,
)
from preprocessing_pipeline.other_utils import (
    apply_l2_qa_mask,
    align_images_ecc,
    satellite_pre_norm_masked,
    satellite_pre_norm,
    relative_normalize,
)


# ===================================================================
# Helpers
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


def _discover_images(input_dir: str, extensions: list) -> list:
    """Return a sorted list of absolute image paths in *input_dir*."""
    extensions = {e.lower() for e in extensions}
    paths = []
    for fname in sorted(os.listdir(input_dir)):
        if os.path.splitext(fname)[1].lower() in extensions:
            paths.append(os.path.join(input_dir, fname))
    return paths


_WARP_MODES = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean":   cv2.MOTION_EUCLIDEAN,
    "affine":      cv2.MOTION_AFFINE,
    "homography":  cv2.MOTION_HOMOGRAPHY,
}


# ===================================================================
# Tiling
# ===================================================================

def _extract_and_save_tiles(img_hr, img_lr, mask, fname, cfg):
    """Slide a window over HR/LR images and save tiles passing the mask filter.

    Parameters
    ----------
    img_hr, img_lr : np.ndarray  (uint8, HxWxC or HxW)
    mask : np.ndarray  (uint8, HxW, 1 = valid)
    fname : str  (original filename, used as tile-name prefix)
    cfg : dict

    Returns
    -------
    int : number of tiles saved.
    """
    tile_cfg = cfg["tiling"]
    crop_size = tile_cfg["crop_size"]
    step = tile_cfg["step"]
    max_invalid = tile_cfg["max_invalid_ratio"]
    save_ext = "." + tile_cfg.get("save_format", "png")

    out_hr_dir = cfg["_abs_output_hr_dir"]
    out_lr_dir = cfg["_abs_output_lr_dir"]
    name, _ = os.path.splitext(fname)

    h, w = img_hr.shape[:2]
    saved = 0

    for y in range(0, h - crop_size + 1, step):
        for x in range(0, w - crop_size + 1, step):
            mask_tile = mask[y:y + crop_size, x:x + crop_size]
            invalid_ratio = 1.0 - (np.count_nonzero(mask_tile) / (crop_size * crop_size))

            if invalid_ratio <= max_invalid:
                hr_tile = img_hr[y:y + crop_size, x:x + crop_size]
                lr_tile = img_lr[y:y + crop_size, x:x + crop_size]

                tile_name = f"{name}_{y:05d}_{x:05d}{save_ext}"
                imsave(hr_tile, os.path.join(out_hr_dir, tile_name))
                imsave(lr_tile, os.path.join(out_lr_dir, tile_name))
                saved += 1

    return saved


# ===================================================================
# Per-image worker
# ===================================================================

def _process_one(args: tuple) -> str:
    """Process a single HR/LR (+ optional QA) triplet.

    Parameters
    ----------
    args : (hr_path, lr_path, qa_path_or_None, cfg_dict, worker_seed)

    Returns
    -------
    str : status message.
    """
    hr_path, lr_path, qa_path, cfg, worker_seed = args

    # Per-worker reproducibility for degradation randomness
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))

    fname = os.path.basename(hr_path)
    n_channels = cfg["n_channels"]

    # ==================================================================
    # 0.  Read images
    # ==================================================================
    img_hr = imread_uint(hr_path, n_channels)
    img_lr = imread_uint(lr_path, n_channels)

    # Ensure LR is same spatial size as HR (required for pairing)
    if img_lr.shape[:2] != img_hr.shape[:2]:
        img_lr = cv2.resize(img_lr, (img_hr.shape[1], img_hr.shape[0]),
                            interpolation=cv2.INTER_CUBIC)

    # ==================================================================
    # 1.  Cloud / Shadow / Snow Masking
    # ==================================================================
    mask_cfg = cfg.get("masking", {})
    if mask_cfg.get("enabled", False):
        method = mask_cfg.get("method", "qa_band")

        if method == "qa_band":
            if qa_path is not None:
                qa_band = cv2.imread(qa_path, cv2.IMREAD_GRAYSCALE)
                if qa_band is None:
                    return f"  [WARN] [{fname}] Could not read QA band. Skipped."
                # Resize QA to match HR if needed
                if qa_band.shape[:2] != img_hr.shape[:2]:
                    qa_band = cv2.resize(qa_band, (img_hr.shape[1], img_hr.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                invalid_classes = mask_cfg.get("invalid_classes", [3, 8, 9, 10, 11])
                mask = apply_l2_qa_mask(qa_band, invalid_classes)
            else:
                # No QA file available – treat everything as valid
                mask = np.ones(img_hr.shape[:2], dtype=np.uint8)

        elif method == "s2cloudless":
            # s2cloudless needs 10 bands in (1, H, W, 10) shape
            if img_hr.ndim == 3 and img_hr.shape[2] == 10:
                img_4d = img_hr[np.newaxis, ...].astype(np.float32)
                from preprocessing_pipeline.other_utils import apply_s2cloudless_mask as _s2mask
                masked_data = _s2mask(
                    img_4d,
                    nodata=mask_cfg.get("s2_nodata", 0.0),
                    auto_scale=mask_cfg.get("s2_auto_scale", True),
                    threshold=mask_cfg.get("s2_threshold", 0.4),
                    average_over=mask_cfg.get("s2_average_over", 4),
                    dilation_size=mask_cfg.get("s2_dilation_size", 2),
                )
                # Build binary mask from nodata pixels
                nodata_val = mask_cfg.get("s2_nodata", 0.0)
                mask = (masked_data.sum(axis=-1) != nodata_val * 10).astype(np.uint8)
            else:
                return (f"  [WARN] [{fname}] s2cloudless requires 10-band input "
                        f"(got {img_hr.shape}). Skipped.")
        else:
            return f"  [ERROR] [{fname}] Unknown masking method '{method}'."
    else:
        # Masking disabled – all pixels valid
        mask = np.ones(img_hr.shape[:2], dtype=np.uint8)

    # ==================================================================
    # 2.  Relative Normalization
    # ==================================================================
    rel_cfg = cfg.get("relative_normalization", {})
    if rel_cfg.get("enabled", False):
        method = rel_cfg.get("method", "histogram_match")
        direction = rel_cfg.get("direction", "lr_to_hr")
        use_mask = rel_cfg.get("mask_aware", True)

        m_src = mask if use_mask else None
        m_ref = mask if use_mask else None

        if direction == "lr_to_hr":
            img_lr = relative_normalize(img_lr, img_hr, method=method,
                                        mask_src=m_src, mask_ref=m_ref)
        elif direction == "hr_to_lr":
            img_hr = relative_normalize(img_hr, img_lr, method=method,
                                        mask_src=m_ref, mask_ref=m_src)
        else:
            return (f"  [ERROR] [{fname}] Unknown relative_normalization direction "
                    f"'{direction}'. Use 'lr_to_hr' or 'hr_to_lr'.")

    # ==================================================================
    # 3.  Absolute Normalization (Percentile Clipping)
    # ==================================================================
    norm_cfg = cfg.get("normalization", {})
    if norm_cfg.get("enabled", False):
        low_p = norm_cfg.get("low_percentile", 2)
        high_p = norm_cfg.get("high_percentile", 98)

        if norm_cfg.get("mask_aware", True):
            img_hr = satellite_pre_norm_masked(img_hr, mask, low_p, high_p)
            img_lr = satellite_pre_norm_masked(img_lr, mask, low_p, high_p)
        else:
            img_hr = satellite_pre_norm(img_hr, low_p, high_p)
            img_lr = satellite_pre_norm(img_lr, low_p, high_p)

    # ==================================================================
    # 4.  Spatial Co-registration (ECC)
    # ==================================================================
    reg_cfg = cfg.get("registration", {})
    if reg_cfg.get("enabled", False):
        mode_str = reg_cfg.get("warp_mode", "translation")
        warp_mode = _WARP_MODES.get(mode_str)
        if warp_mode is None:
            return (f"  [ERROR] [{fname}] Unknown warp_mode '{mode_str}'. "
                    f"Choose from: {list(_WARP_MODES.keys())}")

        img_lr_aligned, success = align_images_ecc(
            img_hr, img_lr,
            warp_mode=warp_mode,
            num_iters=reg_cfg.get("num_iters", 50),
            eps=reg_cfg.get("eps", 1e-5),
            gauss_filt_size=reg_cfg.get("gauss_filt_size", 5),
        )

        if not success:
            if reg_cfg.get("skip_on_failure", True):
                return f"  [SKIP] [{fname}] ECC registration failed. Pair skipped."
            else:
                # Keep un-aligned LR
                pass
        else:
            img_lr = img_lr_aligned

            # Also warp the mask so that border regions introduced by warping
            # are marked invalid (prevents black border artefacts in tiles)
            if mask_cfg.get("enabled", False):
                if warp_mode == cv2.MOTION_HOMOGRAPHY:
                    # reuse the warp matrix from the last ECC call is not
                    # accessible here, so we rebuild a conservative mask:
                    # any pixel that became 0 in LR after warping → invalid
                    lr_gray = cv2.cvtColor(img_lr, cv2.COLOR_RGB2GRAY) if img_lr.ndim == 3 else img_lr
                    mask = mask & (lr_gray > 0).astype(np.uint8)
                else:
                    lr_gray = cv2.cvtColor(img_lr, cv2.COLOR_RGB2GRAY) if img_lr.ndim == 3 else img_lr
                    mask = mask & (lr_gray > 0).astype(np.uint8)

    # ==================================================================
    # 5.  Degradation (Optional - applies to HR to generate new LR)
    # ==================================================================
    deg_cfg = cfg.get("degradation", {})
    if deg_cfg.get("enabled", False):
        # Convert HR to float32 [0, 1]
        img_hr_f = uint2single(img_hr)

        deg_type = deg_cfg.get("type", "satellite")
        sf = deg_cfg.get("scale", 4)

        try:
            if deg_type == "bsrgan":
                deg_params = deg_cfg.get("bsrgan", {})
                img_lq, img_hq = degrade_bsrgan(
                    img_hr_f,
                    sf=sf,
                    jpeg_prob=deg_params.get("jpeg_prob", 0.9),
                    scale2_prob=deg_params.get("scale2_prob", 0.25),
                    isp_prob=deg_params.get("isp_prob", 0.25),
                    noise_level1=deg_params.get("noise_level1", 2),
                    noise_level2=deg_params.get("noise_level2", 25),
                )

            elif deg_type == "real_esrgan":
                deg_params = deg_cfg.get("real_esrgan", {})
                img_lq, img_hq = degrade_real_esrgan(
                    img_hr_f,
                    sf=sf,
                    blur_prob_1=deg_params.get("blur_prob_1", 1.0),
                    resize_prob_1=deg_params.get("resize_prob_1", 1.0),
                    gaussian_noise_prob_1=deg_params.get("gaussian_noise_prob_1", 0.5),
                    poisson_noise_prob_1=deg_params.get("poisson_noise_prob_1", 0.1),
                    speckle_noise_prob_1=deg_params.get("speckle_noise_prob_1", 0.1),
                    jpeg_prob_1=deg_params.get("jpeg_prob_1", 0.9),
                    noise_level1_s1=deg_params.get("noise_level1_s1", 2),
                    noise_level2_s1=deg_params.get("noise_level2_s1", 25),
                    blur_prob_2=deg_params.get("blur_prob_2", 0.8),
                    resize_prob_2=deg_params.get("resize_prob_2", 1.0),
                    gaussian_noise_prob_2=deg_params.get("gaussian_noise_prob_2", 0.5),
                    poisson_noise_prob_2=deg_params.get("poisson_noise_prob_2", 0.1),
                    speckle_noise_prob_2=deg_params.get("speckle_noise_prob_2", 0.1),
                    jpeg_prob_2=deg_params.get("jpeg_prob_2", 0.8),
                    noise_level1_s2=deg_params.get("noise_level1_s2", 2),
                    noise_level2_s2=deg_params.get("noise_level2_s2", 15),
                    final_jpeg_prob=deg_params.get("final_jpeg_prob", 0.5),
                    resize_back_prob=deg_params.get("resize_back_prob", 0.5),
                    isp_prob=deg_params.get("isp_prob", 0.1),
                )

            elif deg_type == "bsrgan_plus":
                deg_params = deg_cfg.get("bsrgan_plus", {})
                img_lq, img_hq = degrade_bsrgan_plus(
                    img_hr_f,
                    sf=sf,
                    shuffle_prob=deg_params.get("shuffle_prob", 0.5),
                    use_sharp=deg_params.get("use_sharp", False),
                    sharpening_weight=deg_params.get("sharpening_weight", 0.5),
                    sharpening_radius=deg_params.get("sharpening_radius", 50),
                    sharpening_threshold=deg_params.get("sharpening_threshold", 10),
                    poisson_prob=deg_params.get("poisson_prob", 0.1),
                    speckle_prob=deg_params.get("speckle_prob", 0.1),
                    isp_prob=deg_params.get("isp_prob", 0.1),
                    noise_level1=deg_params.get("noise_level1", 2),
                    noise_level2=deg_params.get("noise_level2", 25),
                )

            elif deg_type == "satellite":
                deg_params = deg_cfg.get("satellite", {})
                img_lq, img_hq = degrade_satellite(
                    img_hr_f,
                    sf=sf,
                    blur_prob_1=deg_params.get("blur_prob_1", 1.0),
                    blur_type_1=deg_params.get("blur_type_1", "mtf"),
                    resize_prob_1=deg_params.get("resize_prob_1", 0.75),
                    poisson_prob_1=deg_params.get("poisson_prob_1", 0.75),
                    read_noise_prob_1=deg_params.get("read_noise_prob_1", 0.55),
                    haze_prob_1=deg_params.get("haze_prob_1", 0.45),
                    jpeg_prob_1=deg_params.get("jpeg_prob_1", 0.12),
                    blur_prob_2=deg_params.get("blur_prob_2", 0.92),
                    blur_type_2=deg_params.get("blur_type_2", "mtf"),
                    resize_prob_2=deg_params.get("resize_prob_2", 0.70),
                    poisson_prob_2=deg_params.get("poisson_prob_2", 0.60),
                    read_noise_prob_2=deg_params.get("read_noise_prob_2", 0.45),
                    haze_prob_2=deg_params.get("haze_prob_2", 0.35),
                    jpeg_prob_2=deg_params.get("jpeg_prob_2", 0.08),
                    final_jpeg_prob=deg_params.get("final_jpeg_prob", 0.10),
                    resize_back_prob=deg_params.get("resize_back_prob", 0.35),
                    isp_prob=deg_params.get("isp_prob", 0.08),
                    noise_level1=deg_params.get("noise_level1", 0.8),
                    noise_level2=deg_params.get("noise_level2", 10.0),
                    mtf_sigma_optics_range=tuple(deg_params.get("mtf_sigma_optics_range", [0.8, 2.8])),
                    mtf_detector_width_range=tuple(deg_params.get("mtf_detector_width_range", [0.7, 1.8])),
                    mtf_atm_sigma_range=tuple(deg_params.get("mtf_atm_sigma_range", [0.4, 1.8])),
                )

            else:
                return (f"  [ERROR] [{fname}] Unknown degradation type '{deg_type}'. "
                        f"Choose from: bsrgan | real_esrgan | bsrgan_plus | satellite")

            # Replace LR with degraded version, update HR to modcrop version
            img_lr = single2uint(img_lq)
            img_hr = single2uint(img_hq)

            # Note: The mask may need to be adjusted if HR dimensions changed
            # due to modcrop. Resize mask if needed.
            if mask.shape[:2] != img_hr.shape[:2]:
                mask = cv2.resize(mask, (img_hr.shape[1], img_hr.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        except Exception as e:
            return f"  [ERROR] [{fname}] Degradation failed: {str(e)}"

    # ==================================================================
    # 6.  Tiling & Filtering  OR  Full-image Save
    # ==================================================================
    tile_cfg = cfg.get("tiling", {})
    if tile_cfg.get("enabled", False):
        n_tiles = _extract_and_save_tiles(img_hr, img_lr, mask, fname, cfg)
        return f"  [OK] [{fname}] {n_tiles} valid tiles saved."
    else:
        # Save full-resolution images
        save_ext = "." + cfg.get("save_format", "png")
        name, _ = os.path.splitext(fname)
        imsave(img_hr, os.path.join(cfg["_abs_output_hr_dir"], name + save_ext))
        imsave(img_lr, os.path.join(cfg["_abs_output_lr_dir"], name + save_ext))
        return f"  [OK] [{fname}] Saved full-res  HR {img_hr.shape[:2]}  LR {img_lr.shape[:2]}."


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Level-2 Paired Satellite Preprocessing Pipeline."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config_l2.json")
    args = parser.parse_args()

    cfg = _parse_json(args.config)
    project_root = _PROJECT_ROOT

    print("=" * 70)
    print(f"  Level-2 Paired Pipeline  –  {cfg.get('task', 'L2_SR')}")
    print("=" * 70)

    # ---- Resolve paths ----
    input_hr_dir = _resolve_path(cfg["input_hr_dir"], project_root)
    input_lr_dir = _resolve_path(cfg["input_lr_dir"], project_root)
    input_qa_dir = cfg.get("input_qa_dir")
    if input_qa_dir is not None:
        input_qa_dir = _resolve_path(input_qa_dir, project_root)

    output_hr_dir = _resolve_path(cfg["output_hr_dir"], project_root)
    output_lr_dir = _resolve_path(cfg["output_lr_dir"], project_root)
    os.makedirs(output_hr_dir, exist_ok=True)
    os.makedirs(output_lr_dir, exist_ok=True)

    cfg["_abs_output_hr_dir"] = output_hr_dir
    cfg["_abs_output_lr_dir"] = output_lr_dir

    # ---- Seed ----
    seed = cfg.get("seed", None)
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed % (2**32))

    # ---- Discover & match images ----
    extensions = cfg.get("supported_extensions", [".png", ".tif", ".tiff"])
    hr_paths = _discover_images(input_hr_dir, extensions)

    if not hr_paths:
        print(f"  [ERROR] No HR images found in '{input_hr_dir}'")
        sys.exit(1)

    worker_args = []
    unmatched = []
    for idx, hr_path in enumerate(hr_paths):
        basename = os.path.basename(hr_path)
        lr_path = os.path.join(input_lr_dir, basename)

        if not os.path.exists(lr_path):
            unmatched.append(basename)
            continue

        qa_path = None
        if input_qa_dir is not None:
            qa_candidate = os.path.join(input_qa_dir, basename)
            if os.path.exists(qa_candidate):
                qa_path = qa_candidate

        worker_seed = seed + idx
        worker_args.append((hr_path, lr_path, qa_path, cfg, worker_seed))

    if unmatched:
        print(f"  [WARN] {len(unmatched)} HR images have no matching LR:")
        for n in unmatched[:5]:
            print(f"    - {n}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched) - 5} more")

    if not worker_args:
        print("  [ERROR] No matched HR/LR pairs found.")
        sys.exit(1)

    # ---- Print summary ----
    mask_cfg = cfg.get("masking", {})
    rel_cfg = cfg.get("relative_normalization", {})
    norm_cfg = cfg.get("normalization", {})
    reg_cfg = cfg.get("registration", {})
    deg_cfg = cfg.get("degradation", {})
    tile_cfg = cfg.get("tiling", {})

    print(f"  Input HR         : {input_hr_dir}")
    print(f"  Input LR         : {input_lr_dir}")
    print(f"  Input QA         : {input_qa_dir or '(none)'}")
    print(f"  Output HR        : {output_hr_dir}")
    print(f"  Output LR        : {output_lr_dir}")
    print(f"  Matched pairs    : {len(worker_args)}")
    print(f"  Channels         : {cfg['n_channels']}")
    print(f"  Workers          : {cfg.get('num_workers', 4)}")
    print(f"  Seed             : {seed}")
    print(f"  ---")
    print(f"  Masking          : {'ON – ' + mask_cfg.get('method', '?') if mask_cfg.get('enabled') else 'OFF'}")
    print(f"  Rel. normalize   : {'ON – ' + rel_cfg.get('method', '?') + ' (' + rel_cfg.get('direction', '?') + ')' if rel_cfg.get('enabled') else 'OFF'}")
    print(f"  Abs. normalize   : {'ON – p' + str(norm_cfg.get('low_percentile', '?')) + '/p' + str(norm_cfg.get('high_percentile', '?')) if norm_cfg.get('enabled') else 'OFF'}")
    print(f"  Registration     : {'ON – ' + reg_cfg.get('warp_mode', '?') if reg_cfg.get('enabled') else 'OFF'}")
    print(f"  Degradation      : {'ON – ' + deg_cfg.get('type', '?') + ' (scale=' + str(deg_cfg.get('scale', '?')) + ')' if deg_cfg.get('enabled') else 'OFF'}")
    print(f"  Tiling           : {'ON – ' + str(tile_cfg.get('crop_size', '?')) + 'px, max_inv=' + str(tile_cfg.get('max_invalid_ratio', '?')) if tile_cfg.get('enabled') else 'OFF (full-res save)'}")
    print("-" * 70)

    # ---- Process ----
    t0 = time.time()
    num_workers = cfg.get("num_workers", 4)

    if num_workers <= 1:
        for idx, wa in enumerate(worker_args, 1):
            msg = _process_one(wa)
            print(f"  [{idx}/{len(worker_args)}] {msg}")
    else:
        with Pool(processes=num_workers) as pool:
            results = pool.imap(_process_one, worker_args)
            for idx, msg in enumerate(results, 1):
                print(f"  [{idx}/{len(worker_args)}] {msg}")

    elapsed = time.time() - t0
    print("-" * 70)
    print(f"  Done – {len(worker_args)} pairs processed in {elapsed:.1f}s")
    print(f"  HR → {output_hr_dir}")
    print(f"  LR → {output_lr_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

