"""
Satellite Imagery Preprocessing Pipeline for Super-Resolution
==============================================================
Preprocesses paired HR/LR GeoTIFF or JP2 satellite imagery into
matched 8-bit RGB patch pairs suitable for PyTorch/TensorFlow SR models.

Pipeline stages:
  1. Configuration
  2. Data Loading & Band Extraction
  3. Spatial Coregistration — Matrix Composition
       A. Coarse Global      (Find ORB Homography)
       B. Sub-pixel Global   (Find Phase Translation)
       C. Global Warp        (Combine A & B into one single warp)
  4. Smart Percentile Scaling
  5. Radiometric Normalization
  6. Patch Extraction & Quality Filter
       C. Patch-wise Local   (Combine ECC Affine + Downscale into one warp)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_JSON_PATH: str = "config.json"

CONFIG: dict = {
    "HR_IMAGE_PATH": "Lahore/HR/3df615fc-2ded-4bd0-85de-237901ccb04f_NONE_STD_A/IMG_01_PNEO3_PMS-FS/IMG_PNEO3_STD_202601250555400_PMS-FS_ORT_1e224074-3a2a-4f29-cfe4-ab0a484fbd47_RGB_R1C1.JP2",
    "LR_IMAGE_PATH": "Lahore/LR/IMG_PHR1A_PMS_001/IMG_PHR1A_PMS_202602130556014_ORT_38a71b19-2781-4146-c1f3-561512adaf94_R1C1.JP2",
    "OUTPUT_DIR": "output_4",
    "HR_RGB_BANDS": [1, 2, 3],
    "LR_RGB_BANDS": [3, 2, 1],
    "SCALE_FACTOR": 2,
    "HR_PATCH_SIZE": 256,
    "LR_PATCH_SIZE": 128,
    "STRIDE": 128,
    "NODATA_VALUE": 0,
    "SATURATED_VALUE": 32767,
    "CLIP_PERCENTILES": [2.0, 98.0],
    "MAX_NODATA_FRACTION": 0.05,
    "MIN_VARIANCE": 50.0,
    "COREG_A_ENABLED": True,
    "COREG_A_MAX_FEATURES": 5000,
    "COREG_A_MATCH_RATIO": 0.75,
    "COREG_A_RANSAC_THRESH": 5.0,
    "COREG_A_DOWNSAMPLE": 0.25,
    "COREG_B_ENABLED": True,
    "COREG_B_DOWNSAMPLE": 0.25,
    "COREG_B_UPSAMPLE_FACTOR": 100,
    "COREG_C_ENABLED": True,
    "COREG_C_MAX_ITER": 50,
    "COREG_C_EPS": 1e-4,
    "COREG_C_WARP_MODE": "translation",
    "COREG_C_DISCARD_ON_FAIL": True,
    "MIN_ECC_SCORE": 0.85,
    "MIN_SSIM": 0.60,
    "RADIOMETRIC_BLOCK_SIZE": 256,
    "RADIOMETRIC_RMSE_THRESHOLD": 40.0,
    "RADIOMETRIC_N_SAMPLES": 100_000,
    "RADIOMETRIC_POST_HIST_MATCH": True,
}


def build_config() -> dict:
    cfg = CONFIG.copy()
    json_path = Path(__file__).parent / CONFIG_JSON_PATH
    if json_path.exists():
        with open(json_path, "r") as fh:
            json_cfg = json.load(fh)
        cfg.update(json_cfg)
        logging.info("Loaded config overrides from: %s", json_path)

    assert cfg["HR_PATCH_SIZE"] % cfg["SCALE_FACTOR"] == 0
    cfg["LR_PATCH_SIZE"] = cfg["HR_PATCH_SIZE"] // cfg["SCALE_FACTOR"]
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — DATA LOADING & BAND EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def load_rgb_bands(image_path: str, bands: list = [1, 2, 3]) -> Tuple[np.ndarray, dict]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with rasterio.open(image_path) as src:
        data = src.read(bands).astype(np.uint16)
        profile = src.profile.copy()
        profile.update(count=3)

    array = np.transpose(data, (1, 2, 0))
    logging.info("Loaded %s (bands %s)  →  shape %s", path.name, bands, array.shape)
    return array, profile


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — SPATIAL COREGISTRATION (MATRIX COMPOSITION)
# ─────────────────────────────────────────────────────────────────────────────

def _to_gray_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype != np.uint8:
        a = array.astype(np.float32)
        a = (a - a.min()) / (a.max() - a.min() + 1e-6) * 255.0
        a = a.astype(np.uint8)
    else:
        a = array
    gray = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return gray.astype(np.uint8)


def _initial_reproject(lr_path: str, hr_profile: dict, hr_height: int, hr_width: int, lr_bands: list) -> np.ndarray:
    dst_crs = hr_profile["crs"]
    dst_transform = hr_profile["transform"]
    lr_dst = np.zeros((3, hr_height, hr_width), dtype=np.uint16)

    with rasterio.open(lr_path) as lr_src:
        for band_idx, band_number in enumerate(lr_bands):
            reproject(
                source=rasterio.band(lr_src, band_number),
                destination=lr_dst[band_idx],
                src_transform=lr_src.transform,
                src_crs=lr_src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.cubic,
            )
    lr_reprojected = np.transpose(lr_dst, (1, 2, 0))
    return np.clip(lr_reprojected, 0, 32767).astype(np.uint16)


def get_orb_homography(hr_array: np.ndarray, lr_array: np.ndarray, cfg: dict) -> Optional[np.ndarray]:
    """Calculates Stage A ORB Homography without applying it."""
    scale = cfg["COREG_A_DOWNSAMPLE"]
    H, W = hr_array.shape[:2]
    small_h, small_w = max(1, int(H * scale)), max(1, int(W * scale))

    hr_small = cv2.resize(_to_gray_uint8(hr_array), (small_w, small_h), interpolation=cv2.INTER_AREA)
    lr_small = cv2.resize(_to_gray_uint8(lr_array), (small_w, small_h), interpolation=cv2.INTER_AREA)

    orb = cv2.ORB_create(nfeatures=cfg["COREG_A_MAX_FEATURES"])
    kp_hr, des_hr = orb.detectAndCompute(hr_small, None)
    kp_lr, des_lr = orb.detectAndCompute(lr_small, None)

    if des_hr is None or des_lr is None or len(kp_hr) < 4 or len(kp_lr) < 4:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des_lr, des_hr, k=2)
    good = [m for m, n in raw_matches if m.distance < cfg["COREG_A_MATCH_RATIO"] * n.distance]

    if len(good) < 4:
        return None

    src_pts = np.float32([kp_lr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_hr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_small, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, cfg["COREG_A_RANSAC_THRESH"])

    if H_small is None:
        return None

    S = np.array([[1.0 / scale, 0, 0], [0, 1.0 / scale, 0], [0, 0, 1]], dtype=np.float64)
    S_inv = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    return S @ H_small @ S_inv


def get_phase_translation(hr_gray: np.ndarray, lr_proxy: np.ndarray, cfg: dict) -> np.ndarray:
    """Calculates Stage B Translation matrix."""
    scale = cfg["COREG_B_DOWNSAMPLE"]
    H, W = hr_gray.shape[:2]
    small_h, small_w = max(1, int(H * scale)), max(1, int(W * scale))

    hr_small = cv2.resize(hr_gray.astype(np.float32), (small_w, small_h), interpolation=cv2.INTER_AREA)
    lr_small = cv2.resize(lr_proxy.astype(np.float32), (small_w, small_h), interpolation=cv2.INTER_AREA)

    shift_small, _, _ = phase_cross_correlation(hr_small, lr_small, upsample_factor=cfg["COREG_B_UPSAMPLE_FACTOR"])
    shift_row, shift_col = shift_small[0] / scale, shift_small[1] / scale

    return np.array([
        [1.0, 0.0, shift_col],
        [0.0, 1.0, shift_row],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def coregister_full(lr_path: str, hr_array: np.ndarray, hr_profile: dict, hr_height: int, hr_width: int, lr_bands: list,
                    cfg: dict) -> np.ndarray:
    """Combines A & B into a single global interpolation."""
    logging.info("=== MODULE 3 — Initial rasterio reproject ===")
    lr_reprojected = _initial_reproject(lr_path, hr_profile, hr_height, hr_width, lr_bands)

    H_global = np.eye(3, dtype=np.float64)

    if cfg["COREG_A_ENABLED"]:
        logging.info("=== MODULE 3 — Stage A: Calculating ORB Homography ===")
        H_A = get_orb_homography(hr_array, lr_reprojected, cfg)
        if H_A is not None:
            H_global = H_A

    if cfg["COREG_B_ENABLED"]:
        logging.info("=== MODULE 3 — Stage B: Calculating Phase Correlation ===")
        hr_gray = _to_gray_uint8(hr_array)
        lr_gray = _to_gray_uint8(lr_reprojected)
        # Warp a gray proxy just to find the phase shift (saves memory/time)
        lr_proxy = cv2.warpPerspective(lr_gray, H_global, (hr_width, hr_height), flags=cv2.INTER_CUBIC)
        T_B = get_phase_translation(hr_gray, lr_proxy, cfg)
        # Compose translation onto existing homography
        H_global = T_B @ H_global

    logging.info("=== MODULE 3 — Applying Combined Global Warp ===")
    lr_aligned = np.zeros_like(lr_reprojected)
    for c in range(3):
        lr_aligned[:, :, c] = np.clip(cv2.warpPerspective(
            lr_reprojected[:, :, c].astype(np.float32), H_global, (hr_width, hr_height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        ), 0, 65535).astype(np.uint16)

    return lr_aligned


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 & 5 (UNCHANGED LOGIC, INCLUDED FOR COMPLETENESS)
# ─────────────────────────────────────────────────────────────────────────────

def scale_to_uint8(array: np.ndarray, nodata_value: int, saturated_value: int,
                   clip_percentiles: Tuple[float, float]) -> np.ndarray:
    p_lo, p_hi = clip_percentiles
    float_array = array.astype(np.float32)
    result = np.zeros_like(float_array)
    for c in range(3):
        valid_mask = ((array[:, :, c] > nodata_value) & (array[:, :, c] < saturated_value))
        valid_pixels = float_array[:, :, c][valid_mask]
        if valid_pixels.size == 0: continue
        v_min, v_max = np.percentile(valid_pixels, p_lo), np.percentile(valid_pixels, p_hi)
        if v_max == v_min: continue
        result[:, :, c] = (np.clip(float_array[:, :, c], v_min, v_max) - v_min) / (v_max - v_min)
    return np.clip(np.round(result * 255.0), 0, 255).astype(np.uint8)


def fit_and_apply_radiometric_regression(lr_uint8: np.ndarray, hr_uint8: np.ndarray, cfg: dict) -> np.ndarray:
    block_size, rmse_thresh, n_samples = cfg["RADIOMETRIC_BLOCK_SIZE"], cfg["RADIOMETRIC_RMSE_THRESHOLD"], cfg[
        "RADIOMETRIC_N_SAMPLES"]
    H, W = lr_uint8.shape[:2]
    lr_f, hr_f = lr_uint8.astype(np.float32), hr_uint8.astype(np.float32)

    accepted_blocks = []
    for r in range(0, H - block_size + 1, block_size):
        for c in range(0, W - block_size + 1, block_size):
            diff = lr_f[r:r + block_size, c:c + block_size] - hr_f[r:r + block_size, c:c + block_size]
            if float(np.sqrt(np.mean(diff ** 2))) <= rmse_thresh:
                accepted_blocks.append((r, c))

    if not accepted_blocks:
        accepted_blocks = [(r, c) for r in range(0, H - block_size + 1, block_size) for c in
                           range(0, W - block_size + 1, block_size)]

    pixels_per_block = max(1, n_samples // len(accepted_blocks))
    rng = np.random.default_rng(42)

    lr_samples, hr_samples = [], []
    for r, c in accepted_blocks:
        n_draw = min(pixels_per_block, block_size * block_size)
        idx = rng.choice(block_size * block_size, size=n_draw, replace=False)
        lr_samples.append(lr_f[r:r + block_size, c:c + block_size].reshape(-1, 3)[idx])
        hr_samples.append(hr_f[r:r + block_size, c:c + block_size].reshape(-1, 3)[idx])

    V = np.column_stack(
        [np.concatenate(lr_samples, axis=0), np.ones(len(lr_samples) * len(lr_samples[0]), dtype=np.float32)])
    S = np.concatenate(hr_samples, axis=0)

    weights, _, _, _ = np.linalg.lstsq(V, S, rcond=None)

    V_full = np.column_stack([lr_f.reshape(-1, 3), np.ones(H * W, dtype=np.float32)])
    return np.clip(np.round(V_full @ weights), 0, 255).astype(np.uint8).reshape(H, W, 3)


def normalise_histogram(lr_uint8: np.ndarray, hr_uint8: np.ndarray) -> np.ndarray:
    matched = match_histograms(lr_uint8.astype(np.float32), hr_uint8.astype(np.float32), channel_axis=2)
    return np.clip(np.round(matched), 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — PATCH EXTRACTION & QUALITY FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def get_ecc_matrix(hr_patch: np.ndarray, lr_patch: np.ndarray, cfg: dict) -> Tuple[np.ndarray, bool, float]:
    warp_mode = cv2.MOTION_TRANSLATION if cfg["COREG_C_WARP_MODE"] == "translation" else cv2.MOTION_EUCLIDEAN
    hr_gray = _to_gray_uint8(hr_patch).astype(np.float32)
    lr_gray = _to_gray_uint8(lr_patch).astype(np.float32)
    warp_init = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg["COREG_C_MAX_ITER"], cfg["COREG_C_EPS"])

    try:
        cc_score, warp_matrix = cv2.findTransformECC(hr_gray, lr_gray, warp_init, warp_mode, criteria)
        return warp_matrix, True, float(cc_score)
    except cv2.error:
        return warp_init, False, 0.0


def extract_and_save_patches(hr_uint8: np.ndarray, lr_adjusted: np.ndarray, cfg: dict) -> int:
    hr_patch_size, lr_patch_size, stride = cfg["HR_PATCH_SIZE"], cfg["LR_PATCH_SIZE"], cfg["STRIDE"]
    output_dir = Path(cfg["OUTPUT_DIR"])
    hr_out_dir, lr_out_dir = output_dir / "hr", output_dir / "lr"
    hr_out_dir.mkdir(parents=True, exist_ok=True)
    lr_out_dir.mkdir(parents=True, exist_ok=True)

    H, W = hr_uint8.shape[:2]
    saved_count = 0
    skipped_nodata = 0
    skipped_variance = 0
    skipped_ecc = 0
    skipped_ecc_score = 0
    skipped_ssim = 0

    # Scale matrix to map coordinates from 128x128 space up to 256x256 space for WARP_INVERSE_MAP
    S_up = np.array([
        [cfg["SCALE_FACTOR"], 0, 0],
        [0, cfg["SCALE_FACTOR"], 0],
        [0, 0, 1]
    ], dtype=np.float32)

    row_starts = list(range(0, H - hr_patch_size + 1, stride))
    col_starts = list(range(0, W - hr_patch_size + 1, stride))
    total_windows = len(row_starts) * len(col_starts)

    logging.info(
        "Starting patch extraction: %d candidate windows "
        "(stride=%d, hr_patch=%d, lr_patch=%d, Stage C ECC=%s, "
        "MIN_ECC_SCORE=%.2f, MIN_SSIM=%.2f)",
        total_windows, stride, hr_patch_size, lr_patch_size,
        cfg["COREG_C_ENABLED"],
        cfg.get("MIN_ECC_SCORE", 0.0),
        cfg.get("MIN_SSIM", 0.0),
    )

    pbar = tqdm(total=total_windows, desc="Extracting patches", unit="win")

    for row in row_starts:
        for col in col_starts:
            pbar.update(1)

            hr_patch = hr_uint8[row: row + hr_patch_size, col: col + hr_patch_size, :]

            if np.any(hr_patch == cfg["NODATA_VALUE"], axis=2).mean() > cfg["MAX_NODATA_FRACTION"]:
                skipped_nodata += 1
                continue

            mean_var = float(np.var(hr_patch.astype(np.float32), axis=(0, 1)).mean())
            if mean_var < cfg["MIN_VARIANCE"]:
                skipped_variance += 1
                continue

            lr_patch_full = lr_adjusted[row: row + hr_patch_size, col: col + hr_patch_size, :]

            # Stage C: Calculate local offset but do not warp yet
            if cfg["COREG_C_ENABLED"]:
                M_ecc, ecc_ok, cc_score = get_ecc_matrix(hr_patch, lr_patch_full, cfg)
                if not ecc_ok and cfg["COREG_C_DISCARD_ON_FAIL"]:
                    skipped_ecc += 1
                    continue
                if cc_score < cfg.get("MIN_ECC_SCORE", 0.0):
                    skipped_ecc_score += 1
                    continue
            else:
                M_ecc = np.eye(2, 3, dtype=np.float32)

            # Compose ECC matrix and Downscale matrix
            M_ecc_3x3 = np.vstack([M_ecc, [0, 0, 1]])
            M_combined_3x3 = M_ecc_3x3 @ S_up
            M_combined_2x3 = M_combined_3x3[:2, :]

            # Apply SINGLE warp to apply local alignment AND downscale simultaneously
            lr_patch = np.zeros((lr_patch_size, lr_patch_size, 3), dtype=np.uint8)
            for c in range(3):
                lr_patch[:, :, c] = np.clip(cv2.warpAffine(
                    lr_patch_full[:, :, c].astype(np.float32), M_combined_2x3,
                    (lr_patch_size, lr_patch_size),
                    flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT_101,
                ), 0, 255).astype(np.uint8)

            # SSIM gate evaluated on the final downscaled HR proxy vs final LR patch
            if cfg.get("MIN_SSIM", 0.0) > 0.0:
                hr_patch_small = cv2.resize(hr_patch, (lr_patch_size, lr_patch_size), interpolation=cv2.INTER_AREA)
                patch_ssim = ssim(_to_gray_uint8(hr_patch_small), _to_gray_uint8(lr_patch), data_range=255)
                if patch_ssim < cfg["MIN_SSIM"]:
                    skipped_ssim += 1
                    continue

            # Save Output
            patch_name = f"patch_{saved_count:06d}.png"
            cv2.imwrite(str(hr_out_dir / patch_name), cv2.cvtColor(hr_patch, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(lr_out_dir / patch_name), cv2.cvtColor(lr_patch, cv2.COLOR_RGB2BGR))
            saved_count += 1

    pbar.close()

    total_skipped = skipped_nodata + skipped_variance + skipped_ecc + skipped_ecc_score + skipped_ssim
    logging.info(
        "Patch extraction complete:\n"
        "  Saved              : %d\n"
        "  Skipped (nodata)   : %d\n"
        "  Skipped (variance) : %d\n"
        "  Skipped (ECC fail) : %d\n"
        "  Skipped (ECC score < %.2f) : %d\n"
        "  Skipped (SSIM  < %.2f)    : %d\n"
        "  Total skipped      : %d / %d candidates",
        saved_count,
        skipped_nodata,
        skipped_variance,
        skipped_ecc,
        cfg.get("MIN_ECC_SCORE", 0.0), skipped_ecc_score,
        cfg.get("MIN_SSIM", 0.0),      skipped_ssim,
        total_skipped, total_windows,
    )

    return saved_count


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    cfg = build_config()

    hr_raw, hr_profile = load_rgb_bands(cfg["HR_IMAGE_PATH"], cfg.get("HR_RGB_BANDS", [1, 2, 3]))
    hr_height, hr_width = hr_raw.shape[:2]

    # Combined Stage A + B Global Warp
    lr_coregistered = coregister_full(
        cfg["LR_IMAGE_PATH"], hr_raw, hr_profile, hr_height, hr_width, cfg.get("LR_RGB_BANDS", [3, 2, 1]), cfg
    )

    hr_uint8 = scale_to_uint8(hr_raw, cfg["NODATA_VALUE"], cfg["SATURATED_VALUE"], tuple(cfg["CLIP_PERCENTILES"]))
    lr_uint8 = scale_to_uint8(lr_coregistered, cfg["NODATA_VALUE"], cfg["SATURATED_VALUE"],
                              tuple(cfg["CLIP_PERCENTILES"]))

    # Radiometric normalisation
    lr_adjusted = fit_and_apply_radiometric_regression(lr_uint8, hr_uint8, cfg)
    if cfg.get("RADIOMETRIC_POST_HIST_MATCH", True):
        lr_adjusted = normalise_histogram(lr_adjusted, hr_uint8)

    # Patch extraction (Combined Stage C + Downscale)
    extract_and_save_patches(hr_uint8, lr_adjusted, cfg)


if __name__ == "__main__":
    main()