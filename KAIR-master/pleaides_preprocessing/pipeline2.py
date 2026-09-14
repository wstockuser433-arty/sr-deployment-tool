"""
Satellite Imagery Preprocessing Pipeline for Super-Resolution
==============================================================
Preprocesses paired HR/LR GeoTIFF or JP2 satellite imagery into
matched 8-bit RGB patch pairs suitable for PyTorch/TensorFlow SR models.

Pipeline stages:
  1. Configuration  (JSON file overrides inline CONFIG dict)
  2. Data Loading & Band Extraction    (rasterio, 16-bit RGB only)
  3. Spatial Coregistration — 3-Stage Pipeline
       A. Coarse Global      (ORB keypoint matching + RANSAC homography)
       B. Sub-pixel Global   (phase correlation FFT shift)
       C. Patch-wise Local   (ECC refinement per extracted patch)
  4. Smart Percentile Scaling          (16-bit → 8-bit, per-channel)
  5. Radiometric Normalization         (linear least-squares regression LR→HR,
                                        per-scene fit with block-level RMSE
                                        outlier rejection, after sen2venus §2.5.4)
  6. Patch Extraction & Quality Filter (sliding window, variance + nodata)

Configuration
-------------
  Edit CONFIG below as defaults, or place a config.json next to this script.
  Any key present in the JSON file overrides the corresponding CONFIG entry.
  JSON path can also be changed via CONFIG_JSON_PATH below.

Dependencies
------------
  pip install rasterio numpy scikit-image opencv-python-headless tqdm
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
from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_JSON_PATH: str = "config.json"

CONFIG: dict = {
    # ── Paths ─────────────────────────────────────────────────────────────────
    "HR_IMAGE_PATH": "Lahore/HR/3df615fc-2ded-4bd0-85de-237901ccb04f_NONE_STD_A/IMG_01_PNEO3_PMS-FS/IMG_PNEO3_STD_202601250555400_PMS-FS_ORT_1e224074-3a2a-4f29-cfe4-ab0a484fbd47_RGB_R1C1.JP2",
    "LR_IMAGE_PATH": "Lahore/LR/IMG_PHR1A_PMS_001/IMG_PHR1A_PMS_202602130556014_ORT_38a71b19-2781-4146-c1f3-561512adaf94_R1C1.JP2",
    "OUTPUT_DIR":    "output_2",

    # ── Band Mappings ─────────────────────────────────────────────────────────
    "HR_RGB_BANDS": [1, 2, 3],  # Pleiades Neo:  Red=1, Green=2, Blue=3
    "LR_RGB_BANDS": [3, 2, 1],  # Pleiades 1A:   Red=3, Green=2, Blue=1

    # ── Patch geometry ────────────────────────────────────────────────────────
    "SCALE_FACTOR":   2,
    "HR_PATCH_SIZE":  256,
    "LR_PATCH_SIZE":  128,   # Derived automatically — do not set manually
    "STRIDE":         128,

    # ── Radiometric parameters ────────────────────────────────────────────────
    "NODATA_VALUE":     0,
    "SATURATED_VALUE":  32767,
    "CLIP_PERCENTILES": [2.0, 98.0],

    # ── Quality-filter thresholds ─────────────────────────────────────────────
    "MAX_NODATA_FRACTION": 0.05,
    "MIN_VARIANCE":        50.0,

    # ── Coregistration — Stage A (ORB) ────────────────────────────────────────
    "COREG_A_ENABLED":       True,
    "COREG_A_MAX_FEATURES":  5000,
    "COREG_A_MATCH_RATIO":   0.75,
    "COREG_A_RANSAC_THRESH": 5.0,
    "COREG_A_DOWNSAMPLE":    0.25,

    # ── Coregistration — Stage B (Phase Correlation) ──────────────────────────
    "COREG_B_ENABLED":         True,
    "COREG_B_DOWNSAMPLE":      0.25,
    "COREG_B_UPSAMPLE_FACTOR": 100,

    # ── Coregistration — Stage C (ECC Patch-wise) ────────────────────────────
    "COREG_C_ENABLED":         True,
    "COREG_C_MAX_ITER":        50,
    "COREG_C_EPS":             1e-4,
    "COREG_C_WARP_MODE":       "translation",
    "COREG_C_DISCARD_ON_FAIL": True,

    # ── Post-alignment quality gates ──────────────────────────────────────────
    "MIN_ECC_SCORE": 0.85,
    "MIN_SSIM":      0.60,

    # ── Module 5 — Radiometric Regression (sen2venus §2.5.4) ─────────────────
    # Linear least-squares fit:  [LR_R, LR_G, LR_B, 1] @ W = [HR_R, HR_G, HR_B]
    # W is a (4, 3) weight matrix estimated once per scene from randomly sampled
    # pixels, after discarding spatial blocks with high RMSE (outlier rejection).
    #
    # RADIOMETRIC_BLOCK_SIZE      — Non-overlapping block edge length (HR pixels)
    #   used to compute per-block RMSE before fitting. Blocks with RMSE above the
    #   threshold are excluded from the pixel sampling pool. Smaller blocks give
    #   finer-grained outlier rejection; larger blocks are faster.
    #
    # RADIOMETRIC_RMSE_THRESHOLD  — Maximum per-block RMSE (in 0-255 uint8 units)
    #   a block may have to be included in the regression fitting pool.
    #   Analogous to the "0.2 reflectance count" threshold in the paper, scaled
    #   to 8-bit space. Default 30 ≈ 12 % of full scale; increase to be more
    #   permissive (include noisier blocks), decrease to be stricter.
    #
    # RADIOMETRIC_N_SAMPLES  — Maximum number of pixel pairs to draw
    #   from the accepted blocks for the lstsq solve. More samples give a more
    #   stable fit at the cost of memory. 100 000 is sufficient for full scenes.
    "RADIOMETRIC_BLOCK_SIZE":     256,
    "RADIOMETRIC_RMSE_THRESHOLD": 40.0,
    "RADIOMETRIC_N_SAMPLES":      100_000,
}


def build_config() -> dict:
    """
    Merge configuration (inline CONFIG < JSON file).
    LR_PATCH_SIZE is always re-derived; any JSON value for it is ignored.
    """
    cfg = CONFIG.copy()

    json_path = Path(__file__).parent / CONFIG_JSON_PATH
    if json_path.exists():
        with open(json_path, "r") as fh:
            json_cfg = json.load(fh)
        cfg.update(json_cfg)
        logging.info("Loaded config overrides from: %s", json_path)
    else:
        logging.info(
            "No JSON config found at '%s' — using inline CONFIG defaults.", json_path
        )

    assert cfg["HR_PATCH_SIZE"] % cfg["SCALE_FACTOR"] == 0, (
        f"HR_PATCH_SIZE ({cfg['HR_PATCH_SIZE']}) must be divisible by "
        f"SCALE_FACTOR ({cfg['SCALE_FACTOR']})."
    )
    cfg["LR_PATCH_SIZE"] = cfg["HR_PATCH_SIZE"] // cfg["SCALE_FACTOR"]
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — DATA LOADING & BAND EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def load_rgb_bands(image_path: str, bands: list = [1, 2, 3]) -> Tuple[np.ndarray, dict]:
    """
    Open a GeoTIFF / JP2 image and extract the specified RGB bands.

    Returns
    -------
    array   : np.ndarray (H, W, 3) uint16
    profile : dict — Rasterio dataset profile (CRS, transform, dtype, …).
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with rasterio.open(image_path) as src:
        max_band = max(bands)
        if src.count < max_band:
            raise ValueError(
                f"{image_path} has {src.count} band(s); band {max_band} was requested."
            )
        data    = src.read(bands).astype(np.uint16)
        profile = src.profile.copy()
        profile.update(count=3)

    array = np.transpose(data, (1, 2, 0))
    logging.info(
        "Loaded %s (bands %s)  →  shape %s  dtype %s  CRS %s",
        path.name, bands, array.shape, array.dtype, profile.get("crs")
    )
    return array, profile


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — SPATIAL COREGISTRATION  (3-Stage Pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def _to_gray_uint8(array: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint16 or uint8 → uint8 grayscale via BT.601 weights."""
    if array.dtype != np.uint8:
        a = array.astype(np.float32)
        a = (a - a.min()) / (a.max() - a.min() + 1e-6) * 255.0
        a = a.astype(np.uint8)
    else:
        a = array
    gray = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return gray.astype(np.uint8)


def _initial_reproject(
    lr_path: str,
    hr_profile: dict,
    hr_height: int,
    hr_width: int,
    lr_bands: list,
) -> np.ndarray:
    """
    Reproject the LR image to the HR CRS and pixel grid using rasterio.warp.
    Returns an (H, W, 3) uint16 array in the HR pixel space.
    """
    dst_crs       = hr_profile["crs"]
    dst_transform = hr_profile["transform"]
    lr_dst        = np.zeros((3, hr_height, hr_width), dtype=np.uint16)

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
    lr_reprojected = np.clip(lr_reprojected, 0, 32767).astype(np.uint16)
    logging.info(
        "Initial reproject complete (bands %s)  →  shape %s", lr_bands, lr_reprojected.shape
    )
    return lr_reprojected


def coregister_stage_a_orb(
    hr_array: np.ndarray,
    lr_array: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Stage A — Coarse Global Alignment via ORB Keypoints + RANSAC Homography."""
    if not cfg["COREG_A_ENABLED"]:
        logging.info("Stage A (ORB) disabled — skipping.")
        return lr_array

    scale   = cfg["COREG_A_DOWNSAMPLE"]
    H, W    = hr_array.shape[:2]
    small_h = max(1, int(H * scale))
    small_w = max(1, int(W * scale))

    hr_gray  = _to_gray_uint8(hr_array)
    lr_gray  = _to_gray_uint8(lr_array)
    hr_small = cv2.resize(hr_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    lr_small = cv2.resize(lr_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    orb               = cv2.ORB_create(nfeatures=cfg["COREG_A_MAX_FEATURES"])
    kp_hr, des_hr     = orb.detectAndCompute(hr_small, None)
    kp_lr, des_lr     = orb.detectAndCompute(lr_small, None)

    if des_hr is None or des_lr is None or len(kp_hr) < 4 or len(kp_lr) < 4:
        logging.warning("Stage A: too few keypoints detected — skipping homography.")
        return lr_array

    matcher     = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des_lr, des_hr, k=2)

    good = [m for m, n in raw_matches if m.distance < cfg["COREG_A_MATCH_RATIO"] * n.distance]
    logging.info("Stage A: %d good matches from %d raw pairs.", len(good), len(raw_matches))

    if len(good) < 4:
        logging.warning("Stage A: fewer than 4 good matches — skipping homography.")
        return lr_array

    src_pts  = np.float32([kp_lr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts  = np.float32([kp_hr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_small, inlier_mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, cfg["COREG_A_RANSAC_THRESH"]
    )

    if H_small is None:
        logging.warning("Stage A: RANSAC failed to find a valid homography — skipping.")
        return lr_array

    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    logging.info("Stage A: homography accepted with %d RANSAC inliers.", n_inliers)

    S     = np.array([[1.0/scale, 0, 0], [0, 1.0/scale, 0], [0, 0, 1]], dtype=np.float64)
    S_inv = np.array([[scale, 0, 0],     [0, scale, 0],     [0, 0, 1]], dtype=np.float64)
    H_full = S @ H_small @ S_inv

    lr_warped = np.zeros_like(lr_array)
    for c in range(3):
        lr_warped[:, :, c] = np.clip(cv2.warpPerspective(
            lr_array[:, :, c].astype(np.float32), H_full, (W, H),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ), 0, 65535).astype(np.uint16)

    logging.info("Stage A complete — LR warped with full-resolution homography.")
    return lr_warped


def coregister_stage_b_phase(
    hr_array: np.ndarray,
    lr_array: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Stage B — Sub-pixel Global Correction via Phase Cross-Correlation."""
    if not cfg["COREG_B_ENABLED"]:
        logging.info("Stage B (Phase Correlation) disabled — skipping.")
        return lr_array

    scale   = cfg["COREG_B_DOWNSAMPLE"]
    H, W    = hr_array.shape[:2]
    small_h = max(1, int(H * scale))
    small_w = max(1, int(W * scale))

    hr_gray  = _to_gray_uint8(hr_array).astype(np.float32)
    lr_gray  = _to_gray_uint8(lr_array).astype(np.float32)
    hr_small = cv2.resize(hr_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    lr_small = cv2.resize(lr_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    shift_small, error, _ = phase_cross_correlation(
        hr_small, lr_small,
        upsample_factor=cfg["COREG_B_UPSAMPLE_FACTOR"],
    )

    shift_row = shift_small[0] / scale
    shift_col = shift_small[1] / scale

    logging.info(
        "Stage B: sub-pixel shift  row=%.4f px  col=%.4f px  "
        "(error=%.4f, upsample_factor=%d)",
        shift_row, shift_col, error, cfg["COREG_B_UPSAMPLE_FACTOR"],
    )

    M = np.float32([[1, 0, shift_col],
                    [0, 1, shift_row]])

    lr_shifted = np.zeros_like(lr_array)
    for c in range(3):
        lr_shifted[:, :, c] = np.clip(cv2.warpAffine(
            lr_array[:, :, c].astype(np.float32), M, (W, H),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ), 0, 65535).astype(np.uint16)

    logging.info("Stage B complete — sub-pixel translation applied.")
    return lr_shifted


def _ecc_warp_mode(mode_str: str) -> int:
    modes = {"translation": cv2.MOTION_TRANSLATION, "euclidean": cv2.MOTION_EUCLIDEAN}
    if mode_str not in modes:
        raise ValueError(
            f"COREG_C_WARP_MODE must be 'translation' or 'euclidean', got '{mode_str}'."
        )
    return modes[mode_str]


def coregister_stage_c_patch_ecc(
    hr_patch: np.ndarray,
    lr_patch: np.ndarray,
    cfg: dict,
) -> Tuple[Optional[np.ndarray], bool, float]:
    """
    Stage C — Patch-wise Local Refinement via Enhanced Correlation Coefficient (ECC).

    Returns
    -------
    lr_refined : np.ndarray (H, W, 3) uint8 | None
    success    : bool
    cc_score   : float  — ECC correlation coefficient in [0, 1].
    """
    if not cfg["COREG_C_ENABLED"]:
        return lr_patch, True, 1.0

    warp_mode = _ecc_warp_mode(cfg["COREG_C_WARP_MODE"])
    hr_gray   = _to_gray_uint8(hr_patch).astype(np.float32)
    lr_gray   = _to_gray_uint8(lr_patch).astype(np.float32)

    warp_init = np.eye(2, 3, dtype=np.float32)
    criteria  = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        cfg["COREG_C_MAX_ITER"],
        cfg["COREG_C_EPS"],
    )

    try:
        cc_score, warp_matrix = cv2.findTransformECC(
            hr_gray, lr_gray, warp_init, warp_mode, criteria
        )
    except cv2.error as exc:
        logging.debug("Stage C ECC failed on patch: %s", exc)
        return (None, False, 0.0) if cfg["COREG_C_DISCARD_ON_FAIL"] else (lr_patch, False, 0.0)

    H, W       = hr_patch.shape[:2]
    lr_refined = np.zeros_like(lr_patch)
    for c in range(3):
        lr_refined[:, :, c] = np.clip(cv2.warpAffine(
            lr_patch[:, :, c].astype(np.float32), warp_matrix, (W, H),
            flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        ), 0, 255).astype(np.uint8)

    return lr_refined, True, float(cc_score)


def coregister_full(
    lr_path: str,
    hr_array: np.ndarray,
    hr_profile: dict,
    hr_height: int,
    hr_width: int,
    lr_bands: list,
    cfg: dict,
) -> np.ndarray:
    """Orchestrate Stages A and B. Stage C runs per-patch in Module 6."""
    logging.info("=== MODULE 3 — Initial rasterio reproject ===")
    lr_reprojected = _initial_reproject(
        lr_path, hr_profile, hr_height, hr_width, lr_bands
    )

    logging.info("=== MODULE 3 — Stage A: Coarse Global (ORB) ===")
    lr_after_a = coregister_stage_a_orb(hr_array, lr_reprojected, cfg)

    logging.info("=== MODULE 3 — Stage B: Sub-pixel Global (Phase Correlation) ===")
    lr_after_b = coregister_stage_b_phase(hr_array, lr_after_a, cfg)

    return lr_after_b


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — SMART PERCENTILE SCALING  (16-bit → 8-bit)
# ─────────────────────────────────────────────────────────────────────────────

def scale_to_uint8(
    array: np.ndarray,
    nodata_value: int,
    saturated_value: int,
    clip_percentiles: Tuple[float, float],
) -> np.ndarray:
    """
    Convert a 16-bit RGB array to 8-bit using per-channel percentile stretching.
    """
    p_lo, p_hi  = clip_percentiles
    float_array = array.astype(np.float32)
    result      = np.zeros_like(float_array)

    for c in range(3):
        channel    = float_array[:, :, c]
        valid_mask = (
            (array[:, :, c] > nodata_value) &
            (array[:, :, c] < saturated_value)
        )
        valid_pixels = channel[valid_mask]

        if valid_pixels.size == 0:
            logging.warning("Channel %d has no valid pixels; skipping.", c)
            result[:, :, c] = 0
            continue

        v_min = np.percentile(valid_pixels, p_lo)
        v_max = np.percentile(valid_pixels, p_hi)

        if v_max == v_min:
            logging.warning("Channel %d has zero dynamic range; skipping.", c)
            result[:, :, c] = 0
            continue

        clipped           = np.clip(channel, v_min, v_max)
        normalised        = (clipped - v_min) / (v_max - v_min)
        result[:, :, c]  = normalised

        logging.debug(
            "Channel %d  p%.1f=%.1f  p%.1f=%.1f", c, p_lo, v_min, p_hi, v_max
        )

    return np.clip(np.round(result * 255.0), 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — RADIOMETRIC NORMALISATION  (Linear Least-Squares Regression)
# ─────────────────────────────────────────────────────────────────────────────
#
# Method (after sen2venus dataset paper §2.5.4):
#   A single per-scene linear regression is fitted between all source (LR)
#   and target (HR) bands using randomly sampled pixels:
#
#       S = V @ W
#
#   where:
#     V  — design matrix (n, 4): columns are [LR_R, LR_G, LR_B, 1] (bias term)
#     S  — target matrix (n, 3): columns are [HR_R, HR_G, HR_B]
#     W  — weight matrix (4, 3): solved via least squares
#
#   The cross-band terms (off-diagonal of W) allow the model to compensate for
#   spectral response differences between the two sensors where LR-green energy,
#   for instance, partially bleeds into HR-red.
#
#   Outlier rejection (§2.5.4):
#     The scene is tiled into non-overlapping blocks of RADIOMETRIC_BLOCK_SIZE.
#     For each block, channel-averaged RMSE between LR and HR is computed.
#     Blocks with RMSE > RADIOMETRIC_RMSE_THRESHOLD are excluded from the
#     pixel sampling pool before fitting. This prevents cloud shadows, recent
#     construction, seasonal phenology changes, or temporal snow-melt events
#     from biasing the regression towards the wrong spectral offset.
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_apply_radiometric_regression(
    lr_uint8: np.ndarray,
    hr_uint8: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """
    Fit a linear radiometric model from LR pixel values to HR pixel values
    and apply it to the full LR scene.

    The model is:
        [HR_R, HR_G, HR_B] = [LR_R, LR_G, LR_B, 1] @ W
    where W (4, 3) is solved by ordinary least squares on randomly sampled
    pixels drawn from spatial blocks that pass the RMSE outlier filter.

    Parameters
    ----------
    lr_uint8 : np.ndarray (H, W, 3) uint8 — LR image after percentile scaling.
    hr_uint8 : np.ndarray (H, W, 3) uint8 — HR image after percentile scaling.
    cfg      : dict — Pipeline configuration.

    Returns
    -------
    lr_adjusted : np.ndarray (H, W, 3) uint8
        LR image radiometrically matched to HR via least-squares regression.
    """
    block_size   = cfg["RADIOMETRIC_BLOCK_SIZE"]
    rmse_thresh  = cfg["RADIOMETRIC_RMSE_THRESHOLD"]
    n_samples    = cfg["RADIOMETRIC_N_SAMPLES"]

    H, W         = lr_uint8.shape[:2]
    lr_f         = lr_uint8.astype(np.float32)
    hr_f         = hr_uint8.astype(np.float32)

    # ── Step 1: Block-level RMSE outlier rejection ────────────────────────────
    # Tile the scene into non-overlapping blocks and compute per-block RMSE
    # averaged across all three channels.  Blocks above the threshold are
    # excluded from the regression fitting pool (§2.5.4 outlier rejection).

    accepted_blocks  = []   # list of (row_start, col_start)
    rejected_blocks  = 0
    total_blocks     = 0

    row_starts = range(0, H - block_size + 1, block_size)
    col_starts = range(0, W - block_size + 1, block_size)

    for r in row_starts:
        for c in col_starts:
            total_blocks += 1
            lr_block = lr_f[r:r + block_size, c:c + block_size, :]
            hr_block = hr_f[r:r + block_size, c:c + block_size, :]

            # Mean RMSE across the three channels for this block
            diff    = lr_block - hr_block
            rmse    = float(np.sqrt(np.mean(diff ** 2)))

            if rmse > rmse_thresh:
                rejected_blocks += 1
                logging.debug(
                    "Block (%d, %d) excluded from regression pool: RMSE=%.2f > %.2f",
                    r, c, rmse, rmse_thresh,
                )
            else:
                accepted_blocks.append((r, c))

    n_accepted = len(accepted_blocks)
    logging.info(
        "Radiometric regression — block RMSE filter: "
        "%d / %d blocks accepted (%.1f %%), %d rejected (RMSE > %.1f)",
        n_accepted, total_blocks,
        100.0 * n_accepted / max(total_blocks, 1),
        rejected_blocks, rmse_thresh,
    )

    if n_accepted == 0:
        logging.warning(
            "All blocks exceeded the RMSE threshold. "
            "Falling back to using all blocks for regression. "
            "Consider raising RADIOMETRIC_RMSE_THRESHOLD."
        )
        accepted_blocks = [(r, c) for r in row_starts for c in col_starts]
        n_accepted = len(accepted_blocks)

    # ── Step 2: Random pixel sampling from accepted blocks ────────────────────
    # Determine how many pixels to draw per block for an approximately uniform
    # sample across the scene.

    pixels_per_block = max(1, n_samples // n_accepted)
    block_pixels     = block_size * block_size

    lr_samples_list = []
    hr_samples_list = []

    rng = np.random.default_rng(seed=42)

    for r, c in accepted_blocks:
        lr_block = lr_f[r:r + block_size, c:c + block_size, :].reshape(-1, 3)
        hr_block = hr_f[r:r + block_size, c:c + block_size, :].reshape(-1, 3)

        n_draw = min(pixels_per_block, block_pixels)
        idx    = rng.choice(block_pixels, size=n_draw, replace=False)
        lr_samples_list.append(lr_block[idx])
        hr_samples_list.append(hr_block[idx])

    lr_pixels = np.concatenate(lr_samples_list, axis=0)   # (n_total, 3)
    hr_pixels = np.concatenate(hr_samples_list, axis=0)   # (n_total, 3)
    n_total   = lr_pixels.shape[0]

    logging.info(
        "Radiometric regression — sampled %d pixel pairs from %d accepted blocks.",
        n_total, n_accepted,
    )

    # ── Step 3: Least-squares fit ─────────────────────────────────────────────
    # Build design matrix V = [LR_R, LR_G, LR_B, 1] and solve V @ W ≈ HR_RGB.
    # np.linalg.lstsq returns W of shape (4, 3).

    V = np.column_stack([lr_pixels, np.ones(n_total, dtype=np.float32)])  # (n, 4)
    S = hr_pixels                                                          # (n, 3)

    weights, residuals, rank, sv = np.linalg.lstsq(V, S, rcond=None)      # weights: (4, 3)

    logging.info(
        "Radiometric regression fitted.  Matrix rank: %d  "
        "Residual sum: %s",
        rank,
        f"{residuals.sum():.4f}" if residuals.size > 0 else "N/A (underdetermined)",
    )
    logging.info(
        "Regression weights (4×3 — rows: R_coeff, G_coeff, B_coeff, bias):\n%s",
        np.array2string(weights, precision=5, suppress_small=True),
    )

    # Validate: compute fitted RMSE on the training sample as a sanity check
    lr_pred    = V @ weights                         # (n, 3)
    train_rmse = float(np.sqrt(np.mean((lr_pred - S) ** 2)))
    logging.info("Regression training RMSE (on sampled pixels): %.4f", train_rmse)

    # ── Step 4: Apply regression to entire LR scene ───────────────────────────
    # Reshape LR to (H*W, 3), prepend bias column, multiply by weights, reshape back.

    lr_flat = lr_f.reshape(-1, 3)                                          # (H*W, 3)
    n_px    = int(H) * int(W)                                              # explicit Python int
    V_full  = np.column_stack([lr_flat, np.ones(n_px, dtype=np.float32)]) # (H*W, 4)
    lr_adj  = V_full @ weights                                             # (H*W, 3)

    lr_adjusted = np.clip(np.round(lr_adj), 0, 255).astype(np.uint8).reshape(H, W, 3)

    logging.info(
        "Radiometric regression applied to full LR scene  →  shape %s", lr_adjusted.shape
    )
    return lr_adjusted


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — PATCH EXTRACTION & QUALITY FILTERING  (with Stage C ECC)
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_save_patches(
    hr_uint8: np.ndarray,
    lr_adjusted: np.ndarray,
    cfg: dict,
) -> int:
    """
    Slide a window over the HR scene, extract paired HR/LR patches, apply
    quality filters + Stage C local ECC refinement, and save valid pairs as PNGs.

    Parameters
    ----------
    hr_uint8    : np.ndarray (H, W, 3) uint8 — 8-bit HR scene.
    lr_adjusted : np.ndarray (H, W, 3) uint8 — 8-bit LR scene after regression.
    cfg         : dict — Final merged configuration.

    Returns
    -------
    saved_count : int — Number of patch pairs written to disk.
    """
    hr_patch_size   = cfg["HR_PATCH_SIZE"]
    lr_patch_size   = cfg["LR_PATCH_SIZE"]
    stride          = cfg["STRIDE"]
    nodata_value    = cfg["NODATA_VALUE"]
    max_nodata_frac = cfg["MAX_NODATA_FRACTION"]
    min_variance    = cfg["MIN_VARIANCE"]
    output_dir      = Path(cfg["OUTPUT_DIR"])

    hr_out_dir = output_dir / "hr"
    lr_out_dir = output_dir / "lr"
    hr_out_dir.mkdir(parents=True, exist_ok=True)
    lr_out_dir.mkdir(parents=True, exist_ok=True)

    H, W, _       = hr_uint8.shape
    row_starts    = list(range(0, H - hr_patch_size + 1, stride))
    col_starts    = list(range(0, W - hr_patch_size + 1, stride))
    total_windows = len(row_starts) * len(col_starts)

    saved_count        = 0
    skipped_nodata     = 0
    skipped_variance   = 0
    skipped_ecc        = 0
    skipped_ecc_score  = 0
    skipped_ssim       = 0

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

            # Extract HR patch
            hr_patch = hr_uint8[row : row + hr_patch_size,
                                col : col + hr_patch_size, :]

            # Quality gate: NODATA fraction
            nodata_mask = np.any(hr_patch == nodata_value, axis=2)
            if nodata_mask.mean() > max_nodata_frac:
                skipped_nodata += 1
                continue

            # Quality gate: Variance / texture
            mean_var = float(np.var(hr_patch.astype(np.float32), axis=(0, 1)).mean())
            if mean_var < min_variance:
                skipped_variance += 1
                continue

            # Extract LR patch at HR spatial size (before downscaling)
            lr_patch_full = lr_adjusted[row : row + hr_patch_size,
                                        col : col + hr_patch_size, :]

            # Stage C: patch-wise local ECC refinement
            lr_refined, ecc_ok, cc_score = coregister_stage_c_patch_ecc(
                hr_patch, lr_patch_full, cfg
            )
            if lr_refined is None or (not ecc_ok and cfg["COREG_C_DISCARD_ON_FAIL"]):
                skipped_ecc += 1
                continue

            # Quality gate: ECC correlation coefficient
            min_ecc_score = cfg.get("MIN_ECC_SCORE", 0.0)
            if cc_score < min_ecc_score:
                logging.debug(
                    "Patch (%d,%d) discarded: ECC score %.4f < %.4f",
                    row, col, cc_score, min_ecc_score,
                )
                skipped_ecc_score += 1
                continue

            # Quality gate: SSIM on luminance (post-warp)
            min_ssim = cfg.get("MIN_SSIM", 0.0)
            if min_ssim > 0.0:
                hr_gray_patch = _to_gray_uint8(hr_patch)
                lr_gray_patch = _to_gray_uint8(lr_refined)
                patch_ssim    = ssim(hr_gray_patch, lr_gray_patch, data_range=255)
                if patch_ssim < min_ssim:
                    logging.debug(
                        "Patch (%d,%d) discarded: SSIM %.4f < %.4f",
                        row, col, patch_ssim, min_ssim,
                    )
                    skipped_ssim += 1
                    continue

            # Downsample LR patch to LR_PATCH_SIZE
            lr_patch = cv2.resize(
                lr_refined,
                (lr_patch_size, lr_patch_size),
                interpolation=cv2.INTER_CUBIC,
            )
            lr_patch = np.clip(lr_patch, 0, 255).astype(np.uint8)

            # Save pair — cv2.imwrite expects BGR
            patch_name = f"patch_{saved_count:06d}.png"
            cv2.imwrite(
                str(hr_out_dir / patch_name),
                cv2.cvtColor(hr_patch, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(lr_out_dir / patch_name),
                cv2.cvtColor(lr_patch, cv2.COLOR_RGB2BGR),
            )
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

def _setup_logging(output_dir: str) -> None:
    """
    Configure logging to both stdout and a timestamped log file in output_dir.

    The file handler captures everything from DEBUG level upward so that
    per-patch debug messages (block RMSE values, ECC/SSIM discard reasons)
    are preserved for post-run inspection even when the console only shows INFO.
    """
    import time

    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path  = log_dir / f"pipeline_{timestamp}.log"

    fmt     = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%H:%M:%S"

    # Console handler — INFO and above only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    # File handler — DEBUG and above (captures per-patch discard reasons)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    logging.info("Log file: %s", log_path)


def main() -> None:
    # Step 0: Build config first so we know OUTPUT_DIR before any logging
    # Use a minimal bootstrap logger until the file handler is ready
    logging.basicConfig(level=logging.WARNING)
    cfg = build_config()

    # Step 0b: Set up full logging (stdout + file in OUTPUT_DIR)
    _setup_logging(cfg["OUTPUT_DIR"])

    logging.info("Configuration:\n%s", "\n".join(f"  {k}: {v}" for k, v in cfg.items()))

    # Step 1: Load HR image
    logging.info("=== MODULE 2: Loading HR image ===")
    hr_raw, hr_profile = load_rgb_bands(
        cfg["HR_IMAGE_PATH"], bands=cfg.get("HR_RGB_BANDS", [1, 2, 3])
    )
    hr_height, hr_width = hr_raw.shape[:2]

    # Step 2: Coregistration — Stages A + B (Stage C applied per-patch in Module 6)
    lr_coregistered = coregister_full(
        lr_path=cfg["LR_IMAGE_PATH"],
        hr_array=hr_raw,
        hr_profile=hr_profile,
        hr_height=hr_height,
        hr_width=hr_width,
        lr_bands=cfg.get("LR_RGB_BANDS", [3, 2, 1]),
        cfg=cfg,
    )

    # Step 3: 16-bit → 8-bit scaling
    logging.info("=== MODULE 4: Percentile scaling  HR ===")
    hr_uint8 = scale_to_uint8(
        hr_raw,
        cfg["NODATA_VALUE"],
        cfg["SATURATED_VALUE"],
        tuple(cfg["CLIP_PERCENTILES"]),
    )

    logging.info("=== MODULE 4: Percentile scaling  LR ===")
    lr_uint8 = scale_to_uint8(
        lr_coregistered,
        cfg["NODATA_VALUE"],
        cfg["SATURATED_VALUE"],
        tuple(cfg["CLIP_PERCENTILES"]),
    )

    # Step 4: Linear radiometric regression (replaces histogram matching)
    logging.info("=== MODULE 5: Radiometric regression (sen2venus §2.5.4) ===")
    lr_adjusted = fit_and_apply_radiometric_regression(lr_uint8, hr_uint8, cfg)

    # Step 5: Patch extraction (includes Stage C ECC per patch)
    logging.info("=== MODULE 6: Patch extraction & quality filtering ===")
    n_saved = extract_and_save_patches(hr_uint8, lr_adjusted, cfg)

    logging.info(
        "Pipeline complete. %d patch pairs written to: %s",
        n_saved, cfg["OUTPUT_DIR"]
    )


if __name__ == "__main__":
    main()