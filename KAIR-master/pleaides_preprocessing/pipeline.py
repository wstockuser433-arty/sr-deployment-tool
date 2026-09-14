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
  5. Radiometric Normalization         (histogram matching LR→HR)
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
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Path to the JSON config file. Resolved relative to this script's directory.
# Any key present in the JSON overrides the matching entry in CONFIG below.
CONFIG_JSON_PATH: str = "config.json"

# Inline default configuration dictionary — edit these as your baseline values.
CONFIG: dict = {
    # ── Paths ─────────────────────────────────────────────────────────────────
    "HR_IMAGE_PATH": "Lahore/HR/3df615fc-2ded-4bd0-85de-237901ccb04f_NONE_STD_A/IMG_01_PNEO3_PMS-FS/IMG_PNEO3_STD_202601250555400_PMS-FS_ORT_1e224074-3a2a-4f29-cfe4-ab0a484fbd47_RGB_R1C1.JP2",
    "LR_IMAGE_PATH": "Lahore/LR/IMG_PHR1A_PMS_001/IMG_PHR1A_PMS_202602130556014_ORT_38a71b19-2781-4146-c1f3-561512adaf94_R1C1.JP2",
    "OUTPUT_DIR":    "output_2",

    # ── Band Mappings ─────────────────────────────────────────────────────────
    "HR_RGB_BANDS": [1, 2, 3],  # Pleiades Neo:  Red=1, Green=2, Blue=3
    "LR_RGB_BANDS": [3, 2, 1],  # Pleiades 1A:   Red=3, Green=2, Blue=1

    # ── Patch geometry ────────────────────────────────────────────────────────
    "SCALE_FACTOR":   2,    # Spatial downscaling factor (HR / LR)
    "HR_PATCH_SIZE":  256,  # Output HR patch edge length in pixels
    "LR_PATCH_SIZE":  128,  # Derived automatically — do not set manually
    "STRIDE":         128,  # Sliding-window stride in HR pixels

    # ── Radiometric parameters ────────────────────────────────────────────────
    "NODATA_VALUE":     0,
    "SATURATED_VALUE":  32767,
    "CLIP_PERCENTILES": [2.0, 98.0],

    # ── Quality-filter thresholds ─────────────────────────────────────────────
    "MAX_NODATA_FRACTION": 0.05,
    "MIN_VARIANCE":        50.0,

    # ── Coregistration — Stage A (ORB) ────────────────────────────────────────
    # ORB is run on grayscale versions of the images downscaled to a manageable
    # size. The homography found at that scale is upscaled to full resolution
    # before the warp is applied.
    "COREG_A_ENABLED":       True,
    "COREG_A_MAX_FEATURES":  5000,  # Max ORB keypoints to detect per image
    "COREG_A_MATCH_RATIO":   0.75,  # Lowe's ratio test threshold
    "COREG_A_RANSAC_THRESH": 5.0,   # RANSAC reprojection error threshold (px)
    "COREG_A_DOWNSAMPLE":    0.25,  # Fraction to downsample for feature detection

    # ── Coregistration — Stage B (Phase Correlation) ──────────────────────────
    # Applied after Stage A to correct any residual sub-pixel global shift.
    # Operates on a grayscale luminance image.
    "COREG_B_ENABLED":         True,
    "COREG_B_DOWNSAMPLE":      0.25,  # Fraction to downsample before FFT
    "COREG_B_UPSAMPLE_FACTOR": 100,   # Sub-pixel precision = 1/upsample_factor px

    # ── Coregistration — Stage C (ECC Patch-wise) ────────────────────────────
    # Applied per-patch during extraction to correct local parallax from
    # buildings, bridges, and terrain. ECC runs on the luminance channel only.
    "COREG_C_ENABLED":         True,
    "COREG_C_MAX_ITER":        50,            # ECC iteration limit per patch
    "COREG_C_EPS":             1e-4,          # ECC convergence epsilon
    "COREG_C_WARP_MODE":       "translation", # "translation" or "euclidean"
    "COREG_C_DISCARD_ON_FAIL": True,          # Discard patch pair if ECC diverges

    # ── Post-alignment quality gates ──────────────────────────────────────────
    # Both checks run on the luminance channel of the HR patch vs the locally
    # refined (but not yet downsampled) LR patch, measuring true spatial
    # agreement at the HR pixel scale before any rescaling is applied.
    #
    # ECC Correlation Coefficient — returned directly by cv2.findTransformECC.
    # Range [0, 1]; 1 = perfect normalised cross-correlation.
    # Catches patches where moving objects (cars, construction) pulled the warp
    # away from the background consensus — the ECC score will be low even if
    # the algorithm technically "converged".
    "MIN_ECC_SCORE": 0.85,   # Discard patch if final ECC CC < this value

    # SSIM — Structural Similarity Index (skimage.metrics.structural_similarity).
    # Range [-1, 1]; 1 = identical structure. Applied after the ECC warp so it
    # measures residual structural mismatch ECC could not resolve (a car that
    # moved between acquisitions, a new shadow, or seasonal vegetation change).
    # More sensitive than pixel-level metrics because it jointly compares local
    # luminance, contrast, and structure.
    "MIN_SSIM": 0.60,        # Discard patch if post-warp SSIM < this value
}


def build_config() -> dict:
    """
    Merge configuration in priority order (lowest → highest):
      1. Inline CONFIG dict  (baseline defaults above)
      2. JSON file at CONFIG_JSON_PATH  (overrides any key present in the file)

    LR_PATCH_SIZE is always derived from HR_PATCH_SIZE // SCALE_FACTOR and
    must never be set manually — any value in the JSON for this key is ignored.

    Returns the final merged configuration dict.
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

    # Derived / validated fields (always recomputed, never user-set)
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

    Parameters
    ----------
    image_path : str       — Filesystem path to the source raster.
    bands      : list[int] — 1-based rasterio band indices for [R, G, B].

    Returns
    -------
    array   : np.ndarray (H, W, 3) uint16 — Native 16-bit pixel data.
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
        data    = src.read(bands).astype(np.uint16)   # (3, H, W)
        profile = src.profile.copy()
        profile.update(count=3)

    array = np.transpose(data, (1, 2, 0))             # → (H, W, 3)
    logging.info(
        "Loaded %s (bands %s)  →  shape %s  dtype %s  CRS %s",
        path.name, bands, array.shape, array.dtype, profile.get("crs")
    )
    return array, profile


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — SPATIAL COREGISTRATION  (3-Stage Pipeline)
# ─────────────────────────────────────────────────────────────────────────────
#
# Stage A — Coarse Global (ORB + RANSAC Homography)
# --------------------------------------------------
# Purpose : Resolve large metadata / projection mismatches between sensors.
#           Gets pixel-level alignment to within ~5–10 px so Stage B can
#           converge. Very cheap because it runs on a heavily downscaled image.
# Method  : Detect ORB keypoints on both grayscale images at COREG_A_DOWNSAMPLE
#           resolution. Match descriptors with BFMatcher + Lowe's ratio test.
#           Estimate a homography via RANSAC, scale it back to full resolution,
#           then apply cv2.warpPerspective to the LR image.
#
# Stage B — Sub-pixel Global (Phase Correlation)
# -----------------------------------------------
# Purpose : Correct the residual 0.1–0.5 px shift that remains after Stage A.
#           SR models are extremely sensitive to even sub-pixel misalignment;
#           a half-pixel offset blurs the implicit target the model must learn.
# Method  : Compute phase_cross_correlation (FFT-based) on downscaled
#           luminance images with a high upsample_factor for sub-pixel
#           precision. The shift is scaled back to full resolution and applied
#           via cv2.warpAffine with a pure translation matrix.
#
# Stage C — Patch-wise Local Refinement (ECC)
# --------------------------------------------
# Purpose : Handle local parallax caused by buildings, bridges, and terrain
#           relief that a single global transform cannot model. Without this,
#           the model learns "blur = feature" on misaligned tall structures.
# Method  : For each extracted patch pair, run cv2.findTransformECC on the
#           luminance channels. A translation or euclidean warp matrix is
#           estimated and applied to the LR patch before downsampling.
#           Patches where ECC fails to converge are optionally discarded.
# ─────────────────────────────────────────────────────────────────────────────

def _to_gray_uint8(array: np.ndarray) -> np.ndarray:
    """
    Convert an (H, W, 3) uint16 or uint8 RGB array to a uint8 grayscale image
    using ITU-R BT.601 luminance weights, suitable for cv2 feature / ECC calls.
    """
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
    This resolves projection differences before any image-based coregistration.
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

    lr_reprojected = np.transpose(lr_dst, (1, 2, 0))   # → (H, W, 3)

    # Clamp out bicubic undershoot/overshoot artifacts.
    # The cubic kernel uses negative lobes that can produce values outside the
    # valid sensor DN range [0, 32767]. On uint16 storage, negative values wrap
    # around to 65535 and above, which corrupts downstream percentile scaling.
    # Clamping here ensures no artifact values survive into the pipeline.
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
    """
    Stage A — Coarse Global Alignment via ORB Keypoints + RANSAC Homography.

    Detects ORB features on downscaled grayscale versions of both images,
    matches with Lowe's ratio test, estimates a homography with RANSAC,
    scales it back to full resolution, and warps the LR image channel-by-channel.

    Parameters
    ----------
    hr_array : np.ndarray (H, W, 3) uint16 — Full-res HR image.
    lr_array : np.ndarray (H, W, 3) uint16 — Reprojected LR image (same shape).
    cfg      : dict — Pipeline configuration.

    Returns
    -------
    lr_aligned : np.ndarray (H, W, 3) uint16
        LR after homography warp, or the original lr_array if insufficient matches.
    """
    if not cfg["COREG_A_ENABLED"]:
        logging.info("Stage A (ORB) disabled — skipping.")
        return lr_array

    scale   = cfg["COREG_A_DOWNSAMPLE"]
    H, W    = hr_array.shape[:2]
    small_h = max(1, int(H * scale))
    small_w = max(1, int(W * scale))

    # Downscale and convert to uint8 grayscale for feature detection
    hr_gray  = _to_gray_uint8(hr_array)
    lr_gray  = _to_gray_uint8(lr_array)
    hr_small = cv2.resize(hr_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    lr_small = cv2.resize(lr_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # ORB detection & BFMatcher
    orb               = cv2.ORB_create(nfeatures=cfg["COREG_A_MAX_FEATURES"])
    kp_hr, des_hr     = orb.detectAndCompute(hr_small, None)
    kp_lr, des_lr     = orb.detectAndCompute(lr_small, None)

    if des_hr is None or des_lr is None or len(kp_hr) < 4 or len(kp_lr) < 4:
        logging.warning("Stage A: too few keypoints detected — skipping homography.")
        return lr_array

    matcher     = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des_lr, des_hr, k=2)

    # Lowe's ratio test
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

    # Scale homography from small-image coordinates to full-resolution:
    # H_full = S * H_small * S^-1  where S is the upscale matrix
    S     = np.array([[1.0/scale, 0, 0], [0, 1.0/scale, 0], [0, 0, 1]], dtype=np.float64)
    S_inv = np.array([[scale, 0, 0],     [0, scale, 0],     [0, 0, 1]], dtype=np.float64)
    H_full = S @ H_small @ S_inv

    # Warp each channel independently to preserve uint16 precision
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
    """
    Stage B — Sub-pixel Global Correction via Phase Cross-Correlation.

    Computes the residual global shift between HR and LR at sub-pixel precision
    using an FFT-based phase correlation on downscaled luminance images.
    The measured shift is scaled to full resolution and applied as a pure
    translation warp.

    Parameters
    ----------
    hr_array : np.ndarray (H, W, 3) uint16 — HR image (Stage A reference).
    lr_array : np.ndarray (H, W, 3) uint16 — LR image after Stage A.
    cfg      : dict — Pipeline configuration.

    Returns
    -------
    lr_shifted : np.ndarray (H, W, 3) uint16 — LR after sub-pixel translation.
    """
    if not cfg["COREG_B_ENABLED"]:
        logging.info("Stage B (Phase Correlation) disabled — skipping.")
        return lr_array

    scale   = cfg["COREG_B_DOWNSAMPLE"]
    H, W    = hr_array.shape[:2]
    small_h = max(1, int(H * scale))
    small_w = max(1, int(W * scale))

    # Downscale to grayscale float for phase correlation
    hr_gray  = _to_gray_uint8(hr_array).astype(np.float32)
    lr_gray  = _to_gray_uint8(lr_array).astype(np.float32)
    hr_small = cv2.resize(hr_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    lr_small = cv2.resize(lr_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Phase cross-correlation: returns (row_shift, col_shift) in downscaled coords
    shift_small, error, _ = phase_cross_correlation(
        hr_small, lr_small,
        upsample_factor=cfg["COREG_B_UPSAMPLE_FACTOR"],
    )

    # Scale shift back to full-resolution pixel units
    shift_row = shift_small[0] / scale
    shift_col = shift_small[1] / scale

    logging.info(
        "Stage B: sub-pixel shift  row=%.4f px  col=%.4f px  "
        "(error=%.4f, upsample_factor=%d)",
        shift_row, shift_col, error, cfg["COREG_B_UPSAMPLE_FACTOR"],
    )

    # Apply as a pure translation: tx=col_shift, ty=row_shift in cv2 convention
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
    """Resolve COREG_C_WARP_MODE string to a cv2 motion type constant."""
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

    Called inside the patch extraction loop for every candidate patch pair.
    Estimates a local warp (translation or euclidean) that maximises the
    normalised cross-correlation between the HR and LR luminance patches,
    then applies it to the LR patch.

    ECC is robust to photometric differences between sensors (contrast, gain)
    because it optimises a normalised objective — more accurate than template
    matching for sub-pixel local parallax.

    Parameters
    ----------
    hr_patch : np.ndarray (H, W, 3) uint8 — HR patch.
    lr_patch : np.ndarray (H, W, 3) uint8 — LR patch at HR resolution (pre-downscale).
    cfg      : dict — Pipeline configuration.

    Returns
    -------
    lr_refined : np.ndarray (H, W, 3) uint8 | None
        Locally refined LR patch, or None if ECC failed and COREG_C_DISCARD_ON_FAIL.
    success : bool — True if ECC converged.
    cc_score : float — Final ECC correlation coefficient in [0, 1]. Returns 0.0
        on failure and 1.0 when Stage C is disabled (no penalty applied).
    """
    if not cfg["COREG_C_ENABLED"]:
        return lr_patch, True, 1.0

    warp_mode = _ecc_warp_mode(cfg["COREG_C_WARP_MODE"])
    hr_gray   = _to_gray_uint8(hr_patch).astype(np.float32)
    lr_gray   = _to_gray_uint8(lr_patch).astype(np.float32)

    warp_init = np.eye(2, 3, dtype=np.float32)   # identity for both translation & euclidean
    criteria  = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        cfg["COREG_C_MAX_ITER"],
        cfg["COREG_C_EPS"],
    )

    try:
        # cc_score is the final normalised cross-correlation value in [0, 1].
        # It is returned as the first element but was previously discarded with _.
        # A low score (< MIN_ECC_SCORE) means moving objects dominated the warp.
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
    """
    Orchestrate Stages A and B of the coregistration pipeline.
    Stage C is applied per-patch inside extract_and_save_patches.

    Returns
    -------
    lr_coregistered : np.ndarray (H, W, 3) uint16
        LR image globally aligned to the HR grid, ready for percentile scaling.
    """
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

    Steps
    -----
    1. Build a valid-pixel mask: exclude NODATA and SATURATED pixels.
    2. Per-channel: compute low/high percentiles on valid pixels only.
    3. Clip the full channel to [p_low, p_high].
    4. Normalise to [0.0, 1.0], multiply by 255, round, cast to uint8.

    Parameters
    ----------
    array            : np.ndarray (H, W, 3) uint16
    nodata_value     : int   — Value marking missing data (typically 0).
    saturated_value  : int   — Value marking saturated pixels (typically 32767).
    clip_percentiles : tuple — (low_pct, high_pct), e.g. (1.0, 99.0).

    Returns
    -------
    uint8_array : np.ndarray (H, W, 3) uint8
    """
    p_lo, p_hi  = clip_percentiles
    float_array = array.astype(np.float32)
    result      = np.zeros_like(float_array)

    for c in range(3):
        channel    = float_array[:, :, c]
        valid_mask = (
            (array[:, :, c] > nodata_value) &      # strictly above 0 (excludes NODATA)
            (array[:, :, c] < saturated_value)      # strictly below 32767 (excludes saturated + warp artifacts)
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
# MODULE 5 — RADIOMETRIC NORMALISATION  (Global Histogram Matching)
# ─────────────────────────────────────────────────────────────────────────────

def normalise_histogram(
    lr_uint8: np.ndarray,
    hr_uint8: np.ndarray,
) -> np.ndarray:
    """
    Match the radiometric distribution of the LR image to the HR image
    so the SR model learns spatial structure, not colour correction.

    Uses skimage.exposure.match_histograms on the full scene before patch
    extraction for a globally consistent colour space.

    Parameters
    ----------
    lr_uint8 : np.ndarray (H, W, 3) uint8 — LR image.
    hr_uint8 : np.ndarray (H, W, 3) uint8 — HR reference image.

    Returns
    -------
    lr_matched : np.ndarray (H, W, 3) uint8
    """
    lr_matched = match_histograms(
        lr_uint8.astype(np.float32),
        hr_uint8.astype(np.float32),
        channel_axis=2,
    )
    lr_matched = np.clip(np.round(lr_matched), 0, 255).astype(np.uint8)
    logging.info("Histogram matching complete.")
    return lr_matched


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — PATCH EXTRACTION & QUALITY FILTERING  (with Stage C ECC)
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_save_patches(
    hr_uint8: np.ndarray,
    lr_matched: np.ndarray,
    cfg: dict,
) -> int:
    """
    Slide a window over the HR scene, extract paired HR/LR patches, apply
    quality filters + Stage C local ECC refinement, and save valid pairs as PNGs.

    For every valid window position:
      1. Extract HR_PATCH_SIZE × HR_PATCH_SIZE from HR.
      2. Apply quality filters (nodata fraction, variance).
      3. Extract the same window from the globally aligned LR.
      4. Run Stage C ECC to correct local parallax on the full-size LR window.
      5. Downsample the refined LR window to LR_PATCH_SIZE via bicubic.
      6. Save both as PNGs with matching filenames.

    Parameters
    ----------
    hr_uint8   : np.ndarray (H, W, 3) uint8 — 8-bit HR scene.
    lr_matched : np.ndarray (H, W, 3) uint8 — 8-bit LR scene (histogram-matched).
    cfg        : dict — Final merged configuration.

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
            lr_patch_full = lr_matched[row : row + hr_patch_size,
                                       col : col + hr_patch_size, :]

            # Stage C: patch-wise local ECC refinement
            lr_refined, ecc_ok, cc_score = coregister_stage_c_patch_ecc(
                hr_patch, lr_patch_full, cfg
            )
            if lr_refined is None or (not ecc_ok and cfg["COREG_C_DISCARD_ON_FAIL"]):
                skipped_ecc += 1
                continue

            # Quality gate: ECC correlation coefficient
            # A low CC score means moving objects (cars, construction) dominated
            # the warp, so the alignment is unreliable even though ECC converged.
            min_ecc_score = cfg.get("MIN_ECC_SCORE", 0.0)
            if cc_score < min_ecc_score:
                logging.debug(
                    "Patch (%d,%d) discarded: ECC score %.4f < %.4f",
                    row, col, cc_score, min_ecc_score,
                )
                skipped_ecc_score += 1
                continue

            # Quality gate: SSIM on luminance channel (post-warp)
            # SSIM catches residual structural mismatch that ECC could not resolve:
            # a car that moved between acquisitions, a new shadow, seasonal
            # vegetation change, or fresh construction. It is computed on the
            # HR-resolution LR patch (before downsampling) for maximum sensitivity.
            min_ssim = cfg.get("MIN_SSIM", 0.0)
            if min_ssim > 0.0:
                hr_gray_patch = _to_gray_uint8(hr_patch)
                lr_gray_patch = _to_gray_uint8(lr_refined)
                patch_ssim = ssim(
                    hr_gray_patch, lr_gray_patch,
                    data_range=255,
                )
                if patch_ssim < min_ssim:
                    logging.debug(
                        "Patch (%d,%d) discarded: SSIM %.4f < %.4f",
                        row, col, patch_ssim, min_ssim,
                    )
                    skipped_ssim += 1
                    continue

            # Downsample LR patch to LR_PATCH_SIZE to enforce SCALE_FACTOR
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

    logging.info(
        "Patch extraction complete: %d saved | %d skipped (nodata) | "
        "%d skipped (low variance) | %d skipped (ECC diverged) | "
        "%d skipped (ECC score < %.2f) | %d skipped (SSIM < %.2f) | "
        "%d total candidates",
        saved_count, skipped_nodata, skipped_variance, skipped_ecc,
        skipped_ecc_score, cfg.get("MIN_ECC_SCORE", 0.0),
        skipped_ssim,       cfg.get("MIN_SSIM", 0.0),
        total_windows,
    )
    return saved_count


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Step 0: Configuration
    cfg = build_config()
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

    # Step 4: Histogram matching
    logging.info("=== MODULE 5: Radiometric normalisation (histogram matching) ===")
    lr_matched = normalise_histogram(lr_uint8, hr_uint8)

    # Step 5: Patch extraction (includes Stage C ECC per patch)
    logging.info("=== MODULE 6: Patch extraction & quality filtering ===")
    n_saved = extract_and_save_patches(hr_uint8, lr_matched, cfg)

    logging.info(
        "Pipeline complete. %d patch pairs written to: %s",
        n_saved, cfg["OUTPUT_DIR"]
    )


if __name__ == "__main__":
    main()