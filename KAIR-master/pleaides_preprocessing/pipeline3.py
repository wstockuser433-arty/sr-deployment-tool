"""
Satellite Imagery Preprocessing Pipeline for Super-Resolution
==============================================================
Preprocesses HR/LR GeoTIFF or JP2 satellite imagery into matched 8-bit RGB
patch pairs suitable for PyTorch/TensorFlow SR models.

HR_IMAGE_PATH / LR_IMAGE_PATH may each be a single file OR a directory:

  - Both directories  : every file is matched to its counterpart by filename
                         stem (same convention as preprocessing_pipeline's
                         run_pipeline.py). Each match is processed as a pair.
  - Both single files  : processed as one pair (legacy single-scene mode).
  - HR present, no LR  : the scene is "unpaired HR" — a synthetic LR is
                         generated via the configured degradation model
                         (DEGRADATION_TYPE) instead of coregistering a real
                         LR capture.
  - LR present, no HR  : the scene is "unpaired LR" — patches are extracted
                         from the LR scene alone (no HR counterpart is
                         produced). Useful for real-world / blind test sets.

Pipeline stages (paired scenes):
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

Unpaired HR scenes skip stages 3 and 5 (there is no real LR to align or
radiometrically match) and instead degrade each accepted HR patch directly
via preprocessing_pipeline.degradation_utils (bsrgan | real_esrgan |
bsrgan_plus | satellite). Unpaired LR scenes skip stages 3, 5 and 6's
HR-dependent quality gates (ECC, SSIM) entirely.

Configuration
-------------
  Edit CONFIG below as defaults, or pass --config path/to/config.json.
  Any key present in the JSON file overrides the corresponding CONFIG entry.
  If --config is omitted, a config.json next to this script is used if present.

Usage
-----
    python pipeline3.py [--config path/to/config.json]

Dependencies
------------
  pip install rasterio numpy scikit-image opencv-python-headless tqdm
  (torch is only required if an unpaired-HR scene triggers degradation)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import cv2
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, transform as window_transform
from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_JSON_PATH: str = "config.json"

CONFIG: dict = {
    # ── Paths — each may be a single file OR a directory ───────────────────
    "HR_IMAGE_PATH": "",
    "LR_IMAGE_PATH": "",
    "OUTPUT_DIR":    "output_patches",

    # ── Directory-mode image discovery ──────────────────────────────────────
    "SUPPORTED_EXTENSIONS": [".tif", ".tiff", ".jp2", ".png", ".jpg", ".jpeg", ".bmp"],

# ── Band Mappings (already correct for these sensors) ──────────────────
    "HR_RGB_BANDS": [1, 2, 3],   # Red=1, Green=2, Blue=3 (Neo)
    "LR_RGB_BANDS": [3, 2, 1],   # Red=3, Green=2, Blue=1 (1A)

    # ── Patch geometry — OPTIMAL FOR SR TRAINING ───────────────────────────
    "SCALE_FACTOR":   2,          # fixed for ×2 SR (standard)
    "HR_PATCH_SIZE":  256,        # SEN2VENµS standard; ideal for most SR backbones
    "LR_PATCH_SIZE":  128,        # auto-derived — do NOT change
    "STRIDE":         256,         # ← CHANGED: 75 % overlap → ~4× more patches than default 128

    # ── Radiometric parameters (8-bit output) ──────────────────────────────
    "NODATA_VALUE":     0,
    "SATURATED_VALUE":  32767,
    "CLIP_PERCENTILES": [2.0, 98.0],   # standard, robust to outliers

    # ── Quality-filter thresholds — tightened for SR quality ───────────────
    "MAX_NODATA_FRACTION": 0.05,      # ← tightened from 0.1 (less cloud/edge junk)
    "MIN_VARIANCE":        120.0,     # ← raised from 50 (ensures textured urban detail; flat areas hurt SR learning)

    # ── Coregistration — keep all stages enabled (critical for multi-sensor) ─
    "COREG_A_ENABLED":       True,
    "COREG_A_MAX_FEATURES":  8000,      # ← increased (more robust on urban Lahore)
    "COREG_A_MATCH_RATIO":   0.75,
    "COREG_A_RANSAC_THRESH": 4.0,       # ← tightened
    "COREG_A_DOWNSAMPLE":    0.25,

    "COREG_B_ENABLED":         True,
    "COREG_B_DOWNSAMPLE":      0.25,
    "COREG_B_UPSAMPLE_FACTOR": 100,

    "COREG_C_ENABLED":         True,
    "COREG_C_MAX_ITER":        100,     # ← increased for better convergence
    "COREG_C_EPS":             1e-5,
    "COREG_C_WARP_MODE":       "translation",
    "COREG_C_DISCARD_ON_FAIL": True,

    # ── Post-alignment quality gates — stricter for clean training pairs ─────
    "MIN_ECC_SCORE": 0.78,    # ← raised from 0.70 (excellent local alignment)
    "MIN_SSIM":      0.60,    # ← raised from 0.60 (strong structural match)

    # ── Radiometric Regression (sen2venus §2.5.4 style) ─────────────────────
    "RADIOMETRIC_BLOCK_SIZE":      256,
    "RADIOMETRIC_RMSE_THRESHOLD":  35.0,   # ← tightened from 40 (dates are only ~19 days apart → very similar scenes)
    "RADIOMETRIC_N_SAMPLES":       150_000, # more samples = stabler fit
    "RADIOMETRIC_POST_HIST_MATCH": True,    # ← KEEP ENABLED (corrects NIR leakage & S-curve differences between Neo & 1A)

    # ── Windowed I/O tuning — controls how much is sampled for each estimate ─
    # step (overview/global-transform, percentile thresholds, radiometric fit,
    # histogram LUT) instead of ever loading a full-resolution scene array.
    # Larger values = more accurate estimates, more I/O; defaults are sized
    # for 10,000-20,000px scenes without materially slowing a run down.
    "COREG_PREVIEW_DECIM_DIM":       2000,  # longest edge (px) of the overview used for Stage A/B + previews
    "PERCENTILE_N_SAMPLE_WINDOWS":   20,    # extra full-res windows sampled for percentile thresholds
    "RADIOMETRIC_N_FIT_WINDOWS":     80,    # candidate blocks read for the regression fit (was: every block)
    "HISTOGRAM_N_SAMPLE_WINDOWS":    40,    # windows sampled to accumulate the histogram-match LUT
    "GDAL_CACHE_MB":                 256,   # explicit cap on GDAL's internal block-read cache (see main())

    # ── Visualisation previews (Part A — written to OUTPUT_DIR/_previews/) ───
    "PREVIEW_ENABLED":      True,
    "PREVIEW_MAX_DIM":      1024,  # longest edge (px) for scene-level preview JPEGs
    "PREVIEW_N_PATCHES":    10,    # patches included in the post-extraction contact sheet
    "PREVIEW_JPEG_QUALITY": 85,

    # ── Unpaired-HR degradation (used only when a scene has HR but no LR) ───
    "DEGRADATION_ENABLED": True,
    "DEGRADATION_TYPE": "satellite",   # "bsrgan" | "real_esrgan" | "bsrgan_plus" | "satellite"
    # Per-type parameter overrides (see preprocessing_pipeline/degradation_utils.py
    # for the full parameter list of each). Omitted keys fall back to that
    # function's own defaults. Left empty here — set via JSON config / GUI.
    "bsrgan": {},
    "real_esrgan": {},
    "bsrgan_plus": {},
    "satellite": {},
}

def build_config(config_path: Optional[str] = None) -> dict:
    """
    Merge configuration (inline CONFIG < JSON file).
    LR_PATCH_SIZE is always re-derived; any JSON value for it is ignored.
    """
    cfg = CONFIG.copy()

    if config_path:
        json_path = Path(config_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")
    else:
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
#
# There is no full-scene loader here by design: every stage from this point
# on reads only decimated overviews (estimate_global_transform,
# estimate_percentile_thresholds) or small windows (fit_radiometric_regression,
# estimate_histogram_match_lut, extract_and_save_patches and its hr_only/
# lr_only variants) directly via rasterio, so a 10,000-20,000px scene never
# needs to be materialized as a single in-memory array.
# ─────────────────────────────────────────────────────────────────────────────


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


def _raw_to_uint16(raw: np.ndarray) -> np.ndarray:
    """Convert raw rasterio output to uint16 for the DN-based pipeline.

    Float sources (surface reflectance, normalised imagery, etc.) have NaN
    mapped to 0 (the nodata sentinel used throughout this pipeline) and are
    multiplied by 10 000 so that values in [0, 1] land in [0, 10 000] —
    consistent with the USGS / ESA Level-2 DN convention and well within the
    [0, 32 767] valid range the quality filters expect.
    Integer sources are cast directly.
    """
    if raw.dtype.kind != 'f':
        return raw.astype(np.uint16)
    scaled = np.where(np.isnan(raw), 0.0, raw * 10_000.0)
    return np.clip(scaled, 0, 32767).astype(np.uint16)


def _read_decimated_overviews(
    hr_path: str,
    lr_path: str,
    hr_bands: list,
    lr_bands: list,
    max_dim: int,
) -> dict:
    """
    Read small, CRS-aligned overviews of the HR and LR scenes — never a
    full-resolution array — for global-transform estimation and previews.

    Returns a dict with: hr_overview, lr_overview (both (h,w,3) uint16, same
    decimated shape), hr_profile, hr_height, hr_width (full-res), decim_scale
    (overview_dim / full_dim).
    """
    with rasterio.open(hr_path) as hr_src:
        hr_height, hr_width = hr_src.height, hr_src.width
        hr_profile = hr_src.profile.copy()
        hr_profile.update(count=3)
        decim_scale = min(1.0, max_dim / max(hr_height, hr_width))
        out_h = max(1, int(round(hr_height * decim_scale)))
        out_w = max(1, int(round(hr_width * decim_scale)))
        hr_data = _raw_to_uint16(hr_src.read(
            hr_bands, out_shape=(len(hr_bands), out_h, out_w), resampling=Resampling.average
        ))
    hr_overview = np.transpose(hr_data, (1, 2, 0))

    dst_transform = hr_profile["transform"] * Affine.scale(hr_width / out_w, hr_height / out_h)
    lr_dst = np.zeros((3, out_h, out_w), dtype=np.uint16)
    with rasterio.open(lr_path) as lr_src:
        has_crs = (lr_src.crs is not None) and (hr_profile.get("crs") is not None)
        if has_crs:
            for band_idx, band_number in enumerate(lr_bands):
                reproject(
                    source=rasterio.band(lr_src, band_number),
                    destination=lr_dst[band_idx],
                    src_transform=lr_src.transform,
                    src_crs=lr_src.crs,
                    dst_transform=dst_transform,
                    dst_crs=hr_profile["crs"],
                    resampling=Resampling.cubic,
                )
        else:
            # No CRS (plain JPG/PNG): read each band and resize to overview dims
            for band_idx, band_number in enumerate(lr_bands):
                raw = lr_src.read(band_number)
                resized = cv2.resize(raw.astype(np.float32), (out_w, out_h),
                                     interpolation=cv2.INTER_CUBIC)
                lr_dst[band_idx] = np.clip(resized, 0, 65535).astype(np.uint16)
    lr_overview = np.clip(np.transpose(lr_dst, (1, 2, 0)), 0, 32767).astype(np.uint16)

    logging.info(
        "Decimated overview: full=%dx%d  overview=%dx%d  decim_scale=%.4f",
        hr_height, hr_width, out_h, out_w, decim_scale,
    )
    return {
        "hr_overview": hr_overview,
        "lr_overview": lr_overview,
        "hr_profile": hr_profile,
        "hr_height": hr_height,
        "hr_width": hr_width,
        "decim_scale": decim_scale,
    }


def coregister_stage_a_orb(
    hr_small: np.ndarray,
    lr_small: np.ndarray,
    cfg: dict,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Stage A — Coarse Global Alignment via ORB Keypoints + RANSAC Homography.
    Operates directly on already-decimated overview arrays (no further
    internal downsampling — the overview itself is the "small" image).

    Returns
    -------
    lr_small_after_a : np.ndarray — lr_small warped by the estimated homography (or unchanged).
    H                : np.ndarray (3,3) | None — homography in OVERVIEW pixel space, or None if skipped/failed.
    """
    if not cfg["COREG_A_ENABLED"]:
        logging.info("Stage A (ORB) disabled — skipping.")
        return lr_small, None

    height, width = hr_small.shape[:2]
    hr_gray = _to_gray_uint8(hr_small)
    lr_gray = _to_gray_uint8(lr_small)

    orb               = cv2.ORB_create(nfeatures=cfg["COREG_A_MAX_FEATURES"])
    kp_hr, des_hr     = orb.detectAndCompute(hr_gray, None)
    kp_lr, des_lr     = orb.detectAndCompute(lr_gray, None)

    if des_hr is None or des_lr is None or len(kp_hr) < 4 or len(kp_lr) < 4:
        logging.warning("Stage A: too few keypoints detected — skipping homography.")
        return lr_small, None

    matcher     = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des_lr, des_hr, k=2)

    good = [m for m, n in raw_matches if m.distance < cfg["COREG_A_MATCH_RATIO"] * n.distance]
    logging.info("Stage A: %d good matches from %d raw pairs.", len(good), len(raw_matches))

    if len(good) < 4:
        logging.warning("Stage A: fewer than 4 good matches — skipping homography.")
        return lr_small, None

    src_pts = np.float32([kp_lr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_hr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, cfg["COREG_A_RANSAC_THRESH"]
    )

    if H is None:
        logging.warning("Stage A: RANSAC failed to find a valid homography — skipping.")
        return lr_small, None

    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    logging.info("Stage A: homography accepted with %d RANSAC inliers.", n_inliers)

    lr_warped = np.zeros_like(lr_small)
    for c in range(3):
        lr_warped[:, :, c] = np.clip(cv2.warpPerspective(
            lr_small[:, :, c].astype(np.float32), H, (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ), 0, 65535).astype(np.uint16)

    logging.info("Stage A complete — overview warped with estimated homography.")
    return lr_warped, H


def coregister_stage_b_phase(
    hr_small: np.ndarray,
    lr_small: np.ndarray,
    cfg: dict,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Stage B — Sub-pixel Global Correction via Phase Cross-Correlation.
    Operates directly on already-decimated overview arrays.

    Returns
    -------
    lr_small_after_b : np.ndarray
    shift            : (shift_row, shift_col) in OVERVIEW pixel units — (0.0, 0.0) if skipped.
    """
    if not cfg["COREG_B_ENABLED"]:
        logging.info("Stage B (Phase Correlation) disabled — skipping.")
        return lr_small, (0.0, 0.0)

    height, width = hr_small.shape[:2]
    hr_gray = _to_gray_uint8(hr_small).astype(np.float32)
    lr_gray = _to_gray_uint8(lr_small).astype(np.float32)

    shift, error, _ = phase_cross_correlation(
        hr_gray, lr_gray, upsample_factor=cfg["COREG_B_UPSAMPLE_FACTOR"],
    )
    shift_row, shift_col = float(shift[0]), float(shift[1])

    logging.info(
        "Stage B: sub-pixel shift  row=%.4f px  col=%.4f px  "
        "(error=%.4f, upsample_factor=%d)",
        shift_row, shift_col, error, cfg["COREG_B_UPSAMPLE_FACTOR"],
    )

    M = np.float32([[1, 0, shift_col],
                    [0, 1, shift_row]])

    lr_shifted = np.zeros_like(lr_small)
    for c in range(3):
        lr_shifted[:, :, c] = np.clip(cv2.warpAffine(
            lr_small[:, :, c].astype(np.float32), M, (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ), 0, 65535).astype(np.uint16)

    logging.info("Stage B complete — sub-pixel translation applied.")
    return lr_shifted, (shift_row, shift_col)


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


def estimate_global_transform(
    hr_path: str,
    lr_path: str,
    hr_bands: list,
    lr_bands: list,
    cfg: dict,
) -> Tuple[np.ndarray, dict]:
    """
    Estimate a residual correction homography that aligns the LR scene
    (already CRS-reprojected onto the HR pixel grid) to the HR scene, using
    only a small decimated overview — never a full-resolution array. Stages
    A and B are both estimated at the SAME overview resolution (unlike the
    old per-stage internal downsampling), then rescaled once into full
    HR-pixel-space. Stage C still runs per-patch in Module 6.

    Returns
    -------
    residual_homography : np.ndarray (3,3) float64 — maps CRS-reprojected LR
        pixel coords -> HR pixel coords, valid at FULL resolution. Identity
        if both Stage A and Stage B are disabled or fail to find a fit.
    diagnostics : dict — small uint8 overview arrays + scene metadata, used
        both internally (by callers needing hr_profile/hr_height/hr_width)
        and for Part-A previews:
        {"hr_overview", "lr_overview", "lr_overview_after_a", "lr_overview_after_b",
         "hr_profile", "hr_height", "hr_width", "decim_scale"}
    """
    max_dim = cfg.get("COREG_PREVIEW_DECIM_DIM", 2000)
    overviews = _read_decimated_overviews(hr_path, lr_path, hr_bands, lr_bands, max_dim)
    decim_scale = overviews["decim_scale"]

    logging.info("=== MODULE 3 — Stage A: Coarse Global (ORB) ===")
    lr_after_a, H_small = coregister_stage_a_orb(
        overviews["hr_overview"], overviews["lr_overview"], cfg
    )

    logging.info("=== MODULE 3 — Stage B: Sub-pixel Global (Phase Correlation) ===")
    lr_after_b, (shift_row, shift_col) = coregister_stage_b_phase(
        overviews["hr_overview"], lr_after_a, cfg
    )

    # Compose Stage A's homography with Stage B's translation, both already
    # estimated at the SAME overview resolution.
    H_combined = np.eye(3, dtype=np.float64) if H_small is None else H_small.astype(np.float64)
    T_b = np.array([[1, 0, shift_col], [0, 1, shift_row], [0, 0, 1]], dtype=np.float64)
    H_combined = T_b @ H_combined

    # Rescale once from overview-pixel-space to full HR-pixel-space (same
    # S @ H @ S_inv rescale pattern the old code used per-stage).
    S     = np.diag([1.0 / decim_scale, 1.0 / decim_scale, 1.0]).astype(np.float64)
    S_inv = np.diag([decim_scale, decim_scale, 1.0]).astype(np.float64)
    residual_homography = S @ H_combined @ S_inv

    diagnostics = {
        "hr_overview": overviews["hr_overview"],
        "lr_overview": overviews["lr_overview"],
        "lr_overview_after_a": lr_after_a,
        "lr_overview_after_b": lr_after_b,
        "hr_profile": overviews["hr_profile"],
        "hr_height": overviews["hr_height"],
        "hr_width": overviews["hr_width"],
        "decim_scale": decim_scale,
    }
    return residual_homography, diagnostics


def _read_lr_window_to_hr_grid(
    lr_src: "rasterio.DatasetReader",
    lr_bands: list,
    hr_profile: dict,
    residual_homography: np.ndarray,
    row: int,
    col: int,
    height: int,
    width: int,
    margin: int = 32,
) -> np.ndarray:
    """
    Read the LR data corresponding to one HR window (row, col, height, width
    in full-scene HR pixel coordinates): CRS-reproject it onto that window's
    pixel grid, then apply the residual Stage A/B correction homography
    (translated into this window's local coordinates). A small margin is
    read/warped and cropped away afterward to avoid border artefacts from
    the residual correction. This is the shared windowed-read primitive used
    by radiometric fitting, histogram-LUT estimation, and patch extraction —
    the full-scene equivalent of the old _initial_reproject + Stage A/B warp,
    applied to one small window at a time instead of the whole scene.

    Parameters
    ----------
    lr_src : an already-open rasterio dataset for the LR file. Callers open
        it once per scene (outside their per-patch loop) and pass the same
        handle through — re-opening per patch measurably grows GDAL's
        internal driver/cache state over thousands of calls.

    Returns
    -------
    np.ndarray (height, width, 3) uint16
    """
    pad_row = max(0, row - margin)
    pad_col = max(0, col - margin)
    pad_h   = height + (row - pad_row) + margin
    pad_w   = width + (col - pad_col) + margin

    window        = Window(pad_col, pad_row, pad_w, pad_h)
    dst_transform = window_transform(window, hr_profile["transform"])

    lr_dst = np.zeros((3, pad_h, pad_w), dtype=np.uint16)
    has_crs = (lr_src.crs is not None) and (hr_profile.get("crs") is not None)
    if has_crs:
        for band_idx, band_number in enumerate(lr_bands):
            reproject(
                source=rasterio.band(lr_src, band_number),
                destination=lr_dst[band_idx],
                src_transform=lr_src.transform,
                src_crs=lr_src.crs,
                dst_transform=dst_transform,
                dst_crs=hr_profile["crs"],
                resampling=Resampling.cubic,
            )
    else:
        # No CRS (plain JPG/PNG): map HR window back to LR coords by scale
        scale_x = hr_profile["width"] / lr_src.width
        scale_y = hr_profile["height"] / lr_src.height
        lr_window = Window(
            col_off=pad_col / scale_x,
            row_off=pad_row / scale_y,
            width=pad_w / scale_x,
            height=pad_h / scale_y,
        )
        for band_idx, band_number in enumerate(lr_bands):
            raw = lr_src.read(
                band_number,
                window=lr_window,
                out_shape=(pad_h, pad_w),
                resampling=Resampling.cubic,
            )
            lr_dst[band_idx] = np.clip(raw, 0, 65535).astype(np.uint16)
    lr_reprojected = np.clip(np.transpose(lr_dst, (1, 2, 0)), 0, 32767).astype(np.uint16)

    # Translate residual_homography (full-scene HR coords) into this padded
    # window's local coordinates: H_local = T_to_local @ H @ T_to_full.
    to_full  = np.array([[1, 0, pad_col], [0, 1, pad_row], [0, 0, 1]], dtype=np.float64)
    to_local = np.array([[1, 0, -pad_col], [0, 1, -pad_row], [0, 0, 1]], dtype=np.float64)
    H_local  = to_local @ residual_homography @ to_full

    lr_warped = np.zeros_like(lr_reprojected)
    for c in range(3):
        lr_warped[:, :, c] = np.clip(cv2.warpPerspective(
            lr_reprojected[:, :, c].astype(np.float32), H_local, (pad_w, pad_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ), 0, 65535).astype(np.uint16)

    crop_row = row - pad_row
    crop_col = col - pad_col
    return lr_warped[crop_row:crop_row + height, crop_col:crop_col + width, :]


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — SMART PERCENTILE SCALING  (16-bit → 8-bit)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_percentile_thresholds(
    path: str,
    bands: list,
    nodata_value: int,
    saturated_value: int,
    clip_percentiles: Tuple[float, float],
    cfg: dict,
    overview: Optional[np.ndarray] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Estimate per-channel (v_min, v_max) percentile thresholds for 16-bit ->
    8-bit scaling without loading the full scene. Pools valid pixels from an
    already-read decimated overview (if supplied — reused from
    estimate_global_transform, avoiding extra I/O) plus a bounded number of
    randomly-positioned full-resolution windows read directly from the file.

    Returns
    -------
    {channel_index: (v_min, v_max)}
    """
    p_lo, p_hi       = clip_percentiles
    n_sample_windows = cfg.get("PERCENTILE_N_SAMPLE_WINDOWS", 20)
    sample_size      = 256

    samples: List[List[np.ndarray]] = [[] for _ in range(3)]

    if overview is not None:
        for c in range(3):
            channel = overview[:, :, c].astype(np.float32)
            valid   = (overview[:, :, c] > nodata_value) & (overview[:, :, c] < saturated_value)
            samples[c].append(channel[valid])

    with rasterio.open(path) as src:
        height, width = src.height, src.width
        win_h = min(sample_size, height)
        win_w = min(sample_size, width)
        rng   = np.random.default_rng(seed=42)

        for _ in range(n_sample_windows):
            row  = int(rng.integers(0, max(1, height - win_h + 1)))
            col  = int(rng.integers(0, max(1, width - win_w + 1)))
            data = _raw_to_uint16(src.read(bands, window=Window(col, row, win_w, win_h)))
            data = np.transpose(data, (1, 2, 0))
            for c in range(3):
                channel = data[:, :, c].astype(np.float32)
                valid   = (data[:, :, c] > nodata_value) & (data[:, :, c] < saturated_value)
                samples[c].append(channel[valid])

    thresholds: Dict[int, Tuple[float, float]] = {}
    for c in range(3):
        pooled = np.concatenate(samples[c]) if samples[c] else np.array([], dtype=np.float32)

        if pooled.size == 0:
            logging.warning("Channel %d has no valid sampled pixels; thresholds default to (0, 1).", c)
            thresholds[c] = (0.0, 1.0)
            continue

        v_min = float(np.percentile(pooled, p_lo))
        v_max = float(np.percentile(pooled, p_hi))

        if v_max == v_min:
            logging.warning("Channel %d has zero dynamic range in sample; widening by 1.", c)
            v_max = v_min + 1.0

        thresholds[c] = (v_min, v_max)
        logging.info(
            "Percentile thresholds  channel %d  p%.1f=%.1f  p%.1f=%.1f  (n=%d sampled pixels)",
            c, p_lo, v_min, p_hi, v_max, pooled.size,
        )

    return thresholds


def apply_percentile_scaling(
    window_array: np.ndarray,
    thresholds: Dict[int, Tuple[float, float]],
) -> np.ndarray:
    """
    Convert a 16-bit RGB window to 8-bit using precomputed per-channel
    percentile thresholds (see estimate_percentile_thresholds). Same
    clip/normalize math the old full-scene scale_to_uint8 used, applied here
    to one small window at a time.
    """
    float_array = window_array.astype(np.float32)
    result      = np.zeros_like(float_array)

    for c in range(3):
        v_min, v_max    = thresholds[c]
        clipped         = np.clip(float_array[:, :, c], v_min, v_max)
        result[:, :, c] = (clipped - v_min) / (v_max - v_min)

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

def fit_radiometric_regression(
    hr_path: str,
    lr_path: str,
    hr_profile: dict,
    hr_height: int,
    hr_width: int,
    hr_bands: list,
    lr_bands: list,
    residual_homography: np.ndarray,
    hr_thresholds: Dict[int, Tuple[float, float]],
    lr_thresholds: Dict[int, Tuple[float, float]],
    cfg: dict,
) -> np.ndarray:
    """
    Fit a linear radiometric model from LR pixel values to HR pixel values by
    sampling a bounded number of windowed block reads — never the full scene.
    Same block-RMSE-outlier-rejection + random-pixel-sampling + least-squares
    math as before; only WHICH blocks get read from disk changes (a random
    subset of size RADIOMETRIC_N_FIT_WINDOWS, instead of every block in the
    full in-memory grid).

    The model is:
        [HR_R, HR_G, HR_B] = [LR_R, LR_G, LR_B, 1] @ W

    Returns
    -------
    weights : np.ndarray (4, 3) — solved via ordinary least squares.
    """
    block_size    = cfg["RADIOMETRIC_BLOCK_SIZE"]
    rmse_thresh   = cfg["RADIOMETRIC_RMSE_THRESHOLD"]
    n_samples     = cfg["RADIOMETRIC_N_SAMPLES"]
    n_fit_windows = cfg.get("RADIOMETRIC_N_FIT_WINDOWS", 80)

    row_starts_all = list(range(0, hr_height - block_size + 1, block_size))
    col_starts_all = list(range(0, hr_width - block_size + 1, block_size))
    all_blocks     = [(r, c) for r in row_starts_all for c in col_starts_all]

    if not all_blocks:
        raise RuntimeError(
            f"Scene ({hr_height}x{hr_width}) is smaller than RADIOMETRIC_BLOCK_SIZE "
            f"({block_size}) — cannot fit radiometric regression."
        )

    rng          = np.random.default_rng(seed=42)
    n_candidates = min(n_fit_windows, len(all_blocks))
    candidate_blocks = [
        all_blocks[i] for i in rng.choice(len(all_blocks), size=n_candidates, replace=False)
    ]

    # ── Read + scale every candidate block once, then classify by block RMSE ──
    # (§2.5.4 outlier rejection — rejects cloud shadows, recent construction,
    # phenology/snow-melt changes that would bias the regression.)
    sampled_blocks = []  # (r, c, hr_block_uint8, lr_block_uint8, rmse)
    with rasterio.open(hr_path) as hr_src, rasterio.open(lr_path) as lr_src:
        for r, c in candidate_blocks:
            hr_raw = np.transpose(
                _raw_to_uint16(hr_src.read(hr_bands, window=Window(c, r, block_size, block_size))),
                (1, 2, 0),
            )
            hr_block = apply_percentile_scaling(hr_raw, hr_thresholds)

            lr_raw = _read_lr_window_to_hr_grid(
                lr_src, lr_bands, hr_profile, residual_homography, r, c, block_size, block_size,
            )
            lr_block = apply_percentile_scaling(lr_raw, lr_thresholds)

            rmse = float(np.sqrt(np.mean(
                (lr_block.astype(np.float32) - hr_block.astype(np.float32)) ** 2
            )))
            sampled_blocks.append((r, c, hr_block, lr_block, rmse))

    accepted = [b for b in sampled_blocks if b[4] <= rmse_thresh]
    logging.info(
        "Radiometric regression — block RMSE filter: %d / %d sampled blocks accepted "
        "(%.1f %%), %d rejected (RMSE > %.1f)",
        len(accepted), len(sampled_blocks),
        100.0 * len(accepted) / max(len(sampled_blocks), 1),
        len(sampled_blocks) - len(accepted), rmse_thresh,
    )

    if not accepted:
        logging.warning(
            "All %d sampled blocks exceeded the RMSE threshold. Falling back to using "
            "every sampled block for regression. Consider raising RADIOMETRIC_RMSE_THRESHOLD "
            "or RADIOMETRIC_N_FIT_WINDOWS.", len(sampled_blocks),
        )
        accepted = sampled_blocks

    # ── Random pixel sampling from accepted blocks ─────────────────────────────
    pixels_per_block = max(1, n_samples // len(accepted))
    block_pixels     = block_size * block_size

    lr_samples_list, hr_samples_list = [], []
    for r, c, hr_block, lr_block, _ in accepted:
        lr_flat = lr_block.reshape(-1, 3).astype(np.float32)
        hr_flat = hr_block.reshape(-1, 3).astype(np.float32)
        n_draw  = min(pixels_per_block, block_pixels)
        idx     = rng.choice(block_pixels, size=n_draw, replace=False)
        lr_samples_list.append(lr_flat[idx])
        hr_samples_list.append(hr_flat[idx])

    lr_pixels = np.concatenate(lr_samples_list, axis=0)   # (n_total, 3)
    hr_pixels = np.concatenate(hr_samples_list, axis=0)   # (n_total, 3)
    n_total   = lr_pixels.shape[0]

    logging.info(
        "Radiometric regression — sampled %d pixel pairs from %d accepted blocks.",
        n_total, len(accepted),
    )

    # ── Least-squares fit ───────────────────────────────────────────────────────
    # Build design matrix V = [LR_R, LR_G, LR_B, 1] and solve V @ W ≈ HR_RGB.
    V = np.column_stack([lr_pixels, np.ones(n_total, dtype=np.float32)])  # (n, 4)
    S = hr_pixels                                                          # (n, 3)

    weights, residuals, rank, sv = np.linalg.lstsq(V, S, rcond=None)      # weights: (4, 3)

    logging.info(
        "Radiometric regression fitted.  Matrix rank: %d  Residual sum: %s",
        rank, f"{residuals.sum():.4f}" if residuals.size > 0 else "N/A (underdetermined)",
    )
    logging.info(
        "Regression weights (4×3 — rows: R_coeff, G_coeff, B_coeff, bias):\n%s",
        np.array2string(weights, precision=5, suppress_small=True),
    )

    lr_pred    = V @ weights
    train_rmse = float(np.sqrt(np.mean((lr_pred - S) ** 2)))
    logging.info("Regression training RMSE (on sampled pixels): %.4f", train_rmse)

    return weights


def apply_radiometric_weights(lr_window_uint8: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Apply previously-fitted radiometric regression weights (see
    fit_radiometric_regression) to one small LR window. Same math as the old
    full-scene "apply to entire LR scene" step (Step 4), called per-window.
    """
    height, width = lr_window_uint8.shape[:2]
    lr_flat = lr_window_uint8.astype(np.float32).reshape(-1, 3)
    n_px    = height * width
    v_full  = np.column_stack([lr_flat, np.ones(n_px, dtype=np.float32)])
    adjusted = v_full @ weights
    return np.clip(np.round(adjusted), 0, 255).astype(np.uint8).reshape(height, width, 3)


def _build_match_lut(source_hist: np.ndarray, reference_hist: np.ndarray) -> np.ndarray:
    """
    Given two 256-bin histograms, build a 256-entry LUT mapping source pixel
    values to matched reference-distribution values via interpolated CDF
    matching — the same algorithm skimage.exposure.match_histograms uses
    internally, applied directly to accumulated 256-bin histograms instead
    of full pixel arrays.
    """
    src_cdf = np.cumsum(source_hist).astype(np.float64)
    src_cdf /= src_cdf[-1] if src_cdf[-1] > 0 else 1.0

    ref_cdf = np.cumsum(reference_hist).astype(np.float64)
    ref_cdf /= ref_cdf[-1] if ref_cdf[-1] > 0 else 1.0

    ref_values = np.arange(256, dtype=np.float64)
    lut = np.interp(src_cdf, ref_cdf, ref_values)
    return np.clip(np.round(lut), 0, 255).astype(np.uint8)


def estimate_histogram_match_lut(
    hr_path: str,
    lr_path: str,
    hr_profile: dict,
    hr_height: int,
    hr_width: int,
    hr_bands: list,
    lr_bands: list,
    residual_homography: np.ndarray,
    hr_thresholds: Dict[int, Tuple[float, float]],
    lr_thresholds: Dict[int, Tuple[float, float]],
    radiometric_weights: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """
    Stage 5B — Estimate a per-channel 256-entry LUT that matches the LR
    scene's (post radiometric-regression) tonal distribution to the HR
    scene's, by accumulating histograms over a bounded number of sampled
    windows instead of needing the full scene in memory.

    Corrects tonal differences that no affine spectral model can capture:
      • Pleiades 1A Red band leaking ~100 nm into NIR uniquely darkens
        dense vegetation at high radiance — a content-dependent offset.
      • Per-channel S-curve differences in the sensor transfer functions
        compress or expand mid-tones in a way a single gain+offset cannot undo.

    Histogram matching on 8-bit data is always expressible as a 256-entry
    LUT, and 256 discrete bins is a well-conditioned estimation target even
    from a sample (unlike the percentile/regression estimates, which trade
    off against true outlier sensitivity).

    Returns
    -------
    lut : np.ndarray (3, 256) uint8 — per-channel value -> matched value.
    """
    n_sample_windows = cfg.get("HISTOGRAM_N_SAMPLE_WINDOWS", 40)
    sample_size      = 256

    hr_hist = np.zeros((3, 256), dtype=np.float64)
    lr_hist = np.zeros((3, 256), dtype=np.float64)

    rng   = np.random.default_rng(seed=42)
    win_h = min(sample_size, hr_height)
    win_w = min(sample_size, hr_width)

    with rasterio.open(hr_path) as hr_src, rasterio.open(lr_path) as lr_src:
        for _ in range(n_sample_windows):
            row = int(rng.integers(0, max(1, hr_height - win_h + 1)))
            col = int(rng.integers(0, max(1, hr_width - win_w + 1)))

            hr_raw = np.transpose(
                _raw_to_uint16(hr_src.read(hr_bands, window=Window(col, row, win_w, win_h))),
                (1, 2, 0),
            )
            hr_block = apply_percentile_scaling(hr_raw, hr_thresholds)

            lr_raw = _read_lr_window_to_hr_grid(
                lr_src, lr_bands, hr_profile, residual_homography, row, col, win_h, win_w,
            )
            lr_block_scaled = apply_percentile_scaling(lr_raw, lr_thresholds)
            lr_block        = apply_radiometric_weights(lr_block_scaled, radiometric_weights)

            for c in range(3):
                hr_hist[c] += np.bincount(hr_block[:, :, c].ravel(), minlength=256)
                lr_hist[c] += np.bincount(lr_block[:, :, c].ravel(), minlength=256)

    lut = np.zeros((3, 256), dtype=np.uint8)
    for c in range(3):
        lut[c] = _build_match_lut(lr_hist[c], hr_hist[c])

    logging.info("Stage 5B histogram-match LUT estimated from %d sampled windows.", n_sample_windows)
    return lut


def apply_histogram_lut(window_uint8: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a previously-estimated per-channel LUT (see estimate_histogram_match_lut)."""
    result = np.empty_like(window_uint8)
    for c in range(3):
        result[:, :, c] = lut[c][window_uint8[:, :, c]]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5b — UNPAIRED-HR DEGRADATION  (synthetic LR generation)
# ─────────────────────────────────────────────────────────────────────────────
#
# Used only for scenes that have an HR image but no matching real LR capture.
# Imports preprocessing_pipeline.degradation_utils lazily so that pipeline3.py
# does not require torch unless this branch actually runs.
# ─────────────────────────────────────────────────────────────────────────────

def _select_degrade_fn(deg_type: str):
    from preprocessing_pipeline.degradation_utils import (
        degrade_bsrgan, degrade_bsrgan_plus, degrade_real_esrgan, degrade_satellite,
    )
    fns = {
        "bsrgan": degrade_bsrgan,
        "real_esrgan": degrade_real_esrgan,
        "bsrgan_plus": degrade_bsrgan_plus,
        "satellite": degrade_satellite,
    }
    if deg_type not in fns:
        raise ValueError(
            f"Unknown DEGRADATION_TYPE '{deg_type}'. Choose from: {sorted(fns)}"
        )
    return fns[deg_type]


def degrade_hr_patch(
    hr_patch_uint8: np.ndarray,
    sf: int,
    deg_type: str,
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesize an LR patch from an HR patch using the configured degradation
    model. Mirrors preprocessing_pipeline/run_pipeline.py's degradation
    parameter wiring so the same per-type config (bsrgan / real_esrgan /
    bsrgan_plus / satellite) works from either pipeline.

    Returns
    -------
    lr_uint8         : np.ndarray — degraded LR patch, uint8.
    hr_uint8_modcrop : np.ndarray — HR patch mod-cropped to a multiple of sf, uint8.
    """
    fn      = _select_degrade_fn(deg_type)
    deg_cfg = cfg.get(deg_type, {}) or {}
    img_f   = hr_patch_uint8.astype(np.float32) / 255.0

    if deg_type == "bsrgan":
        img_lq, img_hq = fn(
            img_f, sf=sf,
            jpeg_prob=deg_cfg.get("jpeg_prob", 0.9),
            scale2_prob=deg_cfg.get("scale2_prob", 0.25),
            isp_prob=deg_cfg.get("isp_prob", 0.25),
            noise_level1=deg_cfg.get("noise_level1", 2),
            noise_level2=deg_cfg.get("noise_level2", 25),
        )
    elif deg_type == "real_esrgan":
        img_lq, img_hq = fn(
            img_f, sf=sf,
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
        img_lq, img_hq = fn(
            img_f, sf=sf,
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
    else:  # "satellite"
        img_lq, img_hq = fn(
            img_f, sf=sf,
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

    lr_uint8 = np.clip(np.round(img_lq * 255.0), 0, 255).astype(np.uint8)
    hq_uint8 = np.clip(np.round(img_hq * 255.0), 0, 255).astype(np.uint8)
    return lr_uint8, hq_uint8


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — PATCH EXTRACTION & QUALITY FILTERING  (with Stage C ECC)
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_save_patches(
    hr_path: str,
    lr_path: str,
    hr_profile: dict,
    hr_height: int,
    hr_width: int,
    hr_bands: list,
    lr_bands: list,
    residual_homography: np.ndarray,
    hr_thresholds: Dict[int, Tuple[float, float]],
    lr_thresholds: Dict[int, Tuple[float, float]],
    radiometric_weights: np.ndarray,
    histogram_lut: np.ndarray,
    cfg: dict,
    patch_prefix: str = "patch",
) -> Tuple[int, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Slide a window over the HR scene, reading each HR/LR window directly
    from disk via rasterio (never a pre-loaded full-scene array), apply
    quality filters + Stage C local ECC refinement, and save valid pairs as
    PNGs.

    Parameters
    ----------
    hr_path, lr_path           : paths to the open-able HR/LR rasters.
    hr_profile, hr_height, hr_width, hr_bands, lr_bands : scene metadata (see estimate_global_transform).
    residual_homography        : (3,3) — see estimate_global_transform.
    hr_thresholds, lr_thresholds : per-channel percentile thresholds (see estimate_percentile_thresholds).
    radiometric_weights        : (4,3) — see fit_radiometric_regression.
    histogram_lut               : (3,256) — see estimate_histogram_match_lut.
    cfg, patch_prefix            : as before.

    Returns
    -------
    saved_count    : int — number of patch pairs written to disk.
    sample_patches : list of (hr_patch_uint8, lr_patch_uint8) — a spread of
                     up to PREVIEW_N_PATCHES saved pairs, for a contact-sheet preview.
    """
    hr_patch_size   = cfg["HR_PATCH_SIZE"]
    lr_patch_size   = cfg["LR_PATCH_SIZE"]
    stride          = cfg["STRIDE"]
    nodata_value    = cfg["NODATA_VALUE"]
    max_nodata_frac = cfg["MAX_NODATA_FRACTION"]
    min_variance    = cfg["MIN_VARIANCE"]
    output_dir      = Path(cfg["OUTPUT_DIR"])
    n_preview       = cfg.get("PREVIEW_N_PATCHES", 10)

    hr_out_dir = output_dir / "hr"
    lr_out_dir = output_dir / "lr"
    hr_out_dir.mkdir(parents=True, exist_ok=True)
    lr_out_dir.mkdir(parents=True, exist_ok=True)

    row_starts     = list(range(0, hr_height - hr_patch_size + 1, stride))
    col_starts     = list(range(0, hr_width - hr_patch_size + 1, stride))
    total_windows  = len(row_starts) * len(col_starts)
    preview_stride = max(1, total_windows // n_preview) if n_preview > 0 else 0

    saved_count       = 0
    skipped_nodata    = 0
    skipped_variance  = 0
    skipped_ecc       = 0
    skipped_ecc_score = 0
    skipped_ssim      = 0
    sample_patches: List[Tuple[np.ndarray, np.ndarray]] = []

    logging.info(
        "Starting patch extraction: %d candidate windows "
        "(stride=%d, hr_patch=%d, lr_patch=%d, Stage C ECC=%s, "
        "MIN_ECC_SCORE=%.2f, MIN_SSIM=%.2f)",
        total_windows, stride, hr_patch_size, lr_patch_size,
        cfg["COREG_C_ENABLED"],
        cfg.get("MIN_ECC_SCORE", 0.0),
        cfg.get("MIN_SSIM", 0.0),
    )

    pbar = tqdm(total=total_windows, desc=f"Extracting {patch_prefix}", unit="win")
    window_idx = 0

    with rasterio.open(hr_path) as hr_src, rasterio.open(lr_path) as lr_src:
        for row in row_starts:
            for col in col_starts:
                pbar.update(1)
                window_idx += 1

                hr_raw = np.transpose(
                    _raw_to_uint16(hr_src.read(
                        hr_bands, window=Window(col, row, hr_patch_size, hr_patch_size)
                    )),
                    (1, 2, 0),
                )
                hr_patch = apply_percentile_scaling(hr_raw, hr_thresholds)

                # Quality gate: NODATA fraction (on the scaled uint8 patch — a
                # raw nodata value clips to v_min and normalises near 0)
                nodata_mask = np.any(hr_patch == nodata_value, axis=2)
                if nodata_mask.mean() > max_nodata_frac:
                    skipped_nodata += 1
                    continue

                # Quality gate: Variance / texture
                mean_var = float(np.var(hr_patch.astype(np.float32), axis=(0, 1)).mean())
                if mean_var < min_variance:
                    skipped_variance += 1
                    continue

                # Read + align the corresponding LR window (CRS reproject +
                # residual homography), then percentile/radiometric/histogram
                # correct it — all per-window, never a full-scene array.
                lr_raw = _read_lr_window_to_hr_grid(
                    lr_src, lr_bands, hr_profile, residual_homography,
                    row, col, hr_patch_size, hr_patch_size,
                )
                lr_patch_full = apply_percentile_scaling(lr_raw, lr_thresholds)
                lr_patch_full = apply_radiometric_weights(lr_patch_full, radiometric_weights)
                lr_patch_full = apply_histogram_lut(lr_patch_full, histogram_lut)

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
                patch_name = f"{patch_prefix}_patch_{saved_count:06d}.png"
                cv2.imwrite(
                    str(hr_out_dir / patch_name),
                    cv2.cvtColor(hr_patch, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(lr_out_dir / patch_name),
                    cv2.cvtColor(lr_patch, cv2.COLOR_RGB2BGR),
                )

                if preview_stride and window_idx % preview_stride == 0 and len(sample_patches) < n_preview:
                    sample_patches.append((hr_patch.copy(), lr_patch.copy()))

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
    return saved_count, sample_patches


def extract_patches_hr_only(
    hr_path: str,
    hr_bands: list,
    hr_height: int,
    hr_width: int,
    hr_thresholds: Dict[int, Tuple[float, float]],
    cfg: dict,
    patch_prefix: str = "patch",
) -> Tuple[int, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Sliding window over an HR scene that has no matching LR image, reading
    each HR window directly from disk. Each candidate patch passes the same
    nodata/variance quality gates used in paired mode; there is no real LR
    to coregister against, so Stage C / ECC / SSIM gates do not apply.
    Instead, the configured degradation model (DEGRADATION_TYPE) synthesizes
    an LR patch directly from the HR patch.

    Returns
    -------
    saved_count    : int
    sample_patches : list of (hr_patch_uint8, lr_patch_uint8), for the contact-sheet preview.
    """
    if not cfg.get("DEGRADATION_ENABLED", True):
        logging.warning(
            "'%s' is HR-only and DEGRADATION_ENABLED=False — skipping "
            "(no LR available to pair with).", patch_prefix,
        )
        return 0, []

    hr_patch_size   = cfg["HR_PATCH_SIZE"]
    stride          = cfg["STRIDE"]
    nodata_value    = cfg["NODATA_VALUE"]
    max_nodata_frac = cfg["MAX_NODATA_FRACTION"]
    min_variance    = cfg["MIN_VARIANCE"]
    scale           = cfg["SCALE_FACTOR"]
    deg_type        = cfg.get("DEGRADATION_TYPE", "satellite")
    output_dir      = Path(cfg["OUTPUT_DIR"])
    n_preview       = cfg.get("PREVIEW_N_PATCHES", 10)

    hr_out_dir = output_dir / "hr"
    lr_out_dir = output_dir / "lr"
    hr_out_dir.mkdir(parents=True, exist_ok=True)
    lr_out_dir.mkdir(parents=True, exist_ok=True)

    row_starts     = list(range(0, hr_height - hr_patch_size + 1, stride))
    col_starts     = list(range(0, hr_width - hr_patch_size + 1, stride))
    total_windows  = len(row_starts) * len(col_starts)
    preview_stride = max(1, total_windows // n_preview) if n_preview > 0 else 0

    saved_count      = 0
    skipped_nodata   = 0
    skipped_variance = 0
    sample_patches: List[Tuple[np.ndarray, np.ndarray]] = []

    logging.info(
        "Starting HR-only degradation extraction for '%s': %d candidate "
        "windows (degradation_type=%s, stride=%d, hr_patch=%d)",
        patch_prefix, total_windows, deg_type, stride, hr_patch_size,
    )

    pbar = tqdm(total=total_windows, desc=f"Degrading {patch_prefix}", unit="win")
    window_idx = 0
    with rasterio.open(hr_path) as hr_src:
        for row in row_starts:
            for col in col_starts:
                pbar.update(1)
                window_idx += 1

                hr_raw = np.transpose(
                    _raw_to_uint16(hr_src.read(
                        hr_bands, window=Window(col, row, hr_patch_size, hr_patch_size)
                    )),
                    (1, 2, 0),
                )
                hr_patch = apply_percentile_scaling(hr_raw, hr_thresholds)

                nodata_mask = np.any(hr_patch == nodata_value, axis=2)
                if nodata_mask.mean() > max_nodata_frac:
                    skipped_nodata += 1
                    continue

                mean_var = float(np.var(hr_patch.astype(np.float32), axis=(0, 1)).mean())
                if mean_var < min_variance:
                    skipped_variance += 1
                    continue

                lr_patch, hr_patch_mc = degrade_hr_patch(hr_patch, scale, deg_type, cfg)

                patch_name = f"{patch_prefix}_patch_{saved_count:06d}.png"
                cv2.imwrite(str(hr_out_dir / patch_name), cv2.cvtColor(hr_patch_mc, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(lr_out_dir / patch_name), cv2.cvtColor(lr_patch, cv2.COLOR_RGB2BGR))

                if preview_stride and window_idx % preview_stride == 0 and len(sample_patches) < n_preview:
                    sample_patches.append((hr_patch_mc.copy(), lr_patch.copy()))

                saved_count += 1

    pbar.close()

    logging.info(
        "HR-only degradation extraction for '%s' complete: saved=%d, "
        "skipped_nodata=%d, skipped_variance=%d / %d windows",
        patch_prefix, saved_count, skipped_nodata, skipped_variance, total_windows,
    )
    return saved_count, sample_patches


def extract_patches_lr_only(
    lr_path: str,
    lr_bands: list,
    lr_height: int,
    lr_width: int,
    lr_thresholds: Dict[int, Tuple[float, float]],
    cfg: dict,
    patch_prefix: str = "patch",
) -> Tuple[int, List[Tuple[Optional[np.ndarray], np.ndarray]]]:
    """
    Sliding window over a standalone LR scene with no matching HR reference
    (e.g. a real-world low-resolution capture used only for blind / test-time
    inference), reading each LR window directly from disk at its own native
    resolution. Patches are written to OUTPUT_DIR/lr only — no HR pair
    exists, so coregistration, radiometric normalisation, and the ECC/SSIM
    gates are not applicable. Nodata/variance gates are evaluated on the LR
    patch itself, at LR_PATCH_SIZE / a proportionally scaled-down stride.

    Returns
    -------
    saved_count    : int
    sample_patches : list of (None, lr_patch_uint8) — no HR counterpart exists.
    """
    lr_patch_size   = cfg["LR_PATCH_SIZE"]
    stride          = max(1, cfg["STRIDE"] // cfg["SCALE_FACTOR"])
    nodata_value    = cfg["NODATA_VALUE"]
    max_nodata_frac = cfg["MAX_NODATA_FRACTION"]
    min_variance    = cfg["MIN_VARIANCE"]
    output_dir      = Path(cfg["OUTPUT_DIR"])
    n_preview       = cfg.get("PREVIEW_N_PATCHES", 10)

    lr_out_dir = output_dir / "lr"
    lr_out_dir.mkdir(parents=True, exist_ok=True)

    if lr_height < lr_patch_size or lr_width < lr_patch_size:
        logging.warning(
            "'%s' (%dx%d) is smaller than LR_PATCH_SIZE=%d — skipping.",
            patch_prefix, lr_height, lr_width, lr_patch_size,
        )
        return 0, []

    row_starts     = list(range(0, lr_height - lr_patch_size + 1, stride))
    col_starts     = list(range(0, lr_width - lr_patch_size + 1, stride))
    total_windows  = len(row_starts) * len(col_starts)
    preview_stride = max(1, total_windows // n_preview) if n_preview > 0 else 0

    saved_count      = 0
    skipped_nodata   = 0
    skipped_variance = 0
    sample_patches: List[Tuple[Optional[np.ndarray], np.ndarray]] = []

    logging.info(
        "Starting LR-only extraction for '%s': %d candidate windows "
        "(stride=%d, lr_patch=%d)",
        patch_prefix, total_windows, stride, lr_patch_size,
    )

    pbar = tqdm(total=total_windows, desc=f"Extracting {patch_prefix} (LR-only)", unit="win")
    window_idx = 0
    with rasterio.open(lr_path) as lr_src:
        for row in row_starts:
            for col in col_starts:
                pbar.update(1)
                window_idx += 1

                lr_raw = np.transpose(
                    _raw_to_uint16(lr_src.read(
                        lr_bands, window=Window(col, row, lr_patch_size, lr_patch_size)
                    )),
                    (1, 2, 0),
                )
                lr_patch = apply_percentile_scaling(lr_raw, lr_thresholds)

                nodata_mask = np.any(lr_patch == nodata_value, axis=2)
                if nodata_mask.mean() > max_nodata_frac:
                    skipped_nodata += 1
                    continue

                mean_var = float(np.var(lr_patch.astype(np.float32), axis=(0, 1)).mean())
                if mean_var < min_variance:
                    skipped_variance += 1
                    continue

                patch_name = f"{patch_prefix}_patch_{saved_count:06d}.png"
                cv2.imwrite(str(lr_out_dir / patch_name), cv2.cvtColor(lr_patch, cv2.COLOR_RGB2BGR))

                if preview_stride and window_idx % preview_stride == 0 and len(sample_patches) < n_preview:
                    sample_patches.append((None, lr_patch.copy()))

                saved_count += 1

    pbar.close()

    logging.info(
        "LR-only extraction for '%s' complete: saved=%d, skipped_nodata=%d, "
        "skipped_variance=%d / %d windows",
        patch_prefix, saved_count, skipped_nodata, skipped_variance, total_windows,
    )
    return saved_count, sample_patches


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6b — VISUALISATION PREVIEWS  (OUTPUT_DIR/_previews/)
# ─────────────────────────────────────────────────────────────────────────────
#
# Lightweight JPEG previews written at key stages so the GUI can show visual
# feedback while a job runs. Each save_* call logs a "PREVIEW_READY <file>
# <stage> <scene>" marker via the normal logging.info() pipe (already
# captured by job_manager's stdout reader) so the frontend can discover new
# previews without polling a separate endpoint. Preview generation must never
# break the actual pipeline run, so failures here are caught and logged, not
# raised.
# ─────────────────────────────────────────────────────────────────────────────

def _read_single_overview(path: str, bands: list, max_dim: int) -> np.ndarray:
    """Read a small decimated overview of one raster — no reprojection. Used
    for HR-only / LR-only load previews, which have no counterpart scene to
    align against."""
    with rasterio.open(path) as src:
        height, width = src.height, src.width
        scale = min(1.0, max_dim / max(height, width))
        out_h = max(1, int(round(height * scale)))
        out_w = max(1, int(round(width * scale)))
        data = _raw_to_uint16(src.read(
            bands, out_shape=(len(bands), out_h, out_w), resampling=Resampling.average
        ))
    return np.transpose(data, (1, 2, 0))


def _emit_preview_marker(rel_path: Path, stage: str, scene_name: str) -> None:
    logging.info("PREVIEW_READY %s %s %s", rel_path.as_posix(), stage, scene_name)


def _downscale_for_preview(arr_uint8: np.ndarray, max_dim: int) -> np.ndarray:
    height, width = arr_uint8.shape[:2]
    scale = min(1.0, max_dim / max(height, width))
    if scale >= 1.0:
        return arr_uint8
    new_h, new_w = max(1, int(height * scale)), max(1, int(width * scale))
    return cv2.resize(arr_uint8, (new_w, new_h), interpolation=cv2.INTER_AREA)


def save_preview(
    arr_uint8: np.ndarray,
    output_dir: Path,
    scene_name: str,
    stage: str,
    cfg: dict,
) -> Optional[Path]:
    """
    Write a downscaled JPEG preview of an 8-bit RGB array to
    OUTPUT_DIR/_previews/{scene_name}_{stage}.jpg and emit the
    PREVIEW_READY marker. Returns None (and logs a warning, never raises) on
    any failure or when PREVIEW_ENABLED is False.
    """
    if not cfg.get("PREVIEW_ENABLED", True):
        return None
    try:
        preview_dir = output_dir / "_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)

        small    = _downscale_for_preview(arr_uint8, cfg.get("PREVIEW_MAX_DIM", 1024))
        filename = f"{scene_name}_{stage}.jpg"
        path     = preview_dir / filename
        cv2.imwrite(
            str(path), cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, cfg.get("PREVIEW_JPEG_QUALITY", 85)],
        )
        _emit_preview_marker(Path(filename), stage, scene_name)
        return path
    except Exception as exc:
        logging.warning("Could not save preview '%s' for '%s': %s", stage, scene_name, exc)
        return None


def save_band_wise_overview(
    path: str,
    output_dir: Path,
    scene_name: str,
    stage: str,
    cfg: dict,
) -> Optional[Path]:
    """
    Read every band in *path*, apply independent per-band percentile scaling,
    and save them as a horizontal grayscale tile grid so every spectral channel
    (including those not mapped to RGB, e.g. NIR) can be inspected.
    Emits a PREVIEW_READY marker so the GUI picks it up automatically.
    """
    if not cfg.get("PREVIEW_ENABLED", True):
        return None
    try:
        max_dim    = cfg.get("COREG_PREVIEW_DECIM_DIM", 2000)
        nodata     = cfg.get("NODATA_VALUE", 0)
        saturated  = cfg.get("SATURATED_VALUE", 32767)
        lo_pct, hi_pct = cfg.get("CLIP_PERCENTILES", [2.0, 98.0])
        preview_dir = output_dir / "_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(path) as src:
            n_bands        = src.count
            height, width  = src.height, src.width
            scale          = min(1.0, max_dim / max(height, width))
            out_h          = max(1, int(round(height * scale)))
            out_w          = max(1, int(round(width  * scale)))
            raw = src.read(
                list(range(1, n_bands + 1)),
                out_shape=(n_bands, out_h, out_w),
                resampling=Resampling.average,
            ).astype(np.float32)

        label_h = 22
        tiles = []
        for i in range(n_bands):
            band  = raw[i]
            valid = band[(band != nodata) & (band < saturated)]
            lo    = float(np.percentile(valid, lo_pct))  if valid.size else 0.0
            hi    = float(np.percentile(valid, hi_pct))  if valid.size else 1.0
            if hi <= lo:
                hi = lo + 1.0
            grey = np.clip((band - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            grey_rgb = np.stack([grey, grey, grey], axis=2)

            bar = np.zeros((label_h, out_w, 3), dtype=np.uint8)
            cv2.putText(bar, f"Band {i + 1}", (4, label_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
            tiles.append(np.vstack([bar, grey_rgb]))

        sheet    = np.hstack(tiles)
        filename = f"{scene_name}_{stage}.jpg"
        path_out = preview_dir / filename
        cv2.imwrite(
            str(path_out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, cfg.get("PREVIEW_JPEG_QUALITY", 85)],
        )
        _emit_preview_marker(Path(filename), stage, scene_name)
        return path_out
    except Exception as exc:
        logging.warning("Could not save band-wise overview '%s' for '%s': %s", stage, scene_name, exc)
        return None


def save_patch_contact_sheet(
    patch_pairs: List[Tuple[Optional[np.ndarray], np.ndarray]],
    output_dir: Path,
    scene_name: str,
    cfg: dict,
) -> Optional[Path]:
    """
    Tile a spread of (hr_patch, lr_patch) pairs collected during extraction
    into one contact-sheet JPEG — since a 10,000-20,000px scene obviously
    can't be shown directly, this is the closest thing to "what did the
    extracted patches actually look like". HR-only entries have no LR
    counterpart and are rendered as a black tile in that slot. Returns None
    (and logs a warning, never raises) on any failure, when PREVIEW_ENABLED
    is False, or when there are no sample patches to show.
    """
    if not cfg.get("PREVIEW_ENABLED", True) or not patch_pairs:
        return None
    try:
        preview_dir = output_dir / "_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)

        tile = 128
        rows = []
        for hr_patch, lr_patch in patch_pairs:
            lr_resized = cv2.resize(lr_patch, (tile, tile), interpolation=cv2.INTER_NEAREST)
            if hr_patch is not None:
                hr_resized = cv2.resize(hr_patch, (tile, tile), interpolation=cv2.INTER_AREA)
            else:
                hr_resized = np.zeros((tile, tile, 3), dtype=np.uint8)
            rows.append(np.hstack([hr_resized, lr_resized]))

        sheet    = np.vstack(rows)
        filename = f"{scene_name}_patches_contact.jpg"
        path     = preview_dir / filename
        cv2.imwrite(
            str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, cfg.get("PREVIEW_JPEG_QUALITY", 85)],
        )
        _emit_preview_marker(Path(filename), "patches", scene_name)
        return path
    except Exception as exc:
        logging.warning("Could not save patch contact sheet for '%s': %s", scene_name, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7 — WORK-ITEM DISCOVERY  (single file OR directory, paired/unpaired)
# ─────────────────────────────────────────────────────────────────────────────

def _discover_images(
    dir_path: Path,
    extensions: List[str],
    class_filter: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Return {key: path} for matching image files.

    Flat mode: images directly in dir_path → key = stem (original behaviour).
    Class-folder mode: if no images exist at top level but subdirectories do,
    descend one level and use 'subfolder__stem' as the key.
    class_filter restricts which subfolders are included (empty/None = all).
    """
    exts = {e.lower() for e in extensions}
    out: Dict[str, Path] = {}

    # Try flat mode first (preserves original behaviour for non-classed datasets)
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out[p.stem] = p
    if out:
        return out

    # Class-folder mode: descend one level into subdirectories.
    # Key uses "__" separator so the name is safe as a flat filename prefix.
    for sub in sorted(dir_path.iterdir()):
        if not sub.is_dir():
            continue
        if class_filter and sub.name not in class_filter:
            continue
        for p in sorted(sub.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                out[f"{sub.name}__{p.stem}"] = p

    return out


def resolve_work_items(cfg: dict) -> List[dict]:
    """
    Determine the list of scenes to process from CONFIG['HR_IMAGE_PATH'] and
    CONFIG['LR_IMAGE_PATH']. Each may be a single file, a directory, or
    empty/omitted. Returns a list of:
        {"name": str, "hr_path": Path|None, "lr_path": Path|None}
    """
    hr_cfg = (cfg.get("HR_IMAGE_PATH") or "").strip()
    lr_cfg = (cfg.get("LR_IMAGE_PATH") or "").strip()
    extensions = cfg.get("SUPPORTED_EXTENSIONS", [".tif", ".tiff", ".jp2", ".png", ".jpg", ".jpeg", ".bmp"])

    hr_path = Path(hr_cfg) if hr_cfg else None
    lr_path = Path(lr_cfg) if lr_cfg else None

    hr_is_dir = hr_path is not None and hr_path.is_dir()
    lr_is_dir = lr_path is not None and lr_path.is_dir()
    class_filter: List[str] = cfg.get("CLASS_FILTER") or []

    if hr_path is None and lr_path is None:
        raise ValueError("At least one of HR_IMAGE_PATH or LR_IMAGE_PATH must be set.")

    # ── Directory mode: at least one side is a directory ───────────────────
    if hr_is_dir or lr_is_dir:
        if hr_path is not None and not hr_is_dir:
            raise ValueError(
                "LR_IMAGE_PATH is a directory, so HR_IMAGE_PATH must also be a "
                "directory (or omitted) — mixing a single file with a directory "
                "is not supported."
            )
        if lr_path is not None and not lr_is_dir:
            raise ValueError(
                "HR_IMAGE_PATH is a directory, so LR_IMAGE_PATH must also be a "
                "directory (or omitted) — mixing a single file with a directory "
                "is not supported."
            )

        hr_index = _discover_images(hr_path, extensions, class_filter) if hr_is_dir else {}
        lr_index = _discover_images(lr_path, extensions, class_filter) if lr_is_dir else {}

        all_names = sorted(set(hr_index) | set(lr_index))
        if not all_names:
            base = hr_path if hr_is_dir else lr_path
            hint = (
                f" (class filter active: {class_filter})" if class_filter
                else " Check that the directory contains image files or class subfolders with images."
            )
            raise RuntimeError(
                f"No images with extensions {extensions} found in {base}.{hint}"
            )

        items = [
            {"name": name, "hr_path": hr_index.get(name), "lr_path": lr_index.get(name)}
            for name in all_names
        ]
        n_paired  = sum(1 for it in items if it["hr_path"] and it["lr_path"])
        n_hr_only = sum(1 for it in items if it["hr_path"] and not it["lr_path"])
        n_lr_only = sum(1 for it in items if it["lr_path"] and not it["hr_path"])
        logging.info(
            "Directory scan complete: %d paired, %d HR-only (will degrade), "
            "%d LR-only (standalone). Total scenes = %d",
            n_paired, n_hr_only, n_lr_only, len(items),
        )
        return items

    # ── Single-file mode (legacy behaviour, also covers HR-only / LR-only) ──
    if hr_path is not None:
        if not hr_path.exists():
            raise FileNotFoundError(f"HR_IMAGE_PATH not found: {hr_path}")
        item = {"name": hr_path.stem, "hr_path": hr_path, "lr_path": None}
        if lr_path is not None:
            if not lr_path.exists():
                raise FileNotFoundError(f"LR_IMAGE_PATH not found: {lr_path}")
            item["lr_path"] = lr_path
        return [item]

    if not lr_path.exists():
        raise FileNotFoundError(f"LR_IMAGE_PATH not found: {lr_path}")
    return [{"name": lr_path.stem, "hr_path": None, "lr_path": lr_path}]


def process_item(item: dict, cfg: dict) -> dict:
    """
    Run the appropriate pipeline branch for one scene. All three branches
    are now windowed end-to-end: estimation steps read decimated overviews
    or a bounded number of sampled windows, and extraction reads/writes one
    small patch window at a time — a full-resolution scene array is never
    held in memory, regardless of source image dimensions.

    Returns a stats dict: {"name", "mode", "n_saved"}.
    """
    name    = item["name"]
    hr_path = item["hr_path"]
    lr_path = item["lr_path"]

    output_dir = Path(cfg["OUTPUT_DIR"])

    if hr_path and lr_path:
        logging.info("=== Processing '%s' (paired HR+LR) ===", name)
        hr_bands = cfg.get("HR_RGB_BANDS", [1, 2, 3])
        lr_bands = cfg.get("LR_RGB_BANDS", [3, 2, 1])

        residual_homography, diagnostics = estimate_global_transform(
            str(hr_path), str(lr_path), hr_bands, lr_bands, cfg
        )
        hr_profile = diagnostics["hr_profile"]
        hr_height  = diagnostics["hr_height"]
        hr_width   = diagnostics["hr_width"]

        hr_thresholds = estimate_percentile_thresholds(
            str(hr_path), hr_bands, cfg["NODATA_VALUE"], cfg["SATURATED_VALUE"],
            tuple(cfg["CLIP_PERCENTILES"]), cfg, overview=diagnostics["hr_overview"],
        )
        lr_thresholds = estimate_percentile_thresholds(
            str(lr_path), lr_bands, cfg["NODATA_VALUE"], cfg["SATURATED_VALUE"],
            tuple(cfg["CLIP_PERCENTILES"]), cfg,
        )

        # Previews 1-3: loaded HR/LR overviews + Stage A/B coregistration,
        # all reusing the small overview arrays already read above — no
        # extra I/O beyond percentile-scaling them for display.
        save_preview(apply_percentile_scaling(diagnostics["hr_overview"], hr_thresholds),
                     output_dir, name, "load_hr", cfg)
        save_band_wise_overview(str(hr_path), output_dir, name, "bands_hr", cfg)
        save_preview(apply_percentile_scaling(diagnostics["lr_overview"], lr_thresholds),
                     output_dir, name, "load_lr", cfg)
        save_band_wise_overview(str(lr_path), output_dir, name, "bands_lr", cfg)
        save_preview(apply_percentile_scaling(diagnostics["lr_overview_after_a"], lr_thresholds),
                     output_dir, name, "coreg_a", cfg)
        save_preview(apply_percentile_scaling(diagnostics["lr_overview_after_b"], lr_thresholds),
                     output_dir, name, "coreg_b", cfg)

        radiometric_weights = fit_radiometric_regression(
            str(hr_path), str(lr_path), hr_profile, hr_height, hr_width,
            hr_bands, lr_bands, residual_homography, hr_thresholds, lr_thresholds, cfg,
        )

        if cfg.get("RADIOMETRIC_POST_HIST_MATCH", True):
            histogram_lut = estimate_histogram_match_lut(
                str(hr_path), str(lr_path), hr_profile, hr_height, hr_width,
                hr_bands, lr_bands, residual_homography, hr_thresholds, lr_thresholds,
                radiometric_weights, cfg,
            )
        else:
            logging.info("MODULE 5B: Histogram matching disabled (RADIOMETRIC_POST_HIST_MATCH=False).")
            histogram_lut = np.tile(np.arange(256, dtype=np.uint8), (3, 1))

        # Preview 4: radiometric + histogram correction applied to one
        # representative scene-center window — a visualisation-only
        # approximation; the real per-patch correction during extraction
        # below is exact for every patch, not just this one preview window.
        try:
            patch_size  = cfg["HR_PATCH_SIZE"]
            center_row  = max(0, (hr_height - patch_size) // 2)
            center_col  = max(0, (hr_width - patch_size) // 2)
            with rasterio.open(str(lr_path)) as lr_src:
                center_lr_raw = _read_lr_window_to_hr_grid(
                    lr_src, lr_bands, hr_profile, residual_homography,
                    center_row, center_col, patch_size, patch_size,
                )
            center_lr = apply_percentile_scaling(center_lr_raw, lr_thresholds)
            center_lr = apply_radiometric_weights(center_lr, radiometric_weights)
            center_lr = apply_histogram_lut(center_lr, histogram_lut)
            save_preview(center_lr, output_dir, name, "radiometric", cfg)
        except Exception as exc:
            logging.warning("Could not build radiometric preview for '%s': %s", name, exc)

        n_saved, sample_patches = extract_and_save_patches(
            str(hr_path), str(lr_path), hr_profile, hr_height, hr_width, hr_bands, lr_bands,
            residual_homography, hr_thresholds, lr_thresholds, radiometric_weights,
            histogram_lut, cfg, patch_prefix=name,
        )
        save_patch_contact_sheet(sample_patches, output_dir, name, cfg)
        return {"name": name, "mode": "paired", "n_saved": n_saved}

    if hr_path and not lr_path:
        logging.info("=== Processing '%s' (HR-only — synthesizing LR) ===", name)
        hr_bands = cfg.get("HR_RGB_BANDS", [1, 2, 3])
        with rasterio.open(str(hr_path)) as hr_src:
            hr_height, hr_width = hr_src.height, hr_src.width

        hr_thresholds = estimate_percentile_thresholds(
            str(hr_path), hr_bands, cfg["NODATA_VALUE"], cfg["SATURATED_VALUE"],
            tuple(cfg["CLIP_PERCENTILES"]), cfg,
        )

        try:
            overview = _read_single_overview(str(hr_path), hr_bands, cfg.get("COREG_PREVIEW_DECIM_DIM", 2000))
            save_preview(apply_percentile_scaling(overview, hr_thresholds), output_dir, name, "load_hr", cfg)
            save_band_wise_overview(str(hr_path), output_dir, name, "bands_hr", cfg)
        except Exception as exc:
            logging.warning("Could not build load preview for '%s': %s", name, exc)

        n_saved, sample_patches = extract_patches_hr_only(
            str(hr_path), hr_bands, hr_height, hr_width, hr_thresholds, cfg, patch_prefix=name,
        )
        save_patch_contact_sheet(sample_patches, output_dir, name, cfg)
        return {"name": name, "mode": "hr_only_degraded", "n_saved": n_saved}

    if lr_path and not hr_path:
        logging.info("=== Processing '%s' (LR-only — no HR available) ===", name)
        lr_bands = cfg.get("LR_RGB_BANDS", [3, 2, 1])
        with rasterio.open(str(lr_path)) as lr_src:
            lr_height, lr_width = lr_src.height, lr_src.width

        lr_thresholds = estimate_percentile_thresholds(
            str(lr_path), lr_bands, cfg["NODATA_VALUE"], cfg["SATURATED_VALUE"],
            tuple(cfg["CLIP_PERCENTILES"]), cfg,
        )

        try:
            overview = _read_single_overview(str(lr_path), lr_bands, cfg.get("COREG_PREVIEW_DECIM_DIM", 2000))
            save_preview(apply_percentile_scaling(overview, lr_thresholds), output_dir, name, "load_lr", cfg)
            save_band_wise_overview(str(lr_path), output_dir, name, "bands_lr", cfg)
        except Exception as exc:
            logging.warning("Could not build load preview for '%s': %s", name, exc)

        n_saved, sample_patches = extract_patches_lr_only(
            str(lr_path), lr_bands, lr_height, lr_width, lr_thresholds, cfg, patch_prefix=name,
        )
        save_patch_contact_sheet(sample_patches, output_dir, name, cfg)
        return {"name": name, "mode": "lr_only", "n_saved": n_saved}

    raise ValueError(f"Work item '{name}' has neither an HR nor an LR path.")


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
    parser = argparse.ArgumentParser(
        description="Preprocess satellite HR/LR imagery into SR training patches."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file overriding the inline CONFIG defaults. "
             "If omitted, a config.json next to this script is used if present.",
    )
    args = parser.parse_args()

    # Step 0: Build config first so we know OUTPUT_DIR before any logging
    # Use a minimal bootstrap logger until the file handler is ready
    logging.basicConfig(level=logging.WARNING)
    cfg = build_config(args.config)

    # Step 0b: Set up full logging (stdout + file in OUTPUT_DIR)
    _setup_logging(cfg["OUTPUT_DIR"])

    logging.info("Configuration:\n%s", "\n".join(f"  {k}: {v}" for k, v in cfg.items()))

    # Step 1: Resolve HR_IMAGE_PATH / LR_IMAGE_PATH into one or more scenes,
    # classifying each as paired / HR-only / LR-only.
    work_items = resolve_work_items(cfg)
    logging.info("Resolved %d scene(s) to process.", len(work_items))

    # GDAL keeps an internal block-read cache shared across every open
    # dataset in the process; left at its default (historically a fraction
    # of system RAM) it grows as windowed reads touch more of a large scene,
    # which would defeat the point of never loading a full array. Cap it
    # explicitly so peak memory stays bounded regardless of host RAM or
    # scene size — a window/patch is only ever a few MB, so a modest cache
    # is plenty to avoid re-reading the same disk blocks for adjacent windows.
    gdal_cache_mb = cfg.get("GDAL_CACHE_MB", 256)
    results = []
    total_saved = 0
    with rasterio.Env(GDAL_CACHEMAX=gdal_cache_mb * 1024 * 1024):
        for item in tqdm(work_items, desc="Scenes", unit="scene"):
            try:
                result = process_item(item, cfg)
            except Exception as exc:
                logging.error("Failed to process '%s': %s", item["name"], exc, exc_info=True)
                result = {"name": item["name"], "mode": "error", "n_saved": 0}
            results.append(result)
            total_saved += result["n_saved"]

    logging.info("=" * 60)
    logging.info(
        "Pipeline complete. %d scene(s) processed, %d patch(es) written to: %s",
        len(results), total_saved, cfg["OUTPUT_DIR"],
    )
    for r in results:
        logging.info("  %-50s mode=%-18s saved=%d", r["name"], r["mode"], r["n_saved"])


if __name__ == "__main__":
    main()
