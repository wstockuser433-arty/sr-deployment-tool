"""
raw_inference.py
================
End-to-end inference on raw satellite (or standard) images.

Two operating modes
-------------------
  paired   -- Takes a raw LR image and a matched HR image.
              Optionally runs the full pipeline3.py pre-processing stack
              (3-stage coregistration, percentile scaling, radiometric
              regression, histogram matching) to produce a properly aligned
              LR image.  The aligned LR is EXACTLY rescaled to
              (HR_H // scale_factor) x (HR_W // scale_factor), enforcing
              an integer ratio even when the real sensor resolution ratio
              is fractional (e.g. 1.7x, 2.3x).
              SwinIR runs on overlapping patches; SR tiles are stitched
              with a Hann-window blend to eliminate seams.
              8 quality metrics are computed against the HR ground truth.

  lr_only  -- Takes only a raw LR image.  No alignment is possible; the
              image is loaded + optionally percentile-scaled, then SR runs
              directly.  No metrics are computed.

Outputs (written to output_dir):
  lr_display.png   -- 8-bit RGB LR for frontend display
  sr_display.png   -- 8-bit RGB SR output
  hr_display.png   -- 8-bit RGB HR  (paired only)
  metrics.json     -- SR / bicubic / delta metrics (paired only)
  run_<ts>.log     -- timestamped log

Usage
-----
  python raw_inference.py --config path/to/config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Project root on sys.path so "from utils import utils_image" works
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils import utils_image as util
from utils import utils_visual_report as vrep

# ---------------------------------------------------------------------------
# Optional rasterio -- required only for GeoTIFF / JP2 inputs
# ---------------------------------------------------------------------------
try:
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.windows import Window, transform as window_transform
    from skimage.registration import phase_cross_correlation
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEOSPATIAL_EXTENSIONS = {".tif", ".tiff", ".jp2", ".img", ".vrt"}
METRIC_NAMES    = ["psnr", "ssim", "it_ssim", "sam", "uiqi", "rmse", "fsim", "srer"]
LOWER_IS_BETTER = {"sam", "rmse"}

DEFAULT_CONFIG: dict = {
    "mode":                         "paired",
    "lr_path":                      "",
    "hr_path":                      "",
    "lr_bands":                     [3, 2, 1],
    "hr_bands":                     [1, 2, 3],
    "enable_preprocessing":         True,
    "model_path":                   "",
    "output_dir":                   "testsets/raw_inference_output",
    "patch_size":                   128,
    "overlap":                      32,
    "scale_factor":                 2,
    # Percentile scaling
    "nodata_value":                 0,
    "saturated_value":              32767,
    "clip_percentiles":             [2.0, 98.0],
    "percentile_n_sample_windows":  20,
    # Stage A (ORB)
    "coreg_a_enabled":              True,
    "coreg_a_max_features":         8000,
    "coreg_a_match_ratio":          0.75,
    "coreg_a_ransac_thresh":        4.0,
    # Stage B (phase correlation)
    "coreg_b_enabled":              True,
    "coreg_b_upsample_factor":      100,
    # Radiometric regression
    "radiometric_enabled":          True,
    "radiometric_block_size":       256,
    "radiometric_rmse_threshold":   35.0,
    "radiometric_n_samples":        150_000,
    "radiometric_n_fit_windows":    80,
    "radiometric_post_hist_match":  True,
    "histogram_n_sample_windows":   40,
    # Overview
    "coreg_preview_decim_dim":      2000,
    # SwinIR architecture (overridden by "model_config" key in JSON)
    "model_config": {
        "upscale":          2,
        "in_chans":         3,
        "img_size":         128,
        "window_size":      8,
        "img_range":        1.0,
        "depths":           [6, 6, 6, 6, 6, 6],
        "embed_dim":        180,
        "num_heads":        [6, 6, 6, 6, 6, 6],
        "mlp_ratio":        2,
        "upsampler":        "pixelshuffle",
        "resi_connection":  "1conv",
    },
}


# ===========================================================================
# MODULE 1 -- IMAGE LOADING
# ===========================================================================

def _is_geospatial(path: str) -> bool:
    return Path(path).suffix.lower() in GEOSPATIAL_EXTENSIONS


def load_standard_image(path: str) -> np.ndarray:
    """Load PNG / BMP / TIFF via cv2 -> uint8 HxWx3 RGB."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ===========================================================================
# MODULE 2 -- PIPELINE3-STYLE ALIGNMENT FUNCTIONS
#   Copied / adapted from pleaides_preprocessing/pipeline3.py.
# ===========================================================================

def _raw_to_uint16(raw: np.ndarray) -> np.ndarray:
    if raw.dtype.kind != "f":
        return raw.astype(np.uint16)
    scaled = np.where(np.isnan(raw), 0.0, raw * 10_000.0)
    return np.clip(scaled, 0, 32767).astype(np.uint16)


def _to_gray_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype != np.uint8:
        a = array.astype(np.float32)
        rng = a.max() - a.min()
        if rng > 0:
            a = (a - a.min()) / rng * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    else:
        a = array
    gray = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return gray.astype(np.uint8)


def estimate_percentile_thresholds(
    path: str,
    bands: list,
    cfg: dict,
    overview: Optional[np.ndarray] = None,
) -> Dict[int, Tuple[float, float]]:
    """Estimate per-channel percentile thresholds for 16-bit -> 8-bit scaling."""
    if not _RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio is required for geospatial images.")
    nodata_value     = cfg.get("nodata_value", 0)
    saturated_value  = cfg.get("saturated_value", 32767)
    p_lo, p_hi       = cfg.get("clip_percentiles", [2.0, 98.0])
    n_sample_windows = cfg.get("percentile_n_sample_windows", 20)
    sample_size      = 256

    samples: List[List[np.ndarray]] = [[] for _ in range(3)]
    if overview is not None:
        for c in range(3):
            channel = overview[:, :, c].astype(np.float32)
            valid   = (overview[:, :, c] > nodata_value) & (overview[:, :, c] < saturated_value)
            if valid.any():
                samples[c].append(channel[valid])

    with rasterio.open(path) as src:
        height, width = src.height, src.width
        win_h = min(sample_size, height)
        win_w = min(sample_size, width)
        rng   = np.random.default_rng(seed=42)
        for _ in range(n_sample_windows):
            row  = int(rng.integers(0, max(1, height - win_h + 1)))
            col  = int(rng.integers(0, max(1, width  - win_w + 1)))
            data = _raw_to_uint16(src.read(bands, window=Window(col, row, win_w, win_h)))
            data = np.transpose(data, (1, 2, 0))
            for c in range(3):
                channel = data[:, :, c].astype(np.float32)
                valid   = (data[:, :, c] > nodata_value) & (data[:, :, c] < saturated_value)
                if valid.any():
                    samples[c].append(channel[valid])

    thresholds: Dict[int, Tuple[float, float]] = {}
    for c in range(3):
        pooled = np.concatenate(samples[c]) if samples[c] else np.array([], dtype=np.float32)
        if pooled.size == 0:
            logging.warning("Channel %d: no valid pixels, using (0, 1).", c)
            thresholds[c] = (0.0, 1.0)
            continue
        v_min = float(np.percentile(pooled, p_lo))
        v_max = float(np.percentile(pooled, p_hi))
        if v_max <= v_min:
            v_max = v_min + 1.0
        thresholds[c] = (v_min, v_max)
        logging.info("Percentile thresholds channel %d: [%.1f, %.1f]", c, v_min, v_max)
    return thresholds


def apply_percentile_scaling(
    window_array: np.ndarray,
    thresholds: Dict[int, Tuple[float, float]],
) -> np.ndarray:
    float_array = window_array.astype(np.float32)
    result      = np.zeros_like(float_array)
    for c in range(3):
        v_min, v_max    = thresholds[c]
        clipped         = np.clip(float_array[:, :, c], v_min, v_max)
        result[:, :, c] = (clipped - v_min) / (v_max - v_min)
    return np.clip(np.round(result * 255.0), 0, 255).astype(np.uint8)


def _read_decimated_overviews(
    hr_path: str, lr_path: str, hr_bands: list, lr_bands: list, max_dim: int,
) -> dict:
    with rasterio.open(hr_path) as hr_src:
        hr_height, hr_width = hr_src.height, hr_src.width
        hr_profile   = hr_src.profile.copy()
        hr_profile.update(count=3)
        decim_scale  = min(1.0, max_dim / max(hr_height, hr_width))
        out_h = max(1, int(round(hr_height * decim_scale)))
        out_w = max(1, int(round(hr_width  * decim_scale)))
        hr_data = _raw_to_uint16(hr_src.read(
            hr_bands, out_shape=(len(hr_bands), out_h, out_w),
            resampling=Resampling.average,
        ))
    hr_overview = np.transpose(hr_data, (1, 2, 0))

    dst_transform = hr_profile["transform"] * Affine.scale(hr_width / out_w, hr_height / out_h)
    lr_dst = np.zeros((3, out_h, out_w), dtype=np.uint16)
    with rasterio.open(lr_path) as lr_src:
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
    lr_overview = np.clip(np.transpose(lr_dst, (1, 2, 0)), 0, 32767).astype(np.uint16)
    logging.info(
        "Decimated overview: full=%dx%d  overview=%dx%d  scale=%.4f",
        hr_height, hr_width, out_h, out_w, decim_scale,
    )
    return {
        "hr_overview": hr_overview, "lr_overview": lr_overview,
        "hr_profile":  hr_profile,  "hr_height":   hr_height,
        "hr_width":    hr_width,    "decim_scale": decim_scale,
    }


def coregister_stage_a_orb(
    hr_small: np.ndarray, lr_small: np.ndarray, cfg: dict,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Stage A: Coarse Global Alignment via ORB + RANSAC Homography."""
    if not cfg.get("coreg_a_enabled", True):
        logging.info("Stage A disabled.")
        return lr_small, None
    height, width = hr_small.shape[:2]
    hr_gray = _to_gray_uint8(hr_small)
    lr_gray = _to_gray_uint8(lr_small)
    orb           = cv2.ORB_create(nfeatures=cfg.get("coreg_a_max_features", 8000))
    kp_hr, des_hr = orb.detectAndCompute(hr_gray, None)
    kp_lr, des_lr = orb.detectAndCompute(lr_gray, None)
    if des_hr is None or des_lr is None or len(kp_hr) < 4 or len(kp_lr) < 4:
        logging.warning("Stage A: too few keypoints -- skipping.")
        return lr_small, None
    matcher     = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des_lr, des_hr, k=2)
    good = [m for m, n in raw_matches if m.distance < cfg.get("coreg_a_match_ratio", 0.75) * n.distance]
    logging.info("Stage A: %d good matches.", len(good))
    if len(good) < 4:
        logging.warning("Stage A: fewer than 4 good matches -- skipping.")
        return lr_small, None
    src_pts = np.float32([kp_lr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_hr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, cfg.get("coreg_a_ransac_thresh", 4.0))
    if H is None:
        logging.warning("Stage A: RANSAC failed -- skipping.")
        return lr_small, None
    logging.info("Stage A: homography accepted (%d inliers).", int(inlier_mask.sum()) if inlier_mask is not None else 0)
    lr_warped = np.zeros_like(lr_small)
    for c in range(3):
        lr_warped[:, :, c] = np.clip(cv2.warpPerspective(
            lr_small[:, :, c].astype(np.float32), H, (width, height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ), 0, 65535).astype(np.uint16)
    logging.info("Stage A complete.")
    return lr_warped, H


def coregister_stage_b_phase(
    hr_small: np.ndarray, lr_small: np.ndarray, cfg: dict,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Stage B: Sub-pixel correction via Phase Cross-Correlation."""
    if not cfg.get("coreg_b_enabled", True):
        logging.info("Stage B disabled.")
        return lr_small, (0.0, 0.0)
    if not _RASTERIO_AVAILABLE:
        logging.warning("Stage B: skimage not available -- skipping.")
        return lr_small, (0.0, 0.0)
    height, width = hr_small.shape[:2]
    hr_gray = _to_gray_uint8(hr_small).astype(np.float32)
    lr_gray = _to_gray_uint8(lr_small).astype(np.float32)
    shift, error, _ = phase_cross_correlation(
        hr_gray, lr_gray, upsample_factor=cfg.get("coreg_b_upsample_factor", 100),
    )
    shift_row, shift_col = float(shift[0]), float(shift[1])
    logging.info("Stage B: shift row=%.4f col=%.4f (error=%.4f)", shift_row, shift_col, error)
    M = np.float32([[1, 0, shift_col], [0, 1, shift_row]])
    lr_shifted = np.zeros_like(lr_small)
    for c in range(3):
        lr_shifted[:, :, c] = np.clip(cv2.warpAffine(
            lr_small[:, :, c].astype(np.float32), M, (width, height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ), 0, 65535).astype(np.uint16)
    logging.info("Stage B complete.")
    return lr_shifted, (shift_row, shift_col)


def estimate_global_transform(
    hr_path: str, lr_path: str, hr_bands: list, lr_bands: list, cfg: dict,
) -> Tuple[np.ndarray, dict]:
    """Estimate residual homography from Stages A+B on decimated overviews."""
    max_dim     = cfg.get("coreg_preview_decim_dim", 2000)
    overviews   = _read_decimated_overviews(hr_path, lr_path, hr_bands, lr_bands, max_dim)
    decim_scale = overviews["decim_scale"]

    logging.info("=== Stage A: Coarse Global (ORB) ===")
    lr_after_a, H_small = coregister_stage_a_orb(overviews["hr_overview"], overviews["lr_overview"], cfg)

    logging.info("=== Stage B: Sub-pixel (Phase Correlation) ===")
    lr_after_b, (shift_row, shift_col) = coregister_stage_b_phase(overviews["hr_overview"], lr_after_a, cfg)

    H_combined = np.eye(3, dtype=np.float64) if H_small is None else H_small.astype(np.float64)
    T_b        = np.array([[1, 0, shift_col], [0, 1, shift_row], [0, 0, 1]], dtype=np.float64)
    H_combined = T_b @ H_combined
    S          = np.diag([1.0 / decim_scale, 1.0 / decim_scale, 1.0]).astype(np.float64)
    S_inv      = np.diag([decim_scale, decim_scale, 1.0]).astype(np.float64)
    residual_homography = S @ H_combined @ S_inv

    return residual_homography, {
        "hr_overview": overviews["hr_overview"], "lr_overview": overviews["lr_overview"],
        "lr_overview_after_a": lr_after_a,       "lr_overview_after_b": lr_after_b,
        "hr_profile":  overviews["hr_profile"],  "hr_height": overviews["hr_height"],
        "hr_width":    overviews["hr_width"],    "decim_scale": decim_scale,
    }


def _read_lr_window_to_hr_grid(
    lr_src, lr_bands: list, hr_profile: dict, residual_homography: np.ndarray,
    row: int, col: int, height: int, width: int, margin: int = 32,
) -> np.ndarray:
    """CRS-reproject one HR window from the LR source, apply residual homography.
    Mirrors pipeline3.py's _read_lr_window_to_hr_grid exactly."""
    pad_row = max(0, row - margin)
    pad_col = max(0, col - margin)
    pad_h   = height + (row - pad_row) + margin
    pad_w   = width  + (col - pad_col) + margin
    window        = Window(pad_col, pad_row, pad_w, pad_h)
    dst_transform = window_transform(window, hr_profile["transform"])
    lr_dst = np.zeros((3, pad_h, pad_w), dtype=np.uint16)
    for band_idx, band_number in enumerate(lr_bands):
        reproject(
            source=rasterio.band(lr_src, band_number),
            destination=lr_dst[band_idx],
            src_transform=lr_src.transform, src_crs=lr_src.crs,
            dst_transform=dst_transform,    dst_crs=hr_profile["crs"],
            resampling=Resampling.cubic,
        )
    lr_reprojected = np.clip(np.transpose(lr_dst, (1, 2, 0)), 0, 32767).astype(np.uint16)
    to_full  = np.array([[1, 0, pad_col], [0, 1, pad_row], [0, 0, 1]], dtype=np.float64)
    to_local = np.array([[1, 0, -pad_col], [0, 1, -pad_row], [0, 0, 1]], dtype=np.float64)
    H_local  = to_local @ residual_homography @ to_full
    lr_warped = np.zeros_like(lr_reprojected)
    for c in range(3):
        lr_warped[:, :, c] = np.clip(cv2.warpPerspective(
            lr_reprojected[:, :, c].astype(np.float32), H_local, (pad_w, pad_h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ), 0, 65535).astype(np.uint16)
    crop_row = row - pad_row
    crop_col = col - pad_col
    return lr_warped[crop_row:crop_row + height, crop_col:crop_col + width, :]


def fit_radiometric_regression(
    hr_path: str, lr_path: str, hr_profile: dict, hr_height: int, hr_width: int,
    hr_bands: list, lr_bands: list, residual_homography: np.ndarray,
    hr_thresholds: Dict[int, Tuple[float, float]],
    lr_thresholds: Dict[int, Tuple[float, float]],
    cfg: dict,
) -> np.ndarray:
    """Fit linear radiometric model LR->HR. Returns weights (4,3)."""
    block_size    = cfg.get("radiometric_block_size", 256)
    rmse_thresh   = cfg.get("radiometric_rmse_threshold", 35.0)
    n_samples     = cfg.get("radiometric_n_samples", 150_000)
    n_fit_windows = cfg.get("radiometric_n_fit_windows", 80)
    row_starts_all = list(range(0, hr_height - block_size + 1, block_size))
    col_starts_all = list(range(0, hr_width  - block_size + 1, block_size))
    all_blocks     = [(r, c) for r in row_starts_all for c in col_starts_all]
    if not all_blocks:
        logging.warning("Scene too small for radiometric blocks -- using identity.")
        return np.vstack([np.eye(3, dtype=np.float32), np.zeros((1, 3), dtype=np.float32)])
    rng          = np.random.default_rng(seed=42)
    n_candidates = min(n_fit_windows, len(all_blocks))
    candidate_blocks = [all_blocks[i] for i in rng.choice(len(all_blocks), size=n_candidates, replace=False)]
    sampled_blocks = []
    with rasterio.open(hr_path) as hr_src, rasterio.open(lr_path) as lr_src:
        for r, c in candidate_blocks:
            hr_raw   = np.transpose(_raw_to_uint16(hr_src.read(hr_bands, window=Window(c, r, block_size, block_size))), (1, 2, 0))
            hr_block = apply_percentile_scaling(hr_raw, hr_thresholds)
            lr_raw   = _read_lr_window_to_hr_grid(lr_src, lr_bands, hr_profile, residual_homography, r, c, block_size, block_size)
            lr_block = apply_percentile_scaling(lr_raw, lr_thresholds)
            rmse     = float(np.sqrt(np.mean((lr_block.astype(np.float32) - hr_block.astype(np.float32)) ** 2)))
            sampled_blocks.append((r, c, hr_block, lr_block, rmse))
    accepted = [b for b in sampled_blocks if b[4] <= rmse_thresh]
    logging.info("Radiometric: %d/%d blocks accepted (RMSE <= %.1f)", len(accepted), len(sampled_blocks), rmse_thresh)
    if not accepted:
        logging.warning("All blocks exceed RMSE threshold -- using all.")
        accepted = sampled_blocks
    pixels_per_block = max(1, n_samples // len(accepted))
    block_pixels     = block_size * block_size
    lr_list, hr_list = [], []
    for r, c, hr_block, lr_block, _ in accepted:
        lr_flat = lr_block.reshape(-1, 3).astype(np.float32)
        hr_flat = hr_block.reshape(-1, 3).astype(np.float32)
        idx     = rng.choice(block_pixels, size=min(pixels_per_block, block_pixels), replace=False)
        lr_list.append(lr_flat[idx]);  hr_list.append(hr_flat[idx])
    lr_pixels = np.concatenate(lr_list, axis=0)
    hr_pixels = np.concatenate(hr_list, axis=0)
    V = np.column_stack([lr_pixels, np.ones(lr_pixels.shape[0], dtype=np.float32)])
    weights, _, rank, _ = np.linalg.lstsq(V, hr_pixels, rcond=None)
    logging.info("Radiometric regression fitted. Rank: %d", rank)
    return weights


def apply_radiometric_weights(lr_uint8: np.ndarray, weights: np.ndarray) -> np.ndarray:
    h, w    = lr_uint8.shape[:2]
    lr_flat = lr_uint8.astype(np.float32).reshape(-1, 3)
    v_full  = np.column_stack([lr_flat, np.ones(h * w, dtype=np.float32)])
    adj     = v_full @ weights
    return np.clip(np.round(adj), 0, 255).astype(np.uint8).reshape(h, w, 3)


def _build_match_lut(source_hist: np.ndarray, reference_hist: np.ndarray) -> np.ndarray:
    src_cdf = np.cumsum(source_hist).astype(np.float64);  src_cdf /= src_cdf[-1] or 1.0
    ref_cdf = np.cumsum(reference_hist).astype(np.float64); ref_cdf /= ref_cdf[-1] or 1.0
    lut = np.interp(src_cdf, ref_cdf, np.arange(256, dtype=np.float64))
    return np.clip(np.round(lut), 0, 255).astype(np.uint8)


def estimate_histogram_match_lut(
    hr_path: str, lr_path: str, hr_profile: dict, hr_height: int, hr_width: int,
    hr_bands: list, lr_bands: list, residual_homography: np.ndarray,
    hr_thresholds: Dict[int, Tuple[float, float]],
    lr_thresholds: Dict[int, Tuple[float, float]],
    radiometric_weights: np.ndarray, cfg: dict,
) -> np.ndarray:
    """Estimate per-channel 256-entry LUT for histogram matching. Returns (3,256) uint8."""
    n_sample_windows = cfg.get("histogram_n_sample_windows", 40)
    sample_size      = 256
    hr_hist = np.zeros((3, 256), dtype=np.float64)
    lr_hist = np.zeros((3, 256), dtype=np.float64)
    rng   = np.random.default_rng(seed=42)
    win_h = min(sample_size, hr_height);  win_w = min(sample_size, hr_width)
    with rasterio.open(hr_path) as hr_src, rasterio.open(lr_path) as lr_src:
        for _ in range(n_sample_windows):
            row = int(rng.integers(0, max(1, hr_height - win_h + 1)))
            col = int(rng.integers(0, max(1, hr_width  - win_w + 1)))
            hr_raw   = np.transpose(_raw_to_uint16(hr_src.read(hr_bands, window=Window(col, row, win_w, win_h))), (1, 2, 0))
            hr_block = apply_percentile_scaling(hr_raw, hr_thresholds)
            lr_raw         = _read_lr_window_to_hr_grid(lr_src, lr_bands, hr_profile, residual_homography, row, col, win_h, win_w)
            lr_block_scaled = apply_percentile_scaling(lr_raw, lr_thresholds)
            lr_block        = apply_radiometric_weights(lr_block_scaled, radiometric_weights)
            for c in range(3):
                hr_hist[c] += np.bincount(hr_block[:, :, c].ravel(), minlength=256)
                lr_hist[c] += np.bincount(lr_block[:, :, c].ravel(), minlength=256)
    lut = np.zeros((3, 256), dtype=np.uint8)
    for c in range(3):
        lut[c] = _build_match_lut(lr_hist[c], hr_hist[c])
    logging.info("Histogram LUT estimated from %d windows.", n_sample_windows)
    return lut


def apply_histogram_lut(window_uint8: np.ndarray, lut: np.ndarray) -> np.ndarray:
    result = np.empty_like(window_uint8)
    for c in range(3):
        result[:, :, c] = lut[c][window_uint8[:, :, c]]
    return result


# ===========================================================================
# MODULE 3 -- ALIGNMENT ORCHESTRATOR
# ===========================================================================

def align_lr_to_hr(
    hr_path: str, lr_path: str, hr_bands: list, lr_bands: list, cfg: dict,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Run full pipeline3 alignment stack and return full-scene uint8 arrays.

    Returns
    -------
    aligned_lr_uint8 : (HR_H, HR_W, 3) uint8 RGB -- LR on HR pixel grid
    hr_uint8         : (HR_H, HR_W, 3) uint8 RGB -- HR percentile-scaled
    hr_height, hr_width
    """
    if not _RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio is required for geospatial alignment.")

    logging.info("=== Estimating global transform (Stages A + B) ===")
    residual_homography, diagnostics = estimate_global_transform(hr_path, lr_path, hr_bands, lr_bands, cfg)
    hr_profile = diagnostics["hr_profile"]
    hr_height  = diagnostics["hr_height"]
    hr_width   = diagnostics["hr_width"]

    logging.info("=== Estimating percentile thresholds ===")
    hr_thresholds = estimate_percentile_thresholds(hr_path, hr_bands, cfg, overview=diagnostics["hr_overview"])
    lr_thresholds = estimate_percentile_thresholds(lr_path, lr_bands, cfg)

    radiometric_weights = None
    histogram_lut       = None
    if cfg.get("radiometric_enabled", True):
        logging.info("=== Fitting radiometric regression ===")
        radiometric_weights = fit_radiometric_regression(
            hr_path, lr_path, hr_profile, hr_height, hr_width,
            hr_bands, lr_bands, residual_homography, hr_thresholds, lr_thresholds, cfg,
        )
        if cfg.get("radiometric_post_hist_match", True):
            logging.info("=== Estimating histogram-match LUT ===")
            histogram_lut = estimate_histogram_match_lut(
                hr_path, lr_path, hr_profile, hr_height, hr_width,
                hr_bands, lr_bands, residual_homography,
                hr_thresholds, lr_thresholds, radiometric_weights, cfg,
            )

    TILE = 512
    logging.info("=== Building full-scene aligned arrays (%dx%d, tile=%d) ===", hr_height, hr_width, TILE)
    hr_full = np.zeros((hr_height, hr_width, 3), dtype=np.uint8)
    lr_full = np.zeros((hr_height, hr_width, 3), dtype=np.uint8)

    with rasterio.open(hr_path) as hr_src, rasterio.open(lr_path) as lr_src:
        row_starts = list(range(0, hr_height, TILE))
        col_starts = list(range(0, hr_width,  TILE))
        total      = len(row_starts) * len(col_starts)
        done       = 0
        for row in row_starts:
            for col in col_starts:
                h = min(TILE, hr_height - row);  w = min(TILE, hr_width - col)
                hr_raw  = np.transpose(_raw_to_uint16(hr_src.read(hr_bands, window=Window(col, row, w, h))), (1, 2, 0))
                hr_tile = apply_percentile_scaling(hr_raw, hr_thresholds)
                hr_full[row:row + h, col:col + w, :] = hr_tile
                lr_raw  = _read_lr_window_to_hr_grid(lr_src, lr_bands, hr_profile, residual_homography, row, col, h, w)
                lr_tile = apply_percentile_scaling(lr_raw, lr_thresholds)
                if radiometric_weights is not None:
                    lr_tile = apply_radiometric_weights(lr_tile, radiometric_weights)
                if histogram_lut is not None:
                    lr_tile = apply_histogram_lut(lr_tile, histogram_lut)
                lr_full[row:row + h, col:col + w, :] = lr_tile
                done += 1
                if done % max(1, total // 10) == 0:
                    logging.info("  Tiling: %d / %d", done, total)

    return lr_full, hr_full, hr_height, hr_width


# ===========================================================================
# MODULE 4 -- EXACT SCALE ENFORCEMENT
# ===========================================================================

def enforce_exact_scale(
    lr_img: np.ndarray, hr_height: int, hr_width: int, scale_factor: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Resize LR to EXACTLY (hr_height // scale_factor) x (hr_width // scale_factor).

    Guarantees an integer x scale_factor ratio regardless of the real-world
    sensor resolution difference (1.7x, 2.3x, etc.).
    Mirrors LR_PATCH_SIZE = HR_PATCH_SIZE // SCALE_FACTOR in pipeline3.py.

    Returns: resized_lr, lr_h, lr_w
    """
    lr_h = hr_height // scale_factor
    lr_w = hr_width  // scale_factor
    if lr_img.shape[0] != lr_h or lr_img.shape[1] != lr_w:
        logging.info(
            "Enforcing exact scale: LR %dx%d -> %dx%d (x%d from HR %dx%d)",
            lr_img.shape[1], lr_img.shape[0], lr_w, lr_h,
            scale_factor, hr_width, hr_height,
        )
        lr_img = cv2.resize(lr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        lr_img = np.clip(lr_img, 0, 255).astype(np.uint8)
    return lr_img, lr_h, lr_w


# ===========================================================================
# MODULE 5 -- SWINIR MODEL
# ===========================================================================

def load_model(checkpoint_path: str, model_cfg: dict, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path:
        raise FileNotFoundError("model_path must point to a .pth checkpoint.")
    cp = Path(checkpoint_path)
    if not cp.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cp}")
    from models.network_swinir import SwinIR as net
    model = net(
        upscale         = model_cfg["upscale"],
        in_chans        = model_cfg.get("in_chans", 3),
        img_size        = model_cfg.get("img_size", 128),
        window_size     = model_cfg.get("window_size", 8),
        img_range       = model_cfg.get("img_range", 1.0),
        depths          = model_cfg.get("depths", [6, 6, 6, 6, 6, 6]),
        embed_dim       = model_cfg.get("embed_dim", 180),
        num_heads       = model_cfg.get("num_heads", [6, 6, 6, 6, 6, 6]),
        mlp_ratio       = model_cfg.get("mlp_ratio", 2),
        upsampler       = model_cfg.get("upsampler", "pixelshuffle"),
        resi_connection = model_cfg.get("resi_connection", "1conv"),
    ).to(device)
    state_dict = torch.load(cp, map_location=device)
    if isinstance(state_dict, dict):
        for key in ("params_ema", "params", "state_dict"):
            if key in state_dict:
                state_dict = state_dict[key]; break
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logging.info("Model loaded: %s", cp)
    return model


def _image_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    img = np.ascontiguousarray(np.transpose(image_rgb[:, :, :3], (2, 0, 1)))
    return torch.from_numpy(img).float().unsqueeze(0).div(255.0)


def _tensor_to_rgb_uint8(output: torch.Tensor) -> np.ndarray:
    img = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return (img * 255.0).round().astype(np.uint8)


def _pad_to_window_size(image: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
    _, _, h, w = image.size()
    h_pad = (window_size - h % window_size) % window_size
    w_pad = (window_size - w % window_size) % window_size
    if h_pad > 0:
        image = torch.cat([image, torch.flip(image, [2])], 2)[:, :, :h + h_pad, :]
    if w_pad > 0:
        image = torch.cat([image, torch.flip(image, [3])], 3)[:, :, :, :w + w_pad]
    return image, h, w


# ===========================================================================
# MODULE 6 -- PATCH -> SR -> STITCH
# ===========================================================================

def _make_hann_window(size: int) -> np.ndarray:
    h = np.hanning(size).astype(np.float32)
    return np.outer(h, h)[:, :, np.newaxis]


def run_sr_on_image(
    lr_img: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    scale: int,
    window_size: int,
    patch_size: int,
    overlap: int,
) -> np.ndarray:
    """
    Run SwinIR on overlapping patches with Hann-window stitching.
    lr_img : uint8 HxWx3 RGB
    Returns: uint8 (H*scale) x (W*scale) x 3 RGB
    """
    lr_h, lr_w = lr_img.shape[:2]
    sr_h, sr_w = lr_h * scale, lr_w * scale
    sr_canvas  = np.zeros((sr_h, sr_w, 3), dtype=np.float32)
    wt_canvas  = np.zeros((sr_h, sr_w, 1), dtype=np.float32)
    hann       = _make_hann_window(patch_size * scale)
    stride     = patch_size - overlap

    row_starts = list(range(0, lr_h - patch_size, stride))
    if not row_starts or row_starts[-1] + patch_size < lr_h:
        row_starts.append(max(0, lr_h - patch_size))
    col_starts = list(range(0, lr_w - patch_size, stride))
    if not col_starts or col_starts[-1] + patch_size < lr_w:
        col_starts.append(max(0, lr_w - patch_size))

    total = len(row_starts) * len(col_starts)
    done  = 0
    logging.info("SR: %d patches (%dx%d stride=%d)", total, patch_size, patch_size, stride)

    with torch.no_grad():
        for row in row_starts:
            for col in col_starts:
                patch = lr_img[row:row + patch_size, col:col + patch_size, :]
                ph, pw = patch.shape[:2]
                t = _image_to_tensor(patch).to(device)
                t, orig_h, orig_w = _pad_to_window_size(t, window_size)
                out      = model(t)
                out      = out[..., :orig_h * scale, :orig_w * scale]
                sr_patch = _tensor_to_rgb_uint8(out)   # uint8 RGB
                sr_ph, sr_pw = ph * scale, pw * scale
                sr_row,  sr_col  = row * scale, col * scale
                blend = hann[:sr_ph, :sr_pw, :]
                sr_canvas[sr_row:sr_row + sr_ph, sr_col:sr_col + sr_pw, :] += sr_patch.astype(np.float32) * blend
                wt_canvas[sr_row:sr_row + sr_ph, sr_col:sr_col + sr_pw, :] += blend
                done += 1
                if done % max(1, total // 5) == 0:
                    logging.info("  Inference: %d / %d patches", done, total)

    sr_img = sr_canvas / (wt_canvas + 1e-8)
    return np.clip(np.round(sr_img), 0, 255).astype(np.uint8)


# ===========================================================================
# MODULE 7 -- METRICS
# ===========================================================================

def calculate_all_metrics(sr_rgb: np.ndarray, hr_rgb: np.ndarray, border: int) -> dict:
    """
    Compute 8 SR metrics, bicubic-baseline metrics, deltas, and per-channel PSNR/SSIM.
    All images uint8 HxWxC RGB.
    """
    sr_bgr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)
    hr_bgr = cv2.cvtColor(hr_rgb, cv2.COLOR_RGB2BGR)

    # Bicubic baseline: downsample HR then upsample back
    lr_h, lr_w = hr_bgr.shape[0] // 2, hr_bgr.shape[1] // 2
    lr_small    = cv2.resize(hr_bgr, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    bicubic_bgr = cv2.resize(lr_small, (hr_bgr.shape[1], hr_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)

    th, tw    = hr_bgr.shape[:2]
    sr_crop   = sr_bgr[:th, :tw, :].astype(np.uint8)
    bic_crop  = bicubic_bgr[:th, :tw, :].astype(np.uint8)
    hr_u8     = hr_bgr.astype(np.uint8)

    def _m(img):
        return {
            "psnr":    float(util.calculate_psnr(img,    hr_u8, border=border)),
            "ssim":    float(util.calculate_ssim(img,    hr_u8, border=border)),
            "it_ssim": float(util.calculate_it_ssim(img, hr_u8, border=border)),
            "sam":     float(util.calculate_sam(img,     hr_u8, border=border)),
            "uiqi":    float(util.calculate_uiqi(img,    hr_u8, border=border)),
            "rmse":    float(util.calculate_rmse(img,    hr_u8, border=border)),
            "fsim":    float(util.calculate_fsim(img,    hr_u8, border=border)),
            "srer":    float(util.calculate_srer(img,    hr_u8, border=border)),
        }

    sr_metrics = _m(sr_crop)
    lr_metrics = _m(bic_crop)
    deltas     = {k: sr_metrics[k] - lr_metrics[k] for k in METRIC_NAMES}

    # Per-band PSNR and SSIM (channel 0 = first selected band, etc.)
    per_band = {}
    for i in range(sr_crop.shape[2]):
        per_band[f"band_{i + 1}"] = {
            "psnr": float(util.calculate_psnr(sr_crop[:, :, i:i+1], hr_u8[:, :, i:i+1], border=border)),
            "ssim": float(util.calculate_ssim(sr_crop[:, :, i:i+1], hr_u8[:, :, i:i+1], border=border)),
        }

    return {"sr": sr_metrics, "lr_bicubic": lr_metrics, "delta": deltas, "per_band": per_band}


# ===========================================================================
# MODULE 8 -- I/O HELPERS
# ===========================================================================

def save_display_png(array_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(array_rgb, cv2.COLOR_RGB2BGR))
    logging.info("Saved: %s", path)


# ===========================================================================
# MODULE 9 -- ORCHESTRATORS
# ===========================================================================

def run_paired_inference(cfg: dict) -> None:
    """
    Full paired (LR + HR) raw image inference pipeline.

    1. Load / align LR image to HR grid (pipeline3 stages if geospatial + preprocessing enabled)
    2. Enforce EXACT integer scale: LR -> (HR_H // scale) x (HR_W // scale)
    3. Run SwinIR SR with patch + stitch
    4. Compute 8 metrics against HR
    5. Save display images + metrics.json
    """
    lr_path    = cfg["lr_path"]
    hr_path    = cfg["hr_path"]
    lr_bands   = cfg.get("lr_bands", [3, 2, 1])
    hr_bands   = cfg.get("hr_bands", [1, 2, 3])
    scale      = int(cfg.get("scale_factor", 2))
    patch_size = int(cfg.get("patch_size", 128))
    overlap    = int(cfg.get("overlap", 32))
    output_dir = Path(cfg["output_dir"])
    model_cfg  = cfg.get("model_config", DEFAULT_CONFIG["model_config"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load / Align
    geo_lr = _is_geospatial(lr_path)
    geo_hr = _is_geospatial(hr_path)

    if cfg.get("enable_preprocessing", True) and geo_lr and geo_hr and _RASTERIO_AVAILABLE:
        logging.info("Preprocessing ENABLED -- running pipeline3 alignment stack.")
        lr_aligned, hr_uint8, hr_height, hr_width = align_lr_to_hr(hr_path, lr_path, hr_bands, lr_bands, cfg)
    else:
        if cfg.get("enable_preprocessing", True):
            logging.info("Preprocessing requested but images are standard format -- skipping coregistration.")
        else:
            logging.info("Preprocessing DISABLED -- loading images directly.")
        hr_uint8 = load_standard_image(hr_path)
        hr_height, hr_width = hr_uint8.shape[:2]
        lr_raw   = load_standard_image(lr_path)
        # For geospatial-but-preprocessing-disabled, apply percentile scaling
        if geo_lr and _RASTERIO_AVAILABLE:
            lr_thresholds = estimate_percentile_thresholds(lr_path, lr_bands, cfg)
            with rasterio.open(lr_path) as src:
                data = _raw_to_uint16(src.read(lr_bands))
            lr_raw   = apply_percentile_scaling(np.transpose(data, (1, 2, 0)), lr_thresholds)
        lr_aligned = lr_raw

    # Step 2: ENFORCE EXACT INTEGER SCALE
    # Regardless of the real sensor resolution ratio (1.7x, 2.3x, etc.),
    # the LR fed to the model is always exactly (HR_H // scale) x (HR_W // scale).
    # This mirrors pipeline3.py: LR_PATCH_SIZE = HR_PATCH_SIZE // SCALE_FACTOR.
    lr_scaled, lr_h, lr_w = enforce_exact_scale(lr_aligned, hr_height, hr_width, scale)

    # HR mod-cropped to match SR output dimensions
    hr_display = hr_uint8[:lr_h * scale, :lr_w * scale, :]

    # Step 3: Run SR
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)
    model      = load_model(cfg["model_path"], model_cfg, device)
    window_size = int(model_cfg.get("window_size", 8))
    logging.info("Running SR: %dx%d LR -> %dx%d SR", lr_w, lr_h, lr_w * scale, lr_h * scale)
    sr_img = run_sr_on_image(lr_scaled, model, device, scale, window_size, patch_size, overlap)

    # Step 4: Metrics
    logging.info("Computing metrics...")
    metrics  = calculate_all_metrics(sr_img, hr_display, border=scale)
    sr_m, lr_m, d_m = metrics["sr"], metrics["lr_bicubic"], metrics["delta"]
    logging.info("SR metrics:  " + "  ".join(f"{k.upper()} {sr_m[k]:.4f}" for k in METRIC_NAMES))
    logging.info("LR (bic):    " + "  ".join(f"{k.upper()} {lr_m[k]:.4f}" for k in METRIC_NAMES))
    logging.info("Delta:       " + "  ".join(
        f"d{k.upper()} {d_m[k]:+.4f}" + ("(down=better)" if k in LOWER_IS_BETTER else "(up=better)")
        for k in METRIC_NAMES))

    # Step 4b: Optional "5-layer" visual assessment (visual grids, FFT spectrum,
    # error/residual maps) on a random sample of patch-sized crops within the
    # stitched scene. Off by default -- configs that predate this option
    # (every existing caller) behave exactly as before.
    if bool(cfg.get("visual_report_enabled", False)):
        try:
            n_samples = int(cfg.get("visual_report_samples", 10))
            seed      = int(cfg.get("visual_report_seed", 23))
            report_patch = min(256, hr_display.shape[0], hr_display.shape[1])
            boxes = vrep.sample_patch_boxes(hr_display.shape[0], hr_display.shape[1], report_patch, n_samples, seed)
            logging.info("Visual assessment enabled: %d sample patch(es), seed=%d", len(boxes), seed)
            preview_dir = output_dir / "_previews"
            lr_up_bgr = cv2.cvtColor(
                cv2.resize(lr_scaled, (hr_display.shape[1], hr_display.shape[0]), interpolation=cv2.INTER_CUBIC),
                cv2.COLOR_RGB2BGR,
            )
            sr_bgr_full = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
            hr_bgr_full = cv2.cvtColor(hr_display, cv2.COLOR_RGB2BGR)
            for i, (r, c) in enumerate(boxes):
                name = f"sample_{i:02d}_r{r}_c{c}"
                vrep.generate_sample_report(
                    lr_up_bgr[r:r + report_patch, c:c + report_patch],
                    sr_bgr_full[r:r + report_patch, c:c + report_patch],
                    hr_bgr_full[r:r + report_patch, c:c + report_patch],
                    preview_dir, name,
                )
        except Exception as exc:
            logging.warning("Visual assessment step failed (non-fatal): %s", exc)

    # Step 5: Save
    save_display_png(lr_scaled,  output_dir / "lr_display.png")
    save_display_png(sr_img,     output_dir / "sr_display.png")
    save_display_png(hr_display, output_dir / "hr_display.png")

    # Per-band grayscale images (channel i+1 of the RGB display)
    for i in range(lr_scaled.shape[2]):
        cv2.imwrite(str(output_dir / f"lr_band_{i + 1}.png"), lr_scaled[:, :, i])
    for i in range(sr_img.shape[2]):
        cv2.imwrite(str(output_dir / f"sr_band_{i + 1}.png"), sr_img[:, :, i])
    for i in range(hr_display.shape[2]):
        cv2.imwrite(str(output_dir / f"hr_band_{i + 1}.png"), hr_display[:, :, i])

    metrics["lr_bands"] = lr_bands
    metrics["hr_bands"] = hr_bands
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logging.info("Metrics -> %s", output_dir / "metrics.json")
    logging.info("Paired inference complete. Outputs: %s", output_dir)


def run_lr_only_inference(cfg: dict) -> None:
    """
    LR-only inference -- no HR ground truth, no metrics.
    """
    lr_path    = cfg["lr_path"]
    lr_bands   = cfg.get("lr_bands", [1, 2, 3])
    scale      = int(cfg.get("scale_factor", 2))
    patch_size = int(cfg.get("patch_size", 128))
    overlap    = int(cfg.get("overlap", 32))
    output_dir = Path(cfg["output_dir"])
    model_cfg  = cfg.get("model_config", DEFAULT_CONFIG["model_config"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load LR
    if _is_geospatial(lr_path) and _RASTERIO_AVAILABLE:
        logging.info("Loading geospatial LR with percentile scaling.")
        lr_thresholds = estimate_percentile_thresholds(lr_path, lr_bands, cfg)
        with rasterio.open(lr_path) as src:
            data = _raw_to_uint16(src.read(lr_bands))
        lr_img = apply_percentile_scaling(np.transpose(data, (1, 2, 0)), lr_thresholds)
    else:
        logging.info("Loading standard image via cv2.")
        lr_img = load_standard_image(lr_path)

    lr_h, lr_w = lr_img.shape[:2]

    # Run SR
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)
    model      = load_model(cfg["model_path"], model_cfg, device)
    window_size = int(model_cfg.get("window_size", 8))
    logging.info("Running SR: %dx%d LR -> %dx%d SR", lr_w, lr_h, lr_w * scale, lr_h * scale)
    sr_img = run_sr_on_image(lr_img, model, device, scale, window_size, patch_size, overlap)

    save_display_png(lr_img, output_dir / "lr_display.png")
    save_display_png(sr_img, output_dir / "sr_display.png")

    for i in range(lr_img.shape[2]):
        cv2.imwrite(str(output_dir / f"lr_band_{i + 1}.png"), lr_img[:, :, i])
    for i in range(sr_img.shape[2]):
        cv2.imwrite(str(output_dir / f"sr_band_{i + 1}.png"), sr_img[:, :, i])

    logging.info("LR-only inference complete. Outputs: %s", output_dir)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def _setup_logging(output_dir: str) -> None:
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts       = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"raw_inference_{ts}.log"
    fmt      = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt  = "%H:%M:%S"
    root     = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(fh)
    logging.info("Log: %s", log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw satellite image SR inference.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        user_cfg = json.load(fh)

    cfg = {**DEFAULT_CONFIG, **user_cfg}
    if "model_config" in user_cfg:
        cfg["model_config"] = {**DEFAULT_CONFIG["model_config"], **user_cfg["model_config"]}

    _setup_logging(cfg["output_dir"])
    logging.info("raw_inference.py -- mode=%s", cfg.get("mode", "paired"))

    mode = cfg.get("mode", "paired").strip().lower()
    if mode == "paired":
        if not cfg.get("lr_path") or not cfg.get("hr_path"):
            raise ValueError("Paired mode requires both lr_path and hr_path.")
        run_paired_inference(cfg)
    elif mode == "lr_only":
        if not cfg.get("lr_path"):
            raise ValueError("lr_only mode requires lr_path.")
        run_lr_only_inference(cfg)
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'paired' or 'lr_only'.")


if __name__ == "__main__":
    main()
