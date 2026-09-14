"""
utils_visual_report.py
=======================
Additive visual-assessment helpers implementing the client's "5-layer" QA
framework (visual grids, FFT/radial-spectrum comparison, error maps, residual
maps) on top of whatever SR/HR/LR arrays an inference script already has in
memory.

Design contract, so callers can adopt this without risk:
  - Every public function catches its own exceptions internally, logs a
    warning, and returns None (or a tuple of Nones) -- a broken plot must
    never take down an inference run that would otherwise have succeeded.
  - All image inputs are BGR uint8 (OpenCV-native), matching
    main_test_swinir_config.py's internal convention. Callers working in RGB
    (e.g. raw_inference.py) convert to BGR before calling in.
  - Nothing here is imported or executed unless a caller explicitly opts in;
    this module has no side effects at import time beyond the optional
    matplotlib import guard below.
"""
import logging
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless -- no GUI backend required
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


def matplotlib_available() -> bool:
    return _MATPLOTLIB_AVAILABLE


def emit_preview_marker(filename: str, stage: str, scene_name: str) -> None:
    """Log the PREVIEW_READY marker the GUI's LogConsole already knows how to parse."""
    logging.info("PREVIEW_READY %s %s %s", filename, stage, scene_name)


def sample_names(names: Iterable[str], n_samples: int, seed: int) -> List[str]:
    """Deterministically pick up to n_samples names from an iterable, order-shuffled."""
    pool = list(names)
    random.Random(seed).shuffle(pool)
    return pool[: max(0, int(n_samples))]


def sample_patch_boxes(
    height: int, width: int, patch_size: int, n_samples: int, seed: int,
) -> List[Tuple[int, int]]:
    """Pick up to n_samples random (row, col) top-left corners for patch_size crops
    that fit within (height, width). Returns fewer than n_samples if the image
    is too small to fit any patch."""
    if patch_size <= 0 or patch_size > height or patch_size > width:
        return []
    rng = random.Random(seed)
    max_row = height - patch_size
    max_col = width - patch_size
    boxes = []
    for _ in range(max(0, int(n_samples))):
        boxes.append((rng.randint(0, max_row), rng.randint(0, max_col)))
    return boxes


def _radial_profile(mag: np.ndarray) -> np.ndarray:
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), mag.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def save_visual_grid(
    lr_bgr: np.ndarray, sr_bgr: np.ndarray, hr_bgr: np.ndarray,
    preview_dir: Path, scene_name: str, stage: str = "grid",
) -> Optional[Path]:
    """3-panel LR(bicubic-upsampled to HR size)/SR/HR side-by-side comparison JPEG."""
    try:
        h, w = hr_bgr.shape[:2]
        lr_up = cv2.resize(lr_bgr, (w, h), interpolation=cv2.INTER_CUBIC) if lr_bgr.shape[:2] != (h, w) else lr_bgr
        tiles = []
        for panel, label in ((lr_up, "LR (bicubic)"), (sr_bgr, "SR"), (hr_bgr, "HR")):
            tile = panel.copy()
            cv2.putText(tile, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            tiles.append(tile)
        sheet = np.hstack(tiles)
        preview_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{scene_name}_{stage}.jpg"
        path = preview_dir / filename
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        emit_preview_marker(filename, stage, scene_name)
        return path
    except Exception as exc:
        logging.warning("Visual grid failed for '%s': %s", scene_name, exc)
        return None


def save_fft_comparison(
    lr_bgr: np.ndarray, sr_bgr: np.ndarray, hr_bgr: np.ndarray,
    preview_dir: Path, scene_name: str, stage: str = "fft",
) -> Optional[Path]:
    """Radial-averaged frequency-spectrum comparison: LR vs SR vs HR.
    Returns None (with a logged warning) if matplotlib is unavailable."""
    if not _MATPLOTLIB_AVAILABLE:
        logging.warning("matplotlib not installed -- skipping FFT comparison for '%s'.", scene_name)
        return None
    try:
        h, w = hr_bgr.shape[:2]
        lr_up = cv2.resize(lr_bgr, (w, h), interpolation=cv2.INTER_CUBIC) if lr_bgr.shape[:2] != (h, w) else lr_bgr

        def spectrum(img_bgr):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            f = np.fft.fftshift(np.fft.fft2(gray))
            mag = np.log1p(np.abs(f))
            return _radial_profile(mag)

        lr_prof, sr_prof, hr_prof = spectrum(lr_up), spectrum(sr_bgr), spectrum(hr_bgr)
        fig, ax = plt.subplots(figsize=(5, 4), dpi=110)
        ax.plot(hr_prof, label="HR", color="black", linewidth=1.5)
        ax.plot(sr_prof, label="SR", color="tab:blue", linewidth=1.2)
        ax.plot(lr_prof, label="LR (bicubic)", color="tab:red", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Spatial frequency (radius, px)")
        ax.set_ylabel("Log power")
        ax.set_title(f"Radial frequency spectrum — {scene_name}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        preview_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{scene_name}_{stage}.jpg"
        path = preview_dir / filename
        fig.savefig(path, format="jpg")
        plt.close(fig)
        emit_preview_marker(filename, stage, scene_name)
        return path
    except Exception as exc:
        logging.warning("FFT comparison failed for '%s': %s", scene_name, exc)
        return None


def save_error_and_residual_maps(
    lr_bgr: np.ndarray, sr_bgr: np.ndarray, hr_bgr: np.ndarray,
    preview_dir: Path, scene_name: str,
    error_stage: str = "errormap", residual_stage: str = "residual",
) -> Tuple[Optional[Path], Optional[Path]]:
    """Error maps (|SR-HR| vs |LR-HR|) and RdBu-diverging residual maps (SR-HR, LR-HR).
    Returns (None, None) (with a logged warning) if matplotlib is unavailable."""
    if not _MATPLOTLIB_AVAILABLE:
        logging.warning("matplotlib not installed -- skipping error/residual maps for '%s'.", scene_name)
        return None, None
    err_path = res_path = None
    try:
        h, w = hr_bgr.shape[:2]
        lr_up = (cv2.resize(lr_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
                  if lr_bgr.shape[:2] != (h, w) else lr_bgr).astype(np.float32)
        sr_f = sr_bgr.astype(np.float32)
        hr_f = hr_bgr.astype(np.float32)

        sr_err = np.mean(np.abs(sr_f - hr_f), axis=2)
        lr_err = np.mean(np.abs(lr_up - hr_f), axis=2)
        vmax = float(max(sr_err.max(), lr_err.max(), 1e-3))

        preview_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=110)
        axes[0].imshow(lr_err, cmap="inferno", vmin=0, vmax=vmax)
        axes[0].set_title("|LR - HR|"); axes[0].axis("off")
        im1 = axes[1].imshow(sr_err, cmap="inferno", vmin=0, vmax=vmax)
        axes[1].set_title("|SR - HR|"); axes[1].axis("off")
        fig.colorbar(im1, ax=axes, shrink=0.8, label="Abs error")
        err_filename = f"{scene_name}_{error_stage}.jpg"
        err_path = preview_dir / err_filename
        fig.savefig(err_path, format="jpg")
        plt.close(fig)
        emit_preview_marker(err_filename, error_stage, scene_name)

        sr_res = np.mean(sr_f - hr_f, axis=2)
        lr_res = np.mean(lr_up - hr_f, axis=2)
        rmax = float(max(abs(sr_res).max(), abs(lr_res).max(), 1e-3))

        fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4), dpi=110)
        axes2[0].imshow(lr_res, cmap="RdBu", vmin=-rmax, vmax=rmax)
        axes2[0].set_title("Residual: LR - HR"); axes2[0].axis("off")
        im3 = axes2[1].imshow(sr_res, cmap="RdBu", vmin=-rmax, vmax=rmax)
        axes2[1].set_title("Residual: SR - HR"); axes2[1].axis("off")
        fig2.colorbar(im3, ax=axes2, shrink=0.8, label="Signed residual")
        res_filename = f"{scene_name}_{residual_stage}.jpg"
        res_path = preview_dir / res_filename
        fig2.savefig(res_path, format="jpg")
        plt.close(fig2)
        emit_preview_marker(res_filename, residual_stage, scene_name)
    except Exception as exc:
        logging.warning("Error/residual maps failed for '%s': %s", scene_name, exc)
    return err_path, res_path


def generate_sample_report(
    lr_bgr: np.ndarray, sr_bgr: np.ndarray, hr_bgr: np.ndarray,
    preview_dir: Path, scene_name: str,
) -> None:
    """Convenience wrapper: grid + FFT + error/residual maps for one sample.
    Never raises -- every step is independently guarded."""
    save_visual_grid(lr_bgr, sr_bgr, hr_bgr, preview_dir, scene_name)
    save_fft_comparison(lr_bgr, sr_bgr, hr_bgr, preview_dir, scene_name)
    save_error_and_residual_maps(lr_bgr, sr_bgr, hr_bgr, preview_dir, scene_name)
