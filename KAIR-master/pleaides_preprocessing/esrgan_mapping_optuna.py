"""
satellite_degradation_optimizer.py

Optuna-based optimization of satellite-specific ESRGAN degradation parameters.
Learns the HR→LR mapping from paired Pleiades Neo (HR) / Pleiades (LR) imagery,
saves the result to JSON, and exposes a drop-in degrade_satellite() function
that uses those learned parameters.

Satellite-specific adjustments vs. vanilla Real-ESRGAN
-------------------------------------------------------
  DISABLED : JPEG compression  – satellites use lossless / raw encoding
  DISABLED : speckle noise     – SAR artefact; irrelevant for optical sensors
  DISABLED : ISP model         – no consumer ISP pipeline in satellite sensors
  RETAINED : Gaussian blur     – diffraction PSF + atmospheric turbulence
  RETAINED : resize            – sensor pixel integration / Nyquist aliasing
  RETAINED : Gaussian noise    – electronic read-noise + dark current
  RETAINED : Poisson noise     – dominant photon shot-noise in optical sensors
  RETAINED : resize-back       – inter-stage aliasing / scale-mismatch artifacts

Metrics optimized (all defined in your utils.py)
-------------------------------------------------
  PSNR, SSIM, SAM, UIQI, FSIM, RMSE, IT-SSIM, SRER

Usage
-----
  Edit CONFIG below, then:  python satellite_degradation_optimizer.py

  Use learned params in training:
    from satellite_degradation_optimizer import degrade_satellite, load_config
    params = load_config("best_degradation.json")
    lr, hr = degrade_satellite(hr_img, **params)
"""

import os
import json
import random
import logging
import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import optuna


# ===========================================================================
# CONFIG  –  edit this block before running
# ===========================================================================
CONFIG = {
    # ── Paths ───────────────────────────────────────────────────────────────
    "hr_dir":     "Lahore_3/hr",   # HR images (e.g. 0.3 m Pleiades Neo)
    "lr_dir":     "Lahore_3/lr",        # Co-registered LR images (e.g. 0.5 m Pleiades)
    "out_config": "best_degradation.json",    # Where to write the learned params

    # ── Sensor / scale ──────────────────────────────────────────────────────
    "sf": 2,                    # Downscale factor between HR and LR sensors

    # ── Patch sampling ──────────────────────────────────────────────────────
    "patch_size": 256,          # Spatial size of evaluation patches (pixels)
    "n_patches":  1000,          # Total patches loaded from all image pairs

    # ── Optuna study ────────────────────────────────────────────────────────
    "n_trials":      200,       # Total Optuna trials
    "n_eval_samples": 100,       # Patch pairs evaluated per trial (speed vs accuracy)
    "study_name":  "satellite_deg",
    "seed":        42,
}

from utils.utils_image import (          # noqa – your existing utils.py
    imread_uint,
    uint2single,
    single2uint,
    calculate_psnr,
    calculate_ssim,
    calculate_sam,
    calculate_uiqi,
    calculate_fsim,
    calculate_rmse,
    calculate_it_ssim,
    calculate_srer,
)
from preprocessing_pipeline.degradation_utils import (   # noqa – your existing degradations.py
    add_blur,
    add_resize,
    add_Gaussian_noise,
    add_Poisson_noise,
)

# ---------------------------------------------------------------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 1.  Default search bounds  (satellite-physics-informed)
# ===========================================================================

SATELLITE_PARAM_BOUNDS: Dict[str, Tuple] = {
    # Stage 1 ----------------------------------------------------------------
    "blur_prob_1":           ("float",  0.7,  1.0),   # PSF always present
    "resize_prob_1":         ("float",  0.8,  1.0),
    "gaussian_noise_prob_1": ("float",  0.1,  0.5),   # read-noise
    "poisson_noise_prob_1":  ("float",  0.3,  0.8),   # shot-noise dominant
    "noise_level1_s1":       ("int",    1,    5),      # DN floor
    "noise_level2_s1":       ("int",    5,   20),      # DN ceiling
    # Stage 2 ----------------------------------------------------------------
    "blur_prob_2":           ("float",  0.0,  0.4),   # optional second pass
    "resize_prob_2":         ("float",  0.0,  0.5),
    "gaussian_noise_prob_2": ("float",  0.0,  0.3),
    "poisson_noise_prob_2":  ("float",  0.0,  0.4),
    "noise_level1_s2":       ("int",    1,    3),
    "noise_level2_s2":       ("int",    3,   10),
    # Inter-stage ------------------------------------------------------------
    "resize_back_prob":      ("float",  0.0,  0.4),   # Nyquist aliasing
}

# Metric weights for the composite objective (must sum to 1.0)
METRIC_WEIGHTS: Dict[str, float] = {
    "psnr":    0.20,
    "ssim":    0.20,
    "sam":     0.15,
    "uiqi":    0.10,
    "fsim":    0.15,
    "rmse":    0.10,
    "it_ssim": 0.05,
    "srer":    0.05,
}

# Normalisation ranges  (metric → (min_val, max_val, higher_is_better))
# Values outside these ranges are clamped to [0, 1] after normalisation.
METRIC_NORM: Dict[str, Tuple] = {
    "psnr":    (20.0, 50.0,  True),
    "ssim":    (0.0,  1.0,   True),
    "sam":     (0.0,  30.0,  False),   # degrees; lower = better
    "uiqi":    (-1.0, 1.0,   True),
    "fsim":    (0.0,  1.0,   True),
    "rmse":    (0.0,  50.0,  False),   # lower = better
    "it_ssim": (0.0,  1.0,   True),
    "srer":    (0.0,  40.0,  True),
}


# ===========================================================================
# 2.  Data loading
# ===========================================================================

def load_paired_patches(
    hr_dir: str,
    lr_dir: str,
    sf: int,
    patch_size: int = 256,
    n_patches: int = 100,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load matched (HR, LR) patch pairs for metric evaluation.

    Both directories must contain identically named files.  The LR image is
    upsampled to HR spatial size for pixel-aligned comparison.  Images are
    assumed to be pre-normalised (no histogram matching applied).

    Parameters
    ----------
    hr_dir : str
        Directory containing HR images (e.g. Pleiades Neo, 0.3 m).
    lr_dir : str
        Directory containing co-registered LR images (e.g. Pleiades, 0.5 m).
    sf : int
        Nominal downscale factor (used only for logging).
    patch_size : int
        Spatial size of square patches extracted for evaluation.
    n_patches : int
        Total number of (HR, LR) patches to sample across all image pairs.
    seed : int
        Random seed for reproducible patch sampling.

    Returns
    -------
    List of (hr_patch_uint8, lr_patch_uint8) tuples, both HxWxC uint8 [0,255].
    """
    rng = random.Random(seed)
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    hr_files = sorted(f for f in os.listdir(hr_dir)
                      if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp')))
    lr_files = sorted(f for f in os.listdir(lr_dir)
                      if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp')))

    common = sorted(set(hr_files) & set(lr_files))
    if not common:
        raise FileNotFoundError(
            f"No common filenames found between:\n  HR: {hr_dir}\n  LR: {lr_dir}"
        )

    per_image = max(1, n_patches // len(common))
    logger.info(f"Found {len(common)} image pairs; sampling {per_image} patches each.")

    for fname in common:
        hr_img = imread_uint(os.path.join(hr_dir, fname), n_channels=3)  # uint8 HWC RGB
        lr_img = imread_uint(os.path.join(lr_dir, fname), n_channels=3)

        # Upsample LR to HR spatial size for pixel-level comparison
        lr_up = cv2.resize(
            lr_img,
            (hr_img.shape[1], hr_img.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        h, w = hr_img.shape[:2]
        if h < patch_size or w < patch_size:
            logger.warning(f"  {fname}: image smaller than patch_size; skipping.")
            continue

        for _ in range(per_image):
            y = rng.randint(0, h - patch_size)
            x = rng.randint(0, w - patch_size)
            hr_patch = hr_img[y:y + patch_size, x:x + patch_size]
            lr_patch = lr_up[y:y + patch_size, x:x + patch_size]
            pairs.append((hr_patch, lr_patch))

    logger.info(f"Loaded {len(pairs)} (HR, LR) patch pairs.")
    return pairs


# ===========================================================================
# 3.  Metric computation
# ===========================================================================

def _safe_metric(fn, img1, img2, **kwargs):
    """Call a metric function; return NaN on failure (e.g. FSIM with tiny patch)."""
    try:
        return float(fn(img1, img2, **kwargs))
    except Exception as e:
        logger.debug(f"Metric {fn.__name__} failed: {e}")
        return float("nan")


def compute_all_metrics(img1_u8: np.ndarray, img2_u8: np.ndarray) -> Dict[str, float]:
    """
    Compute all 8 image quality metrics between two uint8 [0,255] images.

    Parameters
    ----------
    img1_u8 : np.ndarray
        Reference image, HxWxC uint8.
    img2_u8 : np.ndarray
        Comparison image, HxWxC uint8.

    Returns
    -------
    Dict with keys: psnr, ssim, sam, uiqi, fsim, rmse, it_ssim, srer.
    """
    return {
        "psnr":    _safe_metric(calculate_psnr,    img1_u8, img2_u8),
        "ssim":    _safe_metric(calculate_ssim,    img1_u8, img2_u8),
        "sam":     _safe_metric(calculate_sam,     img1_u8, img2_u8),
        "uiqi":    _safe_metric(calculate_uiqi,    img1_u8, img2_u8),
        "fsim":    _safe_metric(calculate_fsim,    img1_u8, img2_u8),
        "rmse":    _safe_metric(calculate_rmse,    img1_u8, img2_u8),
        "it_ssim": _safe_metric(calculate_it_ssim, img1_u8, img2_u8),
        "srer":    _safe_metric(calculate_srer,    img1_u8, img2_u8),
    }


def _normalise_metric(name: str, value: float) -> float:
    """Map a raw metric value to [0, 1] where 1 = best."""
    lo, hi, higher_better = METRIC_NORM[name]
    norm = (value - lo) / (hi - lo + 1e-8)
    norm = float(np.clip(norm, 0.0, 1.0))
    return norm if higher_better else (1.0 - norm)


def composite_score(metrics: Dict[str, float]) -> float:
    """
    Weighted sum of normalised metrics; returns a value in [0, 1].
    Higher = better degradation match.  Optuna minimises the negative.
    """
    total, weight_sum = 0.0, 0.0
    for name, weight in METRIC_WEIGHTS.items():
        val = metrics.get(name, float("nan"))
        if np.isnan(val):
            continue
        total += weight * _normalise_metric(name, val)
        weight_sum += weight
    return total / (weight_sum + 1e-8)


# ===========================================================================
# 4.  Satellite degradation  (JPEG / speckle / ISP disabled)
# ===========================================================================

def degrade_satellite(
    img: np.ndarray,
    sf: int = 2,
    # ── Stage 1 ──────────────────────────────────────────────────────────
    blur_prob_1: float = 1.0,
    resize_prob_1: float = 1.0,
    gaussian_noise_prob_1: float = 0.3,
    poisson_noise_prob_1: float = 0.6,
    noise_level1_s1: int = 1,
    noise_level2_s1: int = 12,
    # ── Stage 2 ──────────────────────────────────────────────────────────
    blur_prob_2: float = 0.2,
    resize_prob_2: float = 0.3,
    gaussian_noise_prob_2: float = 0.1,
    poisson_noise_prob_2: float = 0.2,
    noise_level1_s2: int = 1,
    noise_level2_s2: int = 6,
    # ── Inter-stage ───────────────────────────────────────────────────────
    resize_back_prob: float = 0.25,
    **kwargs,    # absorbs any extra keys loaded from JSON (e.g. metadata)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Satellite-physics-aware dual-stage degradation pipeline for ESRGAN training.

    Derived from Real-ESRGAN (Wang et al., ICCVW 2021) with the following
    modifications for optical satellite imagery:

      - JPEG compression  → removed (lossless on-board encoding)
      - Speckle noise     → removed (SAR artefact only)
      - ISP model         → removed (no consumer ISP in satellite sensors)
      - Poisson noise     → elevated probability (dominant shot-noise source)
      - Gaussian noise    → retained at lower probability (read + dark current)

    The noise gate in each stage is mutually exclusive:
        p(Gaussian) | p(Poisson) | p(neither)
    where p(neither) = 1 - gaussian_prob - poisson_prob.

    Parameters
    ----------
    img : np.ndarray
        HR input, HxWxC float32 in [0, 1].
    sf : int
        Integer downscale factor.
    blur_prob_1 / blur_prob_2 : float
        Probability of applying an isotropic Gaussian blur in each stage.
    resize_prob_1 / resize_prob_2 : float
        Probability of a random intermediate resize in each stage.
    gaussian_noise_prob_1 / _2 : float
        Probability of additive Gaussian noise (read noise model).
    poisson_noise_prob_1 / _2 : float
        Probability of Poisson noise (shot noise model).  Applied only if
        Gaussian gate does not fire.
    noise_level1_s1 / _s2 : int
        Lower bound of noise sigma (in 0–255 scale) for each stage.
    noise_level2_s1 / _s2 : int
        Upper bound of noise sigma for each stage.
    resize_back_prob : float
        Probability of an up→down resize pass between stages to introduce
        Nyquist-aliasing artefacts.
    **kwargs
        Ignored; allows direct unpacking of a loaded JSON config dict.

    Returns
    -------
    img_lq : np.ndarray
        Degraded LR image, (H/sf)×(W/sf)×C, float32 [0, 1].
    img_hq : np.ndarray
        Mod-cropped HR reference aligned to img_lq, float32 [0, 1].
    """
    # ── mod-crop ──────────────────────────────────────────────────────────
    h0, w0 = img.shape[:2]
    img = img.copy()[:h0 - h0 % sf, :w0 - w0 % sf, ...]
    hq = img.copy()
    target_h = hq.shape[0] // sf
    target_w = hq.shape[1] // sf

    # =====================================================================
    # Stage 1 : blur → resize → noise (Gaussian | Poisson)
    # =====================================================================
    if random.random() < blur_prob_1:
        img = add_blur(img, sf=sf)

    if random.random() < resize_prob_1:
        img = add_resize(img, sf=sf)

    rnd = random.random()
    if rnd < gaussian_noise_prob_1:
        img = add_Gaussian_noise(
            img,
            noise_level1=noise_level1_s1,
            noise_level2=noise_level2_s1,
        )
    elif rnd < gaussian_noise_prob_1 + poisson_noise_prob_1:
        img = add_Poisson_noise(img)

    img = np.clip(img, 0.0, 1.0)

    # ── optional inter-stage resize-back (Nyquist aliasing) ───────────────
    if random.random() < resize_back_prob:
        img = cv2.resize(
            img, (hq.shape[1], hq.shape[0]),
            interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]),
        )
        img = cv2.resize(
            img, (target_w, target_h),
            interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]),
        )
        img = np.clip(img, 0.0, 1.0)

    # =====================================================================
    # Stage 2 : blur → resize → noise (Gaussian | Poisson)
    # =====================================================================
    if random.random() < blur_prob_2:
        img = add_blur(img, sf=sf)

    if random.random() < resize_prob_2:
        img = add_resize(img, sf=sf)

    rnd = random.random()
    if rnd < gaussian_noise_prob_2:
        img = add_Gaussian_noise(
            img,
            noise_level1=noise_level1_s2,
            noise_level2=noise_level2_s2,
        )
    elif rnd < gaussian_noise_prob_2 + poisson_noise_prob_2:
        img = add_Poisson_noise(img)

    img = np.clip(img, 0.0, 1.0)

    # ── final downscale to target LQ resolution ───────────────────────────
    img = cv2.resize(
        img, (target_w, target_h),
        interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]),
    )
    img_lq = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img_lq, hq


# ===========================================================================
# 5.  Optuna objective
# ===========================================================================

def _build_params_from_trial(trial: optuna.Trial) -> Dict:
    """Suggest parameter values within satellite-physics-informed bounds."""
    params = {}
    for name, (kind, lo, hi) in SATELLITE_PARAM_BOUNDS.items():
        if kind == "float":
            params[name] = trial.suggest_float(name, lo, hi)
        elif kind == "int":
            params[name] = trial.suggest_int(name, lo, hi)
    return params


def _evaluate_params(
    params: Dict,
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    sf: int,
    n_eval_samples: int,
    seed: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Apply degrade_satellite to a subset of HR patches and compare with real LR.

    Returns (composite_score, mean_metrics_dict).
    """
    rng = random.Random(seed)
    subset = rng.sample(pairs, min(n_eval_samples, len(pairs)))

    all_metrics: Dict[str, List[float]] = {k: [] for k in METRIC_WEIGHTS}

    for hr_u8, lr_u8 in subset:
        hr_f32 = hr_u8.astype(np.float32) / 255.0
        synth_lr_f32, _ = degrade_satellite(hr_f32, sf=sf, **params)

        # Resize synthetic LR back to patch_size for pixel-aligned comparison
        synth_lr_u8 = single2uint(
            cv2.resize(synth_lr_f32, (hr_u8.shape[1], hr_u8.shape[0]),
                       interpolation=cv2.INTER_CUBIC)
        )

        m = compute_all_metrics(lr_u8, synth_lr_u8)
        for k, v in m.items():
            if not np.isnan(v):
                all_metrics[k].append(v)

    mean_metrics = {k: float(np.mean(v)) if v else float("nan")
                    for k, v in all_metrics.items()}
    score = composite_score(mean_metrics)
    return score, mean_metrics


def make_objective(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    sf: int,
    n_eval_samples: int = 30,
):
    """Factory that returns an Optuna objective closure."""

    def objective(trial: optuna.Trial) -> float:
        params = _build_params_from_trial(trial)

        # Guard: poisson + gaussian probs must not exceed 1.0
        if (params["gaussian_noise_prob_1"] + params["poisson_noise_prob_1"]) > 1.0:
            raise optuna.TrialPruned()
        if (params["gaussian_noise_prob_2"] + params["poisson_noise_prob_2"]) > 1.0:
            raise optuna.TrialPruned()

        score, metrics = _evaluate_params(
            params, pairs, sf, n_eval_samples, seed=trial.number
        )

        # Store per-trial metrics as user attributes for later inspection
        for k, v in metrics.items():
            trial.set_user_attr(k, round(v, 6) if not np.isnan(v) else None)
        trial.set_user_attr("composite_score", round(score, 6))

        return -score  # Optuna minimises; we want to maximise composite_score

    return objective


# ===========================================================================
# 6.  JSON save / load
# ===========================================================================

def save_config(params: Dict, path: str, extra_meta: Optional[Dict] = None) -> None:
    """
    Persist learned degradation parameters to a JSON file.

    Parameters
    ----------
    params : Dict
        Optimised parameter dictionary (output of optimise_degradation_params).
    path : str
        Destination file path.
    extra_meta : Dict, optional
        Additional metadata to embed (e.g. n_trials, best_score, timestamp).
    """
    payload = {"degradation_params": params}
    if extra_meta:
        payload["meta"] = extra_meta
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Config saved → {path}")


def load_config(path: str) -> Dict:
    """
    Load degradation parameters from a JSON file.

    Returns a flat dict ready for unpacking into degrade_satellite():
        params = load_config("best_degradation.json")
        lr, hr = degrade_satellite(hr_img, **params)
    """
    with open(path) as f:
        payload = json.load(f)
    params = payload.get("degradation_params", payload)  # backward-compat
    # Cast int params back from JSON (JSON has no int/float distinction)
    int_keys = {"noise_level1_s1", "noise_level2_s1", "noise_level1_s2", "noise_level2_s2", "sf"}
    for k in int_keys:
        if k in params:
            params[k] = int(params[k])
    logger.info(f"Config loaded ← {path}")
    return params


# ===========================================================================
# 7.  Main optimisation entry-point
# ===========================================================================

def optimise_degradation_params(
    hr_dir: str,
    lr_dir: str,
    sf: int = 2,
    n_trials: int = 200,
    n_eval_samples: int = 30,
    patch_size: int = 256,
    n_patches: int = 120,
    out_config: str = "best_degradation.json",
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None,
    study_name: str = "satellite_deg",
    seed: int = 42,
) -> Dict:
    """
    Run Optuna study to find the best degradation parameters.

    Parameters
    ----------
    hr_dir : str
        Path to HR image directory (e.g. Pleiades Neo).
    lr_dir : str
        Path to matching LR image directory (e.g. Pleiades).
    sf : int
        Downscale factor between the two sensors.
    n_trials : int
        Number of Optuna trials.
    n_eval_samples : int
        Number of patch pairs evaluated per trial (trade-off: speed vs accuracy).
    patch_size : int
        Spatial size of evaluation patches.
    n_patches : int
        Total patches loaded from disk.
    out_config : str
        Output JSON path.
    sampler : optuna.samplers.BaseSampler, optional
        Defaults to TPESampler (Bayesian).
    pruner : optuna.pruners.BasePruner, optional
        Defaults to MedianPruner.
    study_name : str
        Optuna study name.
    seed : int
        Global random seed.

    Returns
    -------
    best_params : Dict
        Flat dict of optimised parameters (also written to out_config).
    """
    # ── load data ─────────────────────────────────────────────────────────
    pairs = load_paired_patches(
        hr_dir, lr_dir, sf,
        patch_size=patch_size,
        n_patches=n_patches,
        seed=seed,
    )

    # ── create study ──────────────────────────────────────────────────────
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    if pruner is None:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(pairs, sf, n_eval_samples=n_eval_samples)

    logger.info(f"Starting Optuna study: {n_trials} trials, {n_eval_samples} eval samples/trial")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ── collect results ───────────────────────────────────────────────────
    best_trial = study.best_trial
    best_params = best_trial.params
    best_params["sf"] = sf  # embed sf for convenience

    best_score = best_trial.user_attrs.get("composite_score", -best_trial.value)
    best_metrics = {k: best_trial.user_attrs.get(k) for k in METRIC_WEIGHTS}

    logger.info("─" * 60)
    logger.info(f"Best composite score : {best_score:.4f}")
    logger.info("Per-metric values at best trial:")
    for k, v in best_metrics.items():
        logger.info(f"  {k:10s}: {v}")
    logger.info("Best parameters:")
    for k, v in best_params.items():
        logger.info(f"  {k:30s}: {v}")
    logger.info("─" * 60)

    # ── save ─────────────────────────────────────────────────────────────
    save_config(
        params=best_params,
        path=out_config,
        extra_meta={
            "composite_score": best_score,
            "metrics":         best_metrics,
            "n_trials":        n_trials,
            "n_pairs":         len(pairs),
            "hr_dir":          hr_dir,
            "lr_dir":          lr_dir,
            "sf":              sf,
            "timestamp":       datetime.datetime.now().isoformat(),
            "metric_weights":  METRIC_WEIGHTS,
        },
    )
    return best_params


# ===========================================================================
# 8.  Entry-point  (reads CONFIG defined at the top of this file)
# ===========================================================================

if __name__ == "__main__":
    optimise_degradation_params(
        hr_dir        = CONFIG["hr_dir"],
        lr_dir        = CONFIG["lr_dir"],
        sf            = CONFIG["sf"],
        n_trials      = CONFIG["n_trials"],
        n_eval_samples= CONFIG["n_eval_samples"],
        patch_size    = CONFIG["patch_size"],
        n_patches     = CONFIG["n_patches"],
        out_config    = CONFIG["out_config"],
        study_name    = CONFIG["study_name"],
        seed          = CONFIG["seed"],
    )