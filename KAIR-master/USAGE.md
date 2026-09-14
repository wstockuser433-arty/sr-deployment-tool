# Usage Guide

This file covers all usage instructions for SwinIR training, testing, and remote-sensing imagery preprocessing in this repository.

---

## Table of Contents

1. [Setup](#0-setup)
2. [Preprocessing](#1-preprocessing)
3. [Training](#2-training)
4. [Testing](#3-testing)
5. [Visual Comparison](#4-visual-comparison)
6. [GUI](#5-gui)

---

## Setup

### Step 1 — Create a virtual environment

It is strongly recommended to use a Python virtual environment to isolate dependencies.

```bash
# Create the environment (run once)
python -m venv .venv

# Activate — Windows (Command Prompt)
.venv\Scripts\activate.bat

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate — Linux / macOS
source .venv/bin/activate
```

The prompt will change to show `(.venv)` when the environment is active. All subsequent `pip install` commands should be run inside this environment.

---

### Step 2 — Check your CUDA version (skip if CPU-only)

PyTorch must be installed with a build that matches the CUDA version installed on your machine.

**On Windows (Command Prompt or PowerShell):**
```
nvidia-smi
```

Look for the `CUDA Version` value in the top-right of the output, e.g. `CUDA Version: 12.1`. This is the maximum CUDA version your driver supports.

---

### Step 3 — Install PyTorch

Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select your OS, package manager, and CUDA version to get the correct install command.

Common example:

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify the install:
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())      # should be True if GPU is set up correctly
print(torch.cuda.get_device_name(0))  # should print your GPU name
```

---

### Step 4 — Install remaining dependencies

```bash
pip install -r requirement.txt
```

> **Note:** `torch` and `torchvision` lines in `requirement.txt` will be skipped or reinstalled without CUDA support if run without the `--index-url` flag above. Always install PyTorch first using the command from Step 2 before running this.

---

## 1. Preprocessing

### 1.1 `pleaides_preprocessing/pipeline3.py` — Main Pipeline

Preprocesses paired HR/LR Pleiades satellite GeoTIFF/JP2 images into matched 8-bit RGB patch pairs suitable for SR training. Handles spatial coregistration, radiometric normalisation, and patch extraction with quality filtering.

**Run:**
```bash
python pleaides_preprocessing/pipeline3.py
```

**Configuration:** Edit the `CONFIG` dict at the top of `pipeline3.py`, or place a `config.json` file next to the script to override keys.

**JSON Configuration:**

| Key | Description |
|-----|-------------|
| `HR_IMAGE_PATH` | Path to the HR GeoTIFF/JP2 |
| `LR_IMAGE_PATH` | Path to the LR GeoTIFF/JP2 |
| `OUTPUT_DIR` | Output directory; patches saved to `<OUTPUT_DIR>/hr/` and `<OUTPUT_DIR>/lr/` |
| `HR_RGB_BANDS`, `LR_RGB_BANDS` | Band indices to use as RGB (e.g. `[2, 1, 0]`) |
| `SCALE_FACTOR` | Upscaling factor (usually `2`) |
| `HR_PATCH_SIZE` | Dimensions of HR patches (e.g., `256`) |
| `STRIDE` | Stride for patch extraction (controls overlap) |
| `COREG_A_ENABLED` | Enable Stage A coregistration (ORB + RANSAC homography) |
| `COREG_B_ENABLED` | Enable Stage B coregistration (phase cross-correlation, sub-pixel) |
| `COREG_C_ENABLED` | Enable Stage C coregistration (ECC, per-patch local refinement) |
| `COREG_A_MAX_FEATURES` | Max ORB features for Stage A |
| `COREG_A_RANSAC_THRESH` | RANSAC inlier threshold for Stage A |
| `MIN_ECC_SCORE` | Minimum ECC score to keep a patch |
| `MIN_SSIM` | Minimum SSIM to keep a patch |
| `MIN_VARIANCE` | Minimum variance to reject blank patches |
| `MAX_NODATA_FRACTION` | Maximum fraction of nodata pixels allowed |

**Outputs:** Patch PNGs under `<OUTPUT_DIR>/hr/` and `<OUTPUT_DIR>/lr/`, plus a timestamped log file.

> **Note:** Configuration must be updated in accordance to the imagery files to be preprocessed.

---

### 1.2 `pleaides_preprocessing/verify_coregistration.py` — Coregistration Verification

Diagnostic suite to verify that HR and LR patches produced by the pipeline are properly aligned. Produces GIFs, diff heatmaps, checkerboard overlays, SSIM heatmaps, and a summary report.

**Run:**
```bash
# Use OUTPUT_DIR from pipeline config
python pleaides_preprocessing/verify_coregistration.py

# Override output dir and sample size
python pleaides_preprocessing/verify_coregistration.py --output Lahore_4 --n 50
```

**JSON Configuration:**

| Key | Description |
|-----|-------------|
| `OUTPUT_DIR` | Output directory from `pipeline3.py` to verify |
| `N_SAMPLES` | Number of patches to analyze (null = all) |
| `SAMPLE_SEED` | Random seed for reproducible patch selection |
| `BLINKER_FPS` | Frames per second for the blinker GIF |
| `CHECKER_GRID` | NxN checkerboard grid divisions |
| `DIFF_CMAP` | Matplotlib colormap for difference maps |
| `SSIM_CMAP` | Matplotlib colormap for SSIM heatmaps |
| `DISPLAY_SIZE` | Pixel size for saved diagnostic images (upscale LR for clarity) |

**Outputs:** Under `<OUTPUT_DIR>/verification/`:
- `blinker/` — GIFs toggling HR_down ↔ LR (spot sub-pixel shifts)
- `diff/` — Per-patch absolute difference heatmaps (MAE, RMSE, p95, max)
- `checkerboard/` — Interleaved tiles for edge continuity
- `ssim/` — SSIM heatmaps and quadrant metrics
- `summary/` — Combined grids, GIFs, and `report.txt`

**Interpretation:**
- Mean SSIM ≳ 0.72 and small MAE → good alignment.
- Blinker: static edges → excellent; breathing/jitter → residual shift.
- If global shift persists: re-tune Stage A (`COREG_A_MAX_FEATURES`, `COREG_A_RANSAC_THRESH`) or Stage C ECC settings.

---

### 1.3 `pleaides_preprocessing/esrgan_mapping_optuna.py` — Learn Satellite Degradation

Uses Optuna to learn a satellite-aware degradation model (blur, resize, noise) that matches real LR images. Writes learned parameters to `best_degradation.json`.

**Run:**
```bash
python pleaides_preprocessing/esrgan_mapping_optuna.py
```

**JSON Configuration:**

| Key | Description |
|-----|-------------|
| `hr_dir` | Path to source HR patches |
| `lr_dir` | Path to co-registered real LR patches |
| `out_config` | Destination path for learned parameters (default: `best_degradation.json`) |
| `sf` | Scale factor (downsampling ratio) |
| `patch_size` | Spatial size of patches for evaluation |
| `n_patches` | Total quantity of patches to load for the study |
| `n_trials` | Number of Optuna trials to run |
| `n_eval_samples` | Number of patches to evaluate per trial |
| `seed` | Random seed for reproducibility |

Configure paired HR/LR folder paths at the top of the file. The output `best_degradation.json` contains `degradation_params` and `meta`.

### 1.4 `pleaides_preprocessing/apply_esrgan_degradation.py` — Apply Learned Degradation

Applies `best_degradation.json` parameters to a directory of HR images to create synthetic LR images for ESRGAN/Real-ESRGAN-style training.

**Run:**
```bash
python pleaides_preprocessing/apply_esrgan_degradation.py
```

**Primary `CONFIG` keys:**

| Key | Description |
|-----|-------------|
| `hr_dir` | Source HR images |
| `lr_dir` | Destination for synthetic LR |
| `params_path` | JSON from Optuna study (default: `best_degradation.json`) |
| `sf` | Scale factor |
| `seed` | Random seed |
| `overwrite` | Whether to overwrite existing LR files |
| `recursive` | Search `hr_dir` recursively |
| `save_ext` | Output extension (e.g. `".png"`) |

**Programmatic use:**
```python
from pleaides_preprocessing.esrgan_mapping_optuna import load_config, degrade_satellite
params = load_config('pleaides_preprocessing/best_degradation.json')
lr, _ = degrade_satellite(hr_img, **params)  # hr_img: float32 H×W×3 in [0,1]
```

---

### Preprocessing Checklist

1. Run `pipeline3.py` — confirm `<OUTPUT_DIR>/hr` and `<OUTPUT_DIR>/lr` contain patch PNGs.
2. Run `verify_coregistration.py --output <OUTPUT_DIR> --n 50` — inspect `report.txt` and summary grids.
3. Confirm mean SSIM and MAE are within acceptable ranges.
4. If alignment is poor, tune Stage A/C coregistration parameters and re-run.

---

## 2. Training

### 2.1 `main_train_swinir.py` — PSNR Training

Trains a SwinIR model using a pixel-level loss (L1 by default). Suitable for classical SR and lightweight SR. Produces PSNR-optimised weights.

**Run:**
```bash
python main_train_swinir.py
```

To use a different config, can pass the path in the CLI or as an argument to the main function `json_path='options/swinir/train_swinir_sr_classical.json'`:
```bash
python main_train_swinir.py --opt options/swinir/train_swinir_sr_realworld_x4_psnr.json
```

For multi-GPU distributed training:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 main_train_swinir.py --dist True
```

Default config: `options/swinir/train_swinir_sr_classical.json`

---

### 2.2 `main_train_swinir_gan.py` — GAN Training

Trains a SwinIR model using adversarial (GAN) + perceptual losses. Produces perceptually sharper outputs at the cost of lower PSNR. Use after PSNR pre-training.

**Run:**
```bash
python main_train_swinir_gan.py
```

To use a different config:
```bash
python main_train_swinir_gan.py --opt options/swinir/train_swinir_sr_realworld_x4_gan.json
```

Default config: `options/swinir/train_swinir_sr_realworld_x2_gan.json`

---

### 2.3 Available Config Files

All configs are in `options/swinir/`:

| File | Task |
|------|------|
| `train_swinir_sr_classical.json` | Classical SR (x2/x3/x4/x8) |
| `train_swinir_sr_lightweight.json` | Lightweight SR |
| `train_swinir_sr_realworld_x4_psnr.json` | Real-world SR, PSNR loss |
| `train_swinir_sr_realworld_x4_gan.json` | Real-world SR, GAN loss |
| `train_swinir_sr_realworld_x2_gan.json` | Real-world SR x2, GAN loss |

---

### 2.4 Configuration Reference

All settings below are in the JSON config file.

#### Basic Task Settings

```json
"task": "swinir_sr_classical_patch48_x2",
"scale": 2,
"n_channels": 3
```

| Key | Description |
|-----|-------------|
| `task` | Experiment name — determines output directory under `superresolution/` |
| `scale` | Upscaling factor: `2`, `3`, `4`, or `8` |
| `n_channels` | `1` for grayscale, `3` for RGB |

#### GPU Settings

```json
"gpu_ids": [0],
"dist": false
```

- Single GPU: `"gpu_ids": [0]`, `"dist": false`
- Multi-GPU: `"gpu_ids": [0,1,2,3]`, `"dist": true`

#### Dataset Paths

```json
"datasets": {
  "train": {
    "dataroot_H": "trainsets/trainH",
    "dataroot_L": "trainsets/trainL",
    "H_size": 96,
    "dataloader_batch_size": 32,
    "dataloader_num_workers": 16
  },
  "test": {
    "dataroot_H": "testsets/Set5/HR",
    "dataroot_L": "testsets/Set5/LR_bicubic/X2"
  }
}
```

| Key | Description |
|-----|-------------|
| `dataroot_H` | HR training images. Required. |
| `dataroot_L` | LR training images. Omit to generate LR via bicubic downsampling on the fly. |
| `H_size` | HR patch crop size. Must be divisible by `window_size` (usually 8). |
| `dataloader_batch_size` | Batch size. Reduce if OOM. |
| `dataloader_num_workers` | Data loading threads. |

**LR/HR filename matching:** LR and HR filenames must be identical (same stem). The dataloader matches pairs by name.

**Dataset types:**
- `dataset_sr` — Standard SR dataset. Uses bicubic if LR not provided; applies geometric and radiometric augmentations on the fly.
- `dataset_blindsr` — Blind SR dataset using BSRGAN degradation on the fly.

#### Pre-trained Model

```json
"path": {
  "root": "superresolution",
  "pretrained_netG": null
}
```

Set `pretrained_netG` to a `.pth` path for fine-tuning, or `null` to train from scratch.

#### Network Architecture (`netG`)

```json
"netG": {
  "net_type": "swinir",
  "upscale": 2,
  "in_chans": 3,
  "img_size": 48,
  "window_size": 8,
  "depths": [6, 6, 6, 6, 6, 6],
  "embed_dim": 180,
  "num_heads": [6, 6, 6, 6, 6, 6],
  "mlp_ratio": 2,
  "upsampler": "pixelshuffle",
  "resi_connection": "1conv"
}
```

| Key | Description |
|-----|-------------|
| `upscale` | Must match top-level `scale` |
| `in_chans` | Must match `n_channels` |
| `img_size` | LR patch size = `H_size / scale` |
| `window_size` | Swin Transformer window size (default: 8) |
| `depths` | Transformer blocks per stage. Longer = more capacity. |
| `embed_dim` | Feature dimension. Must be divisible by all values in `num_heads`. |
| `num_heads` | Attention heads per stage. Length must match `depths`. |
| `upsampler` | `"pixelshuffle"` (classical) or `"nearest+conv"` (real-world) |
| `resi_connection` | `"1conv"` or `"3conv"` |

**Model size guide:**
- Small/Fast: `embed_dim: 60`, `depths: [6, 6, 6, 6]`
- Medium (default): `embed_dim: 180`, `depths: [6, 6, 6, 6, 6, 6]`
- Large: `embed_dim: 240`, `depths: [6, 6, 6, 6, 6, 6, 6, 6]`

#### Training Hyperparameters

```json
"train": {
  "G_lossfn_type": "l1",
  "G_lossfn_weight": 1.0,
  "G_optimizer_type": "adam",
  "G_optimizer_lr": 2e-4,
  "G_optimizer_wd": 0,
  "G_scheduler_type": "MultiStepLR",
  "G_scheduler_milestones": [250000, 400000, 450000, 475000, 500000],
  "G_scheduler_gamma": 0.5,
  "E_decay": 0.999
}
```

| Key | Description |
|-----|-------------|
| `G_lossfn_type` | `"l1"` (preferred), `"l2"`, `"ssim"`, `"charbonnier"` |
| `G_optimizer_lr` | Initial learning rate |
| `G_scheduler_milestones` | Iterations at which LR is multiplied by `gamma` |
| `G_scheduler_gamma` | LR decay factor at each milestone (e.g. `0.5` halves LR) |
| `E_decay` | EMA decay rate. Set to `0` to disable EMA. |

#### Checkpointing

```json
"checkpoint_test": 5000,
"checkpoint_save": 5000,
"checkpoint_print": 200
```

For quick tests on small datasets, set these lower (e.g. `1000`).

---

### 2.5 Output Structure

```
superresolution/
└── <task_name>/
    ├── models/
    │   ├── 5000_G.pth
    │   └── ...
    ├── images/       ← SR images saved during validation
    ├── log/
    │   └── train.log
    └── options/
        └── train.json
```

---

### 2.6 Common Issues

| Issue | Fix |
|-------|-----|
| CUDA out of memory | Reduce `dataloader_batch_size`, `H_size`, or `embed_dim` |
| No training data found | Check `dataroot_H` path; confirm it contains images |
| Training too slow | Increase `dataloader_num_workers`; use SSD storage; enable distributed training |
| Loss is NaN | Reduce learning rate; check data normalisation; reduce batch size |

---

## 3. Testing

### 3.1 `main_test_swinir_config.py` — Inference + Metrics

Runs a trained SwinIR model on LR images to produce SR outputs, saves them to disk, and reports average metrics (PSNR, SSIM, IT-SSIM, SAM, UIQI, RMSE, FSIM, SRER) against HR ground truth. Config is in-file.

**Run:**
```bash
python main_test_swinir_config.py
```

**`CONFIG` settings (edit at top of file):**

```python
CONFIG = {
    "model_path": "superresolution/<task>/models/<iter>_E.pth",
    "lr_dir": "testsets/my_test/lr",
    "hr_dir": "testsets/my_test/hr",
    "sr_dir": "testsets/my_test/sr",
    "tile": None,        # None to process full image; integer for tiled inference (multiple of window_size)
    "tile_overlap": 32,  # Overlap between tiles in pixels
    "overwrite_sr": True,
    "log_dir": "testsets/my_test",
}
```

**`MODEL_CONFIG` settings (must match the trained checkpoint):**

```python
MODEL_CONFIG = {
    "upscale": 2,
    "in_chans": 3,
    "img_size": 128,
    "window_size": 8,
    "img_range": 1.0,
    "depths": [6, 6, 6, 6, 6, 6],
    "embed_dim": 180,
    "num_heads": [6, 6, 6, 6, 6, 6],
    "mlp_ratio": 2,
    "upsampler": "pixelshuffle",
    "resi_connection": "1conv",
}
```

These must exactly match the architecture used during training.

> **Tip:** Recommended architecture settings can be retrieved from the `train.log` file (or `options/train.json`) located in the training output directory: `super-resolution/taskname/train.log` as mentioned in the [Output Structure](#25-output-structure) section above.

**File matching:** LR and HR images are matched by filename stem. Files with no common stem are skipped. SR outputs are saved as `<name>_SwinIR.png` in `sr_dir`.

**Tile mode:** Set `"tile"` to an integer (multiple of `window_size`) to run tiled inference on large images that don't fit in GPU memory. `tile_overlap` controls overlap to reduce tile boundary artefacts.

**Metrics computed:** PSNR, SSIM, IT-SSIM, SAM, UIQI, RMSE, FSIM, SRER — averaged across all test images and logged to `<log_dir>/<sr_dir_name>.log`.

---

### 3.2 `main_evaluate_swinir_by_class.py` — Class-Wise Evaluation

Evaluates SR results by class for UCMerced-style datasets. HR images are expected flat in `hr_dir`, named `ucmerced_{class}{digits}.png`. Automatically picks the highest-iteration SR file for each image. Produces per-image, per-class, and global average metrics in a log file.

**Run:**
```bash
python main_evaluate_swinir_by_class.py
```

**`CONFIG` settings:**

```python
CONFIG = {
    "hr_dir": "testsets/<dataset>/hr",
    "sr_base_dir": "superresolution/<task>/images",
    "log_file": "superresolution/<task>/classwise_evaluation_log.txt",
    "border": 2,  # Pixels to exclude from border when computing metrics
}
```

**Expected SR directory layout:**
```
sr_base_dir/
└── ucmerced_agricultural06/
    └── ucmerced_agricultural06_175000.png   ← iteration number in filename
```

The script picks the file with the highest iteration number automatically.

**Output:** A structured log file with per-image results, per-class summaries, and a global summary table across all classes.

> **Note:** Class extraction strips the `ucmerced_` prefix and trailing digits from the filename stem. If your dataset uses a different naming scheme, update `extract_ucmerced_class()` in the script.

---

### 3.3 `raw_inference.py` — End-to-end Raw Inference

Performs full end-to-end inference on raw satellite images (or standard images) with integrated pipeline3 coregistration, patching, stitching, and metric evaluation.

**Run:**
```bash
python raw_inference.py --config config.json
```

**JSON Configuration Example:**
```json
{
    "mode": "paired",
    "lr_path": "path/to/lr.tif",
    "hr_path": "path/to/hr.tif",
    "lr_bands": [3, 2, 1],
    "hr_bands": [1, 2, 3],
    "model_path": "superresolution/task/models/best_E.pth",
    "output_dir": "testsets/raw_inference_output",
    "scale_factor": 2,
    "patch_size": 128,
    "overlap": 32,
    "enable_preprocessing": true,
    "nodata_value": 0,
    "saturated_value": 32767,
    "clip_percentiles": [2.0, 98.0],
    "coreg_a_enabled": true,
    "coreg_a_max_features": 8000,
    "coreg_a_match_ratio": 0.75,
    "coreg_a_ransac_thresh": 4.0,
    "coreg_b_enabled": true,
    "coreg_b_upsample_factor": 100,
    "radiometric_enabled": true,
    "radiometric_block_size": 256,
    "radiometric_rmse_threshold": 35.0,
    "radiometric_post_hist_match": true,
    "model_config": {
        "upscale": 2, "in_chans": 3, "img_size": 128, "window_size": 8,
        "img_range": 1.0,
        "depths": [6,6,6,6,6,6], "embed_dim": 180, "num_heads": [6,6,6,6,6,6],
        "mlp_ratio": 2, "upsampler": "pixelshuffle", "resi_connection": "1conv"
    }
}
```

**Configuration Key Reference:**

| Key | Description |
|-----|-------------|
| `mode` | `"paired"` (LR + HR, with metrics) or `"lr_only"` (LR only, no metrics) |
| `lr_path` | Path to the LR image (GeoTIFF, JP2, PNG, etc.) |
| `hr_path` | Path to the HR image (`"paired"` mode only) |
| `lr_bands` | 1-indexed band numbers for LR RGB extraction (e.g. `[3, 2, 1]` for Pleiades 1A). Defaults to `[3,2,1]`. |
| `hr_bands` | 1-indexed band numbers for HR RGB extraction (e.g. `[1, 2, 3]` for Pleiades Neo). Defaults to `[1,2,3]`. |
| `model_path` | Path to the trained `.pth` checkpoint |
| `output_dir` | Directory where all output files are written |
| `scale_factor` | SR upscaling factor — must match the trained model's `upscale` |
| `patch_size` | LR patch size in pixels for tiled inference |
| `overlap` | Overlap between adjacent patches in pixels (reduces seam artefacts) |
| `enable_preprocessing` | If `true`, runs the full pipeline3 alignment stack before SR |
| `nodata_value` | Pixel value treated as nodata in GeoTIFF images |
| `saturated_value` | Pixel value treated as saturated (excluded from percentile estimation) |
| `clip_percentiles` | `[low, high]` percentile bounds for 16-bit to 8-bit scaling |
| `coreg_a_enabled` | Enable Stage A ORB coregistration |
| `coreg_a_max_features` | Max ORB features to detect |
| `coreg_a_match_ratio` | Lowe ratio test threshold |
| `coreg_a_ransac_thresh` | RANSAC inlier threshold (pixels in overview space) |
| `coreg_b_enabled` | Enable Stage B phase correlation |
| `coreg_b_upsample_factor` | Sub-pixel refinement factor |
| `radiometric_enabled` | Enable radiometric regression (LR to HR normalisation) |
| `radiometric_block_size` | Block size for regression fitting |
| `radiometric_rmse_threshold` | Max block RMSE; blocks above are excluded from the fit |
| `radiometric_post_hist_match` | Apply histogram matching after linear regression |
| `model_config` | SwinIR architecture dict — must match the trained checkpoint exactly |

**Operating Modes:**
- `"paired"`: Takes both an LR and HR image. Optionally runs full `pipeline3` stages (ORB, Phase Correlation, Radiometric Normalisation, Histogram Matching) to align the LR image exactly to the HR grid, enforcing an exact integer scale ratio regardless of the real sensor resolution ratio. Runs SwinIR on overlapping patches, stitches them with a Hann-window blend, and computes 8 metrics (PSNR, SSIM, IT-SSIM, SAM, UIQI, RMSE, FSIM, SRER) against the HR ground truth.
- `"lr_only"`: Takes only a single LR image. Runs SwinIR on overlapping patches and saves the SR output. No HR ground truth required; no metrics are computed.

**Outputs** (saved in `output_dir`):

| File | Description |
|------|-------------|
| `lr_display.png` | LR image rendered as an 8-bit RGB composite using `lr_bands` |
| `sr_display.png` | SR image rendered as an 8-bit RGB composite |
| `hr_display.png` | HR image rendered as an 8-bit RGB composite (`"paired"` only) |
| `lr_band_N.png` | Grayscale PNG for each individual LR channel (N = 1, 2, …) |
| `sr_band_N.png` | Grayscale PNG for each individual SR channel |
| `hr_band_N.png` | Grayscale PNG for each individual HR channel (`"paired"` only) |
| `metrics.json` | All computed metrics (`"paired"` only — see structure below) |
| `raw_inference_*.log` | Full run log |

**`metrics.json` structure** (`"paired"` mode):
```json
{
  "sr":  { "psnr": 32.1, "ssim": 0.91, "sam": 0.04, ... },
  "lr_bicubic": { "psnr": 28.5, "ssim": 0.83, ... },
  "delta": { "psnr": 3.6, "ssim": 0.08, ... },
  "per_band": {
    "band_1": { "psnr": 33.2, "ssim": 0.92 },
    "band_2": { "psnr": 31.8, "ssim": 0.90 },
    "band_3": { "psnr": 31.4, "ssim": 0.89 }
  },
  "lr_bands": [3, 2, 1],
  "hr_bands": [1, 2, 3]
}
```

- `sr` / `lr_bicubic` / `delta` — full-image composite metrics (8 metrics each)
- `per_band` — PSNR and SSIM broken down per output channel
- `lr_bands` / `hr_bands` — the band indices used so results can be mapped back to spectral bands

---

## 4. Visual Comparison

### `compare_images_side_by_side.py`

Creates side-by-side comparison images showing LR | SR | HR for each patch. Handles both flat and per-image subdirectory SR layouts. Supports random or manual patch selection.

**Run:**
```bash
python compare_images_side_by_side.py
```

**`CONFIG` settings:**

```python
CONFIG = {
    "lr_dir": "testsets/my_test/lr",
    "hr_dir": "testsets/my_test/hr",
    "sr_dir": "testsets/my_test/sr",
    "output_dir": "testsets/my_test/comparisons",
    "sr_iteration": "",        # Specific iteration number string, or "" to auto-pick latest
    "fallback_to_latest_sr_when_iter_missing": True,
    "image_extensions": [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
    "recursive_lr": False,
    "recursive_hr": False,
    "resize_lr_to_hr": True,   # Upscale LR to HR size for visual alignment
    "resize_sr_to_hr": True,   # Resize SR to HR size if dimensions differ
    "padding": 10,             # White padding pixels between panels
    "font_size": 18,
    "label_lr": "LR",
    "label_sr": "SR",
    "label_hr": "HR",
    "patch_count": 100,        # Max number of comparisons to generate; None for all
    "include_patches": [],     # List of specific patch names to always include
    "random_seed": 42,
    "output_suffix": "_comparison",
    "output_ext": ".png",
}
```

**SR file lookup:** The script searches for SR files in two layouts automatically:
1. **Subdirectory layout:** `sr_dir/<name>/<name>[_iter].ext`
2. **Flat layout:** `sr_dir/<name>[_iter].ext`

Set `sr_iteration` to a specific iteration number string to select an exact checkpoint's output, or leave empty to auto-select the highest iteration available.

**Output:** One PNG per patch saved to `output_dir`, named `<patch_name><output_suffix>.png`.

---

## 5. GUI

A full web-based GUI is available in the `gui/` directory. It covers all three workflows (Preprocessing, Training, Inference) with live log streaming, model auto-configuration, and results display.

See **[gui/GUI_USAGE.md](gui/GUI_USAGE.md)** for installation, startup, and a complete feature reference.

**Quick start summary:**
```bash
# Terminal 1 — backend (activate KAIR venv first)
uvicorn gui.backend.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — frontend
cd gui/frontend && npm run dev
```
Open `http://localhost:5173` in your browser.

### 5.1 Phase A — Preprocessing

Open the **Preprocessing** page and select the appropriate tab:
- **Tab A — Pleiades Pipeline**: for paired HR+LR GeoTIFF/JP2. Enter paths, toggle ORB / Phase Correlation / ECC stages, set quality thresholds, and click **Run Pipeline**. A temp config is written to `gui/backend/tmp/pipeline3_config.json` and passed to `pipeline3.py` via `--config`.
- **Tab B — HR Degradation Pipeline**: for HR-only or HR+LR pairs with synthetic degradation (BSRGAN, Real-ESRGAN, BSRGAN+, Satellite MTF). Use **↑ Load from file** to restore a saved degradation config or **↓ Save config** to export the current form values.
- **Tab C — Step Preview**: shows image previews emitted by Pipeline A jobs in real time.

### 5.2 Phase B — Training

Open the **Training** page. Select **PSNR Training** or **GAN Training**, set the task name, load a preset config if needed, configure paths and hyperparameters, then click **Start Training**. A temp config JSON is written to `gui/backend/tmp/` and passed via `--opt`.

### 5.3 Phase C — Inference

The **Inference** page has three tabs:
- **Tab 1 — Patched Images** (`main_test_swinir_config.py`): batch inference on aligned patch directories; reports 8 averaged metrics.
- **Tab 2 — Raw HR+LR** (`raw_inference.py`, `mode=paired`): full-resolution paired inference with optional coregistration. Band selection, per-band image grid, and per-band PSNR/SSIM table are shown after completion.
- **Tab 3 — LR-Only** (`raw_inference.py`, `mode=lr_only`): single LR image, no HR required. SR composite and per-band images are shown after completion.