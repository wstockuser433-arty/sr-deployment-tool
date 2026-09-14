# GUI Usage Guide

This guide covers how to install, run, and use the KAIR Super-Resolution Studio — a web GUI for training, inference, and preprocessing using SwinIR.

---

## Prerequisites

- Python ≥ 3.9 with the KAIR virtualenv already set up (see [USAGE.md](USAGE.md))
- Node.js ≥ 18 and npm ≥ 9

---

## 1. Install Backend Dependencies

Activate your KAIR virtual environment first, then install the extra GUI packages:

```bash
# Activate venv (if not already active)
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate.bat       # Windows

# Install GUI backend dependencies
pip install -r gui/backend/requirements.txt
```

---

## 2. Install Frontend Dependencies

```bash
cd gui/frontend
npm install
cd ../..
```

---

## 3. Run the Application

Open **two terminals** from the KAIR project root.

### Terminal 1 — Start the FastAPI backend

```bash
source .venv/bin/activate

# Run from the project root (so relative paths like superresolution/ resolve correctly)
uvicorn gui.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Terminal 2 — Start the React frontend

```bash
cd gui/frontend
npm run dev
```

Open **`http://localhost:5173`** in your browser.

> The Vite dev server automatically proxies all `/api/*` requests to `http://localhost:8000`, so no CORS issues during development.

---

## 4. Pages Overview

### Training (`/training`)

| Field | Description |
|---|---|
| Mode tabs | **PSNR Training** → `main_train_swinir.py` · **GAN Training** → `main_train_swinir_gan.py` |
| Task name | Becomes the output folder under `superresolution/` |
| Scale | Upscaling factor: ×2 / ×3 / ×4 / ×8 |
| Load preset config | Load defaults from any `options/swinir/*.json` |
| Dataset section | HR/LR train & test dirs, batch size, workers, patch size |
| Network section | `embed_dim`, `depths`, `num_heads`, `upsampler`, `resi_connection` |
| Hyperparameters | Loss type, LR, EMA decay, milestones, checkpoint intervals |
| GAN section | Discriminator, perceptual loss, GAN type (only shown in GAN mode) |
| Recent runs | Live list of `superresolution/` task dirs with iteration count |

A temporary config JSON is written to `gui/backend/tmp/` and passed to the training script via `--opt`. The real `options/swinir/` configs are never modified.

---

### Inference (`/inference`)

Features a 3-tab layout:

**Tab 1 — Patched Images (`main_test_swinir_config.py`)**
Runs batch inference on a directory of already-aligned LR patches against an HR ground truth directory.
- Model selection (latest from task or custom `.pth`) with autofilled `MODEL_CONFIG`
- Input/Output dir paths
- Reports 8 metrics across the batch (PSNR, SSIM, IT-SSIM, SAM, UIQI, RMSE, FSIM, SRER)

> [!NOTE]
> **Model Auto-fill Directory Format**: To automatically detect and auto-fill model architectures in the GUI (e.g. embed dimension, window size, depths), checkpoints must be placed in a specific structure under `superresolution/`:
> ```
> superresolution/
> └── <task_name>/
>     ├── models/
>     │   └── <iteration>_E.pth (or _G.pth)
>     ├── options/
>     │   └── train.json (containing SwinIR architecture settings under "netG")
>     └── log/
>         └── train.log
> ```
> Training runs executed via the GUI automatically produce this format. If importing pretrained model weights downloaded from external sources, please replicate this folder layout so the backend model scanner can parse the config.


**Tab 2 — Raw HR+LR Inference (`raw_inference.py`)**
Performs end-to-end inference on a paired raw satellite image.
- Configure band mappings per sensor (HR and LR band indices) using the sensor profile selector
- Automatically coregisters (ORB + Phase Correlation) and radiometrically matches LR to HR
- Enforces an exact integer scaling factor (e.g. ×2) regardless of sensor resolution differences
- Patches the full image, runs SR, stitches with a Hann-window blend, and computes metrics
- **Band selection**: When a valid image path is entered, the GUI fetches band count and dimensions from the backend and displays a `MetaPill` (e.g. "4 bands · 1024×1024 px · geospatial"). Band selection switches from a text array editor to toggle buttons — click to add/remove bands; the order you click sets the display channel order (badges show 1/2/3 position).
- Resulting SR/LR/HR RGB composites are displayed directly in the GUI
- **Per-band images**: Below the composite viewer, a spectral band grid shows each individual grayscale channel for LR, SR, and HR (click any thumbnail to open full-size). Labelled "Spectral N" using the configured band indices.
- **Per-band metrics table**: PSNR and SSIM broken down per output channel, labelled by the actual spectral band number from the band selection config.

**Tab 3 — LR-Only Inference (`raw_inference.py`)**
Runs SR on a single unlabelled LR image (no HR ground truth).
- Full image patching and stitching
- Band selection with metadata auto-load (same `MetaPill` + toggle-button UX as Tab 2)
- Per-band grayscale SR images displayed below the RGB composite result
- Result SR image is displayed directly in the GUI

**Model Architecture card** — shown in the right column for all three tabs (outside the form, always visible while configuring bands and paths).

**Band Classes reference** — a "Band Classes" button in the sidebar opens a modal listing spectral band profiles for common satellite sensors (Panchromatic, RGB, Multispectral 4-band, Multispectral 8-band, Sentinel-2). Each profile shows band index, name, wavelength range, and description. Use this as a quick reference when setting `lr_bands` / `hr_bands`.

---

### Preprocessing (`/preprocessing`)

#### Tab A — Pleiades Pipeline (`pipeline3.py`)

For **paired HR + LR GeoTIFF / JP2** satellite imagery. Performs:
1. 3-stage coregistration: ORB → Phase Correlation → ECC (each stage independently toggleable)
2. Radiometric regression (linear LR→HR normalisation)
3. Optional histogram matching
4. Sliding-window patch extraction with quality filters (variance, SSIM, ECC score, nodata)
5. Optional post-processing **train/test split** (configurable ratio)

A temporary config JSON is written to `gui/backend/tmp/pipeline3_config.json` and passed to `pipeline3.py` via `--config`. The script's inline `CONFIG` dict and `pleaides_preprocessing/config.json` are **not modified**.

#### Tab B — HR Degradation Pipeline (`run_pipeline.py`)

For **HR-only** images (generates synthetic LR) or **existing HR+LR pairs** (preprocessing only):

| Mode | Description |
|---|---|
| `hr_only` | Reads HR, applies normalisation, degrades to LR |
| `hr_lr_pair` | Reads both HR and LR, preprocesses without degradation |

**Degradation types:**
- `bsrgan` — BSRGAN-style (13-step degradation shuffle)
- `real_esrgan` — Real-ESRGAN two-stage pipeline
- `bsrgan_plus` — Extended BSRGAN with USM sharpening
- `satellite` — MTF-based optics/sensor/atmospheric degradation (recommended for satellite data)

Optional **train/test split** moves a configurable fraction of output images to separate `*_test` directories.

**Degradation file browser** — at the top of the Degradation section:
- **↑ Load from file**: loads a previously saved `.json` degradation config and merges all degradation parameters into the current form state (useful for reusing a config learned by `esrgan_mapping_optuna.py`).
- **↓ Save config**: downloads the current degradation form values as a named `.json` file for reuse or version control.

#### Tab C — Step Preview

Displays image previews emitted by Pipeline A jobs as they run. Each preprocessing stage that produces a preview image (e.g. coregistration overlays, patch quality visualisations) appears as a labelled thumbnail grid. Thumbnails are updated in real time as the job progresses. Click any thumbnail to view it full-size.

---
