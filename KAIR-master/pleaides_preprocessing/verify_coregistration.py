"""
Coregistration Verification Suite
==================================
Four diagnostic tests on saved HR/LR patch pairs from the preprocessing pipeline.

  Test 1 — Blinker GIF       : animated toggle HR_down ↔ LR (eyes detect 0.5 px shift)
  Test 2 — Difference Map    : |HR_down − LR| per-channel, with per-patch stats
  Test 3 — Checkerboard      : 4×4 tile interleave of HR_down and LR
  Test 4 — SSIM Heatmap      : structural similarity map with full=True

Usage
-----
  python verify_coregistration.py                    # uses CONFIG defaults
  python verify_coregistration.py --output Lahore_3  # override output dir
  python verify_coregistration.py --n 20             # sample 20 patches

Output layout
-------------
  <OUTPUT_DIR>/verification/
    blinker/      patch_000000.gif  ...
    diff/         patch_000000.png  ...
    checkerboard/ patch_000000.png  ...
    ssim/         patch_000000.png  ...
    summary/      grid_blinker.gif  grid_diff.png  grid_checker.png  grid_ssim.png
    report.txt    per-patch SSIM / MAE / max-diff table + global stats

Dependencies
------------
  pip install numpy opencv-python-headless scikit-image Pillow matplotlib tqdm
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")           # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (mirrors the main pipeline's CONFIG / config.json)
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_JSON_PATH = "config.json"

CONFIG = {
    "OUTPUT_DIR":     "Lahore_4",
    "HR_PATCH_SIZE":  256,
    "LR_PATCH_SIZE":  128,
    "SCALE_FACTOR":   2,
    "NODATA_VALUE":   0,
}

# Verification-specific settings
VERIFY_CONFIG = {
    "N_SAMPLES":        50,     # Number of patches to analyze (None = all)
    "SAMPLE_SEED":      42,     # RNG seed for reproducible patch selection
    "BLINKER_FPS":      3,      # Frames per second for the blinker GIF
    "BLINKER_LOOPS":    0,      # 0 = loop forever
    "CHECKER_GRID":     4,      # NxN checkerboard grid divisions
    "DIFF_CMAP":        "hot",  # Matplotlib colormap for difference maps
    "SSIM_CMAP":        "RdYlGn",  # Colormap for SSIM: green=similar, red=different
    "SUMMARY_COLS":     5,      # Columns in the summary grid images
    "DISPLAY_SIZE":     256,    # Pixel size used for all saved images (upscale LR for clarity)
}


def build_config(output_dir_override: str = None, n_override: int = None) -> Tuple[dict, dict]:
    cfg = CONFIG.copy()
    json_path = Path(__file__).parent / CONFIG_JSON_PATH
    if json_path.exists():
        with open(json_path) as fh:
            cfg.update(json.load(fh))
    cfg["LR_PATCH_SIZE"] = cfg["HR_PATCH_SIZE"] // cfg["SCALE_FACTOR"]

    vcfg = VERIFY_CONFIG.copy()
    if output_dir_override:
        cfg["OUTPUT_DIR"] = output_dir_override
    if n_override:
        vcfg["N_SAMPLES"] = n_override
    return cfg, vcfg


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_pair(hr_path: Path, lr_path: Path, display_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one HR/LR pair.  Returns:
      hr_full : (256,256,3) uint8   — original HR patch
      hr_down : (128,128,3) uint8   — HR downsampled to LR size (bicubic, the comparison baseline)
      lr      : (128,128,3) uint8   — LR patch as saved by the pipeline
    """
    hr_full = cv2.cvtColor(cv2.imread(str(hr_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    lr      = cv2.cvtColor(cv2.imread(str(lr_path),  cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    hr_down = cv2.resize(hr_full, (lr.shape[1], lr.shape[0]), interpolation=cv2.INTER_CUBIC)
    hr_down = np.clip(hr_down, 0, 255).astype(np.uint8)
    return hr_full, hr_down, lr


def to_display(img: np.ndarray, size: int) -> np.ndarray:
    """Upscale a small array to `size`×`size` (nearest) for visual output."""
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def save_png(path: Path, img: np.ndarray) -> None:
    """Save an RGB uint8 array as PNG (cv2 writes BGR)."""
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — BLINKER GIF
# ─────────────────────────────────────────────────────────────────────────────

def make_blinker_gif(
    hr_down: np.ndarray,
    lr: np.ndarray,
    out_path: Path,
    fps: int,
    loops: int,
    display_size: int,
) -> None:
    """
    Create a 2-frame animated GIF toggling between HR_down and LR.
    Both frames are upscaled to `display_size` with nearest-neighbour so
    individual pixels are clearly visible.
    """
    frame_a = Image.fromarray(to_display(hr_down, display_size)).convert("RGB")
    frame_b = Image.fromarray(to_display(lr,      display_size)).convert("RGB")

    # Add a 1-pixel coloured border so the viewer knows which frame is which
    # (green border = HR_down reference, red border = LR target)
    def add_border(pil_img: Image.Image, colour: Tuple[int, int, int], thickness: int = 4) -> Image.Image:
        arr = np.array(pil_img)
        arr[:thickness, :] = colour
        arr[-thickness:, :] = colour
        arr[:, :thickness] = colour
        arr[:, -thickness:] = colour
        return Image.fromarray(arr)

    frame_a = add_border(frame_a, (50, 200, 50))   # green = HR reference
    frame_b = add_border(frame_b, (200, 50, 50))   # red   = LR

    duration_ms = int(1000 / fps)
    frame_a.save(
        str(out_path),
        save_all=True,
        append_images=[frame_b],
        duration=duration_ms,
        loop=loops,
        optimize=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — DIFFERENCE MAP
# ─────────────────────────────────────────────────────────────────────────────

def make_diff_map(
    hr_down: np.ndarray,
    lr: np.ndarray,
    out_path: Path,
    cmap_name: str,
    display_size: int,
) -> dict:
    """
    Compute the absolute difference |HR_down − LR| averaged over channels.
    Renders as a false-colour heatmap and returns per-patch error statistics.
    """
    diff = np.abs(hr_down.astype(np.float32) - lr.astype(np.float32))  # (H,W,3)
    diff_mean = diff.mean(axis=2)                                        # (H,W)

    stats = {
        "mae":      float(diff_mean.mean()),
        "max_diff": float(diff_mean.max()),
        "p95_diff": float(np.percentile(diff_mean, 95)),
        "rmse":     float(np.sqrt((diff_mean ** 2).mean())),
    }

    # Normalise to [0,1] using the theoretical max (255) so colours are
    # comparable across patches (not auto-scaled per patch).
    norm_diff = diff_mean / 255.0

    cmap = plt.get_cmap(cmap_name)
    heatmap = (cmap(norm_diff)[:, :, :3] * 255).astype(np.uint8)   # drop alpha

    # Overlay colourbar strip on the right (16 px wide, full height)
    h, w = heatmap.shape[:2]
    bar_w = max(16, w // 16)
    grad  = np.linspace(1.0, 0.0, h)[:, None]                       # bright=high error
    bar   = (cmap(grad)[:, :, :3] * 255).astype(np.uint8)
    bar   = np.repeat(bar, bar_w, axis=1)
    canvas = np.hstack([heatmap, bar])

    save_png(out_path, to_display(canvas, display_size + bar_w) if display_size != h else canvas)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — CHECKERBOARD OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def make_checkerboard(
    hr_down: np.ndarray,
    lr: np.ndarray,
    out_path: Path,
    grid_n: int,
    display_size: int,
) -> None:
    """
    Divide both images into an (grid_n × grid_n) tile grid, then alternate
    tiles between HR_down (the reference) and LR.

    Look at road edges / building outlines that cross tile boundaries —
    a clean straight line means perfect coregistration.
    """
    H, W = hr_down.shape[:2]
    tile_h = H // grid_n
    tile_w = W // grid_n

    canvas = np.zeros_like(hr_down)
    for i in range(grid_n):
        for j in range(grid_n):
            r0, r1 = i * tile_h, (i + 1) * tile_h
            c0, c1 = j * tile_w, (j + 1) * tile_w
            if (i + j) % 2 == 0:
                canvas[r0:r1, c0:c1] = hr_down[r0:r1, c0:c1]   # HR = reference
            else:
                canvas[r0:r1, c0:c1] = lr[r0:r1, c0:c1]         # LR = candidate

    # Draw grid lines so tile boundaries are visible
    disp = to_display(canvas, display_size)
    d    = display_size // grid_n
    for k in range(1, grid_n):
        disp[k * d, :] = [200, 200, 200]
        disp[:, k * d] = [200, 200, 200]

    save_png(out_path, disp)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — SSIM HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def make_ssim_heatmap(
    hr_down: np.ndarray,
    lr: np.ndarray,
    out_path: Path,
    cmap_name: str,
    display_size: int,
) -> dict:
    """
    Compute structural similarity (SSIM) map between HR_down and LR grayscale.
    Returns per-patch SSIM score and saves a heatmap where:
      Bright green → structurally identical
      Dark red      → structurally different (parallax or shift artefact)
    """
    hr_gray = cv2.cvtColor(hr_down, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lr_gray = cv2.cvtColor(lr,      cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    score, ssim_map = ssim(hr_gray, lr_gray, full=True, data_range=1.0)

    # ssim_map is in [-1, 1]; normalise to [0, 1] for display
    ssim_norm = (ssim_map + 1.0) / 2.0

    cmap    = plt.get_cmap(cmap_name)
    heatmap = (cmap(ssim_norm)[:, :, :3] * 255).astype(np.uint8)

    save_png(out_path, to_display(heatmap, display_size))

    # Region-level diagnosis
    H, W  = ssim_map.shape
    q_h, q_w = H // 2, W // 2
    quadrants = {
        "TL": ssim_map[:q_h, :q_w].mean(),
        "TR": ssim_map[:q_h, q_w:].mean(),
        "BL": ssim_map[q_h:, :q_w].mean(),
        "BR": ssim_map[q_h:, q_w:].mean(),
    }
    return {"ssim": float(score), "quadrant_ssim": quadrants}


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY GRID IMAGES
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_grid(image_paths: List[Path], out_path: Path, n_cols: int) -> None:
    """Tile a list of PNG images into a single grid PNG."""
    imgs = [np.array(Image.open(p).convert("RGB")) for p in image_paths]
    if not imgs:
        return

    h, w = imgs[0].shape[:2]
    n_cols = min(n_cols, len(imgs))
    n_rows = (len(imgs) + n_cols - 1) // n_cols

    canvas = np.full((n_rows * h, n_cols * w, 3), 30, dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, n_cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img[:h, :w]

    save_png(out_path, canvas)


def make_summary_blinker_gif(gif_paths: List[Path], out_path: Path, fps: int, loops: int) -> None:
    """
    Build a summary GIF where each frame composites N blinker GIFs into a grid.
    All input GIFs must be 2-frame (HR_down frame, then LR frame).
    """
    gifs = [Image.open(p) for p in gif_paths]
    if not gifs:
        return

    n   = len(gifs)
    n_c = min(5, n)
    n_r = (n + n_c - 1) // n_c
    W, H = gifs[0].size

    out_frames = []
    for frame_idx in range(2):    # 2-frame GIF: frame 0 = HR, frame 1 = LR
        canvas = Image.new("RGB", (n_c * W, n_r * H), (30, 30, 30))
        for idx, gif in enumerate(gifs):
            gif.seek(frame_idx)
            frame = gif.convert("RGB")
            r, c  = divmod(idx, n_c)
            canvas.paste(frame, (c * W, r * H))
        out_frames.append(canvas)

    duration_ms = int(1000 / fps)
    out_frames[0].save(
        str(out_path),
        save_all=True,
        append_images=out_frames[1:],
        duration=duration_ms,
        loop=loops,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REPORT WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_report(records: list, out_path: Path, cfg: dict, vcfg: dict) -> None:
    """Write a plain-text report with per-patch stats and global diagnostics."""

    lines = [
        "=" * 78,
        "  COREGISTRATION VERIFICATION REPORT",
        "=" * 78,
        f"  Output dir  : {cfg['OUTPUT_DIR']}",
        f"  HR patch    : {cfg['HR_PATCH_SIZE']}×{cfg['HR_PATCH_SIZE']} px",
        f"  LR patch    : {cfg['LR_PATCH_SIZE']}×{cfg['LR_PATCH_SIZE']} px",
        f"  Comparison  : HR downsampled (bicubic) vs LR as-saved",
        f"  Patches     : {len(records)} sampled",
        "",
        f"  {'Patch':<20}  {'SSIM':>6}  {'MAE':>7}  {'RMSE':>7}  {'p95 diff':>9}  {'max diff':>9}",
        "  " + "-" * 62,
    ]

    ssim_vals  = []
    mae_vals   = []
    rmse_vals  = []

    for rec in records:
        name = rec["name"]
        s    = rec["ssim"]
        mae  = rec["mae"]
        rmse = rec["rmse"]
        p95  = rec["p95_diff"]
        mx   = rec["max_diff"]
        ssim_vals.append(s)
        mae_vals.append(mae)
        rmse_vals.append(rmse)

        # Flag suspicious patches
        flag = ""
        if s < 0.50:
            flag = "  ← POOR (likely parallax or global shift)"
        elif s < 0.70:
            flag = "  ← MARGINAL"
        lines.append(
            f"  {name:<20}  {s:6.4f}  {mae:7.2f}  {rmse:7.2f}  {p95:9.2f}  {mx:9.2f}{flag}"
        )

    lines += [
        "  " + "-" * 62,
        f"  {'MEAN':<20}  {np.mean(ssim_vals):6.4f}  {np.mean(mae_vals):7.2f}  {np.mean(rmse_vals):7.2f}",
        f"  {'STD':<20}  {np.std(ssim_vals):6.4f}  {np.std(mae_vals):7.2f}  {np.std(rmse_vals):7.2f}",
        f"  {'MIN':<20}  {np.min(ssim_vals):6.4f}  {np.min(mae_vals):7.2f}  {np.min(rmse_vals):7.2f}",
        f"  {'MAX (worst)':<20}  {np.max(ssim_vals):6.4f}  {np.max(mae_vals):7.2f}  {np.max(rmse_vals):7.2f}",
        "",
        "  INTERPRETATION GUIDE",
        "  ─────────────────────────────────────────────────────────────",
        "  Blinker GIF   Green border = HR reference, Red = LR.",
        "                Static edges between flickers → good alignment.",
        "                'Breathing' / 'jitter' → residual shift remains.",
        "",
        "  Diff Map      MAE < 5.0 and p95 < 15.0 → acceptable.",
        "                White halos on building edges → parallax.",
        "                Uniform offset on ALL edges → global shift.",
        "                Scattered white dots → temporal change (normal).",
        "",
        "  Checkerboard  Road / runway must continue cleanly across tiles.",
        "                Broken or offset lines → coregistration failure.",
        "",
        "  SSIM Heatmap  Bright green = structurally identical.",
        "                Dark red on building edges only → local parallax.",
        "                Dark red everywhere → global shift or radiometric",
        "                difference too large.",
        "=" * 78,
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Report written to: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Coregistration Verification Suite")
    parser.add_argument("--output", default=None, help="Override OUTPUT_DIR from config")
    parser.add_argument("--n",      type=int, default=None, help="Number of patches to sample")
    args = parser.parse_args()

    cfg, vcfg = build_config(args.output, args.n)

    output_dir  = Path(cfg["OUTPUT_DIR"])
    hr_dir      = output_dir / "hr"
    lr_dir      = output_dir / "lr"

    if not hr_dir.exists() or not lr_dir.exists():
        logging.error(
            "hr/ or lr/ subdirectory not found inside '%s'. "
            "Run the main preprocessing pipeline first.", output_dir
        )
        sys.exit(1)

    # Collect and optionally subsample patch filenames
    all_patches = sorted(hr_dir.glob("*.png"))
    if not all_patches:
        logging.error("No PNG patches found in %s", hr_dir)
        sys.exit(1)

    n = vcfg["N_SAMPLES"]
    if n and n < len(all_patches):
        rng = np.random.default_rng(vcfg["SAMPLE_SEED"])
        selected = sorted(rng.choice(len(all_patches), size=n, replace=False))
        patches  = [all_patches[i] for i in selected]
    else:
        patches = all_patches

    logging.info("Verifying %d / %d patches.", len(patches), len(all_patches))

    # Create output subdirectories
    verify_dir  = output_dir / "verification"
    blinker_dir = verify_dir / "blinker"
    diff_dir    = verify_dir / "diff"
    checker_dir = verify_dir / "checkerboard"
    ssim_dir    = verify_dir / "ssim"
    summary_dir = verify_dir / "summary"
    for d in [blinker_dir, diff_dir, checker_dir, ssim_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    display_size = vcfg["DISPLAY_SIZE"]
    records: list = []

    blinker_gifs  = []
    diff_pngs     = []
    checker_pngs  = []
    ssim_pngs     = []

    for hr_path in tqdm(patches, desc="Running diagnostic tests", unit="patch"):
        name    = hr_path.stem
        lr_path = lr_dir / hr_path.name

        if not lr_path.exists():
            logging.warning("LR patch missing for %s — skipping.", name)
            continue

        hr_full, hr_down, lr = load_pair(hr_path, lr_path, display_size)

        # Test 1 — Blinker GIF
        blinker_path = blinker_dir / f"{name}.gif"
        make_blinker_gif(hr_down, lr, blinker_path,
                         fps=vcfg["BLINKER_FPS"], loops=vcfg["BLINKER_LOOPS"],
                         display_size=display_size)
        blinker_gifs.append(blinker_path)

        # Test 2 — Difference Map
        diff_path = diff_dir / f"{name}.png"
        diff_stats = make_diff_map(hr_down, lr, diff_path,
                                   cmap_name=vcfg["DIFF_CMAP"],
                                   display_size=display_size)
        diff_pngs.append(diff_path)

        # Test 3 — Checkerboard
        checker_path = checker_dir / f"{name}.png"
        make_checkerboard(hr_down, lr, checker_path,
                          grid_n=vcfg["CHECKER_GRID"],
                          display_size=display_size)
        checker_pngs.append(checker_path)

        # Test 4 — SSIM Heatmap
        ssim_path = ssim_dir / f"{name}.png"
        ssim_stats = make_ssim_heatmap(hr_down, lr, ssim_path,
                                       cmap_name=vcfg["SSIM_CMAP"],
                                       display_size=display_size)
        ssim_pngs.append(ssim_path)

        records.append({
            "name":     name,
            "ssim":     ssim_stats["ssim"],
            "mae":      diff_stats["mae"],
            "rmse":     diff_stats["rmse"],
            "p95_diff": diff_stats["p95_diff"],
            "max_diff": diff_stats["max_diff"],
        })

    logging.info("Building summary grids …")
    n_cols = vcfg["SUMMARY_COLS"]
    make_summary_grid(diff_pngs,    summary_dir / "grid_diff.png",        n_cols)
    make_summary_grid(checker_pngs, summary_dir / "grid_checkerboard.png", n_cols)
    make_summary_grid(ssim_pngs,    summary_dir / "grid_ssim.png",         n_cols)
    make_summary_blinker_gif(blinker_gifs, summary_dir / "grid_blinker.gif",
                             fps=vcfg["BLINKER_FPS"], loops=vcfg["BLINKER_LOOPS"])

    write_report(records, verify_dir / "report.txt", cfg, vcfg)

    mean_ssim = np.mean([r["ssim"] for r in records])
    mean_mae  = np.mean([r["mae"]  for r in records])
    logging.info(
        "Verification complete.  Mean SSIM=%.4f  Mean MAE=%.2f  "
        "Results → %s",
        mean_ssim, mean_mae, verify_dir,
    )


if __name__ == "__main__":
    main()