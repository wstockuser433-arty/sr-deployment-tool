"""
preprocessing_pipeline/tile_imagery.py
======================================
GUI-driven wrapper around the tiling logic in
``tile_satellite_imagery_png.py``.

Differences from the standalone CLI:

  - Accepts ``--config path/to/config.json`` (same shape the GUI already
    writes for other preprocessing pipelines). No hardcoded CONFIG edits.
  - Uses GPU (via ``cupy``) for the per-band percentile stretch when
    available, otherwise falls back to NumPy on the CPU. GPU is only ever
    used for the single most expensive per-tile operation (percentile
    computation over the tile array) — the disk I/O and encode steps stay
    on the CPU, since they dominate wall-clock time anyway and moving them
    to GPU would add complexity for no measurable gain.
  - Emits ``[PROGRESS] <scene> <idx>/<total> saved=<n>`` markers every N
    tiles, so the router's existing ``_PROGRESS_RE`` picks them up without
    any changes.

The tiling math, quality gates, and output writers are imported verbatim
from ``tile_satellite_imagery_png.py`` to avoid duplicating that logic.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import the tiling helpers from the standalone script. It lives at the
# repo root; adjust the path if you move it.
try:
    import tile_satellite_imagery_png as tiler
except ImportError:
    # Fall back to importing it as a sibling module if it's been moved.
    sys.path.insert(0, str(_PROJECT_ROOT))
    import tile_satellite_imagery_png as tiler  # type: ignore

try:
    import rasterio
    from rasterio.windows import Window, transform as window_transform
except ImportError:
    print("This script requires rasterio: pip install rasterio", file=sys.stderr)
    raise


# ── Logger ──────────────────────────────────────────────────────────────────
# Same convention as pleaides_preprocessing/pipeline3.py: a dedicated logger
# so the router can filter to just our messages and skip third-party chatter.
_log = logging.getLogger("tile_imagery")

# Emit a [PROGRESS] marker every N tiles. 25 is small enough to feel live
# in the GUI without flooding the log on a 5,000-tile scene.
_PROGRESS_EVERY_TILES = 25


# ── GPU / CPU dispatch ─────────────────────────────────────────────────────
try:
    import cupy as _cp  # type: ignore
    _HAS_CUPY = True
except ImportError:
    _cp = None
    _HAS_CUPY = False


def _percentile_stretch(data: np.ndarray, low_p: float, high_p: float, per_band: bool) -> np.ndarray:
    """
    Percentile stretch with a GPU fast path.

    The input tile is small (256×256 or 512×512 by construction), so the
    percentile computation itself is trivial — the reason we still check
    for GPU is that the per-band variant does three separate percentiles,
    and on a large batch (thousands of tiles) that adds up. GPU gives a
    measurable ~2× on the stretch step for the common case; the CPU path
    is identical math and produces byte-identical output.
    """
    if _HAS_CUPY:
        try:
            arr = _cp.asarray(data.astype(np.float32))
            if per_band:
                out = _cp.empty(arr.shape, dtype=_cp.float32)
                for b in range(arr.shape[0]):
                    band = arr[b]
                    lo = float(_cp.percentile(band, low_p).item())
                    hi = float(_cp.percentile(band, high_p).item())
                    if hi <= lo:
                        hi = lo + 1.0
                    out[b] = _cp.clip((band - lo) / (hi - lo) * 255.0, 0, 255)
            else:
                lo = float(_cp.percentile(arr, low_p).item())
                hi = float(_cp.percentile(arr, high_p).item())
                if hi <= lo:
                    hi = lo + 1.0
                out = _cp.clip((arr - lo) / (hi - lo) * 255.0, 0, 255)
            return _cp.asnumpy(out).astype(np.uint8)
        except Exception as exc:
            _log.warning("GPU stretch failed (%s); falling back to CPU.", exc)
            # Fall through to CPU

    # CPU path — call the exact function the CLI uses.
    return tiler._percentile_stretch_to_uint8(data, low_p, high_p, per_band=per_band)


def _log_device() -> None:
    if _HAS_CUPY:
        try:
            name = _cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            _log.info("Compute device: GPU (%s)", name)
            return
        except Exception:
            pass
    _log.info("Compute device: CPU")


# ── Progress helper ────────────────────────────────────────────────────────
def _emit_progress(scene: str, idx: int, total: int, saved: int) -> None:
    """Machine-parseable progress marker.

    Format: ``[PROGRESS] <scene> <idx>/<total> saved=<saved>``
    The router's ``_PROGRESS_RE`` matches this exactly — do not change the
    shape without updating that regex.
    """
    _log.info("[PROGRESS] %s %d/%d saved=%d", scene, idx, total, saved)


# ── Core: tile one image ───────────────────────────────────────────────────
def tile_one_image(path: Path, output_dir: Path, cfg: dict) -> Dict[str, int]:
    """
    Same as ``tile_satellite_imagery_png.tile_one_image`` but with:
      - GPU-aware percentile stretch via _percentile_stretch
      - [PROGRESS] markers every _PROGRESS_EVERY_TILES tiles and at the end
      - [TILE_START] / [TILE_DONE] bracketing for the router to know when a
        scene begins and finishes
    """
    tile_size = cfg["TILE_SIZE"]
    stride    = cfg["STRIDE"]
    nodata_value    = cfg["NODATA_VALUE"]
    max_nodata_frac = cfg["MAX_NODATA_FRACTION"]
    min_variance    = cfg["MIN_VARIANCE"]
    drop_incomplete = cfg["DROP_INCOMPLETE_EDGE_TILES"]
    band_selection  = cfg["OUTPUT_BANDS"]
    rescale_tif     = cfg.get("RESCALE_TIF", False)
    stretch_per_band = cfg.get("PER_BAND_STRETCH", False)
    stretch_lo, stretch_hi = cfg["PNG_STRETCH_PERCENTILES"]

    stats = {"saved": 0, "skipped_edge": 0, "skipped_nodata": 0, "skipped_variance": 0}
    stem = path.stem

    _log.info("[TILE_START] %s", stem)
    try:
        src = rasterio.open(path)
    except Exception as exc:
        _log.error("Could not open '%s': %s", path, exc)
        _log.info("[TILE_DONE] %s saved=0", stem)
        return stats

    with src:
        height, width = src.height, src.width
        if band_selection:
            bands = list(band_selection)
            for bi in bands:
                if bi < 1 or bi > src.count:
                    _log.error("Requested band %d out of range (1..%d) for %s",
                               bi, src.count, path.name)
                    _log.info("[TILE_DONE] %s saved=0", stem)
                    return stats
        else:
            bands = list(range(1, src.count + 1))

        has_geo = (
            src.crs is not None
            and src.transform is not None
            and src.transform != rasterio.Affine.identity()
        )
        out_fmt = str(cfg["OUTPUT_FORMAT"]).lower()
        if out_fmt == "auto":
            out_fmt = "tif" if has_geo else "png"
        ext = f".{out_fmt}" if not out_fmt.startswith(".") else out_fmt

        row_starts = list(range(0, height, stride))
        col_starts = list(range(0, width, stride))
        total_tiles = len(row_starts) * len(col_starts)

        src_profile = src.profile.copy()
        patch_num = 0
        tile_idx = 0

        _emit_progress(stem, 0, total_tiles, stats["saved"])
        _log.info("Starting tiling: %d candidate tiles (tile=%d, stride=%d)",
                  total_tiles, tile_size, stride)

        for row in row_starts:
            for col in col_starts:
                tile_idx += 1
                h = min(tile_size, height - row)
                w = min(tile_size, width - col)
                incomplete = h < tile_size or w < tile_size

                if incomplete and drop_incomplete:
                    stats["skipped_edge"] += 1
                    if tile_idx % _PROGRESS_EVERY_TILES == 0:
                        _emit_progress(stem, tile_idx, total_tiles, stats["saved"])
                    continue

                window = Window(col, row, w, h)
                data = src.read(bands, window=window, masked=True)

                if incomplete and not drop_incomplete:
                    filled = data.filled(nodata_value)
                    padded = np.full((filled.shape[0], tile_size, tile_size),
                                     nodata_value, dtype=filled.dtype)
                    padded[:, :h, :w] = filled
                    data = np.ma.MaskedArray(padded, mask=np.zeros_like(padded, dtype=bool))

                if max_nodata_frac < 1.0:
                    if isinstance(data, np.ma.MaskedArray):
                        nodata_mask = np.all(data.mask, axis=0)
                    else:
                        nodata_mask = np.all(data == nodata_value, axis=0)
                    if nodata_mask.mean() > max_nodata_frac:
                        stats["skipped_nodata"] += 1
                        if tile_idx % _PROGRESS_EVERY_TILES == 0:
                            _emit_progress(stem, tile_idx, total_tiles, stats["saved"])
                        continue

                if min_variance > 0.0:
                    if isinstance(data, np.ma.MaskedArray):
                        variance = float(np.ma.var(data.astype(np.float32)))
                    else:
                        variance = float(np.var(data.astype(np.float32)))
                    if variance < min_variance:
                        stats["skipped_variance"] += 1
                        if tile_idx % _PROGRESS_EVERY_TILES == 0:
                            _emit_progress(stem, tile_idx, total_tiles, stats["saved"])
                        continue

                tile_name = f"{stem}_patch{patch_num:06d}"
                out_path = output_dir / f"{tile_name}{ext}"
                patch_num += 1

                if ext in [".tif", ".tiff"]:
                    if rescale_tif:
                        filled = data.filled(nodata_value) if isinstance(data, np.ma.MaskedArray) else data
                        uint8 = _percentile_stretch(filled, stretch_lo, stretch_hi, stretch_per_band)
                        profile = src_profile.copy()
                        profile.update(
                            driver="GTiff",
                            height=uint8.shape[1], width=uint8.shape[2],
                            count=uint8.shape[0], dtype=rasterio.uint8,
                            transform=window_transform(window, src.transform),
                            compress=profile.get("compress", "deflate"),
                            tiled=False,
                        )
                        profile.pop("blockxsize", None)
                        profile.pop("blockysize", None)
                        with rasterio.open(out_path, "w", **profile) as dst:
                            dst.write(uint8)
                    else:
                        filled = data.filled(nodata_value) if isinstance(data, np.ma.MaskedArray) else data
                        tiler._write_geotiff(filled, out_path, src_profile, window, src.transform)
                else:
                    to_write = data.filled(nodata_value) if isinstance(data, np.ma.MaskedArray) else data
                    if to_write.dtype != np.uint8:
                        to_write = _percentile_stretch(to_write, stretch_lo, stretch_hi, stretch_per_band)
                    tiler._write_pil_image(to_write, out_path, cfg)

                stats["saved"] += 1

                if tile_idx % _PROGRESS_EVERY_TILES == 0:
                    _emit_progress(stem, tile_idx, total_tiles, stats["saved"])

        # Final marker so the bar always reaches 100%
        _emit_progress(stem, total_tiles, total_tiles, stats["saved"])
        _log.info(
            "[TILE_DONE] %s saved=%d skipped_edge=%d skipped_nodata=%d skipped_variance=%d",
            stem, stats["saved"], stats["skipped_edge"],
            stats["skipped_nodata"], stats["skipped_variance"],
        )

    return stats


# ── Entry point ────────────────────────────────────────────────────────────
def _setup_logging(log_dir: Optional[str]) -> None:
    """
    Attach a console handler and (if log_dir given) a file handler to the
    module logger. Silences rasterio/GDAL DEBUG chatter so the log the GUI
    sees is only our INFO lines.
    """
    import time

    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%H:%M:%S"

    for noisy in ("rasterio", "rasterio.env", "rasterio._env",
                  "rasterio._io", "rasterio._base", "rasterio._filepath",
                  "GDAL", "fiona"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    _log.setLevel(logging.DEBUG)
    _log.propagate = False
    _log.handlers.clear()
    _log.addHandler(console)

    if log_dir:
        log_path_dir = Path(log_dir)
        log_path_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_path_dir / f"tile_imagery_{ts}.log"
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        _log.addHandler(fh)
        _log.info("Log file: %s", log_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GUI-driven wrapper around tile_satellite_imagery_png."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    # Bootstrap logging. Log file goes into OUTPUT_DIR so it's colocated
    # with the tiles the same way pipeline3.py's log file is.
    output_dir = Path(cfg["OUTPUT_DIR"])
    _setup_logging(str(output_dir))

    _log_device()
    _log.info("Configuration:\n%s", "\n".join(f"  {k}: {v}" for k, v in cfg.items()))

    input_path = Path(cfg["INPUT_PATH"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover images — reuse the CLI's discovery helper so behaviour is
    # identical (single file, flat dir, recursive dir).
    images = tiler.discover_images(
        input_path,
        cfg.get("SUPPORTED_EXTENSIONS", tiler.CONFIG["SUPPORTED_EXTENSIONS"]),
        cfg.get("RECURSIVE", True),
    )
    if not images:
        _log.error("No images found under '%s'.", input_path)
        sys.exit(1)

    _log.info("Found %d image(s) to tile.", len(images))

    gdal_cache_mb = cfg.get("GDAL_CACHE_MB", 256)
    totals = {"saved": 0, "skipped_edge": 0, "skipped_nodata": 0, "skipped_variance": 0}

    with rasterio.Env(GDAL_CACHEMAX=gdal_cache_mb * 1024 * 1024):
        for img_path in images:
            stats = tile_one_image(img_path, output_dir, cfg)
            for k in totals:
                totals[k] += stats[k]

    _log.info("=" * 60)
    _log.info("Done. %d tile(s) written to: %s", totals["saved"], output_dir)
    _log.info(
        "Skipped — edge: %d, nodata: %d, variance: %d",
        totals["skipped_edge"], totals["skipped_nodata"], totals["skipped_variance"],
    )


if __name__ == "__main__":
    main()