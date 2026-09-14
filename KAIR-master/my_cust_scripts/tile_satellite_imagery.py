"""
tile_satellite_imagery.py
==========================
Tile satellite imagery (GeoTIFF, plain TIFF, JP2, PNG, JPG, BMP, ...) into
fixed-size square tiles (default 256x256).

Design goals (matches the conventions already used in this project's
preprocessing pipelines — see pleaides_preprocessing/pipeline3.py):

  - Never loads a full-resolution scene into memory. Every tile is read via
    a rasterio *windowed* read, so a 20,000x20,000px (or larger) GeoTIFF
    never needs to fit in RAM regardless of how many tiles it produces.
  - Works on both georeferenced and non-georeferenced inputs. rasterio/GDAL
    opens TIFF, GeoTIFF, JP2, PNG, JPEG, BMP, etc. through the same code
    path — georeferencing (CRS + affine transform) is only used/written
    when it's actually present on the source.
  - Output tiles are GeoTIFF when the source has real georeferencing
    (preserves CRS + a correctly offset affine transform per tile — this
    matters if these tiles ever get used for anything spatial later), or
    PNG otherwise. This can be overridden.
  - Optional quality gates: discard tiles that are mostly nodata, or nearly
    blank/flat (low variance) — same idea as MAX_NODATA_FRACTION /
    MIN_VARIANCE in pipeline3.py, so tiles produced here are compatible
    with the same downstream SR training expectations.
  - Recursively discovers images under a directory, or accepts a single
    file.

Usage
-----
    python tile_satellite_imagery.py --input path/to/scene.tif --output tiles_out
    python tile_satellite_imagery.py --input path/to/scene_dir --output tiles_out --recursive
    python tile_satellite_imagery.py --input scene.tif --output tiles_out --tile-size 256 --stride 256

    # or just edit CONFIG below and run with no arguments:
    python tile_satellite_imagery.py

Dependencies
------------
    pip install rasterio numpy tqdm pillow
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window, transform as window_transform
except ImportError:
    print("This script requires rasterio: pip install rasterio", file=sys.stderr)
    raise

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional — fall back to a no-op wrapper
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # these are legitimately large scientific images
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these as defaults, or override via CLI flags
# ─────────────────────────────────────────────────────────────────────────────

CONFIG: dict = {
    # ── Paths ────────────────────────────────────────────────────────────
    # "INPUT_PATH": "/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/lr-degraded-outputs/maxar-dataset-pak/la-wildfire-59cm/103001010C487900-visual/lr_x2/103001010C487900-visual.png",              # a single file OR a directory
    "INPUT_PATH": "/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/dc_int_070726/ISB/PRSS-1_MSS_0114_0124_20200201_L2AR_0827081604830.tif", 
    "OUTPUT_DIR": "/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/img-patch-tiles/HR/tiles_512/prss-ms",

    # ── Discovery (directory mode) ──────────────────────────────────────
    "SUPPORTED_EXTENSIONS": [
        ".tif", ".tiff", ".jp2", ".png", ".jpg", ".jpeg", ".bmp",
    ],
    "RECURSIVE": True,             # scan subfolders too (Maxar/PRSS-style nested layouts)

    # ── Tile geometry ────────────────────────────────────────────────────
    "TILE_SIZE": 512,
    "STRIDE": 512,                 # == TILE_SIZE -> no overlap; smaller -> overlapping tiles
    "DROP_INCOMPLETE_EDGE_TILES": True,   # if False, incomplete edge tiles are zero-padded to TILE_SIZE

    # ── Output ───────────────────────────────────────────────────────────
    "OUTPUT_FORMAT": "auto",       # "auto" | "tif" | "png"
                                    #   auto -> GeoTIFF if source has real georeferencing, else PNG
    "OUTPUT_FORMAT": "png",        # "auto" | "tif" | "png" | "jpg"
                                    #   "png" / "jpg" -> force non-geo image output
                                    #   "tif" -> force GeoTIFF output
                                    #   "auto" -> GeoTIFF if source has georeferencing, else PNG
    "OUTPUT_BANDS": None,          # None = keep all source bands; or a list of 1-indexed band numbers,
                                    # e.g. [1, 2, 3] to force RGB-only output from a multi-band source
    "PNG_STRETCH_PERCENTILES": [2.0, 98.0],   # used only when writing PNG from >8-bit source data

    # ── Quality filters (set to disable: MAX_NODATA_FRACTION=1.0, MIN_VARIANCE=0.0) ─
    "NODATA_VALUE": 0,
    "MAX_NODATA_FRACTION": 1.0,    # 1.0 = no filtering; 0.05 = discard tiles >5% nodata
    "MIN_VARIANCE": 0.0,           # 0.0 = no filtering; raise to discard flat/blank tiles

    # ── I/O tuning ───────────────────────────────────────────────────────
    "GDAL_CACHE_MB": 256,          # caps GDAL's internal block-read cache (bounded peak memory)
}


# ─────────────────────────────────────────────────────────────────────────────
# DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_images(input_path: Path, extensions: List[str], recursive: bool) -> List[Path]:
    """Return the list of image files to process."""
    exts = {e.lower() for e in extensions}

    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    pattern_fn = input_path.rglob if recursive else input_path.glob
    files = sorted(
        p for p in pattern_fn("*")
        if p.is_file() and p.suffix.lower() in exts
    )
    return files


# ─────────────────────────────────────────────────────────────────────────────
# TILE WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def _percentile_stretch_to_uint8(data: np.ndarray, low_p: float, high_p: float) -> np.ndarray:
    """(bands, H, W) any dtype -> (bands, H, W) uint8, per-call global percentile stretch."""
    out = np.empty(data.shape, dtype=np.uint8)
    flat = data.astype(np.float32)
    lo = np.percentile(flat, low_p)
    hi = np.percentile(flat, high_p)
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((flat - lo) / (hi - lo) * 255.0, 0, 255)
    out[:] = scaled.astype(np.uint8)
    return out


def _write_png(data: np.ndarray, out_path: Path, cfg: dict) -> None:
    """Write a (bands, H, W) array as a PNG. Handles 1, 3, or 4-band uint8/uint16 input."""
    if not _HAS_PIL:
        raise RuntimeError("Writing PNG tiles requires Pillow: pip install pillow")

    if data.dtype != np.uint8:
        low_p, high_p = cfg["PNG_STRETCH_PERCENTILES"]
        data = _percentile_stretch_to_uint8(data, low_p, high_p)

    bands = data.shape[0]
    arr = np.transpose(data, (1, 2, 0))  # (H, W, bands)

    if bands == 1:
        img = Image.fromarray(arr[:, :, 0], mode="L")
    elif bands == 3:
        img = Image.fromarray(arr, mode="RGB")
    elif bands == 4:
        img = Image.fromarray(arr, mode="RGBA")
    else:
        # Unusual band count for PNG (e.g. multispectral) — save first 3 bands as RGB
        # and warn, rather than silently producing a broken image.
        logging.warning(
            "PNG output only supports 1/3/4 bands; source has %d — writing first 3 as RGB (%s)",
            bands, out_path.name,
        )
        img = Image.fromarray(arr[:, :, :3], mode="RGB")

    img.save(out_path)


def _write_geotiff(
    data: np.ndarray,
    out_path: Path,
    src_profile: dict,
    window: Window,
    src_transform,
) -> None:
    """Write a (bands, H, W) array as a georeferenced GeoTIFF tile."""
    profile = src_profile.copy()
    profile.update(
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        transform=window_transform(window, src_transform),
        compress=profile.get("compress", "deflate"),
        tiled=False,
    )
    # BIGTIFF / blockxsize etc. from the source profile can be invalid for a
    # small tile-sized output; strip anything that only makes sense at full-scene scale.
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)


# ─────────────────────────────────────────────────────────────────────────────
# CORE TILING
# ─────────────────────────────────────────────────────────────────────────────

def tile_one_image(path: Path, output_dir: Path, cfg: dict) -> Dict[str, int]:
    """
    Slide a window over one image and write out tile_size x tile_size crops.
    Reads are windowed (rasterio) — the source is never loaded in full.

    Returns a stats dict: {"saved", "skipped_edge", "skipped_nodata", "skipped_variance"}.
    """
    tile_size = cfg["TILE_SIZE"]
    stride = cfg["STRIDE"]
    nodata_value = cfg["NODATA_VALUE"]
    max_nodata_frac = cfg["MAX_NODATA_FRACTION"]
    min_variance = cfg["MIN_VARIANCE"]
    drop_incomplete = cfg["DROP_INCOMPLETE_EDGE_TILES"]
    band_selection = cfg["OUTPUT_BANDS"]

    stats = {"saved": 0, "skipped_edge": 0, "skipped_nodata": 0, "skipped_variance": 0}
    stem = path.stem

    try:
        src = rasterio.open(path)
    except Exception as exc:
        logging.error("Could not open '%s': %s", path, exc)
        return stats

    with src:
        height, width = src.height, src.width
        bands = band_selection if band_selection else list(range(1, src.count + 1))

        has_geo = (
            src.crs is not None
            and src.transform is not None
            and src.transform != rasterio.Affine.identity()
        )
        out_fmt = cfg["OUTPUT_FORMAT"]
        if out_fmt == "auto":
            out_fmt = "tif" if has_geo else "png"
        if out_fmt == "tif" and not has_geo:
            logging.info(
                "'%s' has no georeferencing — writing plain (non-geo) TIFF tiles.", path.name
            )

        row_starts = list(range(0, height, stride))
        col_starts = list(range(0, width, stride))
        total = len(row_starts) * len(col_starts)

        src_profile = src.profile.copy()
        patch_num = 0  # for naming tiles in a way that matches HR/LR pairs, e.g., patch000000.png, patch000001.png, etc.
        for row in tqdm(row_starts, desc=f"{path.name} (rows)", leave=False):
            for col in col_starts:
                h = min(tile_size, height - row)
                w = min(tile_size, width - col)
                incomplete = h < tile_size or w < tile_size

                if incomplete and drop_incomplete:
                    stats["skipped_edge"] += 1
                    continue

                window = Window(col, row, w, h)
                data = src.read(bands, window=window)  # (n_bands, h, w)

                if incomplete and not drop_incomplete:
                    # Zero/nodata-pad up to tile_size so every output tile is uniform.
                    padded = np.full(
                        (data.shape[0], tile_size, tile_size), nodata_value, dtype=data.dtype
                    )
                    padded[:, :h, :w] = data
                    data = padded

                # ── Quality gate: nodata fraction ──────────────────────────
                if max_nodata_frac < 1.0:
                    nodata_mask = np.all(data == nodata_value, axis=0)
                    if nodata_mask.mean() > max_nodata_frac:
                        stats["skipped_nodata"] += 1
                        continue

                # ── Quality gate: variance (flat/blank tiles) ──────────────
                if min_variance > 0.0:
                    variance = float(np.var(data.astype(np.float32)))
                    if variance < min_variance:
                        stats["skipped_variance"] += 1
                        continue

                # tile_name = f"{stem}_r{row:06d}_c{col:06d}"
                tile_name = f"{stem}_patch{patch_num:06d}" # New naming convention for tiles, such that names of HR and LR tiles match for the same scene, e.g., 1040010076369100-visual_patch000000.png
                patch_num += 1
                if out_fmt == "tif":
                    out_path = output_dir / f"{tile_name}.tif"
                    _write_geotiff(data, out_path, src_profile, window, src.transform)
                else:
                    out_path = output_dir / f"{tile_name}.png"
                    _write_png(data, out_path, cfg)

                stats["saved"] += 1

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_cli_config() -> dict:
    parser = argparse.ArgumentParser(
        description="Tile satellite imagery (GeoTIFF/TIFF/etc.) into fixed-size square tiles."
    )
    parser.add_argument("--input", type=str, default=None, help="Input file or directory.")
    parser.add_argument("--output", type=str, default=None, help="Output directory for tiles.")
    parser.add_argument("--tile-size", type=int, default=None, help="Tile size in pixels (default 256).")
    parser.add_argument("--stride", type=int, default=None, help="Stride between tiles (default = tile-size, i.e. no overlap).")
    parser.add_argument("--format", type=str, default=None, choices=["auto", "tif", "png"], help="Output tile format.")
    parser.add_argument("--recursive", action="store_true", default=None, help="Recurse into subdirectories.")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Do not recurse into subdirectories.")
    parser.add_argument("--max-nodata-fraction", type=float, default=None, help="Discard tiles with more than this fraction of nodata pixels (0-1).")
    parser.add_argument("--min-variance", type=float, default=None, help="Discard tiles with pixel variance below this value.")
    parser.add_argument("--keep-edge-tiles", action="store_true", default=None, help="Zero-pad incomplete edge tiles instead of dropping them.")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    if args.input is not None:
        cfg["INPUT_PATH"] = args.input
    if args.output is not None:
        cfg["OUTPUT_DIR"] = args.output
    if args.tile_size is not None:
        cfg["TILE_SIZE"] = args.tile_size
    cfg["STRIDE"] = args.stride if args.stride is not None else cfg.get("STRIDE") or cfg["TILE_SIZE"]
    if args.format is not None:
        cfg["OUTPUT_FORMAT"] = args.format
    if args.recursive is not None:
        cfg["RECURSIVE"] = args.recursive
    if args.max_nodata_fraction is not None:
        cfg["MAX_NODATA_FRACTION"] = args.max_nodata_fraction
    if args.min_variance is not None:
        cfg["MIN_VARIANCE"] = args.min_variance
    if args.keep_edge_tiles:
        cfg["DROP_INCOMPLETE_EDGE_TILES"] = False

    return cfg


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = build_cli_config()

    if not cfg["INPUT_PATH"]:
        logging.error("No input path given. Use --input <file_or_dir>, or set INPUT_PATH in CONFIG.")
        sys.exit(1)

    input_path = Path(cfg["INPUT_PATH"])
    output_dir = Path(cfg["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    assert cfg["TILE_SIZE"] > 0, "TILE_SIZE must be positive"
    assert cfg["STRIDE"] > 0, "STRIDE must be positive"

    images = discover_images(input_path, cfg["SUPPORTED_EXTENSIONS"], cfg["RECURSIVE"])
    if not images:
        logging.error(
            "No images found under '%s' with extensions %s", input_path, cfg["SUPPORTED_EXTENSIONS"]
        )
        sys.exit(1)

    logging.info("Found %d image(s) to tile.", len(images))
    logging.info(
        "tile_size=%d  stride=%d  output_format=%s  output_dir=%s",
        cfg["TILE_SIZE"], cfg["STRIDE"], cfg["OUTPUT_FORMAT"], output_dir,
    )

    gdal_cache_mb = cfg.get("GDAL_CACHE_MB", 256)
    totals = {"saved": 0, "skipped_edge": 0, "skipped_nodata": 0, "skipped_variance": 0}

    with rasterio.Env(GDAL_CACHEMAX=gdal_cache_mb * 1024 * 1024):
        for img_path in tqdm(images, desc="Scenes", unit="scene"):
            stats = tile_one_image(img_path, output_dir, cfg)
            for k in totals:
                totals[k] += stats[k]
            logging.info(
                "  %-45s saved=%-6d skipped(edge=%d nodata=%d variance=%d)",
                img_path.name, stats["saved"], stats["skipped_edge"],
                stats["skipped_nodata"], stats["skipped_variance"],
            )

    logging.info("=" * 60)
    logging.info(
        "Done. %d tile(s) written to: %s", totals["saved"], output_dir
    )
    logging.info(
        "Skipped — edge: %d, nodata: %d, variance: %d",
        totals["skipped_edge"], totals["skipped_nodata"], totals["skipped_variance"],
    )


if __name__ == "__main__":
    main()
