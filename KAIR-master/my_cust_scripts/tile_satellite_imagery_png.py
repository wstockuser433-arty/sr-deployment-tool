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
  - Output tiles format can be configured explicitly (e.g. png, jpg, tif)
    or set to "auto" (GeoTIFF if source has georeferencing, else PNG).
  - Optional quality gates: discard tiles that are mostly nodata, or nearly
    blank/flat (low variance) — same idea as MAX_NODATA_FRACTION /
    MIN_VARIANCE in pipeline3.py, so tiles produced here are compatible
    with the same downstream SR training expectations.
  - Recursively discovers images under a directory, or accepts a single
    file.

Usage
-----
    python tile_satellite_imagery.py --input path/to/scene.tif --output tiles_out --format png
    python tile_satellite_imagery.py --input path/to/scene_dir --output tiles_out --recursive --tile-size 512
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
    "INPUT_PATH": "/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/maxar-dataset-pak/hurricane-fiona-48cm/105001002C5F6600-ms.tif", 
    "OUTPUT_DIR": "/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/img-patch-tiles/HR/tiles_512/hurricane-fiona-48cm",  

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
    "OUTPUT_FORMAT": "png",        # "png" | "jpg" | "tif" | "auto"
                                    #   "png" / "jpg" -> force image format
                                    #   "tif" -> force GeoTIFF output
                                    #   "auto" -> GeoTIFF if source has real georeferencing, else PNG
    "OUTPUT_BANDS": [3, 2, 1],          # None = keep all source bands; or a list of 1-indexed band numbers,
                                    # e.g. [1, 2, 3] to force RGB-only output from a multi-band source
    "PNG_STRETCH_PERCENTILES": [2.0, 98.0],   # used only when writing PNG/JPG from >8-bit source data
    "RESCALE_TIF": True,         # if True, write GeoTIFF tiles rescaled to uint8 for visualization
    "PER_BAND_STRETCH": True,    # if True, scale each band independently (reduces color cast when bands differ)

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

def _percentile_stretch_to_uint8(data: np.ndarray, low_p: float, high_p: float, per_band: bool = False) -> np.ndarray:
    """(bands, H, W) any dtype -> (bands, H, W) uint8.
    If per_band is False: global percentile stretch across all bands (original behavior).
    If per_band is True: compute percentiles per-band and scale each band independently.
    """
    data_f = data.astype(np.float32)
    out = np.empty(data.shape, dtype=np.uint8)

    if per_band:
        for b in range(data.shape[0]):
            band = data_f[b]
            lo = np.percentile(band, low_p)
            hi = np.percentile(band, high_p)
            if hi <= lo:
                hi = lo + 1.0
            scaled = np.clip((band - lo) / (hi - lo) * 255.0, 0, 255)
            out[b] = scaled.astype(np.uint8)
    else:
        flat = data_f
        lo = np.percentile(flat, low_p)
        hi = np.percentile(flat, high_p)
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((flat - lo) / (hi - lo) * 255.0, 0, 255)
        out[:] = scaled.astype(np.uint8)

    return out


def _write_pil_image(data: np.ndarray, out_path: Path, cfg: dict) -> None:
    """Write a (bands, H, W) array using Pillow (supports PNG, JPG, BMP, etc.)."""
    if not _HAS_PIL:
        raise RuntimeError("Writing non-TIFF image tiles requires Pillow: pip install pillow")

    if data.dtype != np.uint8:
        low_p, high_p = cfg["PNG_STRETCH_PERCENTILES"]
        data = _percentile_stretch_to_uint8(data, low_p, high_p, per_band=cfg.get("PER_BAND_STRETCH", False))

    bands = data.shape[0]
    arr = np.transpose(data, (1, 2, 0))  # (H, W, bands)
    ext = out_path.suffix.lower()

    if bands == 1:
        img = Image.fromarray(arr[:, :, 0], mode="L")
    elif bands == 3 or (bands == 4 and ext in [".jpg", ".jpeg"]):
        # JPEG does not support alpha channel — slice to 3 channels
        img = Image.fromarray(arr[:, :, :3], mode="RGB")
    elif bands == 4:
        img = Image.fromarray(arr, mode="RGBA")
    else:
        # Unusual band count for standard images (e.g. multispectral)
        logging.warning(
            "Image output only supports 1/3/4 bands; source has %d — writing first 3 as RGB (%s)",
            bands, out_path.name,
        )
        img = Image.fromarray(arr[:, :, :3], mode="RGB")

    if ext in [".jpg", ".jpeg"] and img.mode == "RGBA":
        img = img.convert("RGB")

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
    # Strip full-scene size configs invalid for small tiles
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
    """
    tile_size = cfg["TILE_SIZE"]
    stride = cfg["STRIDE"]
    nodata_value = cfg["NODATA_VALUE"]
    max_nodata_frac = cfg["MAX_NODATA_FRACTION"]
    min_variance = cfg["MIN_VARIANCE"]
    drop_incomplete = cfg["DROP_INCOMPLETE_EDGE_TILES"]
    band_selection = cfg["OUTPUT_BANDS"]
    rescale_tif = cfg.get("RESCALE_TIF", False)

    stats = {"saved": 0, "skipped_edge": 0, "skipped_nodata": 0, "skipped_variance": 0}
    stem = path.stem

    try:
        src = rasterio.open(path)
    except Exception as exc:
        logging.error("Could not open '%s': %s", path, exc)
        return stats

    with src:
        height, width = src.height, src.width
        # Normalize band selection (1-based indices expected in cfg)
        if band_selection:
            bands = list(band_selection)
            for bi in bands:
                if bi < 1 or bi > src.count:
                    logging.error("Requested band %d out of range (1..%d) for %s", bi, src.count, path.name)
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
        
        if out_fmt in ["tif", "tiff"] and not has_geo:
            logging.info(
                "'%s' has no georeferencing — writing plain (non-geo) TIFF tiles.", path.name
            )

        ext = f".{out_fmt}" if not out_fmt.startswith(".") else out_fmt

        row_starts = list(range(0, height, stride))
        col_starts = list(range(0, width, stride))

        src_profile = src.profile.copy()
        patch_num = 0
        for row in tqdm(row_starts, desc=f"{path.name} (rows)", leave=False):
            for col in col_starts:
                h = min(tile_size, height - row)
                w = min(tile_size, width - col)
                incomplete = h < tile_size or w < tile_size

                if incomplete and drop_incomplete:
                    stats["skipped_edge"] += 1
                    continue

                window = Window(col, row, w, h)
                # Read as masked array to let rasterio mark nodata pixels
                data = src.read(bands, window=window, masked=True)  # (n_bands, h, w) MaskedArray

                if incomplete and not drop_incomplete:
                    filled = data.filled(nodata_value)
                    padded = np.full((filled.shape[0], tile_size, tile_size), nodata_value, dtype=filled.dtype)
                    padded[:, :h, :w] = filled
                    data = np.ma.MaskedArray(padded, mask=np.zeros_like(padded, dtype=bool))

                # ── Quality gate: nodata fraction ──────────────────────────
                if max_nodata_frac < 1.0:
                    if isinstance(data, np.ma.MaskedArray):
                        nodata_mask = np.all(data.mask, axis=0)
                    else:
                        nodata_mask = np.all(data == nodata_value, axis=0)
                    if nodata_mask.mean() > max_nodata_frac:
                        stats["skipped_nodata"] += 1
                        continue

                # ── Quality gate: variance (flat/blank tiles) ──────────────
                if min_variance > 0.0:
                    if isinstance(data, np.ma.MaskedArray):
                        variance = float(np.ma.var(data.astype(np.float32)))
                    else:
                        variance = float(np.var(data.astype(np.float32)))
                    if variance < min_variance:
                        stats["skipped_variance"] += 1
                        continue

                # tile_name = f"{stem}_r{row:06d}_c{col:06d}" # Uncomment it if you want to keep the original naming convention, row and column numbers in the tile name
                tile_name = f"{stem}_patch{patch_num:06d}" # New naming convention for tiles, such that names of HR and LR tiles match for the same scene, e.g., 1040010076369100-visual_patch000000.png
                out_path = output_dir / f"{tile_name}{ext}"
                patch_num += 1
                if ext in [".tif", ".tiff"]:
                    if rescale_tif:
                        # Visualize by rescaling to uint8 using percentile stretch
                        filled = data.filled(nodata_value) if isinstance(data, np.ma.MaskedArray) else data
                        uint8 = _percentile_stretch_to_uint8(
                            filled,
                            cfg["PNG_STRETCH_PERCENTILES"][0],
                            cfg["PNG_STRETCH_PERCENTILES"][1],
                            per_band=cfg.get("PER_BAND_STRETCH", False),
                        )
                        profile = src_profile.copy()
                        profile.update(
                            driver="GTiff",
                            height=uint8.shape[1],
                            width=uint8.shape[2],
                            count=uint8.shape[0],
                            dtype=rasterio.uint8,
                            transform=window_transform(window, src.transform),
                            compress=profile.get("compress", "deflate"),
                            tiled=False,
                        )
                        profile.pop("blockxsize", None)
                        profile.pop("blockysize", None)
                        with rasterio.open(out_path, "w", **profile) as dst:
                            dst.write(uint8)
                    else:
                        _write_geotiff(data.filled(nodata_value) if isinstance(data, np.ma.MaskedArray) else data, out_path, src_profile, window, src.transform)
                else:
                    to_write = data.filled(nodata_value) if isinstance(data, np.ma.MaskedArray) else data
                    _write_pil_image(to_write, out_path, cfg)

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
    parser.add_argument("--stride", type=int, default=None, help="Stride between tiles (default = tile-size).")
    parser.add_argument("--format", type=str, default=None, choices=["auto", "tif", "tiff", "png", "jpg", "jpeg"], help="Output tile format.")
    parser.add_argument("--bands", type=str, default=None, help="Comma-separated 1-based band indices to output, e.g. 3,2,1 for RGB.")
    parser.add_argument("--rescale", action="store_true", default=None, help="Rescale output GeoTIFF tiles to uint8 for visualization.")
    parser.add_argument("--per-band-stretch", action="store_true", default=None, help="Apply percentile stretch per-band instead of globally.")
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
    if args.bands is not None:
        try:
            cfg["OUTPUT_BANDS"] = [int(x) for x in args.bands.split(",") if x.strip()]
        except Exception:
            raise ValueError("Invalid --bands value; expect comma-separated integers like 3,2,1")
    if args.rescale:
        cfg["RESCALE_TIF"] = True
    if args.per_band_stretch:
        cfg["PER_BAND_STRETCH"] = True

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