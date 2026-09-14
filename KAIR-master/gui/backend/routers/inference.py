"""
routers/inference.py
====================
API endpoints for SwinIR inference.
"""
import asyncio
import json
import math
import re
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse

from ..schemas.inference import (
    StartInferenceRequest, JobResponse,
    RawPairedInferenceRequest, LROnlyInferenceRequest,
)
from ..services import config_service, job_manager

router = APIRouter(prefix="/api/inference", tags=["inference"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TMP_DIR = Path(__file__).resolve().parents[1] / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

INFERENCE_SCRIPT = PROJECT_ROOT / "main_test_swinir_config.py"
INFERENCE_WRAPPER = TMP_DIR / "run_inference.py"
RAW_INFERENCE_SCRIPT = PROJECT_ROOT / "raw_inference.py"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_inference_script(req: StartInferenceRequest) -> Path:
    """Write a self-contained inference launcher that overrides CONFIG and MODEL_CONFIG."""
    mc = req.model_network_config
    script = f"""
import sys
sys.path.insert(0, r'{PROJECT_ROOT}')
import main_test_swinir_config as m

m.CONFIG = {{
    "model_path": r'{req.model_path}',
    "lr_dir": r'{req.lr_dir}',
    "hr_dir": r'{req.hr_dir}',
    "sr_dir": r'{req.sr_dir}',
    "tile": {repr(req.tile)},
    "tile_overlap": {req.tile_overlap},
    "overwrite_sr": {req.overwrite_sr},
    "log_dir": r'{req.log_dir}',
    "visual_report_enabled": {req.visual_report_enabled},
    "visual_report_samples": {req.visual_report_samples},
    "visual_report_seed": {req.visual_report_seed},
}}
m.MODEL_CONFIG = {{
    "upscale": {mc.upscale},
    "in_chans": {mc.in_chans},
    "img_size": {mc.img_size},
    "window_size": {mc.window_size},
    "img_range": {mc.img_range},
    "depths": {mc.depths},
    "embed_dim": {mc.embed_dim},
    "num_heads": {mc.num_heads},
    "mlp_ratio": {mc.mlp_ratio},
    "upsampler": '{mc.upsampler}',
    "resi_connection": '{mc.resi_connection}',
}}
m.main()
"""
    wrapper = TMP_DIR / f"run_inference_{hash(req.model_path) & 0xFFFFFF}.py"
    wrapper.write_text(script.strip(), encoding="utf-8")
    return wrapper


def _build_raw_config(req, mode: str) -> tuple[dict, Path]:
    """Build the raw_inference.py JSON config from a request body and write it to a temp file."""
    mc = req.model_network_config
    model_cfg = {
        "upscale": mc.upscale, "in_chans": mc.in_chans, "img_size": mc.img_size,
        "window_size": mc.window_size, "img_range": mc.img_range,
        "depths": mc.depths, "embed_dim": mc.embed_dim, "num_heads": mc.num_heads,
        "mlp_ratio": mc.mlp_ratio, "upsampler": mc.upsampler,
        "resi_connection": mc.resi_connection,
    }

    cfg: dict = {
        "mode": mode, "model_path": req.model_path, "output_dir": req.output_dir,
        "patch_size": req.patch_size, "overlap": req.overlap,
        "scale_factor": req.scale_factor, "model_config": model_cfg,
    }

    if mode == "paired":
        cfg["lr_path"] = req.lr_path
        cfg["hr_path"] = req.hr_path
        cfg["lr_bands"] = req.lr_bands
        cfg["hr_bands"] = req.hr_bands
        coreg = req.coreg
        cfg.update({
            "enable_preprocessing":        coreg.enable_preprocessing,
            "coreg_a_enabled":             coreg.coreg_a_enabled,
            "coreg_a_max_features":        coreg.coreg_a_max_features,
            "coreg_a_match_ratio":         coreg.coreg_a_match_ratio,
            "coreg_a_ransac_thresh":       coreg.coreg_a_ransac_thresh,
            "coreg_b_enabled":             coreg.coreg_b_enabled,
            "coreg_b_upsample_factor":     coreg.coreg_b_upsample_factor,
            "radiometric_enabled":         coreg.radiometric_enabled,
            "radiometric_block_size":      coreg.radiometric_block_size,
            "radiometric_rmse_threshold":  coreg.radiometric_rmse_threshold,
            "radiometric_n_samples":       coreg.radiometric_n_samples,
            "radiometric_n_fit_windows":   coreg.radiometric_n_fit_windows,
            "radiometric_post_hist_match": coreg.radiometric_post_hist_match,
            "histogram_n_sample_windows":  coreg.histogram_n_sample_windows,
            "nodata_value":                coreg.nodata_value,
            "saturated_value":             coreg.saturated_value,
            "clip_percentiles":            coreg.clip_percentiles,
            "percentile_n_sample_windows": coreg.percentile_n_sample_windows,
            "coreg_preview_decim_dim":     coreg.coreg_preview_decim_dim,
        })
        cfg.update({
            "visual_report_enabled": req.visual_report_enabled,
            "visual_report_samples": req.visual_report_samples,
            "visual_report_seed":    req.visual_report_seed,
        })
    else:  # lr_only
        cfg["lr_path"] = req.lr_path
        cfg["lr_bands"] = req.lr_bands

    import hashlib
    cfg_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    cfg_path = TMP_DIR / f"raw_inference_{mode}_{cfg_hash}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg, cfg_path


# ─────────────────────────────────────────────────────────────────────────────
# Task / model discovery
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/tasks")
def list_tasks():
    return config_service.list_training_runs()


@router.get("/latest-model/{task_name}")
def get_latest_model(task_name: str):
    try:
        return config_service.get_latest_model_info(task_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/config-from-options/{options_name}")
def get_config_from_options(options_name: str):
    try:
        model_config = config_service.get_model_config_from_options(options_name)
        return {"source": "options_file", "options_name": options_name, "model_config": model_config}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/config-from-path")
def get_config_from_path(path: str):
    try:
        parts = Path(path).parts
    except Exception:
        return {"source": "not_found", "model_config": {}}
    try:
        sr_idx = next(i for i, p in enumerate(parts) if p.lower() == "superresolution")
    except StopIteration:
        return {"source": "not_found", "model_config": {}}
    if sr_idx + 1 >= len(parts):
        return {"source": "not_found", "model_config": {}}
    task_name = parts[sr_idx + 1]
    train_json = config_service.SUPERRESOLUTION_DIR / task_name / "options" / "train.json"
    if not train_json.exists():
        return {"source": "not_found", "model_config": {}}
    try:
        cfg = config_service.load_kair_json(train_json)
        return {
            "source": "train_json",
            "task_name": task_name,
            "model_config": config_service._extract_model_config(cfg),
        }
    except Exception:
        return {"source": "not_found", "model_config": {}}


# ─────────────────────────────────────────────────────────────────────────────
# Image metadata (expanded)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/image-info")
def get_image_info(path: str = Query(..., description="Absolute or project-relative path to image file")):
    """
    Return band count, pixel dimensions, data type, geospatial flag, and GSD
    for a satellite or standard image. Tries rasterio first, falls back to cv2.
    """
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Try rasterio for geospatial formats
    try:
        import rasterio
        with rasterio.open(str(file_path)) as src:
            # Data type — use the first band's dtype
            try:
                dtype = src.dtypes[0]
            except Exception:
                dtype = None

            # GSD — average of x/y resolution if CRS is projected
            gsd = None
            try:
                if src.crs and src.transform:
                    xres = abs(src.transform.a)
                    yres = abs(src.transform.e)
                    # Only report GSD if CRS is projected (has linear units)
                    if src.crs.is_projected:
                        gsd = round((xres + yres) / 2.0, 4)
            except Exception:
                gsd = None

            # Band descriptions (e.g. "Red", "Green", "Blue")
            descriptions = []
            try:
                for i in range(1, src.count + 1):
                    desc = src.descriptions[i - 1]
                    descriptions.append(desc if desc else None)
            except Exception:
                descriptions = [None] * src.count

            return {
                "bands": src.count,
                "width": src.width,
                "height": src.height,
                "format": "GeoTIFF" if str(src.driver).upper() in ("GTIFF", "COG") else str(src.driver),
                "geospatial": True,
                "crs": str(src.crs) if src.crs else None,
                "gsd": gsd,
                "dtype": dtype,
                "band_descriptions": descriptions,
            }
    except Exception:
        pass

    # Fallback: cv2 for standard image formats
    try:
        import cv2 as _cv2
        img = _cv2.imread(str(file_path), _cv2.IMREAD_UNCHANGED)
        if img is not None:
            h, w = img.shape[:2]
            b = img.shape[2] if img.ndim == 3 else 1
            dtype = str(img.dtype)
            return {
                "bands": b, "width": w, "height": h,
                "format": file_path.suffix.lstrip(".").upper() or "Standard",
                "geospatial": False, "crs": None, "gsd": None,
                "dtype": dtype, "band_descriptions": [None] * b,
            }
    except Exception:
        pass

    raise HTTPException(status_code=400, detail="Could not read image metadata.")


# ─────────────────────────────────────────────────────────────────────────────
# Inference summary — estimated output size + patch count
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/inference-summary")
def inference_summary(
    width: int = Query(..., ge=1),
    height: int = Query(..., ge=1),
    patch_size: int = Query(..., ge=8),
    overlap: int = Query(0, ge=0),
    scale: int = Query(..., ge=1, le=8),
):
    """
    Compute estimated output dimensions and patch count for a tiled inference job.
    Uses a standard sliding-window-with-overlap patch count formula.
    """
    stride = max(1, patch_size - overlap)
    n_x = max(1, math.ceil((width - overlap) / stride))
    n_y = max(1, math.ceil((height - overlap) / stride))
    total_patches = n_x * n_y

    return {
        "input_width": width,
        "input_height": height,
        "output_width": width * scale,
        "output_height": height * scale,
        "patch_size": patch_size,
        "overlap": overlap,
        "stride": stride,
        "patches_x": n_x,
        "patches_y": n_y,
        "total_patches": total_patches,
        "scale": scale,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Progress parsing — lightweight regex over recent job logs
# ─────────────────────────────────────────────────────────────────────────────

_PROGRESS_PATCH_RE = re.compile(
    r'[Pp]atch\s+(\d+)\s*/\s*(\d+)'                       # "Patch 904/1156"
    r'|(\d+)\s*/\s*(\d+)\s+patches?'                      # "904/1156 patches"
    r'|\[(\d+)\s*/\s*(\d+)\]'                             # "[904/1156]"
)
_PROGRESS_PCT_RE = re.compile(r'(\d{1,3}(?:\.\d+)?)\s*%')
_STAGE_RE = re.compile(
    r'(?i)\b(coreg|stage a|stage b|orb|phase corr|radiometric|histogram|'
    r'patch|stitch|infer|sr inference|metric|preview|done|complete)\b'
)

_PROGRESS_TOTAL_RE = re.compile(
    r'(?:SR|HR|inference)[:\s]+(\d+)\s+patches'      # "SR: 1156 patches"
    r'|(\d+)\s+patches\s+\(.*stride'                 # "1156 patches (256x256 stride=256)"
)


@router.get("/progress/{job_id}")
def get_inference_progress(job_id: str):
    """
    Return a minimal, structured progress summary for a running inference job,
    derived from the tail of the job's log buffer.
    """
    job = job_manager.get_job(job_id)
    summary = job_manager.get_job_summary(job_id) or {}

    # Unknown job → return a benign "unknown" payload instead of 404.
    # The frontend polls this endpoint briefly right after launching a job
    # (job registration is async) and doesn't need a hard error.
    if job is None:
        return {
            "job_id": job_id,
            "status": "unknown",
            "stage": None,
            "current": 0,
            "total": 0,
            "percent": None,
            "output_dir": None,
            "last_line": "",
        }

    status = summary.get("status", "unknown")

    # Pull recent log lines (whatever job_manager exposes)
    lines: list[str] = []
    try:
        lines = list(job_manager.get_recent_lines(job_id, max_lines=200))
    except Exception:
        # Fallback: some implementations expose .lines directly
        lines = list(getattr(job, "lines", []) or [])[-200:]

    current = 0
    total = 0
    pct = None
    stage = None
    last_line = ""

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        last_line = line

        m = _PROGRESS_PATCH_RE.search(line)
        if m:
            g = [x for x in m.groups() if x is not None]
            if len(g) == 2:
                try:
                    current = int(g[0]); total = int(g[1])
                except ValueError:
                    pass
        if m == 0:
            mt = _PROGRESS_TOTAL_RE.search(line)
            if mt:
                g = [x for x in mt.groups() if x is not None]
                if g:
                    try:
                        total = int(g[0])
                    except ValueError:
                        pass    

        mp = _PROGRESS_PCT_RE.search(line)
        if mp:
            try:
                pct = float(mp.group(1))
            except ValueError:
                pass

        ms = _STAGE_RE.search(line)
        if ms:
            stage = ms.group(1).lower()
        

    if pct is None and total > 0:
        pct = round(100.0 * current / total, 1)

    return {
        "job_id": job_id,
        "status": status,
        "stage": stage,
        "current": current,
        "total": total,
        "percent": pct,
        "output_dir": summary.get("output_dir")
                      or (summary.get("meta") or {}).get("output_dir"),
        "last_line": last_line,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Job control
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/start", response_model=JobResponse)
async def start_inference(req: StartInferenceRequest):
    wrapper = _write_inference_script(req)
    cmd = [sys.executable, str(wrapper)]
    job_id = job_manager.create_job(cmd=cmd, cwd=str(PROJECT_ROOT), output_dir=req.sr_dir)
    asyncio.create_task(job_manager.launch_job(job_id))
    return JobResponse(job_id=job_id, status="pending")


@router.get("/stream/{job_id}")
async def stream_inference_logs(job_id: str):
    return StreamingResponse(
        job_manager.stream_logs(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/status/{job_id}")
def get_status(job_id: str):
    summary = job_manager.get_job_summary(job_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return summary


@router.post("/stop/{job_id}")
def stop_inference(job_id: str):
    if not job_manager.cancel_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "cancelled", "job_id": job_id}


@router.post("/raw-paired/start", response_model=JobResponse)
async def start_raw_paired_inference(req: RawPairedInferenceRequest):
    if not RAW_INFERENCE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="raw_inference.py not found in project root.")
    _, cfg_path = _build_raw_config(req, mode="paired")
    cmd = [sys.executable, str(RAW_INFERENCE_SCRIPT), "--config", str(cfg_path)]
    job_id = job_manager.create_job(
        cmd=cmd, cwd=str(PROJECT_ROOT),
        meta={"output_dir": req.output_dir, "mode": "paired"},
    )
    asyncio.create_task(job_manager.launch_job(job_id))
    return JobResponse(job_id=job_id, status="pending")


@router.post("/lr-only/start", response_model=JobResponse)
async def start_lr_only_inference(req: LROnlyInferenceRequest):
    if not RAW_INFERENCE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="raw_inference.py not found in project root.")
    _, cfg_path = _build_raw_config(req, mode="lr_only")
    cmd = [sys.executable, str(RAW_INFERENCE_SCRIPT), "--config", str(cfg_path)]
    job_id = job_manager.create_job(
        cmd=cmd, cwd=str(PROJECT_ROOT),
        meta={"output_dir": req.output_dir, "mode": "lr_only"},
    )
    asyncio.create_task(job_manager.launch_job(job_id))
    return JobResponse(job_id=job_id, status="pending")


# ─────────────────────────────────────────────────────────────────────────────
# Result serving
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/preview/{job_id}/{filename}")
def get_inference_preview_image(job_id: str, filename: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = job.output_dir or (job.meta or {}).get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=404, detail="Job has no output_dir recorded.")

    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    base = (PROJECT_ROOT / output_dir / "_previews").resolve()
    candidate = (base / filename).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Preview not found")

    return FileResponse(candidate, media_type="image/jpeg")


_RESULT_FILE_RE = re.compile(r'^(lr|sr|hr)_(display|band_[1-9]\d*)\.png$')


@router.get("/raw/result/{job_id}/{filename}")
def get_raw_result_image(job_id: str, filename: str):
    if not _RESULT_FILE_RE.match(filename):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    summary = job_manager.get_job_summary(job_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = (summary.get("meta") or {}).get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=404, detail="Job has no output_dir recorded.")

    file_path = PROJECT_ROOT / output_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not yet available.")

    return FileResponse(str(file_path), media_type="image/png")


@router.get("/raw/metrics/{job_id}")
def get_raw_metrics(job_id: str):
    summary = job_manager.get_job_summary(job_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = (summary.get("meta") or {}).get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=404, detail="Job has no output_dir recorded.")

    metrics_path = PROJECT_ROOT / output_dir / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not yet available.")

    with open(metrics_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/image-compare")
def compare_images(hr: str = Query(...), lr: str = Query(...)):
    import numpy as np

    def load_capped(path_str, max_dim=1024):
        p = Path(path_str)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path_str}")
        try:
            import rasterio
            from rasterio.enums import Resampling
            with rasterio.open(str(p)) as src:
                scale = min(1.0, max_dim / max(src.width, src.height))
                out_w = max(1, int(src.width * scale))
                out_h = max(1, int(src.height * scale))
                data = src.read(
                    out_shape=(src.count, out_h, out_w),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)
                return data, src.count, src.width, src.height
        except Exception:
            pass
        try:
            import cv2
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                scale = min(1.0, max_dim / max(w, h))
                if scale < 1.0:
                    img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_LINEAR)
                arr = (img[np.newaxis] if img.ndim == 2
                       else np.transpose(img, (2, 0, 1))).astype(np.float32)
                return arr, arr.shape[0], w, h
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Could not read: {path_str}")

    hr_arr, hr_bands, hr_w, hr_h = load_capped(hr)
    lr_arr, lr_bands, lr_w, lr_h = load_capped(lr)

    n = min(hr_arr.shape[0], lr_arr.shape[0])
    hr_c = hr_arr[:n]
    lr_c = lr_arr[:n]

    if hr_c.shape[1:] != lr_c.shape[1:]:
        try:
            import cv2
            lr_c = np.stack([
                cv2.resize(lr_c[i], (hr_c.shape[2], hr_c.shape[1]),
                           interpolation=cv2.INTER_LINEAR)
                for i in range(n)
            ])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Resize failed: {exc}")

    def norm01(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

    mse = float(np.mean((norm01(hr_c) - norm01(lr_c)) ** 2))
    psnr = round(float(10 * np.log10(1.0 / mse)) if mse > 0 else 100.0, 2)

    return {
        "psnr": psnr, "bands_compared": n,
        "hr_bands": hr_bands, "hr_width": hr_w, "hr_height": hr_h,
        "lr_bands": lr_bands, "lr_width": lr_w, "lr_height": lr_h,
    }


