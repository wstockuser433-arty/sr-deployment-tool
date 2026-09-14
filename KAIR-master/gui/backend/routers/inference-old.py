"""
routers/inference.py
====================
API endpoints for SwinIR inference.
"""
import asyncio
import json
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

# Script that accepts CONFIG injected as env vars — we write a temp config file
INFERENCE_SCRIPT = PROJECT_ROOT / "main_test_swinir_config.py"
INFERENCE_WRAPPER = TMP_DIR / "run_inference.py"
RAW_INFERENCE_SCRIPT = PROJECT_ROOT / "raw_inference.py"


def _write_inference_script(req: StartInferenceRequest) -> Path:
    """
    Write a self-contained inference launcher that overrides CONFIG and MODEL_CONFIG
    and calls main_test_swinir_config.main().
    """
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


def _build_raw_config(
    req,
    mode: str,
) -> tuple[dict, Path]:
    """
    Build the raw_inference.py JSON config from a request body and write it
    to a temp file.  Returns (config_dict, config_path).
    """
    mc = req.model_network_config
    model_cfg = {
        "upscale":          mc.upscale,
        "in_chans":         mc.in_chans,
        "img_size":         mc.img_size,
        "window_size":      mc.window_size,
        "img_range":        mc.img_range,
        "depths":           mc.depths,
        "embed_dim":        mc.embed_dim,
        "num_heads":        mc.num_heads,
        "mlp_ratio":        mc.mlp_ratio,
        "upsampler":        mc.upsampler,
        "resi_connection":  mc.resi_connection,
    }

    cfg: dict = {
        "mode":        mode,
        "model_path":  req.model_path,
        "output_dir":  req.output_dir,
        "patch_size":  req.patch_size,
        "overlap":     req.overlap,
        "scale_factor": req.scale_factor,
        "model_config": model_cfg,
    }

    if mode == "paired":
        cfg["lr_path"]  = req.lr_path
        cfg["hr_path"]  = req.hr_path
        cfg["lr_bands"] = req.lr_bands
        cfg["hr_bands"] = req.hr_bands
        # Merge CoregConfig fields into top-level cfg
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
        cfg["lr_path"]  = req.lr_path
        cfg["lr_bands"] = req.lr_bands

    # Write to tmp
    import hashlib
    cfg_hash  = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    cfg_path  = TMP_DIR / f"raw_inference_{mode}_{cfg_hash}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg, cfg_path


@router.get("/tasks")
def list_tasks():
    """List all trained task names from superresolution/."""
    return config_service.list_training_runs()


@router.get("/latest-model/{task_name}")
def get_latest_model(task_name: str):
    """Return the latest model path + autofilled MODEL_CONFIG for a task."""
    try:
        return config_service.get_latest_model_info(task_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/config-from-options/{options_name}")
def get_config_from_options(options_name: str):
    """Extract MODEL_CONFIG fields from a named options/swinir JSON file."""
    try:
        model_config = config_service.get_model_config_from_options(options_name)
        return {"source": "options_file", "options_name": options_name, "model_config": model_config}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/config-from-path")
def get_config_from_path(path: str):
    """
    Auto-detect MODEL_CONFIG from a checkpoint path.
    If the path contains superresolution/<task>/..., loads that task's options/train.json.
    Returns {source, model_config} — source is 'train_json' or 'not_found'.
    """
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


@router.post("/start", response_model=JobResponse)
async def start_inference(req: StartInferenceRequest):
    """Launch the patch-based inference subprocess."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Raw satellite image inference endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/raw-paired/start", response_model=JobResponse)
async def start_raw_paired_inference(req: RawPairedInferenceRequest):
    """
    Launch a paired (LR + HR) raw image inference job.
    Optionally runs pipeline3-style coregistration + radiometric normalisation
    before patch extraction, then stitches SR output and computes metrics.
    """
    if not RAW_INFERENCE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="raw_inference.py not found in project root.")

    _, cfg_path = _build_raw_config(req, mode="paired")
    cmd    = [sys.executable, str(RAW_INFERENCE_SCRIPT), "--config", str(cfg_path)]
    job_id = job_manager.create_job(
        cmd=cmd,
        cwd=str(PROJECT_ROOT),
        meta={"output_dir": req.output_dir, "mode": "paired"},
    )
    asyncio.create_task(job_manager.launch_job(job_id))
    return JobResponse(job_id=job_id, status="pending")


@router.post("/lr-only/start", response_model=JobResponse)
async def start_lr_only_inference(req: LROnlyInferenceRequest):
    """
    Launch an LR-only raw image inference job.
    No HR ground truth — SR is computed and saved; no metrics are produced.
    """
    if not RAW_INFERENCE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="raw_inference.py not found in project root.")

    _, cfg_path = _build_raw_config(req, mode="lr_only")
    cmd    = [sys.executable, str(RAW_INFERENCE_SCRIPT), "--config", str(cfg_path)]
    job_id = job_manager.create_job(
        cmd=cmd,
        cwd=str(PROJECT_ROOT),
        meta={"output_dir": req.output_dir, "mode": "lr_only"},
    )
    asyncio.create_task(job_manager.launch_job(job_id))
    return JobResponse(job_id=job_id, status="pending")


@router.get("/preview/{job_id}/{filename}")
def get_inference_preview_image(job_id: str, filename: str):
    """
    Serve a "5-layer" visual-assessment preview (grid/FFT/error/residual JPEG)
    written by main_test_swinir_config.py or raw_inference.py to
    <output_dir>/_previews/. Mirrors preprocessing's preview endpoint;
    filename is restricted to a bare name (no path separators/traversal) and
    the resolved path must stay inside that job's _previews directory.
    """
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
    """
    Serve a result PNG — display composites or per-band grayscale images.
    Accepted: lr_display.png, sr_display.png, hr_display.png,
              lr_band_N.png, sr_band_N.png, hr_band_N.png
    """
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
    """
    Return the metrics.json produced by a paired raw inference job.
    """
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


@router.get("/image-info")
def get_image_info(path: str = Query(..., description="Absolute or project-relative path to image file")):
    """
    Return band count and pixel dimensions for a satellite or standard image.
    Tries rasterio first (for GeoTIFF/JP2), falls back to cv2.
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
            return {
                "bands": src.count,
                "width": src.width,
                "height": src.height,
                "format": "geospatial",
                "crs": str(src.crs) if src.crs else None,
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
            return {"bands": b, "width": w, "height": h, "format": "standard", "crs": None}
    except Exception:
        pass

    raise HTTPException(status_code=400, detail="Could not read image metadata.")


@router.get("/image-compare")
def compare_images(hr: str = Query(...), lr: str = Query(...)):
    """
    Compute PSNR between an HR and LR image pair.
    Both are resampled to at most 1024×1024 for speed.
    LR is upsampled to HR spatial size before comparison.
    """
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

    # Upsample LR to HR spatial size for comparison
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
        "psnr": psnr,
        "bands_compared": n,
        "hr_bands": hr_bands, "hr_width": hr_w, "hr_height": hr_h,
        "lr_bands": lr_bands, "lr_width": lr_w, "lr_height": lr_h,
    }


