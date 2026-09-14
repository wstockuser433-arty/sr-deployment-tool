"""
routers/training.py
===================
API endpoints for SwinIR training (PSNR and GAN).
"""
import asyncio
import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..schemas.training import StartTrainingRequest, JobResponse
from ..services import config_service, job_manager

router = APIRouter(prefix="/api/training", tags=["training"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TMP_DIR = Path(__file__).resolve().parents[1] / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/configs")
def list_configs():
    """List available SwinIR training config files."""
    return config_service.list_swinir_configs()


@router.get("/config/{config_name}")
def get_config(config_name: str):
    """Load and return a parsed SwinIR config."""
    try:
        return config_service.load_swinir_config(config_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/runs")
def list_runs():
    """List recent training runs from superresolution/."""
    return config_service.list_training_runs()


def _build_config_dict(req: StartTrainingRequest) -> dict:
    """Convert the request model into a KAIR-compatible JSON config dict."""
    cfg: dict = {
        "task": req.task,
        "model": "gan" if req.training_type == "gan" else "plain",
        "gpu_ids": req.gpu_ids,
        "dist": False,
        "scale": req.scale,
        "n_channels": req.n_channels,
        "path": {
            "root": "superresolution",
            "pretrained_netG": req.pretrained_netG,
            "pretrained_netE": req.pretrained_netE,
        },
        "datasets": {
            "train": {
                "name": "train_dataset",
                "dataset_type": req.train_dataset.dataset_type,
                "dataroot_H": req.train_dataset.dataroot_H,
                "dataroot_L": req.train_dataset.dataroot_L,
                "H_size": req.train_dataset.H_size,
                "dataloader_shuffle": req.train_dataset.dataloader_shuffle,
                "dataloader_num_workers": req.train_dataset.dataloader_num_workers,
                "dataloader_batch_size": req.train_dataset.dataloader_batch_size,
                "use_photometric_aug": req.train_dataset.use_photometric_aug,
            },
            "test": {
                "name": "test_dataset",
                "dataset_type": req.test_dataset.dataset_type,
                "dataroot_H": req.test_dataset.dataroot_H,
                "dataroot_L": req.test_dataset.dataroot_L,
            },
        },
        "netG": req.net_g.model_dump(),
        "train": {
            "G_lossfn_type": req.train_params.G_lossfn_type,
            "G_lossfn_weight": req.train_params.G_lossfn_weight,
            "E_decay": req.train_params.E_decay,
            "G_optimizer_type": req.train_params.G_optimizer_type,
            "G_optimizer_lr": req.train_params.G_optimizer_lr,
            "G_optimizer_wd": req.train_params.G_optimizer_wd,
            "G_optimizer_reuse": req.train_params.G_optimizer_reuse,
            "G_scheduler_type": req.train_params.G_scheduler_type,
            "G_scheduler_milestones": req.train_params.G_scheduler_milestones,
            "G_scheduler_gamma": req.train_params.G_scheduler_gamma,
            "G_param_strict": req.train_params.G_param_strict,
            "E_param_strict": req.train_params.E_param_strict,
            "checkpoint_test": req.train_params.checkpoint_test,
            "checkpoint_save": req.train_params.checkpoint_save,
            "checkpoint_print": req.train_params.checkpoint_print,
        },
    }

    # Add GAN-specific dataset extras
    if req.train_dataset.degradation_type:
        cfg["datasets"]["train"]["degradation_type"] = req.train_dataset.degradation_type
    if req.train_dataset.shuffle_prob is not None:
        cfg["datasets"]["train"]["shuffle_prob"] = req.train_dataset.shuffle_prob
    if req.train_dataset.lq_patchsize is not None:
        cfg["datasets"]["train"]["lq_patchsize"] = req.train_dataset.lq_patchsize
    if req.train_dataset.use_sharp is not None:
        cfg["datasets"]["train"]["use_sharp"] = req.train_dataset.use_sharp

    # Inject GAN training extras
    if req.training_type == "gan" and req.gan_extras:
        g = req.gan_extras
        cfg["path"]["pretrained_netD"] = req.pretrained_netD
        cfg["path"]["pretrained_netE"] = req.pretrained_netE_gan
        cfg["netD"] = {
            "net_type": "discriminator_unet",
            "in_nc": req.n_channels,
            "base_nc": 64,
            "n_layers": 3,
            "norm_type": "spectral",
            "init_type": "orthogonal",
            "init_bn_type": "uniform",
            "init_gain": 0.2,
        }
        cfg["train"].update({
            "F_lossfn_type": g.F_lossfn_type,
            "F_lossfn_weight": g.F_lossfn_weight,
            "F_feature_layer": g.F_feature_layer,
            "F_weights": g.F_weights,
            "F_use_input_norm": g.F_use_input_norm,
            "F_use_range_norm": g.F_use_range_norm,
            "gan_type": g.gan_type,
            "D_lossfn_weight": g.D_lossfn_weight,
            "D_optimizer_type": "adam",
            "D_optimizer_lr": g.D_optimizer_lr,
            "D_optimizer_wd": g.D_optimizer_wd,
            "D_scheduler_type": "MultiStepLR",
            "D_scheduler_milestones": g.D_scheduler_milestones,
            "D_scheduler_gamma": g.D_scheduler_gamma,
            "D_optimizer_reuse": False,
            "D_param_strict": True,
            "E_param_strict": True,
            "D_init_iters": g.D_init_iters,
            "G_optimizer_lr": g.G_optimizer_lr,
        })

    return cfg


@router.post("/start", response_model=JobResponse)
async def start_training(req: StartTrainingRequest):
    """Write config JSON and launch the training subprocess."""
    # Build and save config
    cfg = _build_config_dict(req)
    config_path = TMP_DIR / f"train_{req.task}.json"
    config_service.save_json(cfg, config_path)

    # Choose script
    script = "main_train_swinir_gan.py" if req.training_type == "gan" else "main_train_swinir.py"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / script),
        "--opt",
        str(config_path),
    ]

    job_id = job_manager.create_job(cmd=cmd, cwd=str(PROJECT_ROOT))
    asyncio.create_task(job_manager.launch_job(job_id))

    return JobResponse(job_id=job_id, status="pending")


@router.get("/stream/{job_id}")
async def stream_training_logs(job_id: str):
    """SSE endpoint — streams live stdout/stderr from the training process."""
    return StreamingResponse(
        job_manager.stream_logs(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/status/{job_id}")
def get_status(job_id: str):
    summary = job_manager.get_job_summary(job_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return summary


@router.post("/stop/{job_id}")
def stop_training(job_id: str):
    if not job_manager.cancel_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "cancelled", "job_id": job_id}


@router.post("/pause/{job_id}")
def pause_training(job_id: str):
    if not job_manager.pause_job(job_id):
        raise HTTPException(status_code=400, detail="Job not running or not found")
    return {"status": "paused", "job_id": job_id}


@router.post("/resume/{job_id}")
def resume_training(job_id: str):
    if not job_manager.resume_job(job_id):
        raise HTTPException(status_code=400, detail="Job not paused or not found")
    return {"status": "running", "job_id": job_id}
