"""
Pydantic schemas for inference endpoints.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ModelConfig(BaseModel):
    upscale: int = 2
    in_chans: int = 3
    img_size: int = 128
    window_size: int = 8
    img_range: float = 1.0
    depths: List[int] = Field(default_factory=lambda: [6, 6, 6, 6, 6, 6])
    embed_dim: int = 180
    num_heads: List[int] = Field(default_factory=lambda: [6, 6, 6, 6, 6, 6])
    mlp_ratio: int = 2
    upsampler: str = "pixelshuffle"
    resi_connection: str = "1conv"


class StartInferenceRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_path: str
    lr_dir: str
    hr_dir: str
    sr_dir: str
    tile: Optional[int] = None
    tile_overlap: int = 32
    overwrite_sr: bool = True
    log_dir: str
    model_network_config: ModelConfig = Field(default_factory=ModelConfig, alias="model_config")
    # "5-layer" visual assessment (visual grids, FFT spectrum, error/residual maps)
    # for a random sample of patches. Defaults preserve old behaviour (disabled).
    visual_report_enabled: bool = False
    visual_report_samples: int = 10
    visual_report_seed: int = 23


class JobResponse(BaseModel):
    job_id: str
    status: str


# ──────────────────────────────────────────────────────────────────────────────
# Raw image inference schemas
# ──────────────────────────────────────────────────────────────────────────────

class CoregConfig(BaseModel):
    """Pipeline3-style coregistration and radiometric normalisation parameters."""
    enable_preprocessing: bool = True
    # Stage A — ORB keypoint coregistration
    coreg_a_enabled: bool = True
    coreg_a_max_features: int = 8000
    coreg_a_match_ratio: float = 0.75
    coreg_a_ransac_thresh: float = 4.0
    # Stage B — Phase cross-correlation
    coreg_b_enabled: bool = True
    coreg_b_upsample_factor: int = 100
    # Radiometric regression + histogram matching
    radiometric_enabled: bool = True
    radiometric_block_size: int = 256
    radiometric_rmse_threshold: float = 35.0
    radiometric_n_samples: int = 150_000
    radiometric_n_fit_windows: int = 80
    radiometric_post_hist_match: bool = True
    histogram_n_sample_windows: int = 40
    # Percentile scaling parameters
    nodata_value: int = 0
    saturated_value: int = 32767
    clip_percentiles: List[float] = Field(default_factory=lambda: [2.0, 98.0])
    percentile_n_sample_windows: int = 20
    coreg_preview_decim_dim: int = 2000


class RawPairedInferenceRequest(BaseModel):
    """Request body for paired LR+HR raw image inference (with optional alignment)."""
    model_config = ConfigDict(populate_by_name=True)

    lr_path: str
    hr_path: str
    lr_bands: List[int] = Field(default_factory=lambda: [3, 2, 1])
    hr_bands: List[int] = Field(default_factory=lambda: [1, 2, 3])
    model_path: str
    output_dir: str
    patch_size: int = 128
    overlap: int = 32
    scale_factor: int = 2
    coreg: CoregConfig = Field(default_factory=CoregConfig)
    model_network_config: ModelConfig = Field(default_factory=ModelConfig)
    # "5-layer" visual assessment -- paired mode only (needs HR ground truth).
    visual_report_enabled: bool = False
    visual_report_samples: int = 10
    visual_report_seed: int = 23


class LROnlyInferenceRequest(BaseModel):
    """Request body for LR-only raw image inference (no metrics)."""
    model_config = ConfigDict(populate_by_name=True)

    lr_path: str
    lr_bands: List[int] = Field(default_factory=lambda: [1, 2, 3])
    model_path: str
    output_dir: str
    patch_size: int = 128
    overlap: int = 32
    scale_factor: int = 2
    model_network_config: ModelConfig = Field(default_factory=ModelConfig)
