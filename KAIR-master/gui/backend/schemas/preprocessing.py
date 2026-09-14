"""
Pydantic schemas for preprocessing endpoints.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Pipeline A — pipeline3.py (Pleiades HR/LR → patches; paired or unpaired) ──

class CoregStageA(BaseModel):
    enabled: bool = True
    max_features: int = 8000
    match_ratio: float = 0.75
    ransac_thresh: float = 4.0
    downsample: float = 0.25


class CoregStageB(BaseModel):
    enabled: bool = True
    downsample: float = 0.25
    upsample_factor: int = 100


class CoregStageC(BaseModel):
    enabled: bool = True
    max_iter: int = 100
    eps: float = 1e-5
    warp_mode: str = "translation"
    discard_on_fail: bool = True


# ── Degradation configs — shared by Pipeline A (unpaired-HR) and Pipeline B ────

class BsrganConfig(BaseModel):
    jpeg_prob: float = 0.9
    scale2_prob: float = 0.25
    isp_prob: float = 0.25
    noise_level1: float = 2.0
    noise_level2: float = 25.0


class RealEsrganConfig(BaseModel):
    blur_prob_1: float = 1.0
    resize_prob_1: float = 1.0
    gaussian_noise_prob_1: float = 0.5
    poisson_noise_prob_1: float = 0.1
    speckle_noise_prob_1: float = 0.1
    jpeg_prob_1: float = 0.9
    noise_level1_s1: float = 2.0
    noise_level2_s1: float = 25.0
    blur_prob_2: float = 0.8
    resize_prob_2: float = 1.0
    gaussian_noise_prob_2: float = 0.5
    poisson_noise_prob_2: float = 0.1
    speckle_noise_prob_2: float = 0.1
    jpeg_prob_2: float = 0.8
    noise_level1_s2: float = 2.0
    noise_level2_s2: float = 15.0
    final_jpeg_prob: float = 0.5
    resize_back_prob: float = 0.5
    isp_prob: float = 0.1


class BsrganPlusConfig(BaseModel):
    shuffle_prob: float = 0.5
    use_sharp: bool = False
    sharpening_weight: float = 0.5
    sharpening_radius: int = 50
    sharpening_threshold: int = 10
    poisson_prob: float = 0.1
    speckle_prob: float = 0.1
    isp_prob: float = 0.1
    noise_level1: float = 2.0
    noise_level2: float = 25.0


class SatelliteConfig(BaseModel):
    blur_prob_1: float = 1.0
    blur_type_1: str = "mtf"
    resize_prob_1: float = 0.75
    poisson_prob_1: float = 0.75
    read_noise_prob_1: float = 0.55
    haze_prob_1: float = 0.45
    jpeg_prob_1: float = 0.12
    blur_prob_2: float = 0.92
    blur_type_2: str = "mtf"
    resize_prob_2: float = 0.70
    poisson_prob_2: float = 0.60
    read_noise_prob_2: float = 0.45
    haze_prob_2: float = 0.35
    jpeg_prob_2: float = 0.08
    final_jpeg_prob: float = 0.10
    resize_back_prob: float = 0.35
    isp_prob: float = 0.08
    noise_level1: float = 0.8
    noise_level2: float = 5.0
    mtf_sigma_optics_range: List[float] = Field(default_factory=lambda: [0.8, 2.8])
    mtf_detector_width_range: List[float] = Field(default_factory=lambda: [0.7, 1.8])
    mtf_atm_sigma_range: List[float] = Field(default_factory=lambda: [0.4, 1.8])


class Pipeline3Request(BaseModel):
    # hr_image_path / lr_image_path may each be a single file OR a directory.
    # Leave lr_image_path empty for HR-only scenes (synthetic LR via degradation);
    # leave hr_image_path empty for LR-only scenes (standalone patch extraction).
    hr_image_path: str = ""
    lr_image_path: str = ""
    output_dir: str
    supported_extensions: List[str] = Field(
        default_factory=lambda: [".tif", ".tiff", ".jp2", ".png", ".jpg", ".jpeg", ".bmp"]
    )
    hr_rgb_bands: List[int] = Field(default_factory=lambda: [1, 2, 3])
    lr_rgb_bands: List[int] = Field(default_factory=lambda: [3, 2, 1])
    scale_factor: int = 2
    hr_patch_size: int = 256
    stride: int = 256
    nodata_value: int = 0
    saturated_value: int = 32767
    clip_percentiles: List[float] = Field(default_factory=lambda: [2.0, 98.0])
    max_nodata_fraction: float = 0.05
    min_variance: float = 120.0
    min_ecc_score: float = 0.78
    min_ssim: float = 0.60
    radiometric_block_size: int = 256
    radiometric_rmse_threshold: float = 35.0
    radiometric_n_samples: int = 150000
    radiometric_post_hist_match: bool = True
    coreg_a: CoregStageA = Field(default_factory=CoregStageA)
    coreg_b: CoregStageB = Field(default_factory=CoregStageB)
    coreg_c: CoregStageC = Field(default_factory=CoregStageC)
    # Unpaired-HR degradation (synthesizes LR when a scene has no matching LR)
    degradation_enabled: bool = True
    degradation_type: str = "satellite"  # "bsrgan"|"real_esrgan"|"bsrgan_plus"|"satellite"
    bsrgan: BsrganConfig = Field(default_factory=BsrganConfig)
    real_esrgan: RealEsrganConfig = Field(default_factory=RealEsrganConfig)
    bsrgan_plus: BsrganPlusConfig = Field(default_factory=BsrganPlusConfig)
    satellite: SatelliteConfig = Field(default_factory=SatelliteConfig)
    # Class/subfolder filter — empty list means include all subfolders
    class_filter: List[str] = Field(default_factory=list)
    # Train/test split
    train_test_split: bool = False
    train_ratio: float = 0.8
    test_output_dir: Optional[str] = None


# ── Pipeline B — run_pipeline.py (HR-only degradation pipeline) ────────────────

class RunPipelineRequest(BaseModel):
    task: str = "preprocess_sr_x2"
    pipeline_mode: str = "hr_only"  # "hr_only" | "hr_lr_pair"
    degradation_type: str = "satellite"  # "bsrgan"|"real_esrgan"|"bsrgan_plus"|"satellite"
    scale: int = 2
    n_channels: int = 3
    seed: Optional[int] = 42
    num_workers: int = 1
    input_hr_dir: str
    input_lr_dir: Optional[str] = None
    output_hr_dir: str
    output_lr_dir: str
    supported_extensions: List[str] = Field(
        default_factory=lambda: [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    )
    save_format: str = "png"
    save_hr_copy: bool = True
    normalize_enabled: bool = False
    normalize_low_percentile: float = 2.0
    normalize_high_percentile: float = 98.0
    cloud_mask_enabled: bool = False
    cloud_mask_threshold: float = 0.4
    cloud_mask_average_over: int = 4
    cloud_mask_dilation_size: int = 2
    cloud_mask_nodata: float = 0.0
    cloud_mask_auto_scale: bool = True
    bsrgan: BsrganConfig = Field(default_factory=BsrganConfig)
    real_esrgan: RealEsrganConfig = Field(default_factory=RealEsrganConfig)
    bsrgan_plus: BsrganPlusConfig = Field(default_factory=BsrganPlusConfig)
    satellite: SatelliteConfig = Field(default_factory=SatelliteConfig)
    # Train/test split
    train_test_split: bool = False
    train_ratio: float = 0.8


class JobResponse(BaseModel):
    job_id: str
    status: str
