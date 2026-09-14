"""
Pydantic schemas for training endpoints.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class NetGConfig(BaseModel):
    net_type: str = "swinir"
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
    init_type: str = "default"


class TrainDatasetConfig(BaseModel):
    dataroot_H: str
    dataroot_L: Optional[str] = None
    H_size: int = 256
    dataloader_shuffle: bool = True
    dataloader_num_workers: int = 4
    dataloader_batch_size: int = 2
    dataset_type: str = "sr"
    use_photometric_aug: bool = False
    # GAN/blindsr extras
    degradation_type: Optional[str] = None
    shuffle_prob: Optional[float] = None
    lq_patchsize: Optional[int] = None
    use_sharp: Optional[bool] = None


class TestDatasetConfig(BaseModel):
    dataroot_H: str
    dataroot_L: Optional[str] = None
    dataset_type: str = "sr"
    degradation_type: Optional[str] = None
    H_size: Optional[int] = None
    shuffle_prob: Optional[float] = None
    lq_patchsize: Optional[int] = None
    use_sharp: Optional[bool] = None


class TrainingHyperparams(BaseModel):
    G_lossfn_type: str = "l1"
    G_lossfn_weight: float = 1.0
    E_decay: float = 0.999
    G_optimizer_type: str = "adam"
    G_optimizer_lr: float = 1e-4
    G_optimizer_wd: float = 0.0
    G_optimizer_reuse: bool = True
    G_scheduler_type: str = "MultiStepLR"
    G_scheduler_milestones: List[int] = Field(
        default_factory=lambda: [300000, 400000, 500000, 600000, 700000]
    )
    G_scheduler_gamma: float = 0.5
    G_param_strict: bool = True
    E_param_strict: bool = True
    checkpoint_test: int = 10000
    checkpoint_save: int = 10000
    checkpoint_print: int = 1000


class GANExtras(BaseModel):
    """GAN-only training parameters."""
    F_lossfn_type: str = "l1"
    F_lossfn_weight: float = 1.0
    F_feature_layer: List[int] = Field(default_factory=lambda: [2, 7, 16, 25, 34])
    F_weights: List[float] = Field(default_factory=lambda: [0.1, 0.1, 1.0, 1.0, 1.0])
    F_use_input_norm: bool = True
    F_use_range_norm: bool = False
    gan_type: str = "gan"
    D_lossfn_weight: float = 0.1
    D_optimizer_lr: float = 5e-5
    D_optimizer_wd: float = 0.0
    D_scheduler_milestones: List[int] = Field(
        default_factory=lambda: [100000, 200000, 300000, 400000, 500000]
    )
    D_scheduler_gamma: float = 0.5
    D_init_iters: int = 0
    G_optimizer_lr: float = 5e-5


class StartTrainingRequest(BaseModel):
    training_type: str  # "psnr" | "gan"
    task: str
    scale: int = 2
    n_channels: int = 3
    gpu_ids: List[int] = Field(default_factory=lambda: [0])
    pretrained_netG: Optional[str] = None
    pretrained_netE: Optional[str] = None
    pretrained_netD: Optional[str] = None
    pretrained_netE_gan: Optional[str] = None
    train_dataset: TrainDatasetConfig
    test_dataset: TestDatasetConfig
    net_g: NetGConfig = Field(default_factory=NetGConfig)
    train_params: TrainingHyperparams = Field(default_factory=TrainingHyperparams)
    gan_extras: Optional[GANExtras] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
