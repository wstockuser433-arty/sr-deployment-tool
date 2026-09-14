"""
preprocessing_pipeline
======================
KAIR-style configurable image preprocessing pipeline for generating
degraded LR / mod-cropped HR training pairs.

Submodules
----------
degradation_utils
    Four degradation pipelines (BSRGAN, Real-ESRGAN, BSRGAN-Plus, Satellite)
    and low-level primitives (blur, noise, resize, JPEG, MTF, haze, …).

other_utils
    Satellite-specific helpers: percentile normalization and Sentinel-2
    cloud masking via s2cloudless.

run_pipeline
    CLI entry-point: reads a JSON config and batch-processes an HR directory.
"""

# -- Degradation pipelines --------------------------------------------------
from preprocessing_pipeline.degradation_utils import (
    degrade_bsrgan,
    degrade_bsrgan_plus,
    degrade_real_esrgan,
    degrade_satellite,
)

# -- Degradation primitives -------------------------------------------------
from preprocessing_pipeline.degradation_utils import (
    add_blur,
    add_atmospheric_haze,
    add_Gaussian_noise,
    add_JPEG_noise,
    add_mtf_blur,
    add_Poisson_noise,
    add_resize,
    add_sharpening,
    add_speckle_noise,
)

# -- Image I/O helpers ------------------------------------------------------
from preprocessing_pipeline.degradation_utils import (
    imread_uint,
    imsave,
    imresize_np,
    single2uint,
    uint2single,
)

# -- Satellite preprocessing utilities --------------------------------------
from preprocessing_pipeline.other_utils import (
    apply_s2cloudless_mask,
    apply_l2_qa_mask,
    align_images_ecc,
    satellite_pre_norm,
    satellite_pre_norm_masked,
    histogram_match,
    mean_std_transfer,
    relative_normalize,
)

__all__ = [
    # pipelines
    "degrade_bsrgan",
    "degrade_bsrgan_plus",
    "degrade_real_esrgan",
    "degrade_satellite",
    # primitives
    "add_blur",
    "add_atmospheric_haze",
    "add_Gaussian_noise",
    "add_JPEG_noise",
    "add_mtf_blur",
    "add_Poisson_noise",
    "add_resize",
    "add_sharpening",
    "add_speckle_noise",
    # I/O
    "imread_uint",
    "imsave",
    "imresize_np",
    "single2uint",
    "uint2single",
    # satellite utils – masking
    "apply_s2cloudless_mask",
    "apply_l2_qa_mask",
    # satellite utils – registration
    "align_images_ecc",
    # satellite utils – normalization
    "satellite_pre_norm",
    "satellite_pre_norm_masked",
    "histogram_match",
    "mean_std_transfer",
    "relative_normalize",
]

