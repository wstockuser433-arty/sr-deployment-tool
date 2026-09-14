import numpy as np
import cv2

"""
Normalization
"""

# ===========================================================================
# 1. Normalization with percentile clipping
# ===========================================================================
def satellite_pre_norm(img, low_percentile=2, high_percentile=98):
    """Percentile-based normalization for satellite imagery.

    Parameters
    ----------
    img : np.ndarray
        Input image, uint8 or uint16.
    low_percentile : float
        Lower percentile for clipping (removes shadows).  Default: 2.
    high_percentile : float
        Upper percentile for clipping (removes clouds/bright).  Default: 98.

    Returns
    -------
    np.ndarray
        Normalized uint8 image stretched to [0, 255].
    """
    low, high = np.percentile(img, (low_percentile, high_percentile))

    img = np.clip(img, low, high)
    denom = (high - low)
    if denom == 0:
        return img.astype(np.uint8)
    img = (img - low) / denom * 255.0

    return img.astype(np.uint8)



"""
Cloud/Shadow Masking
"""

# ===========================================================================
# 1. Cloud Masking for Sentinel-2 images only
# ===========================================================================
def apply_s2cloudless_mask(
    image_4d: np.ndarray,
    nodata: float = 0.0,
    auto_scale: bool = True,
    threshold: float = 0.4,
    average_over: int = 4,
    dilation_size: int = 2,
) -> np.ndarray:
    """

    Uses the s2cloudless library to detect and mask clouds in Sentinel-2 imagery.
    The detector analyzes 10 spectral bands to identify cloud pixels using a
    machine learning-based approach.
    Sentinel-2 images can be filtered to exclude clouds when retrieving the images from the API

    Requirements:
        - s2cloudless library (pip install s2cloudless)
        - Input images must have exactly 10 bands in the correct order:
          B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12

    Parameters
    ----------
    image_4d : np.ndarray
        Must be shape (1, H, W, 10), float32 or integer.
        Bands MUST be in this exact order:
        B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12
    nodata : float, default 0.0
        Value to set masked pixels to.
    auto_scale : bool, default True
        If True and the array looks like raw Sentinel-2 data
        (integer dtype or max > 1.5), automatically divide by 10_000
        to convert to reflectance [0, 1].
    threshold : float, default 0.4
        Cloud probability threshold (higher = stricter, fewer clouds kept).
    average_over : int, default 4
        Smoothing window size for the cloud detector.
    dilation_size : int, default 2
        Dilation in pixels applied to the cloud mask.

    Returns
    -------
    np.ndarray
        Shape (H, W, 10), same dtype as input (after scaling), with
        cloud pixels set to `nodata`.

    Raises
    ------
    TypeError, ValueError : if input format is wrong.
    ImportError : if s2cloudless is not installed.
    """
    # Lazy import — only needed when cloud masking is actually used
    try:
        from s2cloudless import S2PixelCloudDetector
    except ImportError:
        raise ImportError(
            "s2cloudless is required for cloud masking. "
            "Install it with: pip install s2cloudless"
        )

    # ==================== VALIDATION ====================
    if not isinstance(image_4d, np.ndarray):
        raise TypeError("image_4d must be a numpy ndarray")

    if image_4d.ndim != 4:
        raise ValueError(f"Expected 4D array (1, H, W, 10), got ndim={image_4d.ndim}")

    if image_4d.shape[0] != 1 or image_4d.shape[3] != 10:
        raise ValueError(
            f"Expected shape (1, H, W, 10), got {image_4d.shape}. "
            "First dim must be 1 (batch size), last dim must be 10 bands."
        )

    # Build detector with caller-supplied parameters
    detector = S2PixelCloudDetector(
        threshold=threshold,
        average_over=average_over,
        dilation_size=dilation_size,
    )

    # ==================== PREPROCESSING ====================
    # Work on a copy in float32
    data = image_4d.astype(np.float32, copy=True)

    # Automatic scaling (most common case for raw S2 L1C/L2A)
    if auto_scale:
        max_val = data.max()
        if np.issubdtype(image_4d.dtype, np.integer) or max_val > 1.5:
            data /= 10000.0
            print(f"[s2cloudless] Auto-scaled by /10000 (original max = {max_val:.1f})")
        elif max_val > 1.0:
            raise ValueError(
                "Input values > 1.0 but not integer/raw. "
                "Either set auto_scale=False or manually scale to [0, 1]."
            )

    # Safety clamp (cloud detector expects [0, 1])
    data = np.clip(data, 0.0, 1.0)

    # ==================== MASKING ====================
    cloud_mask = detector.get_cloud_masks(data)          # shape (1, H, W), bool
    mask_expanded = np.broadcast_to(
        cloud_mask[0][..., None], data.shape
    )

    masked = np.where(mask_expanded, nodata, data[0])    # (H, W, 10)

    return masked


"""
Relative Normalization
"""

# ===========================================================================
# Histogram Matching (CDF-based)
# ===========================================================================
def histogram_match(source, reference, mask_src=None, mask_ref=None):
    """Match the histogram of *source* to that of *reference* per channel.

    Uses CDF transfer: for each channel the cumulative distribution of
    valid (unmasked) source pixels is remapped to the cumulative
    distribution of valid reference pixels via ``np.interp``.

    Parameters
    ----------
    source : np.ndarray
        Source image (uint8 or uint16), HxW or HxWxC.
    reference : np.ndarray
        Reference image, same dtype/shape convention as *source*.
    mask_src : np.ndarray or None
        Binary mask for source (1 = valid, 0 = invalid).  Shape HxW.
    mask_ref : np.ndarray or None
        Binary mask for reference.  Shape HxW.

    Returns
    -------
    np.ndarray
        Source image with its histogram matched to reference, same dtype.
    """
    orig_dtype = source.dtype
    src = source.astype(np.float64)
    ref = reference.astype(np.float64)

    if src.ndim == 2:
        src = src[:, :, np.newaxis]
        ref = ref[:, :, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    result = np.empty_like(src)
    n_channels = src.shape[2]

    for c in range(n_channels):
        s_ch = src[:, :, c]
        r_ch = ref[:, :, c]

        # Extract valid pixels
        s_valid = s_ch[mask_src == 1] if mask_src is not None else s_ch.ravel()
        r_valid = r_ch[mask_ref == 1] if mask_ref is not None else r_ch.ravel()

        if s_valid.size == 0 or r_valid.size == 0:
            result[:, :, c] = s_ch
            continue

        # Compute CDFs
        s_values, s_idx, s_counts = np.unique(s_valid, return_inverse=True, return_counts=True)
        s_cdf = np.cumsum(s_counts).astype(np.float64)
        s_cdf /= s_cdf[-1]

        r_values, r_counts = np.unique(r_valid, return_counts=True)
        r_cdf = np.cumsum(r_counts).astype(np.float64)
        r_cdf /= r_cdf[-1]

        # Interpolate: map source CDF → reference values
        interp_values = np.interp(s_cdf, r_cdf, r_values)

        # Build full lookup: for every pixel value in the source channel,
        # find its mapped value
        flat = s_ch.ravel()
        mapped = np.interp(flat, s_values, interp_values)
        result[:, :, c] = mapped.reshape(s_ch.shape)

    if squeeze:
        result = result[:, :, 0]

    # Clip to valid range
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        result = np.clip(result, info.min, info.max)
    return result.astype(orig_dtype)


# ===========================================================================
# Mean / Std Transfer
# ===========================================================================
def mean_std_transfer(source, reference, mask_src=None, mask_ref=None):
    """Match source to reference by transferring per-channel mean and std.

    Formula per channel::

        out = (source - mean_src) / std_src * std_ref + mean_ref

    Parameters
    ----------
    source : np.ndarray
        Source image (uint8 or uint16), HxW or HxWxC.
    reference : np.ndarray
        Reference image.
    mask_src : np.ndarray or None
        Binary mask for source (1 = valid).
    mask_ref : np.ndarray or None
        Binary mask for reference.

    Returns
    -------
    np.ndarray
        Normalised source image, same dtype as input.
    """
    orig_dtype = source.dtype
    src = source.astype(np.float64)
    ref = reference.astype(np.float64)

    if src.ndim == 2:
        src = src[:, :, np.newaxis]
        ref = ref[:, :, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    result = np.empty_like(src)
    n_channels = src.shape[2]

    for c in range(n_channels):
        s_ch = src[:, :, c]
        r_ch = ref[:, :, c]

        s_valid = s_ch[mask_src == 1] if mask_src is not None else s_ch.ravel()
        r_valid = r_ch[mask_ref == 1] if mask_ref is not None else r_ch.ravel()

        if s_valid.size == 0 or r_valid.size == 0:
            result[:, :, c] = s_ch
            continue

        mean_s, std_s = s_valid.mean(), s_valid.std()
        mean_r, std_r = r_valid.mean(), r_valid.std()

        if std_s < 1e-8:
            result[:, :, c] = s_ch
            continue

        result[:, :, c] = (s_ch - mean_s) / std_s * std_r + mean_r

    if squeeze:
        result = result[:, :, 0]

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        result = np.clip(result, info.min, info.max)
    return result.astype(orig_dtype)


# ===========================================================================
# Relative Normalization Dispatcher
# ===========================================================================
def relative_normalize(img_source, img_reference, method="histogram_match",
                       mask_src=None, mask_ref=None):
    """Normalise *img_source* relative to *img_reference*.

    Parameters
    ----------
    img_source : np.ndarray
        Image to be adjusted.
    img_reference : np.ndarray
        Image whose statistics serve as the target.
    method : str
        ``"histogram_match"`` | ``"mean_std_transfer"`` | ``"none"``.
    mask_src : np.ndarray or None
        Binary mask for source (1 = valid).
    mask_ref : np.ndarray or None
        Binary mask for reference.

    Returns
    -------
    np.ndarray
        Adjusted source image.
    """
    if method == "histogram_match":
        return histogram_match(img_source, img_reference,
                               mask_src=mask_src, mask_ref=mask_ref)
    elif method == "mean_std_transfer":
        return mean_std_transfer(img_source, img_reference,
                                 mask_src=mask_src, mask_ref=mask_ref)
    elif method == "none":
        return img_source
    else:
        raise ValueError(
            f"Unknown relative normalization method '{method}'. "
            "Choose from: histogram_match | mean_std_transfer | none"
        )


"""
Masking and co-registration
"""

# ===========================================================================
# Mask-Aware Normalization
# ===========================================================================
def satellite_pre_norm_masked(img, mask=None, low_percentile=2, high_percentile=98):
    """Percentile-based normalization that ignores masked (invalid) pixels."""
    if mask is None:
        # Fallback to standard if no mask is provided
        valid_pixels = img
    else:
        # Extract only valid pixels (where mask == 1)
        valid_pixels = img[mask == 1]
        if valid_pixels.size == 0:
            return np.zeros_like(img, dtype=np.uint8)  # Return black if entirely masked

    # Calculate percentiles only on valid data
    low, high = np.percentile(valid_pixels, (low_percentile, high_percentile))

    img_clipped = np.clip(img, low, high)
    denom = (high - low)
    if denom == 0:
        return img_clipped.astype(np.uint8)

    img_normalized = (img_clipped - low) / denom * 255.0

    # Optional: Set masked areas explicitly to 0 (black) to keep them clean
    if mask is not None:
        if img_normalized.ndim == 3:
            img_normalized[mask == 0] = [0] * img.shape[2]
        else:
            img_normalized[mask == 0] = 0

    return img_normalized.astype(np.uint8)


# ===========================================================================
# Level-2 QA Masking
# ===========================================================================
def apply_l2_qa_mask(qa_band, invalid_classes=None):
    """
    Create a binary mask from a Level-2 QA band (e.g., Sentinel-2 SCL).
    1 = Valid, 0 = Invalid.

    Sentinel-2 SCL Classes:
    3: Cloud Shadows, 8: Cloud Medium Prob, 9: Cloud High Prob
    10: Cirrus, 11: Snow / Ice
    """
    if invalid_classes is None:
        invalid_classes = [3, 8, 9, 10, 11]
    mask = np.ones(qa_band.shape[:2], dtype=np.uint8)
    for invalid_class in invalid_classes:
        mask[qa_band == invalid_class] = 0
    return mask


# ===========================================================================
# ECC Co-Registration
# ===========================================================================
def align_images_ecc(img_hr, img_lr, warp_mode=cv2.MOTION_TRANSLATION,
                     num_iters=50, eps=1e-5, gauss_filt_size=5):
    """Aligns LR image to HR image using Enhanced Correlation Coefficient.

    Parameters
    ----------
    img_hr : np.ndarray
        Reference (HR) image, uint8, HxW or HxWxC.
    img_lr : np.ndarray
        Source (LR) image to align, uint8, same size as img_hr.
    warp_mode : int
        OpenCV warp mode constant (e.g. cv2.MOTION_TRANSLATION).
    num_iters : int
        Maximum ECC iterations.  Default: 50.
    eps : float
        ECC convergence threshold.  Default: 1e-5.
    gauss_filt_size : int
        Gaussian filter size for ECC (must be odd, 0 to disable).
        Default: 5.

    Returns
    -------
    aligned_lr : np.ndarray
        Warped LR image aligned to HR.
    success : bool
        True if ECC converged, False if it failed.
    """
    # Convert to grayscale for feature matching
    if img_hr.ndim == 3:
        gray_hr = cv2.cvtColor(img_hr, cv2.COLOR_RGB2GRAY)
        gray_lr = cv2.cvtColor(img_lr, cv2.COLOR_RGB2GRAY)
    else:
        gray_hr, gray_lr = img_hr, img_lr

    # Define the transformation matrix
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iters, eps)

    try:
        # Run the ECC algorithm. The results are stored in warp_matrix.
        _, warp_matrix = cv2.findTransformECC(gray_hr, gray_lr, warp_matrix, warp_mode, criteria, None, gauss_filt_size)

        # Apply the calculated warp matrix to the multi-band LR image
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            aligned_lr = cv2.warpPerspective(img_lr, warp_matrix, (img_hr.shape[1], img_hr.shape[0]),
                                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            aligned_lr = cv2.warpAffine(img_lr, warp_matrix, (img_hr.shape[1], img_hr.shape[0]),
                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned_lr, True
    except Exception as e:
        # ECC fails if correlation is too low (e.g., entirely clouds)
        return img_lr, False