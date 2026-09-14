"""
degradation_utils.py
======================
LR-image degradation methods for blind super-resolution preprocessing pipelines.

Three degradation strategies are provided:

1. ``degrade_bsrgan``       – Official BSRGAN degradation pipeline.
                              (Kai Zhang et al., ICCV 2021)

2. ``degrade_real_esrgan``  – Real-ESRGAN dual-stage (second-order) pipeline.
                              (Xintao Wang et al., ICCVW 2021)

3. ``degrade_bsrgan_plus``  – Combined BSRGAN + Real-ESRGAN extended pipeline.
"""

import math
import os
import random

import cv2
import numpy as np
import scipy.stats as ss
import torch
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import orth


# ===========================================================================
# Image I/O helpers  (inlined from utils/utils_image.py)
# ===========================================================================

def _cv2_imread_unicode(path, flags):
    """cv2.imread wrapper that handles non-ASCII paths on Windows.

    cv2.imread silently returns None for paths containing non-ASCII characters
    on Windows (cp1252 locale). Fall back to reading raw bytes via numpy so
    that paths with µ, accented letters, CJK characters, etc. work correctly.
    """
    img = cv2.imread(path, flags)
    if img is None:
        try:
            raw = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(raw, flags)
        except Exception:
            pass
    return img


def imread_uint(path, n_channels=3):
    """Read an image from disk as a uint8 HxWxC numpy array (RGB channel order).

    Parameters
    ----------
    path : str
        Path to the image file.
    n_channels : int, optional
        1 for grayscale, 3 for RGB.  Default: 3.

    Returns
    -------
    np.ndarray
        uint8 array, shape HxWx``n_channels``.

    Raises
    ------
    FileNotFoundError
        If the file cannot be opened (e.g. path does not exist or is unreadable).
    """
    if n_channels == 1:
        img = _cv2_imread_unicode(path, 0)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = np.expand_dims(img, axis=2)
    else:
        img = _cv2_imread_unicode(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imsave(img, img_path):
    """Save a uint8 HxWxC (RGB) numpy array to disk.

    Parameters
    ----------
    img : np.ndarray
        uint8 image array in RGB channel order.
    img_path : str
        Destination file path.
    """
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]   # RGB -> BGR for cv2
    # cv2.imwrite silently fails on non-ASCII paths on Windows; use imencode+tofile
    ext = os.path.splitext(img_path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(img_path)
    else:
        raise RuntimeError(f"cv2.imencode failed for {img_path}")


def uint2single(img):
    """Convert uint8 [0, 255] numpy array to float32 [0, 1]."""
    return np.float32(img / 255.0)


def single2uint(img):
    """Convert float32 [0, 1] numpy array to uint8 [0, 255]."""
    return np.uint8((img.clip(0, 1) * 255.0).round())


# ===========================================================================
# Bicubic resize helper  (inlined from utils/utils_image.py)
# ===========================================================================

def _cubic(x):
    """Keys' cubic interpolation kernel (torch tensor version)."""
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + \
           (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (
               ((absx > 1) * (absx <= 2)).type_as(absx))


def _calculate_weights_indices(in_length, out_length, scale, kernel_width, antialiasing):
    """Compute bicubic interpolation weights and indices for one dimension."""
    if scale < 1 and antialiasing:
        kernel_width = kernel_width / scale
    x = torch.linspace(1, out_length, out_length)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = torch.floor(u - kernel_width / 2)
    P = math.ceil(kernel_width) + 2
    indices = left.view(out_length, 1).expand(out_length, P) + \
              torch.linspace(0, P - 1, P).view(1, P).expand(out_length, P)
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    if scale < 1 and antialiasing:
        weights = scale * _cubic(distance_to_center * scale)
    else:
        weights = _cubic(distance_to_center)
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0].item(), 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1].item(), 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize_np(img, scale, antialiasing=True):
    """Bicubic resize of a numpy HxWxC (or HxW) float32 image.

    Parameters
    ----------
    img : np.ndarray
        Input float32 image, HxWxC or HxW, range [0, 1].
    scale : float
        Resize scale factor (e.g. 0.25 for ×4 downscale).
    antialiasing : bool, optional
        Apply anti-aliasing when downscaling.  Default: True.

    Returns
    -------
    np.ndarray
        Resized float32 image.
    """
    img = torch.from_numpy(img)
    need_squeeze = img.dim() == 2
    if need_squeeze:
        img.unsqueeze_(2)
    in_H, in_W, in_C = img.size()
    out_H = math.ceil(in_H * scale)
    out_W = math.ceil(in_W * scale)
    kernel_width = 4

    weights_H, indices_H, sym_len_Hs, sym_len_He = _calculate_weights_indices(
        in_H, out_H, scale, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = _calculate_weights_indices(
        in_W, out_W, scale, kernel_width, antialiasing)

    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)
    sym_patch = img[:sym_len_Hs, :, :]
    img_aug.narrow(0, 0, sym_len_Hs).copy_(
        sym_patch.index_select(0, torch.arange(sym_patch.size(0) - 1, -1, -1).long()))
    sym_patch = img[-sym_len_He:, :, :]
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(
        sym_patch.index_select(0, torch.arange(sym_patch.size(0) - 1, -1, -1).long()))

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kw = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(in_C):
            out_1[i, :, j] = img_aug[idx:idx + kw, :, j].transpose(0, 1).mv(weights_H[i])

    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)
    sym_patch = out_1[:, :sym_len_Ws, :]
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(
        sym_patch.index_select(1, torch.arange(sym_patch.size(1) - 1, -1, -1).long()))
    sym_patch = out_1[:, -sym_len_We:, :]
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(
        sym_patch.index_select(1, torch.arange(sym_patch.size(1) - 1, -1, -1).long()))

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kw = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(in_C):
            out_2[:, i, j] = out_1_aug[:, idx:idx + kw, j].mv(weights_W[i])

    if need_squeeze:
        out_2.squeeze_()
    return out_2.numpy()


# ===========================================================================
# Kernel / blur primitives  (inlined from utils/utils_blindsr.py)
# ===========================================================================

def _gm_blur_kernel(mean, cov, size=15):
    """Gaussian mixture blur kernel evaluated on a grid."""
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)
    return k / np.sum(k)


def _anisotropic_gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """Generate an anisotropic Gaussian blur kernel."""
    v = np.dot(
        np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]]),
        np.array([1.0, 0.0]),
    )
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    return _gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)


def _fspecial_gaussian(hsize, sigma):
    """MATLAB-style fspecial('gaussian', hsize, sigma)."""
    siz = [(hsize - 1.0) / 2.0, (hsize - 1.0) / 2.0]
    [x, y] = np.meshgrid(
        np.arange(-siz[1], siz[1] + 1),
        np.arange(-siz[0], siz[0] + 1),
    )
    arg = -(x * x + y * y) / (2 * sigma * sigma)
    h = np.exp(arg)
    h[h < np.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


def _shift_pixel(x, sf, upper_left=True):
    """Shift pixel positions for aligned SR kernel construction."""
    h, w = x.shape[:2]
    shift = (sf - 1) * 0.5
    xv = np.arange(0, w, 1.0)
    yv = np.arange(0, h, 1.0)
    x1 = (xv + shift) if upper_left else (xv - shift)
    y1 = (yv + shift) if upper_left else (yv - shift)
    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    xx1, yy1 = np.meshgrid(x1, y1)
    pts = np.stack([yy1.ravel(), xx1.ravel()], axis=-1)
    if x.ndim == 2:
        interp = RegularGridInterpolator(
            (yv, xv), x, method='linear', bounds_error=False, fill_value=None)
        x = interp(pts).reshape(h, w)
    elif x.ndim == 3:
        for i in range(x.shape[-1]):
            interp = RegularGridInterpolator(
                (yv, xv), x[:, :, i], method='linear',
                bounds_error=False, fill_value=None)
            x[:, :, i] = interp(pts).reshape(h, w)
    return x


# ===========================================================================
# Primitive degradation operations  (inlined from utils/utils_blindsr.py)
# ===========================================================================

def add_sharpening(img, weight=0.5, radius=50, threshold=10):
    """USM (Unsharp Mask) sharpening applied to a float32 image.

    Parameters
    ----------
    img : np.ndarray
        Float32 HxWxC image in [0, 1].
    weight : float
        Strength of the sharpening residual.  Default: 0.5.
    radius : int
        Gaussian blur kernel radius (must be odd; +1 applied if even).
        Default: 50.
    threshold : int
        Pixel-difference threshold (0–255) for the sharpening mask.
        Default: 10.

    Returns
    -------
    np.ndarray
        Sharpened float32 image in [0, 1].
    """
    if radius % 2 == 0:
        radius += 1
    blurred = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blurred
    mask = (np.abs(residual) * 255 > threshold).astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)
    K = np.clip(img + weight * residual, 0, 1)
    return soft_mask * K + (1 - soft_mask) * img


def add_blur(img, sf=4):
    """Apply a random anisotropic Gaussian or isotropic Gaussian blur.

    Parameters
    ----------
    img : np.ndarray
        Float32 HxWxC image in [0, 1].
    sf : int
        Scale factor (controls kernel size range).  Default: 4.

    Returns
    -------
    np.ndarray
        Blurred image.
    """
    wd2 = 4.0 + sf
    wd  = 2.0 + 0.2 * sf
    if random.random() < 0.5:
        l1 = wd2 * random.random()
        l2 = wd2 * random.random()
        k = _anisotropic_gaussian(
            ksize=2 * random.randint(2, 11) + 3,
            theta=random.random() * np.pi,
            l1=l1, l2=l2,
        )
    else:
        k = _fspecial_gaussian(2 * random.randint(2, 11) + 3, wd * random.random())
    img = ndimage.convolve(img, np.expand_dims(k, axis=2), mode='mirror')
    return img


def add_resize(img, sf=4):
    """Apply a random intermediate resize (up, down, or identity).

    This simulates the resolution mismatch artefacts that arise from
    different camera zoom levels or pre-processing pipelines.

    Parameters
    ----------
    img : np.ndarray
        Float32 HxWxC image in [0, 1].
    sf : int
        Scale factor (controls the lower bound of the resize range).
        Default: 4.

    Returns
    -------
    np.ndarray
        Randomly resized image (same or different spatial size).
    """
    rnum = np.random.rand()
    if rnum > 0.8:
        sf1 = random.uniform(1, 2)
    elif rnum < 0.7:
        sf1 = random.uniform(0.5 / sf, 1)
    else:
        sf1 = 1.0
    img = cv2.resize(
        img,
        (int(sf1 * img.shape[1]), int(sf1 * img.shape[0])),
        interpolation=random.choice([1, 2, 3]),
    )
    return np.clip(img, 0.0, 1.0)


def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    """Add random Gaussian noise (colour, grayscale, or correlated).

    Parameters
    ----------
    img : np.ndarray
        Float32 HxWxC image in [0, 1].
    noise_level1 : int or float
        Minimum noise standard deviation (intensity / 255).  Default: 2.
    noise_level2 : int or float
        Maximum noise standard deviation.  Default: 25.

    Returns
    -------
    np.ndarray
        Noisy image clipped to [0, 1].
    """
    noise_level = random.uniform(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:
        img += np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:
        img += np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2 / 255.0
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal(
            [0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]
        ).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def add_speckle_noise(img, noise_level1=2, noise_level2=25):
    """Add random speckle (multiplicative) noise.

    Parameters
    ----------
    img : np.ndarray
        Float32 HxWxC image in [0, 1].
    noise_level1 : int or float
        Minimum noise level (intensity / 255).  Default: 2.
    noise_level2 : int or float
        Maximum noise level.  Default: 25.

    Returns
    -------
    np.ndarray
        Noisy image clipped to [0, 1].
    """
    noise_level = random.uniform(noise_level1, noise_level2)
    img = np.clip(img, 0.0, 1.0)
    rnum = random.random()
    if rnum > 0.6:
        img += img * np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:
        img += img * np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2 / 255.0
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += img * np.random.multivariate_normal(
            [0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]
        ).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def add_Poisson_noise(img):
    """Add random Poisson noise (colour or grayscale).

    Parameters
    ----------
    img : np.ndarray
        Float32 HxWxC image in [0, 1].

    Returns
    -------
    np.ndarray
        Noisy image clipped to [0, 1].
    """
    img = np.clip((img * 255.0).round(), 0, 255) / 255.0
    vals = 10 ** (2 * random.random() + 2.0)
    if random.random() < 0.5:
        img = np.random.poisson(img * vals).astype(np.float32) / vals
    else:
        img_gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.0
        noise_gray = np.random.poisson(img_gray * vals).astype(np.float32) / vals - img_gray
        img += noise_gray[:, :, np.newaxis]
    return np.clip(img, 0.0, 1.0)


def add_JPEG_noise(img, quality_min=30, quality_max=95):
    """Apply random JPEG compression artefacts.

    Parameters
    ----------
    img : np.ndarray
        Float32 HxWxC image in [0, 1], RGB channel order.
    quality_min : int
        Minimum JPEG quality factor (1–100).  Default: 30.
    quality_max : int
        Maximum JPEG quality factor.  Default: 95.

    Returns
    -------
    np.ndarray
        Float32 image with JPEG artefacts, RGB channel order, range [0, 1].
    """
    quality_factor = random.randint(quality_min, quality_max)
    img_bgr = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
    _, encimg = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img_bgr = cv2.imdecode(encimg, 1)
    return cv2.cvtColor(uint2single(img_bgr), cv2.COLOR_BGR2RGB)

def _satellite_psf(
    ksize=21,
    sigma_optics=1.5,
    detector_width=1.0,
    atm_sigma=1.0,
):
    """Parametric PSF approximating optical satellite sensor MTF.

    Components:
    - Gaussian (diffraction-limited optics + platform jitter)
    - sinc² (square detector integration, per Chung et al. 2023 style)
    - Extra Gaussian (atmospheric turbulence)

    This is the core of realistic satellite blur (see Dong et al. 2022 and Chung et al. 2023).
    """
    half = (ksize - 1) / 2.0
    x = np.linspace(-half, half, ksize)
    xx, yy = np.meshgrid(x, x)

    # Optics + jitter Gaussian
    gauss = np.exp(-(xx**2 + yy**2) / (2 * sigma_optics**2))

    # Detector MTF (separable sinc² for square pixels)
    sinc_x = np.sinc(xx / detector_width) ** 2
    sinc_y = np.sinc(yy / detector_width) ** 2
    detector = sinc_x * sinc_y

    # Atmospheric turbulence Gaussian
    atm = np.exp(-(xx**2 + yy**2) / (2 * atm_sigma**2))

    k = gauss * detector * atm
    k = k / (np.sum(k) + 1e-12)
    return k


def add_mtf_blur(
    img,
    sf=4,
    sigma_optics_range=(0.8, 2.8),
    detector_width_range=(0.7, 1.8),
    atm_sigma_range=(0.4, 1.8),
):
    """Apply realistic satellite MTF/PSF blur (recommended default)."""
    ksize = 2 * random.randint(5, 14) + 1          # 11–29 px, typical for HR→LQ
    sigma_o = random.uniform(*sigma_optics_range)
    det_w = random.uniform(*detector_width_range)
    atm_s = random.uniform(*atm_sigma_range)

    # Slight scaling with sf (larger downsampling → relatively larger physical blur)
    sigma_o *= (sf / 4.0)
    atm_s *= (sf / 4.0)

    k = _satellite_psf(ksize=ksize, sigma_optics=sigma_o,
                       detector_width=det_w, atm_sigma=atm_s)
    img = ndimage.convolve(img, np.expand_dims(k, axis=2), mode='mirror')
    return img


def add_atmospheric_haze(
    img,
    intensity_range=(0.02, 0.18),
    turbulence_scale=0.07,
):
    """Simple atmospheric path radiance + low-frequency turbulence (common in RSISR)."""
    intensity = random.uniform(*intensity_range)

    # Base haze (additive scattering)
    base = intensity * 0.75

    # Turbulence (low-frequency variation)
    turb = np.random.normal(0, turbulence_scale, img.shape[:2]).astype(np.float32)
    turb = cv2.GaussianBlur(turb, (0, 0), sigmaX=random.uniform(9, 32))[:, :, np.newaxis]

    # Apply approximate radiative transfer model
    img = img * (1.0 - intensity * 0.65) + base + turb * intensity * 2.2
    return np.clip(img, 0.0, 1.0)


# ===========================================================================
# 1.  BSRGAN Degradation
# ===========================================================================

def degrade_bsrgan(
    img,
    sf=4,
    isp_model=None,
    jpeg_prob=0.9,
    scale2_prob=0.25,
    isp_prob=0.25,
    noise_level1=2,
    noise_level2=25,
):
    """Apply the BSRGAN degradation pipeline to a full HR image.

    Based on:
        Kai Zhang et al., "Designing a Practical Degradation Model for Deep
        Blind Image Super-Resolution", ICCV 2021.

    The pipeline randomly shuffles seven degradation operations:
    - Two blur passes (anisotropic or isotropic Gaussian)
    - Two downsampling passes (random resize or shifted-kernel + nearest)
    - One Gaussian noise pass
    - One JPEG compression pass (at ``jpeg_prob`` probability)
    - One optional ISP camera-sensor noise pass

    A final JPEG compression is always applied at the end.

    This function operates on the **full HR image** and returns a full LR
    image at 1/sf resolution.  No patching or cropping is performed here;
    those operations are handled by the training dataloader.

    Parameters
    ----------
    img : np.ndarray
        Input HR image, shape HxWxC, dtype float32, range [0, 1].
    sf : int, optional
        Integer downscale factor.  Default: 4.
    isp_model : object or None, optional
        Optional ISP noise model exposing ``.forward(img, hq) -> (img, hq)``.
        Pass ``None`` (default) to skip ISP noise.
    jpeg_prob : float, optional
        Probability of applying JPEG noise in the shuffle (step 5).
        Default: 0.9.
    scale2_prob : float, optional
        When ``sf == 4``, probability of a ×2 pre-downsampling step that
        splits the ×4 factor into two ×2 stages.  Default: 0.25.
    isp_prob : float, optional
        Probability of invoking ``isp_model`` (step 6).  Default: 0.25.
    noise_level1 : int, optional
        Lower bound of the Gaussian noise level range (intensity / 255).
        Default: 2.
    noise_level2 : int, optional
        Upper bound of the Gaussian noise level range.  Default: 25.

    Returns
    -------
    img_lq : np.ndarray
        Degraded LR image, shape (H/sf) x (W/sf) x C, float32, [0, 1].
    img_hq : np.ndarray
        Mod-cropped HR image aligned to ``img_lq``, float32, [0, 1].
    """
    sf_ori = sf

    # mod-crop so dimensions are divisible by sf
    h1, w1 = img.shape[:2]
    img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]
    hq = img.copy()

    # optional ×2 pre-downsampling (splits ×4 into two ×2 stages)
    if sf == 4 and random.random() < scale2_prob:
        if np.random.rand() < 0.5:
            img = cv2.resize(
                img,
                (int(0.5 * img.shape[1]), int(0.5 * img.shape[0])),
                interpolation=random.choice([1, 2, 3]),
            )
        else:
            img = imresize_np(img, 0.5, True)
        img = np.clip(img, 0.0, 1.0)
        sf = 2

    # build a random shuffle of 7 degradation steps, keeping downsample-2
    # (slot 2) before downsample-3 (slot 3) to preserve spatial consistency
    shuffle_order = random.sample(range(7), 7)
    idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
    if idx1 > idx2:
        shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

    a, b = img.shape[1], img.shape[0]   # store pre-downsample dims for slot 3

    for i in shuffle_order:

        if i == 0:
            # blur pass 1
            img = add_blur(img, sf=sf)

        elif i == 1:
            # blur pass 2
            img = add_blur(img, sf=sf)

        elif i == 2:
            # downsample pass 1 – random resize or shifted-kernel + nearest
            if random.random() < 0.75:
                sf1 = random.uniform(1, 2 * sf)
                img = cv2.resize(
                    img,
                    (int(img.shape[1] / sf1), int(img.shape[0] / sf1)),
                    interpolation=random.choice([1, 2, 3]),
                )
            else:
                k = _fspecial_gaussian(25, random.uniform(0.1, 0.6 * sf))
                k_shifted = _shift_pixel(k, sf)
                k_shifted = k_shifted / k_shifted.sum()
                img = ndimage.convolve(img, np.expand_dims(k_shifted, axis=2), mode='mirror')
                img = img[0::sf, 0::sf, ...]   # nearest-neighbour downsample
            img = np.clip(img, 0.0, 1.0)

        elif i == 3:
            # downsample pass 2 – resize to exact target LQ resolution
            img = cv2.resize(
                img,
                (int(a / sf), int(b / sf)),
                interpolation=random.choice([1, 2, 3]),
            )
            img = np.clip(img, 0.0, 1.0)

        elif i == 4:
            # Gaussian noise
            img = add_Gaussian_noise(img, noise_level1=noise_level1, noise_level2=noise_level2)

        elif i == 5:
            # JPEG noise (gated by jpeg_prob)
            if random.random() < jpeg_prob:
                img = add_JPEG_noise(img)

        elif i == 6:
            # optional ISP camera-sensor noise
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)

    # final JPEG compression (always applied)
    img = add_JPEG_noise(img)

    # ensure exact target spatial size
    target_h = int(hq.shape[0] / sf_ori)
    target_w = int(hq.shape[1] / sf_ori)
    if img.shape[0] != target_h or img.shape[1] != target_w:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    img_lq = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img_lq, hq


# ===========================================================================
# 2.  Real-ESRGAN Degradation
# ===========================================================================

def degrade_real_esrgan(
    img,
    sf=4,
    # ---- stage-1 probabilities / ranges -----------------------------------
    blur_prob_1=1.0,
    resize_prob_1=1.0,
    gaussian_noise_prob_1=0.5,
    poisson_noise_prob_1=0.1,
    speckle_noise_prob_1=0.1,
    jpeg_prob_1=0.9,
    noise_level1_s1=2,
    noise_level2_s1=25,
    # ---- stage-2 probabilities / ranges -----------------------------------
    blur_prob_2=0.8,
    resize_prob_2=1.0,
    gaussian_noise_prob_2=0.5,
    poisson_noise_prob_2=0.1,
    speckle_noise_prob_2=0.1,
    jpeg_prob_2=0.8,
    noise_level1_s2=2,
    noise_level2_s2=15,
    # ---- final step --------------------------------------------------------
    final_jpeg_prob=0.5,
    resize_back_prob=0.5,
    isp_model=None,
    isp_prob=0.1,
):
    """Apply the Real-ESRGAN dual-stage (second-order) degradation pipeline.

    Based on:
        Xintao Wang et al., "Real-ESRGAN: Training Real-World Blind
        Super-Resolution with Pure Synthetic Data", ICCVW 2021.

    The second-order model applies the classic degradation sequence::

        blur → resize → noise (Gaussian / Poisson / Speckle) → JPEG

    **twice** in succession before a final resize to the target LQ
    resolution.  This produces more complex, compound degradations that
    better mimic real-world photographs.

    An optional intermediate resize-back step between the two stages
    simulates scale-mismatch / over-sharpening artefacts.

    This function operates on the **full HR image** and returns a full LR
    image at 1/sf resolution.  No patching or cropping is performed here.

    Parameters
    ----------
    img : np.ndarray
        Input HR image, shape HxWxC, dtype float32, range [0, 1].
    sf : int, optional
        Integer downscale factor.  Default: 4.
    blur_prob_1 : float, optional
        Probability of applying blur in stage 1.  Default: 1.0.
    resize_prob_1 : float, optional
        Probability of applying random resize in stage 1.  Default: 1.0.
    gaussian_noise_prob_1 : float, optional
        Probability of Gaussian noise in stage 1.  Default: 0.5.
    poisson_noise_prob_1 : float, optional
        Probability of Poisson noise in stage 1 (applied after Gaussian gate
        fails).  Default: 0.1.
    speckle_noise_prob_1 : float, optional
        Probability of speckle noise in stage 1 (applied after Poisson gate
        fails).  Default: 0.1.
    jpeg_prob_1 : float, optional
        Probability of JPEG compression in stage 1.  Default: 0.9.
    noise_level1_s1 : int, optional
        Lower bound of Gaussian noise level for stage 1.  Default: 2.
    noise_level2_s1 : int, optional
        Upper bound of Gaussian noise level for stage 1.  Default: 25.
    blur_prob_2 : float, optional
        Probability of blur in stage 2.  Default: 0.8.
    resize_prob_2 : float, optional
        Probability of random resize in stage 2.  Default: 1.0.
    gaussian_noise_prob_2 : float, optional
        Probability of Gaussian noise in stage 2.  Default: 0.5.
    poisson_noise_prob_2 : float, optional
        Probability of Poisson noise in stage 2.  Default: 0.1.
    speckle_noise_prob_2 : float, optional
        Probability of speckle noise in stage 2.  Default: 0.1.
    jpeg_prob_2 : float, optional
        Probability of JPEG compression in stage 2.  Default: 0.8.
    noise_level1_s2 : int, optional
        Lower bound of Gaussian noise level for stage 2.  Default: 2.
    noise_level2_s2 : int, optional
        Upper bound of Gaussian noise level for stage 2.  Default: 15.
    final_jpeg_prob : float, optional
        Probability of a final JPEG pass after the last resize.  Default: 0.5.
    resize_back_prob : float, optional
        Probability of an intermediate up→HQ-size then down→LQ-size resize
        between stages 1 and 2 (simulates scale-mismatch artefacts).
        Default: 0.5.
    isp_model : object or None, optional
        Optional ISP noise model.  Default: None.
    isp_prob : float, optional
        Probability of invoking ``isp_model`` when provided.  Default: 0.1.

    Returns
    -------
    img_lq : np.ndarray
        Degraded LR image, shape (H/sf) x (W/sf) x C, float32, [0, 1].
    img_hq : np.ndarray
        Mod-cropped HR image aligned to ``img_lq``, float32, [0, 1].
    """
    # mod-crop
    h1, w1 = img.shape[:2]
    img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]
    hq = img.copy()
    target_h = int(hq.shape[0] / sf)
    target_w = int(hq.shape[1] / sf)

    # =========================================================
    # Stage 1: blur → resize → noise → JPEG
    # =========================================================

    if random.random() < blur_prob_1:
        img = add_blur(img, sf=sf)

    if random.random() < resize_prob_1:
        img = add_resize(img, sf=sf)

    # mutually exclusive noise gate
    rnd = random.random()
    if rnd < gaussian_noise_prob_1:
        img = add_Gaussian_noise(img, noise_level1=noise_level1_s1, noise_level2=noise_level2_s1)
    elif rnd < gaussian_noise_prob_1 + poisson_noise_prob_1:
        img = add_Poisson_noise(img)
    elif rnd < gaussian_noise_prob_1 + poisson_noise_prob_1 + speckle_noise_prob_1:
        img = add_speckle_noise(img)

    if random.random() < jpeg_prob_1:
        img = add_JPEG_noise(img)

    if isp_model is not None and random.random() < isp_prob:
        with torch.no_grad():
            img, hq = isp_model.forward(img.copy(), hq)

    img = np.clip(img, 0.0, 1.0)

    # optional intermediate resize-back (up then down) to simulate aliasing
    if random.random() < resize_back_prob:
        img = cv2.resize(img, (hq.shape[1], hq.shape[0]),
                         interpolation=random.choice([1, 2, 3]))
        img = cv2.resize(img, (target_w, target_h),
                         interpolation=random.choice([1, 2, 3]))
        img = np.clip(img, 0.0, 1.0)

    # =========================================================
    # Stage 2: blur → resize → noise → JPEG
    # =========================================================

    if random.random() < blur_prob_2:
        img = add_blur(img, sf=sf)

    if random.random() < resize_prob_2:
        img = add_resize(img, sf=sf)

    rnd = random.random()
    if rnd < gaussian_noise_prob_2:
        img = add_Gaussian_noise(img, noise_level1=noise_level1_s2, noise_level2=noise_level2_s2)
    elif rnd < gaussian_noise_prob_2 + poisson_noise_prob_2:
        img = add_Poisson_noise(img)
    elif rnd < gaussian_noise_prob_2 + poisson_noise_prob_2 + speckle_noise_prob_2:
        img = add_speckle_noise(img)

    if random.random() < jpeg_prob_2:
        img = add_JPEG_noise(img)

    if isp_model is not None and random.random() < isp_prob:
        with torch.no_grad():
            img, hq = isp_model.forward(img.copy(), hq)

    img = np.clip(img, 0.0, 1.0)

    # final resize to target LQ spatial size
    img = cv2.resize(img, (target_w, target_h), interpolation=random.choice([1, 2, 3]))
    img = np.clip(img, 0.0, 1.0)

    if random.random() < final_jpeg_prob:
        img = add_JPEG_noise(img)

    img_lq = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img_lq, hq


# ===========================================================================
# 3.  BSRGAN-Plus (Combined) Degradation
# ===========================================================================

def degrade_bsrgan_plus(
    img,
    sf=4,
    shuffle_prob=0.5,
    use_sharp=False,
    sharpening_weight=0.5,
    sharpening_radius=50,
    sharpening_threshold=10,
    poisson_prob=0.1,
    speckle_prob=0.1,
    isp_prob=0.1,
    noise_level1=2,
    noise_level2=25,
    isp_model=None,
):
    """Apply the extended BSRGAN-Plus combined degradation pipeline.

    The pipeline has **13 ordered degradation slots** in two symmetrical
    groups (A: slots 0–6, B: slots 7–12).  When the full-shuffle gate
    (``shuffle_prob``) is triggered the entire order is randomised;
    otherwise only the noise/JPEG slots within each group are locally
    shuffled.

    Slot map::

        0  – blur              (group A)
        1  – random resize     (group A)
        2  – Gaussian noise    (group A, locally shuffled)
        3  – Poisson noise     (group A, locally shuffled)
        4  – Speckle noise     (group A, locally shuffled)
        5  – ISP noise         (group A, locally shuffled)
        6  – JPEG              (group A, locally shuffled)
        7  – blur              (group B)
        8  – random resize     (group B)
        9  – Gaussian noise    (group B, locally shuffled)
        10 – Poisson noise     (group B, locally shuffled)
        11 – Speckle noise     (group B, locally shuffled)
        12 – ISP noise         (group B, locally shuffled)

    A final resize to target LQ resolution and a JPEG pass are always
    applied after the loop.

    This function operates on the **full HR image** and returns a full LR
    image at 1/sf resolution.  No patching or cropping is performed here.

    Parameters
    ----------
    img : np.ndarray
        Input HR image, shape HxWxC, dtype float32, range [0, 1].
    sf : int, optional
        Integer downscale factor.  Default: 4.
    shuffle_prob : float, optional
        Probability of fully randomising the 13-step degradation order.
        Default: 0.5.
    use_sharp : bool, optional
        Apply USM sharpening to the HR image **before** the degradation
        loop.  Default: False.
    sharpening_weight : float, optional
        USM residual weight.  Only used when ``use_sharp=True``.
        Default: 0.5.
    sharpening_radius : int, optional
        USM Gaussian blur radius.  Only used when ``use_sharp=True``.
        Default: 50.
    sharpening_threshold : int, optional
        USM pixel-difference threshold (0–255).  Only used when
        ``use_sharp=True``.  Default: 10.
    poisson_prob : float, optional
        Per-slot probability of Poisson noise.  Default: 0.1.
    speckle_prob : float, optional
        Per-slot probability of speckle noise.  Default: 0.1.
    isp_prob : float, optional
        Per-slot probability of ISP noise (requires ``isp_model``).
        Default: 0.1.
    noise_level1 : int, optional
        Lower bound of Gaussian noise level (intensity / 255).  Default: 2.
    noise_level2 : int, optional
        Upper bound of Gaussian noise level.  Default: 25.
    isp_model : object or None, optional
        Optional ISP noise model.  Default: None.

    Returns
    -------
    img_lq : np.ndarray
        Degraded LR image, shape (H/sf) x (W/sf) x C, float32, [0, 1].
    img_hq : np.ndarray
        Mod-cropped HR image aligned to ``img_lq``, float32, [0, 1].
    """
    # mod-crop
    h1, w1 = img.shape[:2]
    img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]

    # optional USM sharpening on HR before degradation
    if use_sharp:
        img = add_sharpening(
            img,
            weight=sharpening_weight,
            radius=sharpening_radius,
            threshold=sharpening_threshold,
        )

    hq = img.copy()

    # build degradation order
    if random.random() < shuffle_prob:
        shuffle_order = random.sample(range(13), 13)
    else:
        shuffle_order = list(range(13))
        shuffle_order[2:7]  = random.sample(shuffle_order[2:7],  5)
        shuffle_order[9:13] = random.sample(shuffle_order[9:13], 4)

    for i in shuffle_order:
        if i == 0:
            img = add_blur(img, sf=sf)
        elif i == 1:
            img = add_resize(img, sf=sf)
        elif i == 2:
            img = add_Gaussian_noise(img, noise_level1=noise_level1, noise_level2=noise_level2)
        elif i == 3:
            if random.random() < poisson_prob:
                img = add_Poisson_noise(img)
        elif i == 4:
            if random.random() < speckle_prob:
                img = add_speckle_noise(img)
        elif i == 5:
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)
        elif i == 6:
            img = add_JPEG_noise(img)
        elif i == 7:
            img = add_blur(img, sf=sf)
        elif i == 8:
            img = add_resize(img, sf=sf)
        elif i == 9:
            img = add_Gaussian_noise(img, noise_level1=noise_level1, noise_level2=noise_level2)
        elif i == 10:
            if random.random() < poisson_prob:
                img = add_Poisson_noise(img)
        elif i == 11:
            if random.random() < speckle_prob:
                img = add_speckle_noise(img)
        elif i == 12:
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)

    # final resize to target LQ spatial size
    target_h = int(hq.shape[0] / sf)
    target_w = int(hq.shape[1] / sf)
    img = cv2.resize(img, (target_w, target_h), interpolation=random.choice([1, 2, 3]))

    # final JPEG compression (always applied)
    img = add_JPEG_noise(img)

    img_lq = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img_lq, hq


# ===========================================================================
# 4.  Satellite-Optimized Degradation
# ===========================================================================

def degrade_satellite(
    img,
    sf=4,
    # --------------------- Stage 1 (sensor-level) -------------------------
    blur_prob_1=1.0,
    blur_type_1="mtf",                    # "mtf" (recommended) or "anisotropic"
    resize_prob_1=0.75,
    poisson_prob_1=0.75,                  # shot noise dominant in satellite
    read_noise_prob_1=0.55,               # thermal/read noise
    haze_prob_1=0.45,                     # atmosphere
    jpeg_prob_1=0.12,                     # very low (most satellite data lossless)
    # --------------------- Stage 2 (transmission/processing) --------------
    blur_prob_2=0.92,
    blur_type_2="mtf",
    resize_prob_2=0.70,
    poisson_prob_2=0.60,
    read_noise_prob_2=0.45,
    haze_prob_2=0.35,
    jpeg_prob_2=0.08,
    # --------------------- Final steps ------------------------------------
    final_jpeg_prob=0.10,
    resize_back_prob=0.35,                # simulate GSD / zoom mismatch
    isp_model=None,
    isp_prob=0.08,
    # --------------------- Noise & MTF ranges -----------------------------
    noise_level1=0.8,                     # lower than consumer cameras
    noise_level2=10.0,
    mtf_sigma_optics_range=(0.8, 2.8),
    mtf_detector_width_range=(0.7, 1.8),
    mtf_atm_sigma_range=(0.4, 1.8),
):
    """Satellite-optimized dual-stage degradation pipeline.

    Designed for optical remote-sensing imagery (e.g. Sentinel-2, WorldView,
    PlanetScope, etc.). Based on:
      • Dong et al. (2022) – practical RS degradation with real-kernel mixing
      • Chung et al. (2023) – explicit MTF-based filters
      • Shinohara et al. (2022) – adaptation of BSRGAN/Real-ESRGAN to satellite

    Key differences from consumer pipelines:
    - Primary blur = parametric MTF/PSF (optics + detector + atmosphere)
    - Dominant noise = Poisson (shot) + low-level Gaussian read noise
    - Atmospheric haze/turbulence explicitly modeled
    - JPEG almost disabled (satellite data rarely uses heavy JPEG)
    - Configurable per-stage probabilities for easy ablation / tuning

    Returns
    -------
    img_lq : np.ndarray   (H/sf × W/sf × C, float32, [0,1])
    img_hq : np.ndarray   (mod-cropped HR, float32, [0,1])
    """
    # mod-crop to exact multiple of sf
    h1, w1 = img.shape[:2]
    img = img.copy()[: h1 - h1 % sf, : w1 - w1 % sf, ...]
    hq = img.copy()
    target_h = int(hq.shape[0] / sf)
    target_w = int(hq.shape[1] / sf)

    # =========================================================
    # Stage 1 – raw sensor / acquisition degradations
    # =========================================================
    if random.random() < blur_prob_1:
        if blur_type_1 == "mtf":
            img = add_mtf_blur(
                img,
                sf=sf,
                sigma_optics_range=mtf_sigma_optics_range,
                detector_width_range=mtf_detector_width_range,
                atm_sigma_range=mtf_atm_sigma_range,
            )
        else:  # fallback to original anisotropic/gaussian
            img = add_blur(img, sf=sf)

    if random.random() < resize_prob_1:
        img = add_resize(img, sf=sf)

    if random.random() < poisson_prob_1:
        img = add_Poisson_noise(img)

    if random.random() < read_noise_prob_1:
        img = add_Gaussian_noise(img, noise_level1=noise_level1, noise_level2=noise_level2)

    if random.random() < haze_prob_1:
        img = add_atmospheric_haze(img)

    if random.random() < jpeg_prob_1:
        img = add_JPEG_noise(img)

    if isp_model is not None and random.random() < isp_prob:
        with torch.no_grad():
            img, hq = isp_model.forward(img.copy(), hq)

    img = np.clip(img, 0.0, 1.0)

    # Optional intermediate resize-back (simulates different GSD pipelines)
    if random.random() < resize_back_prob:
        img = cv2.resize(img, (hq.shape[1], hq.shape[0]),
                         interpolation=random.choice([1, 2, 3]))
        img = cv2.resize(img, (target_w, target_h),
                         interpolation=random.choice([1, 2, 3]))
        img = np.clip(img, 0.0, 1.0)

    # =========================================================
    # Stage 2 – transmission / ground-station processing
    # =========================================================
    if random.random() < blur_prob_2:
        if blur_type_2 == "mtf":
            img = add_mtf_blur(
                img,
                sf=sf,
                sigma_optics_range=mtf_sigma_optics_range,
                detector_width_range=mtf_detector_width_range,
                atm_sigma_range=mtf_atm_sigma_range,
            )
        else:
            img = add_blur(img, sf=sf)

    if random.random() < resize_prob_2:
        img = add_resize(img, sf=sf)

    if random.random() < poisson_prob_2:
        img = add_Poisson_noise(img)

    if random.random() < read_noise_prob_2:
        img = add_Gaussian_noise(img, noise_level1=noise_level1 * 0.85,
                                 noise_level2=noise_level2 * 0.85)

    if random.random() < haze_prob_2:
        img = add_atmospheric_haze(img)

    if random.random() < jpeg_prob_2:
        img = add_JPEG_noise(img)

    if isp_model is not None and random.random() < isp_prob:
        with torch.no_grad():
            img, hq = isp_model.forward(img.copy(), hq)

    img = np.clip(img, 0.0, 1.0)

    # =========================================================
    # Final exact LQ resolution + light JPEG
    # =========================================================
    img = cv2.resize(img, (target_w, target_h),
                     interpolation=random.choice([1, 2, 3]))
    img = np.clip(img, 0.0, 1.0)

    if random.random() < final_jpeg_prob:
        img = add_JPEG_noise(img)

    img_lq = img.astype(np.float32)
    return img_lq, hq