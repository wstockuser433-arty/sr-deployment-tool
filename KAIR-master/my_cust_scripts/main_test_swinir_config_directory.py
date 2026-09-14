"""
SwinIR testing script driven by an in-file config.

The local CONFIG dict provides the LR, HR, SR, and model-weight paths. The
known SwinIR architecture is stored locally, so the option file is no longer
required. The script runs SwinIR on the LR images, saves SR outputs, and
reports the average of 8 metrics against HR:
PSNR, SSIM, IT-SSIM, SAM, UIQI, RMSE, FSIM, and SRER.

Edit CONFIG below and run:

    python main_test_swinir_config.py
"""

import logging
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch

from utils import utils_image as util


CONFIG = {
    # "model_path": "superresolution/swinir_sr_realworld_x2_gan_airbus_ucmerced_both_synth/models/175000_G.pth", #model path needs to be inquired from them or added if known
    "model_path": "/home/hassan/Documents/Hassan/AI/fast_superresolution/KAIR/model_zoo/002_swinir_sr_realworld_x2_gan_airbus_ucmerced_both_synth/03Jun26_185000_E.pth", 
    # "lr_dir": "/home/hassan/Documents/Hassan/AI/fast_superresolution/KAIR/testsets/my_images/LR_x4_bicubic",
    # "lr_dir": "/home/hassan/Documents/Hassan/AI/fast_superresolution/KAIR/testsets/my_images/LR_x4_bicubic/nwpu/test",
    "lr_dir": "/home/hassan/Documents/Hassan/AI/fast_superresolution/KAIR/testsets/my_images/LR_x2_bicubic/uc-merced",
    # "hr_dir": "/home/hassan/Documents/Hassan/AI/fast_superresolution/KAIR/testsets/my_images/HR",
    "hr_dir": "/home/hassan/Documents/Hassan/SR Enhancement Work/Datasets/uc-merced-land-use-dataset/UCMerced_LandUse/Images",
    "sr_dir": "/home/hassan/Documents/Hassan/AI/fast_superresolution/KAIR/testsets/my_images/sr_gan/uc-merced/x2_bicubic",
    "tile": None,   #512 could be a good starting point for satellite imagery
    "tile_overlap": 32,
    "overwrite_sr": True,
    "log_dir": "testsets/xview_test",
}

MODEL_CONFIG = {
    "upscale": 2,
    "in_chans": 3,
    "img_size": 128,
    "window_size": 8,
    "img_range": 1.0,
    "depths": [6, 6, 6, 6, 6, 6],
    "embed_dim": 180,
    "num_heads": [6, 6, 6, 6, 6, 6],
    "mlp_ratio": 2,
    "upsampler": "pixelshuffle",
    "resi_connection": "1conv",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METRIC_NAMES = ["psnr", "ssim", "it_ssim", "sam", "uiqi", "rmse", "fsim", "srer"]


# def list_images(folder: Path):
#     return [path for path in sorted(folder.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]

def list_images(folder: Path):
    return [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

# def build_index(folder: Path):
#     index = {}
#     for path in list_images(folder):
#         index[path.stem] = path
#     return index


def build_index(folder: Path):
    index = {}

    for path in list_images(folder):

        rel_path = path.relative_to(folder)

        key = rel_path.with_suffix("").as_posix()

        index[key] = path

    return index



def build_model(device: torch.device):
    from models.network_swinir import SwinIR as net

    return net(
        upscale=MODEL_CONFIG["upscale"],
        in_chans=MODEL_CONFIG["in_chans"],
        img_size=MODEL_CONFIG["img_size"],
        window_size=MODEL_CONFIG["window_size"],
        img_range=MODEL_CONFIG["img_range"],
        depths=MODEL_CONFIG["depths"],
        embed_dim=MODEL_CONFIG["embed_dim"],
        num_heads=MODEL_CONFIG["num_heads"],
        mlp_ratio=MODEL_CONFIG["mlp_ratio"],
        upsampler=MODEL_CONFIG["upsampler"],
        resi_connection=MODEL_CONFIG["resi_connection"],
    ).to(device)


def load_model(checkpoint_path: str, device: torch.device):
    if not checkpoint_path:
        raise FileNotFoundError("CONFIG['model_path'] must point to the model weights.")

    model = build_model(device)

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    state_dict = torch.load(checkpoint_file, map_location=device)
    if isinstance(state_dict, dict):
        for key in ("params_ema", "params", "state_dict"):
            if key in state_dict:
                state_dict = state_dict[key]
                break

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)
    if image.shape[2] >= 3:
        image = image[:, :, :3][:, :, [2, 1, 0]]
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image).float().unsqueeze(0).div(255.0)


def tensor_to_bgr_uint8(output: torch.Tensor) -> np.ndarray:
    image = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if image.ndim == 3:
        image = np.transpose(image[[2, 1, 0], :, :], (1, 2, 0))
    return (image * 255.0).round().astype(np.uint8)


def pad_to_window_size(image: torch.Tensor, window_size: int):
    _, _, height, width = image.size()
    height_pad = (window_size - height % window_size) % window_size
    width_pad = (window_size - width % window_size) % window_size

    if height_pad > 0:
        image = torch.cat([image, torch.flip(image, [2])], 2)[:, :, : height + height_pad, :]
    if width_pad > 0:
        image = torch.cat([image, torch.flip(image, [3])], 3)[:, :, :, : width + width_pad]
    return image, height, width


def run_model(image: torch.Tensor, model: torch.nn.Module, scale: int, window_size: int, tile, tile_overlap: int):
    if tile is None:
        return model(image)

    batch, channels, height, width = image.size()
    tile = min(tile, height, width)
    if tile % window_size != 0:
        raise ValueError("tile size should be a multiple of window_size")

    stride = tile - tile_overlap
    height_indices = list(range(0, height - tile, stride)) + [height - tile]
    width_indices = list(range(0, width - tile, stride)) + [width - tile]

    output = torch.zeros(batch, channels, height * scale, width * scale, device=image.device, dtype=image.dtype)
    weight = torch.zeros_like(output)

    for height_index in height_indices:
        for width_index in width_indices:
            input_patch = image[..., height_index:height_index + tile, width_index:width_index + tile]
            output_patch = model(input_patch)
            output[..., height_index * scale:(height_index + tile) * scale, width_index * scale:(width_index + tile) * scale].add_(output_patch)
            weight[..., height_index * scale:(height_index + tile) * scale, width_index * scale:(width_index + tile) * scale].add_(1)

    return output.div_(weight)


def calculate_metrics(sr: np.ndarray, hr: np.ndarray, border: int):
    return {
        "psnr": util.calculate_psnr(sr, hr, border=border),
        "ssim": util.calculate_ssim(sr, hr, border=border),
        "it_ssim": util.calculate_it_ssim(sr, hr, border=border),
        "sam": util.calculate_sam(sr, hr, border=border),
        "uiqi": util.calculate_uiqi(sr, hr, border=border),
        "rmse": util.calculate_rmse(sr, hr, border=border),
        "fsim": util.calculate_fsim(sr, hr, border=border),
        "srer": util.calculate_srer(sr, hr, border=border),
    }


def main():
    lr_dir = Path(CONFIG["lr_dir"])
    hr_dir = Path(CONFIG["hr_dir"])
    sr_dir = Path(CONFIG["sr_dir"])
    sr_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(CONFIG["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"{sr_dir.name}.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    if not lr_dir.is_dir():
        raise FileNotFoundError(f"LR folder not found: {lr_dir}")
    if not hr_dir.is_dir():
        raise FileNotFoundError(f"HR folder not found: {hr_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CONFIG["model_path"], device)

    scale = int(MODEL_CONFIG["upscale"])
    window_size = int(MODEL_CONFIG["window_size"])
    border = scale

    lr_index = build_index(lr_dir)
    hr_index = build_index(hr_dir)
    common_names = sorted(set(lr_index.keys()) & set(hr_index.keys()))
    if not common_names:
        raise RuntimeError(f"No matching LR/HR filenames were found in {lr_dir} and {hr_dir}.")

    totals = defaultdict(float)
    processed = 0

    for index, name in enumerate(common_names):
        lr_path = lr_index[name]
        hr_path = hr_index[name]

        lr_image = cv2.imread(str(lr_path), cv2.IMREAD_UNCHANGED)
        hr_image = cv2.imread(str(hr_path), cv2.IMREAD_UNCHANGED)
        if lr_image is None:
            raise RuntimeError(f"Could not read LR image: {lr_path}")
        if hr_image is None:
            raise RuntimeError(f"Could not read HR image: {hr_path}")

        lr_tensor = image_to_tensor(lr_image).to(device)
        lr_tensor, height, width = pad_to_window_size(lr_tensor, window_size)

        with torch.no_grad():
            output = run_model(lr_tensor, model, scale, window_size, CONFIG["tile"], CONFIG["tile_overlap"])
            output = output[..., : height * scale, : width * scale]

        sr_image = tensor_to_bgr_uint8(output)
        # if CONFIG["overwrite_sr"] or not (sr_dir / f"{name}_SwinIR.png").exists():
        #     cv2.imwrite(str(sr_dir / f"{name}_SwinIR.png"), sr_image)

        relative_sr_path = Path(f"{name}_SwinIR.png")
        sr_output_path = sr_dir / relative_sr_path
        sr_output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        if CONFIG["overwrite_sr"] or not sr_output_path.exists():
            cv2.imwrite(str(sr_output_path), sr_image)

        hr_image = hr_image.astype(np.uint8)
        if hr_image.ndim == 3:
            hr_image = hr_image[: height * scale, : width * scale, ...]
        else:
            hr_image = np.squeeze(hr_image[: height * scale, : width * scale, ...])

        metrics = calculate_metrics(sr_image, hr_image, border=border)
        for key, value in metrics.items():
            totals[key] += float(value)
        processed += 1

        metric_text = "; ".join(f"{key.upper()}: {value:.4f}" for key, value in metrics.items())
        # logger.info(f"{index:04d} {name:30s} {metric_text}")
        logger.info(f"{index:04d} {name} {metric_text}")

    logger.info("\n" + "=" * 80)
    logger.info(f"SR folder: {sr_dir}")
    logger.info(f"Total images: {processed}")
    for key in METRIC_NAMES:
        logger.info(f"Average {key.upper()}: {totals[key] / processed:.4f}")


if __name__ == "__main__":
    main()