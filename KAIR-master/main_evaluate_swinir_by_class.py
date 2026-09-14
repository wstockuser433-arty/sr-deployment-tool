"""
SwinIR Evaluation Script with Class-Based Metrics and Per-Class Summaries

Evaluates SwinIR super-resolution on UCMerced images from a mixed dataset.

HR images live flat in hr_dir, named: ucmerced_{class}{digits}.png
  e.g. ucmerced_agricultural06.png, ucmerced_airplane05.png

SR images live in sr_base_dir under per-image subdirectories:
  sr_base_dir/ucmerced_agricultural06/ucmerced_agricultural06_175000.png
The script picks the file with the highest iteration number automatically.

Class names are derived by stripping the 'ucmerced_' prefix and trailing digits,
e.g. ucmerced_agricultural06 -> class 'agricultural'.

Edit CONFIG below and run:

    python main_evaluate_swinir_by_class.py
"""

from pathlib import Path
from collections import defaultdict
import re
import cv2
import numpy as np

from utils import utils_image as util


CONFIG = {
    "hr_dir": "testsets/uc_airbus_both_synth/hr",
    "sr_base_dir": "superresolution/swinir_sr_x2_psnr_airbus_ucmerced_both_synth/images",
    "log_file": "superresolution/swinir_sr_x2_psnr_airbus_ucmerced_both_synth/classwise_evaluation_log.txt",
    "border": 2,
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METRIC_NAMES = ["psnr", "ssim", "it_ssim", "sam", "uiqi", "rmse", "fsim", "srer"]


def list_images(folder: Path):
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def extract_ucmerced_class(stem: str) -> str:
    """
    Extract class name from a ucmerced HR stem.
    e.g. ucmerced_agricultural06 -> 'agricultural'
         ucmerced_airplane05     -> 'airplane'
    """
    prefix = "ucmerced_"
    if stem.startswith(prefix):
        remainder = stem[len(prefix):]
        class_name = re.sub(r"\d+$", "", remainder)
        if class_name:
            return class_name
    # fallback: everything between first and last underscore-separated token
    parts = stem.split("_")
    return "_".join(parts[1:-1]) if len(parts) >= 3 else stem


def find_latest_sr_image(sr_base_dir: Path, hr_stem: str) -> Path | None:
    """
    Return the SR image with the highest iteration number for hr_stem.
    Looks in: sr_base_dir / hr_stem / {hr_stem}_{digits}.png
    """
    sr_image_dir = sr_base_dir / hr_stem
    if not sr_image_dir.exists():
        return None

    pattern = re.compile(rf"^{re.escape(hr_stem)}_(\d+)\.png$")
    candidates = []
    for fp in sr_image_dir.iterdir():
        if fp.is_file() and fp.suffix.lower() == ".png":
            m = pattern.match(fp.name)
            if m:
                candidates.append((int(m.group(1)), fp))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


def calculate_metrics(sr: np.ndarray, hr: np.ndarray, border: int) -> dict:
    return {
        "psnr":    util.calculate_psnr(sr, hr, border=border),
        "ssim":    util.calculate_ssim(sr, hr, border=border),
        "it_ssim": util.calculate_it_ssim(sr, hr, border=border),
        "sam":     util.calculate_sam(sr, hr, border=border),
        "uiqi":    util.calculate_uiqi(sr, hr, border=border),
        "rmse":    util.calculate_rmse(sr, hr, border=border),
        "fsim":    util.calculate_fsim(sr, hr, border=border),
        "srer":    util.calculate_srer(sr, hr, border=border),
    }


def format_metrics(metrics: dict) -> str:
    return " | ".join(f"{name.upper()}: {metrics[name]:.4f}" for name in METRIC_NAMES)


def main():
    hr_dir = Path(CONFIG["hr_dir"])
    sr_base_dir = Path(CONFIG["sr_base_dir"])
    log_file = Path(CONFIG["log_file"])
    border = CONFIG["border"]

    if not hr_dir.is_dir():
        raise FileNotFoundError(f"HR folder not found: {hr_dir}")
    if not sr_base_dir.is_dir():
        raise FileNotFoundError(f"SR base folder not found: {sr_base_dir}")

    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Collect only ucmerced HR images and group by extracted class
    all_hr = list_images(hr_dir)
    if not all_hr:
        raise RuntimeError(f"No HR images found in {hr_dir}")

    class_groups: dict[str, list[Path]] = defaultdict(list)
    for hr_path in all_hr:
        if not hr_path.stem.startswith("ucmerced_"):
            continue
        cls = extract_ucmerced_class(hr_path.stem)
        class_groups[cls].append(hr_path)

    if not class_groups:
        raise RuntimeError(f"No ucmerced_* images found in {hr_dir}")

    per_image_results = []
    class_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    class_counts: dict[str, int] = defaultdict(int)

    with open(log_file, "w") as log:
        log.write("=" * 110 + "\n")
        log.write("SwinIR Real-World x2 GAN — UCMerced Evaluation Results by Class\n")
        log.write("=" * 110 + "\n\n")

        for class_name in sorted(class_groups.keys()):
            hr_paths = sorted(class_groups[class_name])

            log.write(f"\nClass: {class_name}\n")
            log.write("-" * 110 + "\n")

            for hr_path in hr_paths:
                stem = hr_path.stem
                sr_path = find_latest_sr_image(sr_base_dir, stem)

                if sr_path is None:
                    log.write(f"  {stem}: SKIPPED (no SR image found in {sr_base_dir / stem})\n")
                    continue

                hr_image = cv2.imread(str(hr_path), cv2.IMREAD_UNCHANGED)
                sr_image = cv2.imread(str(sr_path), cv2.IMREAD_UNCHANGED)

                if hr_image is None or sr_image is None:
                    log.write(f"  {stem}: SKIPPED (could not read image)\n")
                    continue

                # Align sizes: SR should match HR for metric computation
                if hr_image.shape != sr_image.shape:
                    sr_image = cv2.resize(
                        sr_image,
                        (hr_image.shape[1], hr_image.shape[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )

                metrics = calculate_metrics(sr_image, hr_image, border=border)

                per_image_results.append({
                    "class": class_name,
                    "image": stem,
                    "sr_path": str(sr_path),
                    "metrics": metrics,
                })

                for metric_name, value in metrics.items():
                    class_totals[class_name][metric_name] += float(value)
                class_counts[class_name] += 1

                iter_tag = re.search(r"_(\d+)\.png$", sr_path.name)
                iter_str = f"[iter {iter_tag.group(1)}]" if iter_tag else ""
                log.write(f"  {stem:50s} {iter_str:15s} {format_metrics(metrics)}\n")

            if class_counts[class_name] > 0:
                log.write(f"\n  Class Summary ({class_counts[class_name]} images):\n")
                for metric_name in METRIC_NAMES:
                    avg_value = class_totals[class_name][metric_name] / class_counts[class_name]
                    log.write(f"    Average {metric_name.upper()}: {avg_value:.4f}\n")
                log.write("\n")

        # Overall summary
        total_images = sum(class_counts.values())
        log.write("\n" + "=" * 110 + "\n")
        log.write("OVERALL SUMMARY\n")
        log.write("=" * 110 + "\n\n")
        log.write(f"Total images evaluated: {total_images}\n")
        log.write(f"Total classes: {len(class_groups)}\n\n")

        # Per-class summary table
        col_w = 12
        log.write("Per-Class Summary:\n")
        log.write("-" * 110 + "\n")
        header = f"{'Class':<30} {'Images':<10}"
        for m in METRIC_NAMES:
            header += f"{m.upper():<{col_w}}"
        log.write(header + "\n")
        log.write("-" * 110 + "\n")

        for class_name in sorted(class_groups.keys()):
            row = f"{class_name:<30} {class_counts[class_name]:<10}"
            for m in METRIC_NAMES:
                avg = class_totals[class_name][m] / class_counts[class_name] if class_counts[class_name] > 0 else 0.0
                row += f"{avg:<{col_w}.4f}"
            log.write(row + "\n")

        log.write("\n" + "-" * 110 + "\n")
        global_row = f"{'GLOBAL AVERAGE':<30} {total_images:<10}"
        for m in METRIC_NAMES:
            g_avg = (
                sum(class_totals[cls][m] for cls in class_groups) / total_images
                if total_images > 0 else 0.0
            )
            global_row += f"{g_avg:<{col_w}.4f}"
        log.write(global_row + "\n")
        log.write("\n" + "=" * 110 + "\n")

    print(f"Evaluation complete. Results saved to: {log_file}")
    print(f"Total images evaluated: {total_images}")
    print(f"Total classes: {len(class_groups)}")
    for class_name in sorted(class_groups.keys()):
        print(f"  {class_name}: {class_counts[class_name]} images")


if __name__ == "__main__":
    main()
