import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


CONFIG = {
    "lr_dir": "testsets/xview_test/lr",
    "hr_dir": "testsets/xview_test/hr",
    "sr_dir": "testsets/xview_test/sr_gan",
    "output_dir": "testsets/xview_test/comparisons_side_by_side",
    # set to None/"" if no iter number appended to sr patches
    "sr_iteration": "",
    "fallback_to_latest_sr_when_iter_missing": True,
    "image_extensions": [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
    "recursive_lr": False,
    "recursive_hr": False,
    "resize_lr_to_hr": True,
    "resize_sr_to_hr": True,
    "padding": 10,
    "font_size": 18,
    "label_lr": "LR",
    "label_sr": "SR",
    "label_hr": "HR",
    "patch_count": 100,
    "include_patches": [],
    "random_seed": 42,
    "output_suffix": "_comparison",
    "output_ext": ".png"
}


def normalize_patch_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        return ""
    return Path(text).stem


def choose_patch_names(common_names: List[str], config: Dict) -> List[str]:
    included_raw = config.get("include_patches", [])
    included: List[str] = []
    for item in included_raw:
        patch = normalize_patch_name(item)
        if patch and patch not in included:
            included.append(patch)

    available = set(common_names)
    selected = [name for name in included if name in available]
    missing_included = [name for name in included if name not in available]
    if missing_included:
        print(f"[Warning] Requested include_patches not found in LR/HR: {missing_included}")

    patch_count = config.get("patch_count", None)
    if patch_count is None:
        # No limit: keep all common names, but place selected ones first.
        rest = [name for name in common_names if name not in set(selected)]
        return selected + rest

    try:
        target = int(patch_count)
    except (TypeError, ValueError):
        print(f"[Warning] Invalid patch_count={patch_count}. Falling back to all patches.")
        rest = [name for name in common_names if name not in set(selected)]
        return selected + rest

    if target <= 0:
        print(f"[Warning] patch_count={target} is not positive. Nothing to process.")
        return []

    if target < len(selected):
        print(
            f"[Warning] patch_count={target} is less than include_patches size={len(selected)}. "
            f"Using {len(selected)} to preserve requested includes."
        )
        target = len(selected)

    target = min(target, len(common_names))
    remaining = [name for name in common_names if name not in set(selected)]
    needed = max(0, target - len(selected))

    seed = config.get("random_seed", None)
    rng = random.Random(seed)

    if needed >= len(remaining):
        fill = remaining
    else:
        fill = rng.sample(remaining, needed)

    return selected + fill


def normalize_extensions(exts: List[str]) -> set:
    normalized = set()
    for ext in exts:
        ext = ext.lower().strip()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext)
    return normalized


def list_image_files(folder: Path, allowed_exts: set, recursive: bool = False) -> List[Path]:
    if not folder.exists():
        return []
    if recursive:
        candidates = folder.rglob("*")
    else:
        candidates = folder.glob("*")
    return [p for p in candidates if p.is_file() and p.suffix.lower() in allowed_exts]


def map_stem_to_latest_file(image_paths: List[Path]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in image_paths:
        stem = path.stem
        current = mapping.get(stem)
        if current is None:
            mapping[stem] = path
        else:
            if path.stat().st_mtime > current.stat().st_mtime:
                mapping[stem] = path
    return mapping


def parse_sr_iteration(stem: str, base_name: str) -> Optional[int]:
    prefix = f"{base_name}_"
    if not stem.startswith(prefix):
        return None
    tail = stem[len(prefix):]
    if tail.isdigit():
        return int(tail)
    return None


def _pick_sr_candidate(
    candidates: List[Path],
    base_name: str,
    requested_iter: Optional[str],
    fallback_to_latest: bool,
) -> Optional[Path]:
    """
    From a list of candidate SR files, return the best match for base_name.

    Priority:
      - If requested_iter is set: exact stem "{base_name}_{iter}" wins.
        Falls back to highest-numbered iteration (or any file) when
        fallback_to_latest is True.
      - If requested_iter is None/empty: files with an iteration suffix are
        preferred (highest number wins); files without any suffix (just
        "{base_name}") are accepted as a fallback.  This lets the function
        work correctly when SR outputs have no iteration number in their name.
    """
    iter_str = str(requested_iter).strip() if requested_iter is not None else ""

    if iter_str:
        exact_stem = f"{base_name}_{iter_str}"
        for p in candidates:
            if p.stem == exact_stem:
                return p
        if not fallback_to_latest:
            return None
        print(f"[Warning] Exact SR iteration not found for {base_name} (iter={iter_str}). Falling back.")

    with_iter: List[Tuple[int, Path]] = []
    without_iter: List[Path] = []
    for p in candidates:
        parsed = parse_sr_iteration(p.stem, base_name)
        if parsed is None:
            without_iter.append(p)
        else:
            with_iter.append((parsed, p))

    if with_iter:
        with_iter.sort(key=lambda x: x[0], reverse=True)
        return with_iter[0][1]

    if without_iter:
        without_iter.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return without_iter[0]

    return None


def find_sr_file_for_image(
    sr_root: Path,
    base_name: str,
    allowed_exts: set,
    requested_iter: Optional[str],
    fallback_to_latest: bool,
) -> Optional[Path]:
    """
    Locate the SR output file for base_name, searching two layouts:

    1. Subdirectory layout  — sr_root / base_name / {base_name}[_{iter}].ext
       (used when each image has its own results folder)
    2. Flat layout          — sr_root / {base_name}[_{iter}].ext
       (used when all SR outputs are dumped directly into sr_root)

    In both layouts the function works regardless of whether an iteration
    number is present in the filename.  When requested_iter is None or an
    empty string the exact-match step is skipped and the best available file
    is returned (highest iteration number, or the plain "{base_name}" file).
    """
    # --- 1. Subdirectory layout -------------------------------------------
    image_sr_dir = sr_root / base_name
    if image_sr_dir.exists() and image_sr_dir.is_dir():
        subdir_candidates = [
            p for p in image_sr_dir.iterdir()
            if p.is_file() and p.suffix.lower() in allowed_exts
        ]
        result = _pick_sr_candidate(subdir_candidates, base_name, requested_iter, fallback_to_latest)
        if result is not None:
            return result

    # --- 2. Flat layout ------------------------------------------------------
    # Match files whose stem is exactly base_name or base_name_{anything}.
    if sr_root.exists() and sr_root.is_dir():
        prefix = f"{base_name}_"
        flat_candidates = [
            p for p in sr_root.iterdir()
            if p.is_file()
            and p.suffix.lower() in allowed_exts
            and (p.stem == base_name or p.stem.startswith(prefix))
        ]
        result = _pick_sr_candidate(flat_candidates, base_name, requested_iter, fallback_to_latest)
        if result is not None:
            return result

    return None


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def add_label(image: np.ndarray, text: str, font_size: int) -> np.ndarray:
    out = image.copy()
    font_scale = max(0.4, font_size / 30.0)
    thickness = max(1, int(font_scale * 1.5))
    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = 10, 10 + th

    cv2.rectangle(out, (x - 2, y - th - 2), (x + tw + 2, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(out, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def build_comparison(
    lr_path: Path,
    sr_path: Path,
    hr_path: Path,
    config: Dict,
) -> Optional[np.ndarray]:
    lr = cv2.imread(str(lr_path), cv2.IMREAD_UNCHANGED)
    sr = cv2.imread(str(sr_path), cv2.IMREAD_UNCHANGED)
    hr = cv2.imread(str(hr_path), cv2.IMREAD_UNCHANGED)

    if lr is None or sr is None or hr is None:
        return None

    lr = ensure_bgr(lr)
    sr = ensure_bgr(sr)
    hr = ensure_bgr(hr)

    h, w = hr.shape[:2]
    if config.get("resize_lr_to_hr", True) and lr.shape[:2] != (h, w):
        lr = cv2.resize(lr, (w, h), interpolation=cv2.INTER_NEAREST)
    if config.get("resize_sr_to_hr", True) and sr.shape[:2] != (h, w):
        sr = cv2.resize(sr, (w, h), interpolation=cv2.INTER_CUBIC)

    if not (lr.shape[:2] == hr.shape[:2] == sr.shape[:2]):
        return None

    lr = add_label(lr, config.get("label_lr", "LR"), int(config.get("font_size", 18)))
    sr = add_label(sr, config.get("label_sr", "SR"), int(config.get("font_size", 18)))
    hr = add_label(hr, config.get("label_hr", "HR"), int(config.get("font_size", 18)))

    pad_w = int(config.get("padding", 10))
    pad = np.full((h, pad_w, 3), 255, dtype=np.uint8)
    return np.hstack([lr, pad, sr, pad, hr])


def main() -> None:
    config = CONFIG

    lr_dir = Path(config["lr_dir"])
    hr_dir = Path(config["hr_dir"])
    sr_dir = Path(config["sr_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_exts = normalize_extensions(config.get("image_extensions", CONFIG["image_extensions"]))
    lr_files = list_image_files(lr_dir, allowed_exts, bool(config.get("recursive_lr", False)))
    hr_files = list_image_files(hr_dir, allowed_exts, bool(config.get("recursive_hr", False)))

    lr_map = map_stem_to_latest_file(lr_files)
    hr_map = map_stem_to_latest_file(hr_files)

    common_names = sorted(set(lr_map.keys()) & set(hr_map.keys()))
    if not common_names:
        print("No common LR/HR image names found.")
        return

    selected_names = choose_patch_names(common_names, config)
    if not selected_names:
        print("No patches selected after applying patch_count/include_patches rules.")
        return

    sr_iteration = config.get("sr_iteration", None)
    fallback_to_latest_sr = bool(config.get("fallback_to_latest_sr_when_iter_missing", True))

    saved = 0
    missing_sr = 0
    skipped = 0

    for name in selected_names:
        lr_path = lr_map[name]
        hr_path = hr_map[name]
        sr_path = find_sr_file_for_image(
            sr_dir,
            name,
            allowed_exts,
            sr_iteration,
            fallback_to_latest_sr,
        )

        if sr_path is None:
            missing_sr += 1
            print(f"[Missing SR] {name} (iter={sr_iteration})")
            continue

        merged = build_comparison(lr_path, sr_path, hr_path, config)
        if merged is None:
            skipped += 1
            print(f"[Skipped] Could not build comparison for {name}")
            continue

        out_name = f"{name}{config.get('output_suffix', '_comparison')}{config.get('output_ext', '.png')}"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), merged)
        saved += 1

    print("\nFinished.")
    print(f"  Total LR/HR pairs: {len(common_names)}")
    print(f"  Selected patches: {len(selected_names)}")
    print(f"  Saved comparisons: {saved}")
    print(f"  Missing SR: {missing_sr}")
    print(f"  Skipped: {skipped}")


if __name__ == "__main__":
    main()