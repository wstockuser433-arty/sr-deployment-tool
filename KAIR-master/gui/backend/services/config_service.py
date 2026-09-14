"""
config_service.py
=================
Reads and writes KAIR-style JSON configs (strips // comments).
Scans superresolution/ for training runs and model checkpoints.
"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Project root (KAIR root, two levels up from gui/backend) ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUPERRESOLUTION_DIR = PROJECT_ROOT / "superresolution"
OPTIONS_SWINIR_DIR = PROJECT_ROOT / "options" / "swinir"


def _strip_comments(text: str) -> str:
    """Remove // comments from KAIR-style JSON."""
    # Remove // comments that are not inside strings
    result = []
    i = 0
    in_string = False
    while i < len(text):
        c = text[i]
        if c == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
        if not in_string and c == "/" and i + 1 < len(text) and text[i + 1] == "/":
            # skip to end of line
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        result.append(c)
        i += 1
    return "".join(result)


def load_kair_json(path: Path) -> Dict[str, Any]:
    """Load a KAIR JSON file (with // comments) and return a dict."""
    text = path.read_text(encoding="utf-8")
    clean = _strip_comments(text)
    return json.loads(clean)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Write a dict as standard JSON (no comments)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── SwinIR config files ───────────────────────────────────────────────────────

def list_swinir_configs() -> List[Dict[str, str]]:
    """Return all .json config files in options/swinir/."""
    configs = []
    if OPTIONS_SWINIR_DIR.is_dir():
        for p in sorted(OPTIONS_SWINIR_DIR.glob("*.json")):
            configs.append({"name": p.name, "path": str(p.relative_to(PROJECT_ROOT))})
    return configs


def load_swinir_config(config_name: str) -> Dict[str, Any]:
    """Load a named swinir config file."""
    path = OPTIONS_SWINIR_DIR / config_name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return load_kair_json(path)


def _extract_model_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Pull MODEL_CONFIG-compatible fields from a KAIR config dict."""
    net_g = cfg.get("netG", {})
    return {
        "upscale":         net_g.get("upscale",          cfg.get("scale",      2)),
        "in_chans":        net_g.get("in_chans",         cfg.get("n_channels", 3)),
        "img_size":        net_g.get("img_size",         128),
        "window_size":     net_g.get("window_size",      8),
        "img_range":       net_g.get("img_range",        1.0),
        "depths":          net_g.get("depths",           [6, 6, 6, 6, 6, 6]),
        "embed_dim":       net_g.get("embed_dim",        180),
        "num_heads":       net_g.get("num_heads",        [6, 6, 6, 6, 6, 6]),
        "mlp_ratio":       net_g.get("mlp_ratio",        2),
        "upsampler":       net_g.get("upsampler",        "pixelshuffle"),
        "resi_connection": net_g.get("resi_connection",  "1conv"),
    }


def get_model_config_from_options(options_name: str) -> Dict[str, Any]:
    """Extract MODEL_CONFIG fields from an options/swinir JSON file."""
    cfg = load_swinir_config(options_name)
    return _extract_model_config(cfg)


# ── Training runs ─────────────────────────────────────────────────────────────

def _find_latest_checkpoint(models_dir: Path) -> Tuple[Optional[int], Optional[str]]:
    """
    Scan a models/ directory for the highest-numbered *_E.pth (EMA weights).
    Falls back to *_G.pth if no E weights found.
    Returns (iteration, relative_path) or (None, None).
    """
    best_iter = -1
    best_path = None

    for suffix in ("_E.pth", "_G.pth"):
        for p in models_dir.glob(f"*{suffix}"):
            stem = p.stem  # e.g. "175000_E"
            parts = stem.split("_")
            try:
                iteration = int(parts[0])
            except (ValueError, IndexError):
                continue
            if iteration > best_iter:
                best_iter = iteration
                best_path = p

    if best_path is None:
        return None, None
    return best_iter, str(best_path.relative_to(PROJECT_ROOT))


def _find_options_file(options_dir: Path) -> Optional[Path]:
    """Find a .json options file in the directory. Prefers 'train.json' if it exists, otherwise picks any."""
    if not options_dir.is_dir():
        return None
    train_json = options_dir / "train.json"
    if train_json.exists():
        return train_json
    json_files = sorted(options_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if json_files:
        return json_files[0]
    return None


def _parse_log_metrics(log_path: Path) -> Dict[str, Any]:
    """Parse a train.log for the best PSNR and SSIM seen across all checkpoints."""
    if not log_path.exists():
        return {}
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        best_psnr: Optional[float] = None
        best_ssim: Optional[float] = None
        for line in content.splitlines():
            pm = re.search(r"Average PSNR:\s*([0-9.]+)", line)
            if pm:
                v = float(pm.group(1))
                if best_psnr is None or v > best_psnr:
                    best_psnr = v
            sm = re.search(r"\bSSIM:\s*([0-9.]+)", line)
            if sm:
                v = float(sm.group(1))
                if best_ssim is None or v > best_ssim:
                    best_ssim = v
        return {"best_psnr": best_psnr, "best_ssim": best_ssim}
    except Exception:
        return {}


def list_training_runs() -> List[Dict[str, Any]]:
    """
    Scan superresolution/ for task directories and return metadata about each.
    Sorted by directory modification time (most recent first).
    """
    runs = []
    if not SUPERRESOLUTION_DIR.is_dir():
        return runs

    for task_dir in SUPERRESOLUTION_DIR.iterdir():
        if not task_dir.is_dir():
            continue

        models_dir = task_dir / "models"
        log_path = task_dir / "log" / "train.log"
        options_dir = task_dir / "options"

        latest_iter, latest_model = _find_latest_checkpoint(models_dir) if models_dir.is_dir() else (None, None)

        config_type = "unknown"
        scale: Optional[int] = None
        n_channels: Optional[int] = None
        train_config_path = _find_options_file(options_dir)
        if train_config_path:
            try:
                cfg = load_kair_json(train_config_path)
                config_type = cfg.get("model", "plain")
                scale = cfg.get("scale") or cfg.get("netG", {}).get("upscale")
                n_channels = cfg.get("n_channels") or cfg.get("netG", {}).get("in_chans")
            except Exception:
                pass

        log_metrics = _parse_log_metrics(log_path)

        runs.append({
            "task_name": task_dir.name,
            "latest_iteration": latest_iter,
            "latest_model_path": latest_model,
            "config_type": config_type,
            "has_log": log_path.exists(),
            "scale": scale,
            "n_channels": n_channels,
            "best_psnr": log_metrics.get("best_psnr"),
            "best_ssim": log_metrics.get("best_ssim"),
            "mtime": task_dir.stat().st_mtime,
        })

    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def get_latest_model_info(task_name: str) -> Dict[str, Any]:
    """
    Return the latest model path and autofilled MODEL_CONFIG for a task.
    Parses netG block from options/train.json.
    """
    task_dir = SUPERRESOLUTION_DIR / task_name
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    models_dir = task_dir / "models"
    latest_iter, latest_path = _find_latest_checkpoint(models_dir)

    if latest_path is None:
        raise FileNotFoundError(f"No model checkpoints found in {models_dir}")

    # Parse netG from saved options json
    model_config = {}
    options_file_name = None
    train_json = _find_options_file(task_dir / "options")
    if train_json:
        options_file_name = train_json.name
        try:
            cfg = load_kair_json(train_json)
            model_config = _extract_model_config(cfg)
        except Exception:
            pass

    return {
        "task_name": task_name,
        "latest_iteration": latest_iter,
        "model_path": latest_path,
        "model_config": model_config,
        "options_file": options_file_name,
    }
