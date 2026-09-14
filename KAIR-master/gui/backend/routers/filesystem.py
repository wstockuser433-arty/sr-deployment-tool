"""
routers/filesystem.py
=====================
Endpoint for server-side directory/file browsing used by the GUI path picker.
Only exposes paths that are accessible from the server's filesystem.
"""
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/fs", tags=["filesystem"])

_IMAGE_MEDIA_TYPES = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.bmp': 'image/bmp',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
}

# Resolve the project root so relative paths in the UI resolve sensibly
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _safe_path(raw: str) -> Path:
    """Resolve raw string to an absolute Path. Empty string → project root."""
    if not raw or raw.strip() == "":
        return _PROJECT_ROOT
    p = Path(raw.strip())
    if not p.is_absolute():
        p = (_PROJECT_ROOT / p).resolve()
    return p


@router.get("/list")
def list_directory(
    path: str = Query(default="", description="Absolute or project-relative directory path"),
    mode: str = Query(default="dirs", description="'dirs' | 'files' | 'both'"),
    extensions: str = Query(default="", description="Comma-separated file extensions to filter, e.g. '.pth,.pt'"),
):
    """
    List entries in a directory for the GUI file/folder browser.

    Returns:
        current  — resolved absolute path of the listed directory
        parent   — parent directory (empty string if already at filesystem root)
        entries  — list of { name, path, type } sorted dirs-first then files
    """
    original = _safe_path(path)
    # Surface the resolved type so the frontend can auto-select when the user
    # types an explicit path and clicks Go, rather than just navigating.
    typed_file: str | None = None
    typed_dir: str | None = None
    if original.is_file():
        typed_file = str(original)
        target = original.parent
    elif not original.exists():
        target = _PROJECT_ROOT
    else:
        typed_dir = str(original)
        target = original

    try:
        raw_entries = list(target.iterdir())
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {target}")

    ext_filter = {e.strip().lower() for e in extensions.split(",") if e.strip()} if extensions else set()

    entries = []
    for entry in sorted(raw_entries, key=lambda e: (not e.is_dir(), e.name.lower())):
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue

        if is_dir:
            # Always include dirs for navigation regardless of mode.
            # Selection rules (Select button vs click-file) are enforced in the frontend.
            entries.append({"name": entry.name, "path": str(entry), "type": "dir"})
        else:
            if mode in ("files", "both"):
                suffix = entry.suffix.lower()
                if not ext_filter or suffix in ext_filter:
                    entries.append({"name": entry.name, "path": str(entry), "type": "file"})

    parent_path = str(target.parent) if target.parent != target else ""

    return {
        "current": str(target),
        "parent": parent_path,
        "entries": entries,
        "typed_file": typed_file,
        "typed_dir": typed_dir,
    }


@router.get("/image")
def serve_image(path: str = Query(..., description="Absolute path to a local image file")):
    """Serve a local image file for in-browser preview (used by ClassResultsPanel)."""
    if not path or not path.strip():
        raise HTTPException(status_code=400, detail="Path required")
    p = Path(path.strip())
    if not p.is_absolute() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = _IMAGE_MEDIA_TYPES.get(p.suffix.lower(), "application/octet-stream")
    return FileResponse(str(p), media_type=media_type)
