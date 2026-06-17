"""Atomic file write utilities for Forge."""

from pathlib import Path


def atomic_write_json(path: Path, data: str) -> None:
    """Atomically write JSON data to a file.
    
    Writes to a temporary file first, then atomically renames it to the target path.
    This ensures the target file is never in a partially-written state.
    
    Args:
        path: Target file path
        data: JSON string to write
        
    Raises:
        Exception: If write fails (temp file is cleaned up automatically)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    
    try:
        temp_path.write_text(data)
        temp_path.rename(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
