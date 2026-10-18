"""Shared JSON file operations for the application."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def write_json_atomic(path: Path, value: Any) -> None:
    """Write formatted UTF-8 JSON and atomically replace the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    except Exception:
        logger.exception("Could not write JSON file", extra={"path": str(path)})
        raise
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()