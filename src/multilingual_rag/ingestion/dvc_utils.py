"""Inspect and validate DVC-tracked ingestion inputs."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml


logger = logging.getLogger(__name__)


class DVCInputError(RuntimeError):
    """Raised when an ingestion input is not a clean, committed DVC output."""


@dataclass(frozen=True)
class DVCSource:
    pointer_path: str
    output_path: str
    md5: str
    verified: bool


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DVCInputError(f"Could not verify Git-tracked DVC pointer: {exc}") from exc


def _read_dvc_source(input_dir: Path) -> tuple[Path, DVCSource]:
    pointer = input_dir.parent / f"{input_dir.name}.dvc"
    if not pointer.is_file():
        raise DVCInputError(f"No DVC pointer found for input directory: {pointer}")

    try:
        pointer_data = yaml.safe_load(pointer.read_text(encoding="utf-8"))
        outputs = pointer_data["outs"]
        output = next(
            entry
            for entry in outputs
            if isinstance(entry, dict) and entry.get("path") == input_dir.name
        )
        md5 = output["md5"]
    except (AttributeError, KeyError, OSError, StopIteration, TypeError, yaml.YAMLError) as exc:
        raise DVCInputError(f"Invalid DVC pointer for input directory: {pointer}") from exc

    if not isinstance(md5, str) or not md5:
        raise DVCInputError(f"DVC pointer has no directory hash: {pointer}")
    return pointer, DVCSource(
        pointer_path=pointer.as_posix(),
        output_path=str(output["path"]),
        md5=md5,
        verified=False,
    )


def validate_dvc_input(input_dir: Path) -> DVCSource:
    """Require the input to match its committed DVC pointer, without remote checks."""
    input_dir = input_dir.resolve()
    pointer, source = _read_dvc_source(input_dir)
    repository = _run_git(["rev-parse", "--show-toplevel"], cwd=input_dir).stdout.strip()
    repository_path = Path(repository)
    relative_pointer = pointer.relative_to(repository_path).as_posix()

    try:
        _run_git(["ls-files", "--error-unmatch", "--", relative_pointer], cwd=repository_path)
    except DVCInputError as exc:
        raise DVCInputError(f"DVC pointer is not tracked by Git: {relative_pointer}") from exc
    try:
        _run_git(["diff", "--quiet", "HEAD", "--", relative_pointer], cwd=repository_path)
    except DVCInputError as exc:
        raise DVCInputError(f"DVC pointer has uncommitted changes: {relative_pointer}") from exc

    try:
        result = subprocess.run(
            ["dvc", "status", "--json", "--no-updates", relative_pointer],
            cwd=repository_path,
            capture_output=True,
            check=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", "")
        raise DVCInputError(f"Could not verify DVC input status: {detail or exc}") from exc

    try:
        status = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise DVCInputError("DVC returned invalid JSON while checking input status") from exc
    if status:
        raise DVCInputError(
            f"DVC input is not up to date: {json.dumps(status, ensure_ascii=False)}"
        )

    logger.info("DVC input verified", extra={"pointer_path": relative_pointer, "dvc_md5": source.md5})
    return DVCSource(
        pointer_path=relative_pointer,
        output_path=source.output_path,
        md5=source.md5,
        verified=True,
    )


def get_dvc_source(input_dir: Path) -> DVCSource | None:
    """Read DVC provenance without enforcing freshness, for explicit bypass runs."""
    try:
        pointer, source = _read_dvc_source(input_dir.resolve())
    except DVCInputError:
        return None
    try:
        pointer_path = pointer.relative_to(Path.cwd()).as_posix()
    except ValueError:
        pointer_path = pointer.name
    return DVCSource(
        pointer_path=pointer_path,
        output_path=source.output_path,
        md5=source.md5,
        verified=False,
    )