"""Minimal, Qt-independent descriptions of local training jobs.

These value contracts do not schedule, poll, or execute work.  A future
lisai_runner adapter can publish the same snapshot without changing the GUI.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingLaunchRequest:
    config_path: Path
    workspace_root: Path
    config_root: Path
    mode: str


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str | None
    status: str  # idle, starting, running, stopping, completed, failed
    run_id: str | None
    run_dir: Path | None
    progress: float | None  # None: local process does not publish structured progress
    message: str


__all__ = ["TrainingLaunchRequest", "JobSnapshot"]
