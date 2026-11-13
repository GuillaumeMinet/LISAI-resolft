"""Live runtime construction for training.

This module owns the runtime-side boundary of training. It turns a resolved
experiment config into the mutable process resources used during a training run:
run directory, logger, device, TensorBoard writer, and callbacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch

from lisai.infra.fs import create_tb_folder
from lisai.infra.logging import setup_logger
from lisai.infra.paths import Paths
from lisai.training.callbacks import TensorBoardCallback, ValidationImagesCallback

if TYPE_CHECKING:
    from lisai.config.models import ResolvedExperiment


@dataclass
class TrainingRuntime:
    """Live training infrastructure for a single run.

    This object carries only process/runtime resources such as resolved paths,
    the effective run name and directory, logging, device selection, writers,
    and callbacks. It does not own the experiment configuration or model setup
    decisions.
    """

    paths: Paths
    run_name: str

    run_dir: Path | None = None
    writer: Any | None = None
    logger: logging.Logger | None = None
    device: torch.device | None = None
    callbacks: list[Any] = field(default_factory=list)
    console_filter: Any | None = None
    file_filter: Any | None = None
    enable_console_logs: Callable[[bool], None] | None = None
    enable_file_logs: Callable[[bool], None] | None = None



def initialize_runtime(cfg: ResolvedExperiment, *, paths: Paths) -> TrainingRuntime:
    """Build the live runtime infrastructure needed by the training process.

    From the input ``cfg``, we derive:
        - canonical paths
        - creation/resolution of the effective run directory 
        - run name
        - logger and log filters
        - TensorBoard writer (if allowed)
        - device selection
        - training callbacks.
    """
    from lisai.training.setup.run_dir import prepare_run_dir

    is_volumetric = cfg.model.architecture == "unet3d" # used for shape of val images callbacks

    # initialize runtime object
    runtime = TrainingRuntime(
        paths=paths,
        run_name=cfg.experiment.exp_name,
    )

    # build run_dir
    run_dir, run_name = prepare_run_dir(cfg, runtime)
    runtime.run_dir = run_dir
    runtime.run_name = run_name  # prepare_run_dir may have changed it to have a unique run name

    # define log file path
    log_file = None
    if run_dir is not None:
        log_file = paths.log_file_path(run_dir=run_dir)

    # construct logger
    use_tqdm = bool(getattr(cfg.training, "progress_bar", False))
    logger, console_filter, file_filter = setup_logger(
        name="lisai",
        level=logging.INFO,
        log_file=log_file,
        use_tqdm=use_tqdm,
        file_format="%(asctime)s %(message)s",
        file_datefmt="%Y-%m-%d %H:%M:%S",
        file_enabled=False,
    )

    runtime.logger = logger
    runtime.console_filter = console_filter
    runtime.file_filter = file_filter

    # toggles for selectively disabling/enabling logs to the console and/or log file.
    def _enable_console(enable: bool):
        """Toggle console logging for the current training process."""
        console_filter.enable = bool(enable)

    def _enable_file(enable: bool):
        """Toggle file logging for the current training process."""
        file_filter.enable = bool(enable)

    runtime.enable_console_logs = _enable_console
    runtime.enable_file_logs = _enable_file

    # output first logging infos
    logger.info(f"Training Initialization: {run_name}")
    if run_dir is not None:
        logger.info(f"Saving to: {run_dir}")
    else:
        logger.warning("Saving disabled!")
    if bool(getattr(cfg.training, "early_stop", False)):
        logger.warning("Training is starting with early_stop mode enabled.")

    # tensorboard (if allowed)
    writer = None
    if bool(cfg.tensorboard.enabled):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as e:
            logger.warning(f"TensorBoard requested but unavailable: {type(e).__name__}: {e}")
            SummaryWriter = None

        if SummaryWriter is not None:
            tb_root = paths.tensorboard_dir(
                dataset_name=cfg.data.dataset_name,
                tensorboard_subfolder=cfg.routing.tensorboard_subfolder or cfg.routing.models_subfolder,
            )
            tb_folder, _ = create_tb_folder(
                tb_root, run_name, exist_ok=(run_dir is not None),
                name_format=paths.settings.NAMING.exp_name_format,
            )
            writer = SummaryWriter(log_dir=str(tb_folder))

    runtime.writer = writer

    # device (CPU/GPU)
    device_name = getattr(cfg.training, "device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Switching to CPU.")
        device_name = "cpu"
    runtime.device = torch.device(device_name)

    # callbacks for tensorboard and validation images
    callbacks = []

    # tensorboard
    if writer is not None:
        callbacks.append(TensorBoardCallback(writer, volumetric=is_volumetric))

    # validation images
    images_dir = None
    if runtime.run_dir is not None:
        images_dir = runtime.paths.validation_images_dir(run_dir=runtime.run_dir)

    if images_dir is not None:
        callbacks.append(
            ValidationImagesCallback(
                images_dir=images_dir,
                enabled=bool(getattr(cfg.saving, "validation_images", True)),
                freq=int(getattr(cfg.saving, "validation_freq", 10)),
                volumetric=is_volumetric,
            )
        )

    runtime.callbacks = callbacks
    return runtime


# Backward-compatible alias during the training setup migration.
initialize = initialize_runtime

__all__ = ["TrainingRuntime", "initialize_runtime", "initialize"]
