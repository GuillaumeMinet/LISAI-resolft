"""Top-level training orchestration entrypoint.

This module wires together the clean training boundaries introduced by the
refactor: resolve the experiment config, initialize the runtime, prepare data,
build the model, construct the trainer, and execute the training loop.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from lisai.config import resolve_config, resolve_config_dict
from lisai.config.progress import resolve_progress_bar
from lisai.runs import count_trainable_parameters

from . import setup
from .orchestration.run_monitor import RunMonitor
from .orchestration import post_training
from .runtime import initialize_runtime
from .trainers import get_trainer

if TYPE_CHECKING:
    from lisai.config.models import ResolvedExperiment
    from lisai.training.trainers.base import TrainingOutcome



def _normalize_training_outcome(outcome) -> "TrainingOutcome":
    from lisai.training.trainers.base import TrainingOutcome

    assert isinstance(outcome, TrainingOutcome), "trainer.train() must return TrainingOutcome"
    return outcome


def _is_failed_outcome(outcome: "TrainingOutcome") -> bool:
    return outcome.reason in {
        "failed_retryable_hdn_divergence",
        "failed_nonretryable",
        "setup_failed",
    }


def _apply_progress_bar_preference(cfg: "ResolvedExperiment", progress_bar: bool | None) -> None:
    default = bool(getattr(cfg.training, "progress_bar", False))
    cfg.training.progress_bar = resolve_progress_bar(default, progress_bar)


def _setup_training(cfg, runtime, monitor: RunMonitor):
    try:
        prepared_data = setup.prepare_data(cfg, runtime)
        setup.save_training_config(
            cfg,
            runtime,
            prepared_data.data_norm_prm,
            prepared_data.model_norm_prm,
            prepared_data.split_manifest,
        )

        model, state_dict = setup.build_model(
            cfg,
            runtime.device,
            runtime.paths,
            prepared_data.model_norm_prm,
        )

        trainable_params = count_trainable_parameters(model)
        monitor.update_signature(trainable_params=trainable_params)

        trainer = get_trainer(
            architecture=cfg.model.architecture,
            model=model,
            train_loader=prepared_data.train_loader,
            val_loader=prepared_data.val_loader,
            device=runtime.device,
            cfg=cfg,
            run_dir=runtime.run_dir,
            volumetric=cfg.model.architecture == "unet3d",
            writer=runtime.writer,
            state_dict=state_dict,
            callbacks=runtime.callbacks,
            patch_info=prepared_data.patch_info,
            console_filter=runtime.console_filter,
            file_filter=runtime.file_filter,
        )
        monitor.reset_peak_gpu_memory_stats()
        return trainer, None

    except KeyboardInterrupt:
        from lisai.training.trainers.base import TrainingOutcome

        return None, TrainingOutcome(
            reason="setup_interrupted",
            last_completed_epoch=None,
            retry_eligible=False,
        )

    except Exception as exc:
        from lisai.training.trainers.base import TrainingOutcome

        logger = getattr(runtime, "logger", None)
        error = getattr(logger, "error", None)
        if callable(error):
            error("Training setup failed", exc_info=True)

        return None, TrainingOutcome(
            reason="setup_failed",
            last_completed_epoch=None,
            failure_reason=f"{type(exc).__name__}: {exc}",
            retry_eligible=False,
        )


def _finalize_training_result(
    cfg,
    runtime,
    monitor: RunMonitor,
    outcome: "TrainingOutcome",
    *,
    total_training_time_sec: float | None,
    training_started: bool,
):
    monitor.finalize(outcome)
    monitor.record_failure_reason(outcome)

    if not training_started:
        logger = getattr(runtime, "logger", None)
        info = getattr(logger, "info", None)
        if callable(info):
            info("Setup failed or got interrupted, training never started.")
        return

    post_training.auto_save_loss_plot_image(cfg, runtime, terminal_reason=outcome.reason)
    monitor.update_training_timing(
        total_training_time_sec=total_training_time_sec,
        outcome=outcome,
    )

    try:
        post_training.run_post_training_evaluation(cfg, runtime,outcome)
    except Exception:
        logger = getattr(runtime, "logger", None)
        error = getattr(logger, "error", None)
        if callable(error):
            error("Post-training evaluation failed", exc_info=True)
        raise


def run_training(config_path, *, progress_bar: bool | None = None):
    """Run training end to end from a config path and return the trainer instance."""
    return run_training_from_resolved_config(
        resolve_config(config_path),
        progress_bar=progress_bar,
    )


def run_training_from_config_dict(config: dict, *, progress_bar: bool | None = None):
    """Run training from an in-memory experiment config dictionary."""
    return run_training_from_resolved_config(
        resolve_config_dict(config),
        progress_bar=progress_bar,
    )


def run_training_from_resolved_config(
    cfg: "ResolvedExperiment",
    *,
    progress_bar: bool | None = None,
):
    """Run training from a resolved experiment config and return the trainer instance."""
    _apply_progress_bar_preference(cfg, progress_bar)
    runtime = initialize_runtime(cfg)
    monitor = RunMonitor(cfg, runtime)
    monitor.install()

    trainer = None
    outcome = None
    attempt_start_perf = time.perf_counter()

    try:
        # setup
        trainer, setup_error = _setup_training(cfg, runtime, monitor)

        # setup failure case
        if setup_error is not None:
            outcome = _normalize_training_outcome(setup_error)
            _finalize_training_result(
                cfg,
                runtime,
                monitor,
                outcome,
                total_training_time_sec=None,
                training_started=False,
            )
        
        # otherwise start training
        else:
            outcome = _normalize_training_outcome(trainer.train())
            total_training_time_sec = max(time.perf_counter() - attempt_start_perf, 0.0)
            _finalize_training_result(
                cfg,
                runtime,
                monitor,
                outcome,
                total_training_time_sec=total_training_time_sec,
                training_started=True,
            )

    finally:
        monitor.persist_peak_gpu_memory_stats()
        if runtime.writer:
            runtime.writer.close()

    if outcome is None:
        raise RuntimeError("Training orchestration completed without a TrainingOutcome.")

    if _is_failed_outcome(outcome):
        reason = outcome.failure_reason or outcome.reason
        raise RuntimeError(f"Training failed: {reason}")

    return trainer
