"""Run-monitoring helpers for training orchestration.

This module keeps run metadata / run finalization concerns out of
run_training.py while staying close to the current lisai.runs API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from lisai.runs import (
    RunMetadataCallback,
    build_training_signature_from_resolved_config,
    create_run_metadata,
    finalize_run_completed,
    finalize_run_failed,
    finalize_run_stopped,
    finalize_setup_failed,
    group_path_from_model_subfolder,
    update_run_failure_reason,
    update_run_runtime_details,
)

if TYPE_CHECKING:
    from lisai.training.trainers.base import TrainingOutcome


class RunMonitor:
    """
    Convenience adapter around lisai.runs helpers to update run metadata during 
    the training:
        - initializes metadata 
        - exposes api such as update_signature(), finalize(outcome), etc.
        - constructs RunMetadataCallback and append it to runtime callbacks so that trainer
        can update progress during training
    
    All metadata updates are called via _safe_call, so that update failures are logged as 
    warnings instead of crashing training.
    """

    def __init__(self, cfg, runtime):
        self.cfg = cfg
        self.runtime = runtime
        self.enabled = False

    def install(self) -> bool:
        runtime = self.runtime
        cfg = self.cfg

        if runtime.run_dir is None:
            self.enabled = False
            return False

        model_subfolder = getattr(cfg.routing, "models_subfolder", "")
        group_path = group_path_from_model_subfolder(model_subfolder)
        preserve_existing = getattr(cfg.experiment, "mode", "train") == "continue_training"

        metadata = self._safe_call(
            "initialization",
            create_run_metadata,
            runtime.run_dir,
            paths=runtime.paths,
            dataset=cfg.data.dataset_name,
            model_subfolder=model_subfolder,
            max_epoch=getattr(cfg.training, "n_epochs", None),
            group_path=group_path,
            preserve_existing=preserve_existing,
        )
        if metadata is None:
            self.enabled = False
            return False

        self.enabled = True
        self.update_signature()

        runtime.callbacks.append(
            RunMetadataCallback(
                runtime.run_dir,
                max_epoch=getattr(cfg.training, "n_epochs", None),
                logger=runtime.logger,
            )
        )
        return True

    def update_signature(self, *, trainable_params: int | None = None) -> None:
        if not self.enabled or self.runtime.run_dir is None:
            return

        self._safe_call(
            "training signature update",
            update_run_runtime_details,
            self.runtime.run_dir,
            training_signature=build_training_signature_from_resolved_config(
                self.cfg,
                trainable_params=trainable_params,
            ),
        )

    def update_training_timing(
        self,
        *,
        total_training_time_sec: float | None,
        outcome: "TrainingOutcome",
    ) -> None:
        if not self.enabled or self.runtime.run_dir is None:
            return
        if total_training_time_sec is None:
            return

        training_time_per_epoch_sec = self._training_time_per_epoch_sec(
            total_training_time_sec=total_training_time_sec,
            outcome=outcome,
        )

        self._safe_call(
            "runtime stats update",
            update_run_runtime_details,
            self.runtime.run_dir,
            total_training_time_sec=total_training_time_sec,
            training_time_per_epoch_sec=training_time_per_epoch_sec,
        )

    def reset_peak_gpu_memory_stats(self) -> None:
        if not self.enabled or self.runtime.run_dir is None:
            return

        device = getattr(self.runtime, "device", None)
        if getattr(device, "type", None) != "cuda":
            return
        if not torch.cuda.is_available():
            return

        try:
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(device_index)
        except Exception as exc:
            self._log_warning(
                f"Failed to reset CUDA peak memory stats: {type(exc).__name__}: {exc}"
            )

    def persist_peak_gpu_memory_stats(self) -> None:
        if not self.enabled or self.runtime.run_dir is None:
            return

        device = getattr(self.runtime, "device", None)
        if getattr(device, "type", None) != "cuda":
            return
        if not torch.cuda.is_available():
            return

        try:
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            peak_bytes = int(torch.cuda.max_memory_allocated(device_index))
        except Exception as exc:
            self._log_warning(
                f"Failed to read CUDA peak memory stats: {type(exc).__name__}: {exc}"
            )
            return

        peak_mb = (peak_bytes + (1024 * 1024 - 1)) // (1024 * 1024)
        self._safe_call(
            "runtime stats update",
            update_run_runtime_details,
            self.runtime.run_dir,
            peak_gpu_mem_mb=peak_mb,
        )

    def record_failure_reason(self, outcome: "TrainingOutcome") -> None:
        if not self.enabled or self.runtime.run_dir is None:
            return

        self._safe_call(
            "failure reason update",
            update_run_failure_reason,
            self.runtime.run_dir,
            failure_reason=outcome.failure_reason,
        )

    def finalize(self, outcome: "TrainingOutcome") -> None:
        if not self.enabled or self.runtime.run_dir is None:
            return

        if outcome.reason in {"completed", "early_stopped", "no_epochs"}:
            finalizer = finalize_run_completed
        elif outcome.reason in {"interrupted", "setup_interrupted"}:
            finalizer = finalize_run_stopped
        elif outcome.reason in {"failed_retryable_hdn_divergence", "failed_nonretryable"}:
            finalizer = finalize_run_failed
        elif outcome.reason == "setup_failed":
            finalizer = finalize_setup_failed
        else:
            return

        self._safe_call("finalization", finalizer, self.runtime.run_dir)

    def _training_time_per_epoch_sec(
        self,
        *,
        total_training_time_sec: float | None,
        outcome: "TrainingOutcome",
    ) -> float | None:
        if total_training_time_sec is None:
            return None
        if outcome.last_completed_epoch is None:
            return None

        epochs_completed = int(outcome.last_completed_epoch) + 1
        if epochs_completed <= 0:
            return None

        return float(total_training_time_sec / epochs_completed)

    def _safe_call(self, action: str, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            self._log_warning(f"Run metadata {action} failed: {type(exc).__name__}: {exc}")
            return None

    def _log_warning(self, message: str) -> None:
        logger = getattr(self.runtime, "logger", None)
        if logger is None:
            return

        warning = getattr(logger, "warning", None)
        if callable(warning):
            warning(message)
            return

        info = getattr(logger, "info", None)
        if callable(info):
            info(message)