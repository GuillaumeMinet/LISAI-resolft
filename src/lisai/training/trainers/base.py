import copy
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from lisai.training.errors import HDNDivergenceError
from lisai.config.models import ResolvedExperiment
from lisai.infra.paths import Paths
from lisai.training.checkpointing import CheckpointManager
from lisai.runs.lifecycle import update_run_recovery_info
from lisai.runs.io import read_run_metadata

try:
    from tqdm import tqdm
    _tqdm_available = True
except Exception:
    _tqdm_available = False


TrainingStopReason = Literal[
    "completed",
    "early_stopped",
    "interrupted",
    "setup_interrupted",
    "no_epochs",
    "setup_failed",
    "failed_retryable_hdn_divergence",
    "failed_nonretryable",
    "pause_requested",
    "paused",
    "resuming",
]


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TrainingOutcome:
    reason: TrainingStopReason
    last_completed_epoch: int | None
    failure_reason: str | None = None
    retry_eligible: bool = False


class BaseTrainer(ABC):
    def __init__(
        self,
        *,
        model,
        train_loader,
        val_loader,
        device,
        cfg: ResolvedExperiment,
        paths: Paths,
        run_dir,
        volumetric=False,
        writer=None,
        state_dict=None,
        callbacks=None,
        patch_info=None,
        console_filter=None,
        file_filter=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.cfg = cfg
        self.paths = paths
        self.mode = cfg.experiment.mode
        self.volumetric = volumetric

        self.training_prm = cfg.training.model_dump()
        self.data_cfg = cfg.data
        self.saving_prm = cfg.saving.model_dump()
        self.tensorboard_prm = cfg.tensorboard.model_dump()

        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.exp_name = self.run_dir.name if self.run_dir is not None else cfg.experiment.exp_name
        self.writer = writer

        self.callbacks = callbacks or []
        self.patch_info = patch_info or getattr(self.data_cfg, "patch_info", None)
        self.console_filter = console_filter
        self.file_filter = file_filter
        self._training_log_initialized = False
        
        # training parameters
        self.batch_size = self.training_prm.get("batch_size")
        self.n_epochs = self.training_prm.get("n_epochs")
        self.debug_stop = bool(self.training_prm.get("debug_stop", False))
        self._legacy_early_stop_enabled = bool(self.training_prm.get("early_stop", False))
        # Keep legacy `early_stop` semantics in trainer subclasses for backward compatibility.
        self.early_stop = self._legacy_early_stop_enabled and not self.debug_stop
        self.pos_encod = self.training_prm.get("pos_encod", False)

        warmup_cfg = self.training_prm.get("warmup", {}) or {}
        self._warmup_enabled_requested = bool(warmup_cfg.get("enabled", False))
        self._warmup_steps = max(int(warmup_cfg.get("steps", 0) or 0), 0)
        self._warmup_start_factor = float(warmup_cfg.get("start_factor", 0.1))
        self._warmup_active = False
        self._warmup_base_lrs: list[float] = []
        self._warmup_ignore_warned = False

        auto_stop_cfg = self.training_prm.get("auto_stop", {}) or {}
        self._auto_stop_enabled = bool(auto_stop_cfg.get("enabled", False))
        self._auto_stop_metric = str(auto_stop_cfg.get("metrics", "val_loss"))
        self._auto_stop_patience = max(int(auto_stop_cfg.get("patience", 30) or 0), 0)

        # progress bar policy
        disable_tqdm = _env_truthy("LISAI_DISABLE_TQDM")
        self.pbar = bool(self.training_prm.get("progress_bar")) and _tqdm_available and not disable_tqdm
        # Queue workers disable tqdm via env var; also suppress per-batch log spam in that mode.
        self.update_console = (not self.pbar) and (not disable_tqdm)

        # state dict
        self.state_dict = state_dict if state_dict is not None else {"epoch": -1}
        self._optimizer_step_count = max(int(self.state_dict.get("optimizer_step_count", 0) or 0), 0)
        self._auto_stop_best_metric = self.state_dict.get("auto_stop_best_metric", None)
        if self._auto_stop_best_metric is not None:
            self._auto_stop_best_metric = float(self._auto_stop_best_metric)
        self._auto_stop_bad_epochs = max(int(self.state_dict.get("auto_stop_bad_epochs", 0) or 0), 0)
        self._last_safe_training_state = None
        self._safe_state_confirmation_lag = 1
        self._safe_state_rewind_steps = self._resolve_safe_resume_rewind_steps()
        self._pending_safe_training_states = deque()
        self._confirmed_safe_training_states = deque(maxlen=max(self._safe_state_rewind_steps + 2, 2))

        # logger (system sets handlers)
        self.logger = logging.getLogger("lisai")

        if self._legacy_early_stop_enabled and self.debug_stop:
            self.logger.warning(
                "Both `training.debug_stop` and legacy `training.early_stop` are enabled; "
                "using `debug_stop` (3 full epochs) and disabling legacy batch-level truncation."
            )

        if self._auto_stop_enabled and self._auto_stop_metric not in {"loss", "val_loss"}:
            self.logger.warning(
                f"Invalid training.auto_stop.metrics={self._auto_stop_metric}; "
                "falling back to 'val_loss'."
            )
            self._auto_stop_metric = "val_loss"

        if self.training_prm.get("val_loss_patience", None) is not None:
            self.logger.warning(
                "training.val_loss_patience is deprecated and ignored. "
                "Use training.auto_stop.patience instead."
            )

        self._drop_optimizer_scheduler_state_on_safe_resume()
        self._base_learning_rate_from_config = self._resolve_base_learning_rate_from_config()

        # optimizer + scheduler
        self.optimizer, self.scheduler = self._make_optimizer_and_scheduler()

        # checkpoint manager (file outputs)
        self.ckpt = CheckpointManager(
            self.run_dir,
            self.saving_prm,
            is_lvae=self.is_lvae,
            paths=self.paths,
        )

        self._apply_recovery_overrides()
        self._configure_warmup()

    # PROPERTY SUBCLASSES MUST DEFINE #
    @property
    @abstractmethod
    def is_lvae(self) -> bool:
        ...

    # shared batch utilities 
    def _split_batch(self, batch, *, warn_once: bool = False):
        if int(batch[0].shape[0] // self.batch_size) == 0:
            virtual_batches = [(tensor,) for tensor in batch]
            if warn_once:
                self.logger.warning("batch_size bigger than # of virtual batches, not optimal")
            return virtual_batches
        return [torch.split(tensor, self.batch_size, dim=0) for tensor in batch]

    def _prepare_batch(self, x, y, samp_pos):
        if torch.isnan(y).all().item():
            y = None
        else:
            y = y.to(self.device)

        if self.pos_encod:
            assert samp_pos is not None
            samp_pos = samp_pos.to(self.device)
        else:
            samp_pos = None

        if len(x.shape) == 4 and self.volumetric:
            x = x.unsqueeze(1)
            if y is not None:
                y = y.unsqueeze(1)

        x = x.to(self.device, dtype=torch.float)
        return x, y, samp_pos
    

    
    # OPTIMIZER AND SCHEDULER 
    def _make_optimizer_and_scheduler(self):
        """
        Creates optimizer + scheduler from training_prm.

        Expected config patterns (flexible):
        training:
            optimizer: "Adam" | {"name": "Adam", "lr": 1e-4, ...}
            scheduler: null | "StepLR" | "ReduceLROnPlateau" | {"name": "...", ...}
        """

        prm = self.training_prm or {}

        # extract optimizer name and optional arguments
        opt_cfg = prm.get("optimizer", "Adam")
        if isinstance(opt_cfg, str):
            # if no optimizer arguments given
            opt_name = opt_cfg
            opt_kwargs = {}
        elif isinstance(opt_cfg, dict):
            # separate optional arguments from optimizer name 
            opt_name = opt_cfg.get("name", "Adam")
            opt_kwargs = {k: v for k, v in opt_cfg.items() if k != "name"}
        else:
            raise ValueError(f"Invalid training.optimizer type: {type(opt_cfg)}")
        

        # resolve learning_rate, add it to opt_kwargs
        lr = opt_kwargs.get("lr", None)
        if lr is None:
            lr = self._base_learning_rate_from_config
        opt_kwargs["lr"] = float(lr)

        # init optimizer
        if not hasattr(torch.optim, opt_name):
            raise ValueError(f"Unknown optimizer '{opt_name}'. Available in torch.optim: {dir(torch.optim)}")
        OptimCls = getattr(torch.optim, opt_name)
        optimizer = OptimCls(self.model.parameters(), **opt_kwargs)

        # restore optimizer state if provided (continue training)
        if isinstance(self.state_dict, dict) and "optimizer_state_dict" in self.state_dict:
            try:
                optimizer.load_state_dict(self.state_dict["optimizer_state_dict"])
            except Exception as e:
                self.logger.warning(f"Failed to load optimizer_state_dict: {type(e)} {e}")

        # Scheduler
        sch_cfg = prm.get("scheduler", None)
        scheduler = None
        self._scheduler_name = None  # used in train loop to decide how to step

        if sch_cfg is None or sch_cfg is False:
            return optimizer, None

        if isinstance(sch_cfg, str):
            sch_name = sch_cfg
            sch_kwargs = {}
        elif isinstance(sch_cfg, dict):
            sch_name = sch_cfg.get("name")
            if not sch_name:
                raise ValueError("training.scheduler dict must contain a 'name' key")
            sch_kwargs = {k: v for k, v in sch_cfg.items() if k != "name"}
            params = sch_kwargs.pop("params", None)
            if isinstance(params, dict):
                sch_kwargs = {**params, **sch_kwargs}
        else:
            raise ValueError(f"Invalid training.scheduler type: {type(sch_cfg)}")

        if not hasattr(torch.optim.lr_scheduler, sch_name):
            raise ValueError(
                f"Unknown scheduler '{sch_name}'. Available in torch.optim.lr_scheduler: {dir(torch.optim.lr_scheduler)}"
            )

        SchedCls = getattr(torch.optim.lr_scheduler, sch_name)
        scheduler = SchedCls(optimizer, **sch_kwargs)
        self._scheduler_name = sch_name

        # restore scheduler state if provided
        if isinstance(self.state_dict, dict):
            sch_state = self.state_dict.get("scheduler_state_dict") or self.state_dict.get("scheduler")
            if sch_state is not None:
                try:
                    scheduler.load_state_dict(sch_state)
                except Exception as e:
                    self.logger.warning(f"Failed to load scheduler state: {type(e)} {e}")

        return optimizer, scheduler

    def _resolve_base_learning_rate_from_config(self) -> float:
        prm = self.training_prm or {}
        opt_cfg = prm.get("optimizer", "Adam")

        lr = None
        if isinstance(opt_cfg, dict):
            lr = opt_cfg.get("lr", None)
        if lr is None:
            lr = prm.get("learning_rate", None)
        if lr is None:
            lr = prm.get("lr", None)
        if lr is None:
            lr = 1e-4
        return float(lr)

    def _configure_warmup(self) -> None:
        self._warmup_active = False
        self._warmup_base_lrs = []

        if not self._warmup_enabled_requested:
            return

        if self._warmup_steps <= 0:
            self.logger.warning(
                "Warmup requested but `training.warmup.steps` is <= 0; warmup disabled."
            )
            return

        if getattr(self, "_scheduler_name", None) != "ReduceLROnPlateau":
            self.logger.warning(
                "Warmup is only supported with ReduceLROnPlateau. "
                f"Ignoring warmup for scheduler={getattr(self, '_scheduler_name', None)}."
            )
            self._warmup_ignore_warned = True
            return

        self._warmup_base_lrs = [
            float(group.get("lr", self._base_learning_rate_from_config))
            for group in self.optimizer.param_groups
        ]
        self._warmup_active = self._optimizer_step_count < self._warmup_steps

        if self._warmup_active:
            self._set_warmup_lrs_for_step(self._optimizer_step_count)
            self.logger.info(
                "Warmup enabled: "
                f"steps={self._warmup_steps}, start_factor={self._warmup_start_factor}."
            )

    def _warmup_factor_for_step(self, step_index: int) -> float:
        if self._warmup_steps <= 1:
            return 1.0
        clamped_step = min(max(int(step_index), 0), self._warmup_steps - 1)
        progress = float(clamped_step) / float(self._warmup_steps - 1)
        return float(self._warmup_start_factor + (1.0 - self._warmup_start_factor) * progress)

    def _set_warmup_lrs_for_step(self, step_index: int) -> None:
        if not self._warmup_base_lrs:
            return
        factor = self._warmup_factor_for_step(step_index)
        for group, base_lr in zip(self.optimizer.param_groups, self._warmup_base_lrs, strict=True):
            group["lr"] = float(base_lr * factor)

    def _set_warmup_base_lrs(self) -> None:
        if not self._warmup_base_lrs:
            return
        for group, base_lr in zip(self.optimizer.param_groups, self._warmup_base_lrs, strict=True):
            group["lr"] = float(base_lr)

    def _apply_manual_warmup_if_needed(self) -> None:
        if not self._warmup_active:
            return

        if self._optimizer_step_count >= self._warmup_steps:
            self._set_warmup_base_lrs()
            self._warmup_active = False
            return

        self._set_warmup_lrs_for_step(self._optimizer_step_count)

    def _check_auto_stop(self, *, train_loss: float, val_loss: float) -> bool:
        if not self._auto_stop_enabled:
            return False

        metric_value = float(train_loss if self._auto_stop_metric == "loss" else val_loss)
        if self._auto_stop_best_metric is None or metric_value < float(self._auto_stop_best_metric):
            self._auto_stop_best_metric = metric_value
            self._auto_stop_bad_epochs = 0
            return False

        self._auto_stop_bad_epochs += 1
        return self._auto_stop_bad_epochs >= self._auto_stop_patience



    # =================== #
    #    TRAINING LOOP    #
    # =================== #
    def train(self) -> TrainingOutcome:
        self._align_safe_resume_epoch_with_metadata()
        last_completed_epoch = int(self.state_dict.get("epoch", -1))
        start_epoch = max(last_completed_epoch + 1, 0)
        self._initialize_log_file()
        
        if start_epoch >= self.n_epochs:
            self.logger.info(
                f"No epochs to run: start_epoch={start_epoch}, n_epochs={self.n_epochs}"
            )
            return TrainingOutcome(
                reason="no_epochs",
                last_completed_epoch=last_completed_epoch if last_completed_epoch >= 0 else None,
            )

        best_loss = self.state_dict.get("best_loss", float("inf"))
        last_train_loss = self.state_dict.get("train_loss")
        last_val_loss = self.state_dict.get("val_loss")
        iter_epoch = range(start_epoch, self.n_epochs)
        if self.pbar:
            iter_epoch = tqdm(iter_epoch, position=1, total=self.n_epochs, initial=start_epoch)
            iter_epoch.set_description("Epochs")

        self.logger.info("Starting Training...")
        stop_reason: TrainingStopReason = "completed"

        for epoch in iter_epoch:
            try:
                epoch_start_perf = time.perf_counter()
                train_loss = None
                val_loss = None
                save_val_imgs = bool(self.saving_prm.get("validation_images", False)) and (
                    epoch % int(self.saving_prm.get("validation_freq", 10)) == 0
                )

                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate(epoch, save_imgs=save_val_imgs)

                train_loss = train_metrics["loss"]
                val_loss = val_metrics["loss"]
                last_train_loss = train_loss
                last_val_loss = val_loss
                self._log_epoch_metrics(epoch, train_metrics, val_metrics)

                self.state_dict.update(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "optimizer_step_count": int(self._optimizer_step_count),
                        "auto_stop_best_metric": self._auto_stop_best_metric,
                        "auto_stop_bad_epochs": int(self._auto_stop_bad_epochs),
                    }
                )

                if self.is_lvae:
                    self.state_dict.update(
                        {
                            "train_kl_loss": train_metrics["kl_loss"],
                            "val_kl_loss": val_metrics["kl_loss"],
                            "train_recons_loss": train_metrics["recons_loss"],
                            "val_recons_loss": val_metrics["recons_loss"],
                        }
                    )

                if self.scheduler is not None:
                    self.state_dict["scheduler"] = self.scheduler.state_dict()

                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                    self.logger.info(
                        f"epoch {self._display_epoch(epoch)}: Best model saved with best_val = {best_loss}."
                    )

                if self.scheduler is not None:
                    if getattr(self, "_scheduler_name", None) == "ReduceLROnPlateau":
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                    # keep state_dict in sync for continue_training
                    self.state_dict["scheduler_state_dict"] = self.scheduler.state_dict()

                if self.saving_prm.get("enabled", True) and self.run_dir is not None:
                    self.ckpt.save(state_dict=self.state_dict, model=self.model, best_loss=best_loss, is_best=is_best)
                    self.ckpt.update_loss_file(epoch, train_metrics, val_metrics)

                # callbacks at epoch end
                epoch_duration_s = max(time.perf_counter() - epoch_start_perf, 0.0)
                logs = {"train_loss": train_loss, "val_loss": val_loss, "epoch_duration_s": epoch_duration_s}
                for cb in self.callbacks:
                    cb.on_epoch_end(self, epoch, logs)
                last_completed_epoch = epoch

                auto_stop_triggered = self._check_auto_stop(
                    train_loss=float(train_loss),
                    val_loss=float(val_loss),
                )
                self.state_dict["auto_stop_best_metric"] = self._auto_stop_best_metric
                self.state_dict["auto_stop_bad_epochs"] = int(self._auto_stop_bad_epochs)

                if self.debug_stop and epoch >= 2:
                    self.logger.info("Debug stop enabled: stopping after 3 full epochs.")
                    stop_reason = "early_stopped"
                    break
                if self.early_stop and epoch > 2:
                    self.logger.info("Legacy early_stop active: stopping early in debug mode.")
                    stop_reason = "early_stopped"
                    break
                if auto_stop_triggered:
                    self.logger.info(
                        "Auto-stop triggered: "
                        f"metric={self._auto_stop_metric}, "
                        f"patience={self._auto_stop_patience}, "
                        f"best={float(self._auto_stop_best_metric):.6f}."
                    )
                    stop_reason = "early_stopped"
                    break

            except KeyboardInterrupt:
                self._log_keyboard_interrupt(epoch, best_loss, last_train_loss, last_val_loss)
                return TrainingOutcome(
                    reason="interrupted",
                    last_completed_epoch=last_completed_epoch if last_completed_epoch >= 0 else None,
                )
            
            except HDNDivergenceError as e:
                self.logger.error(
                    f"HDN divergence detected during epoch {self._display_epoch(epoch)}: {e}"
                )
                self._save_last_safe_training_state(epoch=epoch, cause=str(e))
                return TrainingOutcome(
                    reason="failed_retryable_hdn_divergence",
                    last_completed_epoch=last_completed_epoch if last_completed_epoch >= 0 else None,
                    failure_reason=str(e),
                    retry_eligible=True,
                )

            except Exception as e:
                self.logger.error(
                    f"Training stopped during epoch {self._display_epoch(epoch)}, because of error:\n"
                    f"{type(e)}\n{e}\n"
                )
                return TrainingOutcome(
                    reason="failed_nonretryable",
                    last_completed_epoch=last_completed_epoch if last_completed_epoch >= 0 else None,
                    failure_reason=f"{type(e).__name__}: {e}",
                    retry_eligible=False,
                )

        self._log_training_finished(epoch, best_loss, last_val_loss)
        return TrainingOutcome(
            reason=stop_reason,
            last_completed_epoch=last_completed_epoch if last_completed_epoch >= 0 else None,
        )


    # common training helpers

    def _backward_virtual_batch(self, raw_loss: torch.Tensor, num_virtual_batches: int) -> None:
        if num_virtual_batches <= 0:
            raise ValueError(f"num_virtual_batches must be >= 1, got {num_virtual_batches}")
        (raw_loss / num_virtual_batches).backward()
        # raw_loss.backward()

    def _optimizer_step(self) -> float | None:
        grad_norm = None

        max_grad_norm = self.training_prm.get("max_grad_norm", None)
        if max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=max_grad_norm,
                error_if_nonfinite=True,
            )
            grad_norm = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)

        self._apply_manual_warmup_if_needed()
        self.optimizer.step()
        self._optimizer_step_count += 1
        self.optimizer.zero_grad()
        return grad_norm
    
     
    def _cpu_state_dict(self, state_dict: dict) -> dict:
        cpu_state = {}
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                cpu_state[key] = value.detach().cpu().clone()
            elif isinstance(value, dict):
                cpu_state[key] = self._cpu_state_dict(value)
            elif isinstance(value, list):
                cpu_state[key] = [
                    self._cpu_state_dict(v) if isinstance(v, dict)
                    else (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                    for v in value
                ]
            else:
                cpu_state[key] = copy.deepcopy(value)
        return cpu_state

    def _capture_safe_training_state(self, *, epoch: int, batch_id: int) -> None:
        snapshot = {
            "epoch": int(epoch),
            "batch_id": int(batch_id),
            "model_state_dict": self._cpu_state_dict(self.model.state_dict()),
            "optimizer_state_dict": self._cpu_state_dict(self.optimizer.state_dict()),
            "scheduler_state_dict": self._cpu_state_dict(self.scheduler.state_dict()) if self.scheduler is not None else None,
            "train_loss": self.state_dict.get("train_loss"),
            "val_loss": self.state_dict.get("val_loss"),
            "optimizer_step_count": int(self._optimizer_step_count),
            "auto_stop_best_metric": self._auto_stop_best_metric,
            "auto_stop_bad_epochs": int(self._auto_stop_bad_epochs),
        }
        self._last_safe_training_state = snapshot

        pending = getattr(self, "_pending_safe_training_states", None)
        if pending is None:
            pending = deque()
            self._pending_safe_training_states = pending

        confirmed = getattr(self, "_confirmed_safe_training_states", None)
        if confirmed is None:
            rewind_steps = int(getattr(self, "_safe_state_rewind_steps", 1))
            confirmed = deque(maxlen=max(rewind_steps + 2, 2))
            self._confirmed_safe_training_states = confirmed

        pending.append(snapshot)

        confirmation_lag = max(int(getattr(self, "_safe_state_confirmation_lag", 1)), 0)
        while len(pending) > confirmation_lag:
            confirmed.append(pending.popleft())

    def _select_safe_training_state_for_persistence(self) -> tuple[dict | None, str]:
        confirmed = getattr(self, "_confirmed_safe_training_states", None)
        if confirmed:
            confirmed_list = list(confirmed)
            rewind_steps = max(int(getattr(self, "_safe_state_rewind_steps", 1)), 0)
            selected_idx = max(len(confirmed_list) - 1 - rewind_steps, 0)
            return confirmed_list[selected_idx], "confirmed_buffer"

        latest = getattr(self, "_last_safe_training_state", None)
        if latest:
            return latest, "latest_fallback"

        return None, "none"

    def _save_last_safe_training_state(self, *, epoch: int, cause: str) -> None:
        selected_state, selected_source = self._select_safe_training_state_for_persistence()
        if not selected_state:
            self.logger.warning("No safe training state available to save.")
            return
        if not (self.saving_prm.get("enabled", True) and self.run_dir is not None):
            self.logger.warning("Saving disabled; cannot persist safe training state.")
            return

        if selected_source == "latest_fallback":
            self.logger.warning(
                "No confirmed lagged safe state available; falling back to latest captured safe state."
            )
        else:
            self.logger.warning(
                "Using confirmed lagged safe state for divergence recovery "
                f"(rewind_steps={self._safe_state_rewind_steps})."
            )

        safe_state = dict(selected_state)
        last_completed_epoch = int(self.state_dict.get("epoch", -1))
        safe_state_epoch = int(safe_state.get("epoch", last_completed_epoch))
        # Trainer resume semantics are epoch-based: `state_dict["epoch"]` must be the
        # last fully completed epoch to avoid skipping an in-progress failed epoch.
        resume_epoch = min(safe_state_epoch, last_completed_epoch)
        safe_state["safe_state_source"] = selected_source
        safe_state["safe_epoch"] = safe_state_epoch
        safe_state["epoch"] = resume_epoch
        safe_state["failure_epoch"] = int(epoch)
        safe_state["failure_cause"] = cause

        checkpoint_name = self.ckpt.save_emergency_safe_state(
            state_dict=safe_state,
            model=self.model,
            tag="safe_on_divergence",
        )

        safe_resume_fail_count = self._read_safe_resume_fail_count(self.run_dir) + 1

        if self.run_dir is not None:
            update_run_recovery_info(
                self.run_dir,
                failure_reason=cause,
                recovery_checkpoint_filename=checkpoint_name,
                recovery_strategy="hdn_safe_resume_v1",
                last_safe_epoch=resume_epoch,
                last_safe_batch_id=int(selected_state.get("batch_id", -1)),
                safe_resume_fail_count=safe_resume_fail_count,
            )


    # safe resuming related
    def _resolve_safe_resume_rewind_steps(self) -> int:
        recovery_cfg = getattr(self.cfg, "recovery", None)
        if recovery_cfg is None or not hasattr(recovery_cfg, "hdn_safe_resume"):
            return 1

        safe_cfg = recovery_cfg.hdn_safe_resume
        raw_value = getattr(safe_cfg, "rewind_steps", 1)
        try:
            value = int(raw_value)
        except Exception:
            return 1
        return max(value, 0)

    def _drop_optimizer_scheduler_state_on_safe_resume(self) -> None:
        if not isinstance(self.state_dict, dict):
            return
        if not self._is_hdn_safe_resume_active():
            return

        safe_cfg = self.cfg.recovery.hdn_safe_resume
        if not safe_cfg.drop_optimizer_scheduler_state_on_safe_resume:
            return

        dropped = False
        for key in ("optimizer_state_dict", "scheduler_state_dict", "scheduler"):
            if key in self.state_dict:
                self.state_dict.pop(key, None)
                dropped = True

        if dropped:
            self.logger.warning(
                "HDN safe resume active: dropped optimizer/scheduler state and kept model weights only."
            )
        else:
            self.logger.warning(
                "HDN safe resume active: optimizer/scheduler state drop requested, but none found in checkpoint."
            )

    def _align_safe_resume_epoch_with_metadata(self) -> None:
        if not self._is_hdn_safe_resume_active():
            return
        if not isinstance(self.state_dict, dict):
            return

        origin_run_dir = getattr(self.cfg.experiment, "origin_run_dir", None)
        if not origin_run_dir:
            return

        state_epoch = self.state_dict.get("epoch")
        if state_epoch is None:
            return

        try:
            metadata = read_run_metadata(origin_run_dir)
        except Exception:
            return

        last_completed_epoch = getattr(metadata, "last_epoch", None)
        if last_completed_epoch is None:
            return

        state_epoch = int(state_epoch)
        last_completed_epoch = int(last_completed_epoch)
        if state_epoch <= last_completed_epoch:
            return

        self.logger.warning(
            "HDN safe resume active: "
            f"checkpoint epoch={state_epoch} is ahead of last completed epoch={last_completed_epoch}; "
            "clamping resume epoch to prevent skipping an unfinished epoch."
        )
        self.state_dict["epoch"] = last_completed_epoch

    def _is_hdn_safe_resume_active(self) -> bool:
        if self.mode != "continue_training":
            return False

        recovery_cfg = getattr(self.cfg, "recovery", None)
        if recovery_cfg is None or not hasattr(recovery_cfg, "hdn_safe_resume"):
            return False

        safe_cfg = recovery_cfg.hdn_safe_resume
        if not safe_cfg.enabled:
            return False

        origin_run_dir = getattr(self.cfg.experiment, "origin_run_dir", None)
        checkpoint = getattr(self.cfg.load_model, "checkpoint", None)
        checkpoint_filename = getattr(checkpoint, "filename", None) if checkpoint is not None else None

        if not origin_run_dir or not checkpoint_filename:
            return False

        try:
            metadata = read_run_metadata(origin_run_dir)
        except Exception:
            return False

        recovery_checkpoint_filename = getattr(metadata, "recovery_checkpoint_filename", None)
        return (
            recovery_checkpoint_filename is not None
            and checkpoint_filename == recovery_checkpoint_filename
        )

    def _apply_recovery_overrides(self) -> None:
        if not self._is_hdn_safe_resume_active():
            return

        safe_cfg = self.cfg.recovery.hdn_safe_resume

        if safe_cfg.lr_scale is not None:
            lr_scale = float(safe_cfg.lr_scale)
            base_lr = float(getattr(self, "_base_learning_rate_from_config", 1e-4))
            safe_resume_fail_count = self._read_safe_resume_fail_count(
                getattr(self.cfg.experiment, "origin_run_dir", None)
            )
            compound_steps = safe_resume_fail_count
            max_compound_steps = getattr(safe_cfg, "max_compound_steps", None)
            if max_compound_steps is not None:
                compound_steps = min(compound_steps, int(max_compound_steps))

            effective_scale = lr_scale ** compound_steps
            scaled_lr = base_lr * effective_scale
            min_lr = float(getattr(safe_cfg, "min_lr", 0.0))
            target_lr = max(scaled_lr, min_lr)

            new_lrs = []
            for group in self.optimizer.param_groups:
                if "lr" in group:
                    group["lr"] = float(target_lr)
                    new_lrs.append(float(group["lr"]))
            self.logger.warning(
                "HDN safe resume active: "
                f"base_lr={base_lr}, lr_scale={lr_scale}, fail_count={safe_resume_fail_count}, "
                f"compound_steps={compound_steps}, effective_scale={effective_scale}, "
                f"min_lr={min_lr}, new_lr={new_lrs}"
            )

        if safe_cfg.force_grad_clip_max_norm is not None:
            self.training_prm["max_grad_norm"] = float(safe_cfg.force_grad_clip_max_norm)
            self.logger.warning(
                f"HDN safe resume active: forced max_grad_norm={self.training_prm['max_grad_norm']}"
            )

    def _read_safe_resume_fail_count(self, run_dir: str | Path | None) -> int:
        if not run_dir:
            return 0
        try:
            metadata = read_run_metadata(run_dir)
        except Exception:
            return 0

        value = getattr(metadata, "safe_resume_fail_count", 0)
        try:
            parsed = int(value)
        except Exception:
            return 0
        return max(parsed, 0)



    # update console helper
    def _update_console_new_batch(self,epoch,batch_id,total_batches):
        """ Update console with epoch and batch number, without logging into log file."""
        if self.file_filter is not None:
            self.file_filter.enable = False
        self.logger.info(
            f"epochs: {self._display_epoch(epoch)}/{self.n_epochs}, "
            f"batch_id: {batch_id}/{total_batches}"
        )
        if self.file_filter is not None:
            self.file_filter.enable = True

    def _log_epoch_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict):
        epoch_idx = self._display_epoch(epoch)
        message = (
            f"epoch {epoch_idx}/{self.n_epochs}: "
            f"train_loss={float(train_metrics['loss']):.6f} "
            f"val_loss={float(val_metrics['loss']):.6f}"
        )
        if self.is_lvae:
            message += (
                f" train_kl={float(train_metrics['kl_loss']):.6f}"
                f" val_kl={float(val_metrics['kl_loss']):.6f}"
                f" train_recons={float(train_metrics['recons_loss']):.6f}"
                f" val_recons={float(val_metrics['recons_loss']):.6f}"
            )
        self.logger.info(message)

    def _initialize_log_file(self):
        if self._training_log_initialized:
            return

        self._training_log_initialized = True
        if self.file_filter is not None:
            self.file_filter.enable = True

        header = self._build_training_log_header()
        if not header:
            return

        console_enabled = None
        if self.console_filter is not None:
            console_enabled = self.console_filter.enable
            self.console_filter.enable = False

        try:
            self.logger.info(header)
        finally:
            if self.console_filter is not None and console_enabled is not None:
                self.console_filter.enable = console_enabled

    def _build_training_log_header(self) -> str:
        if self.run_dir is None:
            return ""

        computer = os.environ.get("COMPUTERNAME", "Unknown")
        gpu = self._device_name()
        patch_txt = self._patch_info_text()

        if self.mode == "train":
            return (
                f"\nExperiment name: {self.exp_name}\n"
                f"Computer: {computer}\n"
                f"Running on: {gpu}\n\n"
                f"{patch_txt}"
            )

        if self.mode == "retrain":
            return (
                f"\nExperiment name: {self.exp_name}\n"
                f"Computer: {computer}\n"
                f"Running on: {gpu}\n\n"
                "Retrain mode - starting from previously trained model.\n"
                "Check cfg file and 'retrain_origin_model' folder for details. \n\n"
                f"{patch_txt}"
            )

        if self.mode == "continue_training":
            return f"Continue training  mode. Computer: {computer}, with {gpu}.\n"

        return ""

    def _patch_info_text(self) -> str:
        if not self.patch_info:
            return ""

        train_patch = self.patch_info.get("train_patch")
        val_patch = self.patch_info.get("val_patch")
        if train_patch is None or val_patch is None:
            return ""

        return (
            f"Training patches: {train_patch}.\n"
            f"Validation patches {val_patch}.\n\n"
        )

    def _device_name(self) -> str:
        if getattr(self.device, "type", None) != "cuda":
            return "CPU"

        try:
            device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            return torch.cuda.get_device_name(device_index)
        except Exception:
            return "CPU"

    def _log_keyboard_interrupt(self, epoch: int, best_loss: float, train_loss, val_loss):
        epoch_idx = self._display_epoch(epoch)
        if train_loss is None or val_loss is None:
            self.logger.info(f"Training manually stopped during epoch {epoch_idx}.")
            return

        self.logger.info(
            f"Training manually stopped during epoch {epoch_idx}.\n"
            f"Model perf: best_val_loss: {best_loss} - "
            f"current_train_loss: {train_loss} - "
            f"current_val_loss: {val_loss}.\n"
        )

    def _log_training_finished(self, epoch: int, best_loss: float, val_loss):
        epoch_idx = self._display_epoch(epoch)
        if val_loss is None:
            self.logger.info(f"Finished training: {epoch_idx}/{self.n_epochs} epochs.")
            return

        self.logger.info(
            f"Finished training: {epoch_idx}/{self.n_epochs} epochs.\n"
            f"Model perf: best_val_loss: {best_loss} - "
            f"current_val_loss: {val_loss}.\n"
        )

    @staticmethod
    def _display_epoch(epoch: int) -> int:
        return int(epoch) + 1

    # must be implemented in children
    @abstractmethod
    def train_epoch(self, epoch: int) -> dict:
        ...

    @abstractmethod
    def validate(self, epoch: int, save_imgs: bool = False) -> dict:
        ...
