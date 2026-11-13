# src/lisai/training/checkpointing/manager.py

import os
from pathlib import Path

import torch

from lisai.infra.paths import Paths, model_filename


class CheckpointManager:
    def __init__(self, run_dir: Path | None, saving_prm: dict, *, paths: Paths, is_lvae: bool = False):
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.saving_prm = saving_prm or {}
        self.is_lvae = is_lvae

        self.enabled = self.run_dir is not None and self.saving_prm.get("enabled", True)

        self.paths = paths

        self.checkpoints_dir = None
        self.loss_file = None

        if self.enabled:
            self.checkpoints_dir = self.paths.checkpoints_dir(run_dir=self.run_dir)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.loss_file = self.paths.loss_file_path(run_dir=self.run_dir)

    # LOSS.TXT
    def update_loss_file(self, epoch: int, train_metrics: dict, val_metrics: dict):
        if not self.enabled:
            return

        if not os.path.exists(self.loss_file):
            with open(self.loss_file, "w") as f:
                if self.is_lvae:
                    head = "Epoch Train_loss Val_loss Recons_Loss KL_Loss".split()
                    f.write(f"{head[0]:<10} {head[1]:<30} {head[2]:<30} {head[3]:<30} {head[4]:<30}\n")
                else:
                    head = "Epoch Train_loss Val_loss".split()
                    f.write(f"{head[0]:<10} {head[1]:<30} {head[2]:<30}\n")

        with open(self.loss_file, "a") as f:
            if self.is_lvae:
                f.write(
                    f"{epoch:<10} {train_metrics['loss']:<30} {val_metrics['loss']:<30}"
                    f"{train_metrics['recons_loss']:<30} {train_metrics['kl_loss']:<30}\n"
                )
            else:
                f.write(f"{epoch:<10} {train_metrics['loss']:<30} {val_metrics['loss']:<30}\n")

    # CHECKPOINT SAVING 
    def save(self, *, state_dict: dict, model, best_loss: float, is_best: bool):
        if not self.enabled:
            return

        epoch = state_dict["epoch"]
        overwrite_best = self.saving_prm.get("overwrite_best", True)

        # legacy: best name depends on overwrite_best
        if overwrite_best:
            name_best_prefix = "model_best"
        else:
            name_best_prefix = f"model_epoch_{epoch}"

        # state_dict saving
        if self.saving_prm.get("state_dict", False):
            if is_best:
                name = model_filename(load_method="state_dict", best_or_last="best")
                if not overwrite_best:
                    name = model_filename(load_method="state_dict", epoch_number=epoch)
                torch.save(state_dict, self.checkpoints_dir / name)

            # always save last, with best_loss temporarily injected
            name_last = model_filename(load_method="state_dict", best_or_last="last")
            state_dict["best_loss"] = best_loss
            torch.save(state_dict, self.checkpoints_dir / name_last)
            del state_dict["best_loss"]

        # entire model saving
        if self.saving_prm.get("entire_model", False):
            if is_best:
                name = model_filename(load_method="full_model", best_or_last="best")
                if not overwrite_best:
                    name = model_filename(load_method="full_model", epoch_number=epoch)
                torch.save(model, self.checkpoints_dir / name)

            name_last = model_filename(load_method="full_model", best_or_last="last")
            torch.save(model, self.checkpoints_dir / name_last)
    
    def save_emergency_safe_state(self, *, state_dict: dict, model, tag: str = "safe_state"):
        if not self.enabled:
            return

        name = f"{tag}.pth"
        torch.save(state_dict, self.checkpoints_dir / name)
        return name