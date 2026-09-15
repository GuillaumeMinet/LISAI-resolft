

from __future__ import annotations
from typing import TYPE_CHECKING


from lisai.evaluation import run_evaluate
from lisai.infra.cli.prompts import prompt_yes_no
from lisai.runs.plotting import save_loss_plot_for_run

if TYPE_CHECKING:
    from lisai.training.trainers.base import TrainingOutcome


POST_TRAINING_INFERENCE_CONFIG = "post_training"


def _prompt_yes_no(prompt: str) -> bool:
    return bool(prompt_yes_no(prompt, input_fn=input))


def _log_runtime_warning(runtime, message: str):
    logger = getattr(runtime, "logger", None)
    if logger is None:
        return

    warning = getattr(logger, "warning", None)
    if callable(warning):
        warning(message)
        return

    info = getattr(logger, "info", None)
    if callable(info):
        info(message)

def _should_run_post_training_evaluation(cfg, runtime, outcome: "TrainingOutcome") -> bool:
    experiment = getattr(cfg, "experiment", None)
    if not bool(getattr(experiment, "post_training_inference", False)):
        return False
    if runtime.run_dir is None:
        return False
    if outcome.reason in {"completed", "early_stopped"}:
        return True
    if outcome.reason != "interrupted":
        return False
    if outcome.last_completed_epoch is None:
        logger = getattr(runtime, "logger", None)
        info = getattr(logger, "info", None)
        if callable(info):
            info("Skipping post-training evaluation because no completed epoch is available.")
        return False
    return _prompt_yes_no(
        f"Training interrupted. Run evaluation now using '{POST_TRAINING_INFERENCE_CONFIG}'? [y/N]: "
    )

def run_post_training_evaluation(cfg, runtime, outcome: "TrainingOutcome") -> None:
    if not _should_run_post_training_evaluation(cfg, runtime, outcome):
        return

    if runtime.run_dir is None:
        return

    run_evaluate(
        dataset_name=cfg.data.dataset_name,
        model_name=runtime.run_dir.name,
        model_subfolder=cfg.routing.models_subfolder,
        config=POST_TRAINING_INFERENCE_CONFIG,
        progress_bar=bool(getattr(cfg.training, "progress_bar", False)),
    )


def auto_save_loss_plot_image(cfg, runtime, *, terminal_reason: str) -> None:
    if runtime.run_dir is None:
        return

    dataset = getattr(getattr(cfg, "data", None), "dataset_name", None)
    model_subfolder = getattr(getattr(cfg, "routing", None), "models_subfolder", None)
    architecture = getattr(getattr(cfg, "model", None), "architecture", None)
    if not isinstance(architecture, str):
        architecture = None

    try:
        saved_path = save_loss_plot_for_run(
            run_dir=runtime.run_dir,
            dataset=dataset,
            model_subfolder=model_subfolder,
            architecture=architecture,
            stderr=None,
        )
    except Exception as exc:
        _log_runtime_warning(
            runtime,
            f"Automatic loss-plot save failed after '{terminal_reason}': "
            f"{type(exc).__name__}: {exc}",
        )
        return

    if saved_path is None:
        _log_runtime_warning(
            runtime,
            f"Loss plot image was not saved after '{terminal_reason}' "
            "(missing or unreadable loss history).",
        )
        return

    logger = getattr(runtime, "logger", None)
    info = getattr(logger, "info", None)
    if callable(info):
        info(f"Saved loss plot image: {saved_path}")
