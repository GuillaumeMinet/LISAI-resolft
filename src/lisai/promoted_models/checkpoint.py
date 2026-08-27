from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping

import torch

from lisai.config import settings
from lisai.evaluation.saved_run import SavedTrainingRun
from lisai.infra.paths import Paths

PromotionCheckpointSelector = Literal["best", "last"]


def resolve_promotion_checkpoint(
    saved_run: SavedTrainingRun,
    *,
    selector: PromotionCheckpointSelector = "best",
    paths: Paths | None = None,
) -> Path:
    """Resolve the canonical state-dict checkpoint used for promotion."""
    if selector not in {"best", "last"}:
        raise ValueError("selector must be 'best' or 'last'.")
    if "state_dict" not in saved_run.checkpoint_methods:
        raise ValueError(
            "Promoting a model currently requires a state-dict checkpoint, but the saved run "
            "does not enable state_dict saving."
        )

    resolved_paths = paths or Paths(settings)
    checkpoint_path = resolved_paths.checkpoint_path(
        run_dir=saved_run.run_dir,
        load_method="state_dict",
        best_or_last=selector,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find the {selector!r} state-dict checkpoint for promotion: "
            f"{checkpoint_path}"
        )
    return checkpoint_path


def _torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Older torch versions do not expose weights_only.
        return torch.load(path, map_location="cpu")


def extract_model_weights(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    """Extract inference-only model weights from a LISAI training checkpoint.

    Canonical LISAI training checkpoints contain optimizer/scheduler state around a
    ``model_state_dict`` entry. Promoted models only need the model weights. Plain
    state-dict files are also accepted for compatibility with imported/legacy runs.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    loaded = _torch_load_cpu(checkpoint_path)
    if isinstance(loaded, Mapping) and "model_state_dict" in loaded:
        state_dict = loaded["model_state_dict"]
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError(
                f"Checkpoint {checkpoint_path} contains an invalid or empty model_state_dict."
            )
    elif isinstance(loaded, Mapping) and loaded:
        # Support canonical plain state_dict files. Reject arbitrary checkpoint
        # dictionaries so optimizer-only/training-state payloads cannot be promoted
        # accidentally.
        if not all(isinstance(key, str) for key in loaded):
            raise ValueError(f"Checkpoint {checkpoint_path} is not a valid model state dict.")
        if not all(isinstance(value, torch.Tensor) for value in loaded.values()):
            raise ValueError(
                f"Checkpoint {checkpoint_path} does not contain model_state_dict and does not "
                "look like a plain model state dict."
            )
        state_dict = loaded
    else:
        raise ValueError(
            f"Unsupported state-dict checkpoint payload at {checkpoint_path}: {type(loaded)!r}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict), output_path)
    return output_path


__all__ = [
    "PromotionCheckpointSelector",
    "extract_model_weights",
    "resolve_promotion_checkpoint",
]
