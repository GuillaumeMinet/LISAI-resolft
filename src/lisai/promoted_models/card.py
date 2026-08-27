from __future__ import annotations

from typing import Any, Mapping

from .promotion import PromotionPlan
from .schema import PromotedModelManifest


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _parameter_lines(parameters: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in parameters.items():
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            continue
        lines.append(f"- `{key}`: {_format_value(value)}")
    return lines


def render_model_card(plan: PromotionPlan, manifest: PromotedModelManifest) -> str:
    """Render a concise, mostly automatic model card for a promoted model."""
    code = manifest.source.code
    params = plan.saved_run.model_parameters.model_dump(mode="json", exclude_none=True)
    parameter_lines = _parameter_lines(params)

    lines = [
        f"# {manifest.name}",
        "",
        "LISAI promoted model.",
        "",
        "## Model overview",
        "",
        f"- Task: `{manifest.model.task}`",
        f"- Architecture: `{manifest.model.architecture}`",
        f"- Training dataset: `{manifest.training_data.dataset}`",
        f"- Source run: `{manifest.source.run_name}` (`{manifest.source.run_id}`)",
        f"- Source run status: `{manifest.source.run_status}`",
        f"- Promoted checkpoint: `{manifest.source.checkpoint_selector}`",
    ]

    if code is not None:
        lines.extend(
            [
                f"- LISAI version: `{_format_value(code.lisai_version)}`",
                f"- Git commit: `{_format_value(code.git_commit)}`",
                f"- Git branch: `{_format_value(code.git_branch)}`",
                f"- Git working tree dirty at run creation: `{_format_value(code.git_dirty)}`",
            ]
        )

    lines.extend(["", "## Model parameters", ""])
    if parameter_lines:
        lines.extend(parameter_lines)
    else:
        lines.append("See `config_train.yaml` for the complete resolved model configuration.")

    lines.extend(["", "## Training summary", ""])
    if manifest.training.best_val_loss is not None:
        lines.append(f"- Best validation loss: `{manifest.training.best_val_loss}`")
    if manifest.training.last_epoch is not None:
        lines.append(f"- Last recorded epoch: `{manifest.training.last_epoch}`")
    if manifest.training.max_epoch is not None:
        lines.append(f"- Configured maximum epoch: `{manifest.training.max_epoch}`")
    if manifest.artifacts.loss is not None:
        lines.append(f"- Loss history: `{manifest.artifacts.loss}`")
    if manifest.artifacts.loss_plot is not None:
        lines.append(f"- Loss plot: `{manifest.artifacts.loss_plot}`")

    lines.extend(
        [
            "",
            "## Package contents",
            "",
            f"- Inference weights: `{manifest.artifacts.weights}`",
            f"- Resolved training config: `{manifest.artifacts.config}`",
        ]
    )
    if manifest.artifacts.noise_model is not None:
        lines.extend(
            [
                f"- Noise model: `{manifest.artifacts.noise_model.model}`",
                f"- Noise-model normalization: `{manifest.artifacts.noise_model.norm_prm}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Description",
            "",
            "TODO: Add a short scientific description of this promoted model.",
            "",
            "## Intended use",
            "",
            "TODO: Describe the intended microscopy data and use case.",
            "",
            "## Limitations",
            "",
            "TODO: Describe known limitations and conditions under which predictions should be validated.",
            "",
            "## Citation",
            "",
            "TODO: Add the publication citation and/or DOI.",
            "",
            "## Authors and license",
            "",
            "TODO: Add authorship and license information.",
            "",
            "## Reproducibility",
            "",
            "`lisai_model.yaml` contains the machine-readable model identity, source-run provenance, "
            "artifact locations, and SHA256 checksums. `config_train.yaml` is the resolved training "
            "configuration saved by the original LISAI run.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["render_model_card"]
