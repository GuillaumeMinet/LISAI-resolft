from __future__ import annotations

from typing import Any, Mapping

from .promotion import PromotionPlan
from .schema import PromotedModelManifest

AUTO_OVERVIEW_START = "<!-- lisai:auto:overview:start -->"
AUTO_OVERVIEW_END = "<!-- lisai:auto:overview:end -->"

LISAI_GITHUB_URL = "https://github.com/GuillaumeMinet/LISAI-resolft"

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


def render_model_overview(manifest: PromotedModelManifest) -> str:
    code = manifest.source.code

    lines=[
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

    return "\n".join(lines)

def _render_auto_overview_block(manifest: PromotedModelManifest) -> str:
    return "\n".join(
        [
            AUTO_OVERVIEW_START,
            render_model_overview(manifest),
            AUTO_OVERVIEW_END,
        ]
    )


def _dataset_readme_body(readme: str | None) -> str | None:
    if readme is None:
        return None
    lines = readme.strip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
    body = "\n".join(lines).strip()
    return body or None


def render_model_card(
    plan: PromotionPlan,
    manifest: PromotedModelManifest,
    *,
    dataset_description: str | None = None,
    dataset_readme: str | None = None,
) -> str:
    """Render a concise, mostly automatic model card for a promoted model."""

    params = plan.saved_run.model_parameters.model_dump(mode="json", exclude_none=True)
    parameter_lines = _parameter_lines(params)

    lines = [
        f"# {manifest.name}",
        "",
        f"LISAI promoted model. Created and packaged with [LISAI]({LISAI_GITHUB_URL}).",
        "",
        _render_auto_overview_block(manifest),
        "",
        "## Training dataset",
        "",
        f"- Dataset: `{manifest.training_data.dataset}`",
    ]

    if dataset_description:
        lines.append(f"- Description: {dataset_description}")

    dataset_body = _dataset_readme_body(dataset_readme)
    if dataset_body:
        lines.extend(["", dataset_body])

    lines.extend([
        "",
        "## Model parameters",
        "",
    ])

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
            "## Description & Intended use",
            "",
            "TODO: Add a short scientific description of this promoted model.",
            "",
            "## Limitations",
            "",
            "TODO: Describe known limitations and conditions under which predictions should be validated.",
            "",
            "Deep-learning predictions should be validated on representative data against appropriate "
            "experimental references before quantitative or biological interpretation. ",
            "",
            "## Reproducibility",
            "",
            "`lisai_model.yaml` contains the machine-readable model identity, source-run provenance, "
            "artifact locations, and SHA256 checksums. `config_train.yaml` is the resolved training "
            "configuration saved by the original LISAI run. `config_inference.yaml`, when present, "
            "contains the model-specific default inference settings used by `lisai apply --model`; "
            "these settings can be overridden for individual inference runs.",
            "",
            "## Citation & Authorship",
            "",
            "Minet, G., Ray, A., Pennacchietti, F., Coceano, G., Jug, F. & Testa, I."
            "*RESOLFT time lapse imaging empowered by deep learning*.",
            "",
            "Training datasets and trained models: "
            "Zenodo, DOI: 10.5281/zenodo.17132698",
            "",
            "Please cite the associated publication once the final journal citation and DOI become available.",
            "## Authors and license",
            "",
            "TODO: Add authorship and license information.",
            "",
        ]
    )
    return "\n".join(lines)


def update_model_card_overview(
        card_text:str,
        manifest: PromotedModelManifest,
    ) -> str:
    new_block = _render_auto_overview_block(manifest)

    start=card_text.find(AUTO_OVERVIEW_START)
    end=card_text.find(AUTO_OVERVIEW_END)

    if (start == -1) != (end==-1):
        raise ValueError(
            "Model card contains an incomplete LISAI auto-generated overview block."
        )

    if start != -1:
        end += len(AUTO_OVERVIEW_END)

        return (
            card_text[:start]
            + new_block
            + card_text[end:]
        )

    # legacy
    legacy_start_marker = "## Model overview"
    legacy_end_marker = "## Model parameters"

    legacy_start = card_text.find(legacy_start_marker)
    legacy_end = card_text.find(legacy_end_marker)
    if legacy_start == -1 or legacy_end == -1 or legacy_end <= legacy_start:
        raise ValueError(
            "Could not locate the model overview in the model card."
        )

    return (
        card_text[:legacy_start]
        + new_block
        + "\n\n"
        + card_text[legacy_end:]
    )

__all__ = [
    "render_model_card",
    "render_model_overview",
    "update_model_card_overview",
]
