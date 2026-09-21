from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from lisai.config import settings
from lisai.evaluation.saved_run import SavedTrainingRun
from lisai.infra.paths import Paths

from .schema import PromotedNoiseModelArtifacts


@dataclass(frozen=True)
class PromotionFile:
    """One source file that should be copied verbatim into a promoted package."""

    source_path: Path
    package_path: str

    def __post_init__(self):
        source = Path(self.source_path)
        package = PurePosixPath(str(self.package_path).replace("\\", "/"))
        if package.is_absolute() or ".." in package.parts or package.as_posix() == ".":
            raise ValueError("package_path must be a safe relative path.")
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "package_path", package.as_posix())


@dataclass(frozen=True)
class PromotionDependencies:
    files: tuple[PromotionFile, ...] = ()
    noise_model: PromotedNoiseModelArtifacts | None = None


def collect_required_dependencies(
    saved_run: SavedTrainingRun,
    *,
    paths: Paths | None = None,
) -> PromotionDependencies:
    """Collect external inference dependencies required by a saved run.

    At present only LVAE/HDN has an external runtime dependency: its registered
    noise model and matching normalization parameters. Other architectures are
    self-contained once config and model weights are packaged.
    """
    if not saved_run.is_lvae:
        return PromotionDependencies()

    if not saved_run.noise_model_name:
        raise ValueError("Saved LVAE/HDN run is missing noise_model.name and cannot be promoted.")

    resolved_paths = paths or Paths(settings)
    model_path = resolved_paths.noise_model_path(noiseModel_name=saved_run.noise_model_name)
    norm_path = resolved_paths.noise_model_norm_prm_path(
        noiseModel_name=saved_run.noise_model_name
    )
    missing = [path for path in (model_path, norm_path) if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "LVAE/HDN promotion requires a self-contained noise model, but these canonical "
            f"files are missing:\n{joined}"
        )

    model_package_path = "artifacts/noise_model/GMM.npz"
    norm_package_path = "artifacts/noise_model/norm_prm.json"
    return PromotionDependencies(
        files=(
            PromotionFile(model_path, model_package_path),
            PromotionFile(norm_path, norm_package_path),
        ),
        noise_model=PromotedNoiseModelArtifacts(
            name=saved_run.noise_model_name,
            model=model_package_path,
            norm_prm=norm_package_path,
        ),
    )


__all__ = [
    "PromotionDependencies",
    "PromotionFile",
    "collect_required_dependencies",
]
