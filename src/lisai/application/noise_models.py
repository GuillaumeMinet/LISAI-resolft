"""Small workspace-bound noise-model catalog for application and Qt clients.

A matching GMM/normalization file pair is *discovered*, not scientifically
validated. Validation/building remain the responsibility of the existing
noise-model APIs, and are not silently performed by the browser.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NoiseModelEntry:
    name: str
    directory: Path
    model_path: Path
    normalization_path: Path
    model_present: bool
    normalization_present: bool

    @property
    def usable(self) -> bool:
        """Both expected resources exist; does *not* imply calibrated validity."""
        return self.model_present and self.normalization_present


def list_noise_model_entries(settings) -> list[NoiseModelEntry]:
    """List models from the selected workspace, including incomplete folders."""
    from lisai.infra.paths import Paths

    paths = Paths(settings)
    parent = paths.noise_model_dir(noiseModel_name="_lisai_probe_").parent
    if not parent.is_dir():
        return []
    found = []
    for folder in sorted(parent.iterdir(), key=lambda p: p.name.casefold()):
        if not folder.is_dir() or folder.is_symlink():
            continue
        name = folder.name
        model = paths.noise_model_path(noiseModel_name=name)
        norm = paths.noise_model_norm_prm_path(noiseModel_name=name)
        found.append(NoiseModelEntry(
            name=name, directory=folder, model_path=model, normalization_path=norm,
            model_present=model.is_file(), normalization_present=norm.is_file(),
        ))
    return found


__all__ = ["NoiseModelEntry", "list_noise_model_entries"]
