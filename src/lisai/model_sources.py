"""Lightweight, explicit references to sources of inference models.

Kept independent from the training and run-discovery packages so importing
the public LISAI API does not eagerly load those components.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunRef:
    """Explicit reference to a saved training run in the selected workspace.

    Unlike a model-name string, this selects an exact run metadata ID and
    never promotes, exports, or installs the run.
    """

    run_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("RunRef.run_id must be a nonempty run identifier.")
        object.__setattr__(self, "run_id", self.run_id.strip().upper())


__all__ = ["RunRef"]
