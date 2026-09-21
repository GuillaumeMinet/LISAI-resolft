from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Generic, Optional, Type, TypeVar, Union

import numpy as np

from ..core import Item, OutputSpec, Source

if TYPE_CHECKING:
    from ..run_preprocess import PreprocessRun

CfgT = TypeVar("CfgT")

class PipelineConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PipelineResult:
    """
    Summary of a preprocessing run. Pipelines may populate optional fields.
    """
    n_files: int
    n_frames: Optional[int] = None
    snr_levels: Optional[Union[int, list[int]]] = None
    timepoints: Optional[Union[int, list[int]]] = None

class BasePipeline(ABC,Generic[CfgT]):
    """
    Minimal base for pipelines using a dataclass config.

    Subclasses must set:
      - Config: a dataclass type
    """
    Config: ClassVar[Type[CfgT]]

    def __init__(self, cfg: CfgT):
        self.cfg = cfg

    @classmethod
    def parse_cfg(cls, raw: dict | None) -> CfgT:
        raw = {} if raw is None else dict(raw)

        cfg_type = getattr(cls, "Config", None)
        if cfg_type is None or not is_dataclass(cfg_type):
            raise PipelineConfigError(
                f"{cls.__name__}.Config must be a dataclass type"
            )

        allowed = {f.name for f in fields(cfg_type)}
        unknown = sorted(set(raw.keys()) - allowed)
        if unknown:
            raise PipelineConfigError(
                f"Unknown config keys for {cls.__name__}: {unknown}. "
                f"Allowed keys: {sorted(allowed)}"
            )

        try:
            return cfg_type(**raw) 
        except TypeError as e:
            # e.g. wrong constructor shape / unexpected dataclass behavior
            raise PipelineConfigError(
                f"Invalid config for {cls.__name__} ({cfg_type.__name__}): {e}"
            ) from e
    

    # ---- required API ----
    @abstractmethod
    def build_source(self, *, run: PreprocessRun) -> Source:
        """Return the OutputSpec (whatever your project type is)."""
        raise NotImplementedError
    
    @abstractmethod
    def output_spec(self) -> OutputSpec:
        """Return the OutputSpec (whatever your project type is)."""
        raise NotImplementedError

    @abstractmethod
    def process_item(self, *,item: Item) -> Dict[str, np.ndarray]:
        """Process one source item."""
        raise NotImplementedError

    @abstractmethod
    def template_kwargs(self, *, item: Item, outputs: dict[str, np.ndarray]) -> dict[str, object]:
        """Return kwargs used for output path templating / naming."""
        raise NotImplementedError

    @abstractmethod
    def make_result(self, *, n_files: int, stats: dict[str, Any]) -> PipelineResult:
        """Build the final PipelineResult using collected stats."""
        raise NotImplementedError
    
    # stats counting - to implement in child if necessary
    def init_stats(self) -> dict[str, Any]:
        """Initialize a mutable stats dict (pipeline-specific)."""
        return {}

    def update_stats(self, *, stats: dict[str, Any], item: Item, outputs: dict[str, Any]) -> dict[str, Any]:
        """Update stats given the outputs produced for one item."""
        return stats
