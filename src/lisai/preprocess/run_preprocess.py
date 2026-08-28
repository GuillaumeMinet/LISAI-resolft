from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lisai.config import settings

from .core import DatasetRegistry, FolderSource, PreprocessConfig, PreprocessSaver
from .core.config import PreprocessLogConfig, PreprocessSplitConfig
from .core.run_log import PreprocessRunLog
from .core.sources import Item
from .core.split import plan_split, summarize_processed_splits
from .pipelines import PIPELINES_REGISTRY
from .pipelines.base import PipelineResult
from .reporting import (
    NoOpPreprocessReporter,
    PreprocessFinishReport,
    PreprocessItemReport,
    PreprocessReporter,
    PreprocessStartReport,
)

if TYPE_CHECKING:
    from lisai.infra.paths import Paths


@dataclass(frozen=True)
class ExistingPreprocessOutput:
    preprocess_dir: Path
    log_path: Path
    has_log: bool
    has_data: bool

    @property
    def exists(self) -> bool:
        return self.has_log or self.has_data

    def describe(self) -> str:
        reasons: list[str] = []
        if self.has_data:
            reasons.append(f"data in {self.preprocess_dir}")
        if self.has_log:
            reasons.append(f"log file {self.log_path}")
        found = " and ".join(reasons) if reasons else str(self.preprocess_dir)
        return (
            "Existing preprocess output detected. Found "
            f"{found}. Overwrite approval is required to delete the current preprocess content and rerun."
        )


@dataclass
class PreprocessRun:
    """
    Entry point for dataset preprocessing.

    High-level workflow:
    1. Load user config and validate parameters by loading to a Pydantic Config.
    2. Validate pipeline compatibility.
    3. Create OutputSpec (via pipeline).
    4. Create Saver, which handles all the saving, following OutputSpec.
    5. Build the Source.
    6. Optionally plan train/val/test splits.
    7. Iterate over Source items, apply transforms, save outputs, and update the run log.
    8. Update dataset registry.

    See docs/preprocess.md for full architecture details.
    """

    dataset_name: str
    paths: Paths
    logger: logging.Logger
    registry: DatasetRegistry

    data_type: str
    fmt: str
    usage: str
    pipeline_cfg: dict[str, Any]
    pipeline_name: str
    log_cfg: PreprocessLogConfig
    registry_defaults: dict[str, str | None]
    split_cfg: PreprocessSplitConfig

    @classmethod
    def from_cfg(cls, cfg: dict, *, paths: Paths | None = None) -> "PreprocessRun":
        pcfg = PreprocessConfig.model_validate(cfg)

        if paths is None:
            from lisai.infra.paths import Paths

            paths = Paths(settings)
        logger = logging.getLogger(f"lisai.preprocess.{pcfg.dataset_name}")
        registry = DatasetRegistry(paths.dataset_registry_path())
        return cls(
            dataset_name=pcfg.dataset_name,
            pipeline_name=pcfg.pipeline,
            data_type=pcfg.data_type,
            fmt=pcfg.fmt,
            usage=pcfg.usage,
            pipeline_cfg=pcfg.pipeline_cfg,
            paths=paths,
            registry=registry,
            logger=logger,
            log_cfg=pcfg.log,
            registry_defaults=pcfg.registry.defaults.model_dump(exclude_unset=True),
            split_cfg=pcfg.split,
        )

    def existing_output(self) -> ExistingPreprocessOutput:
        preprocess_dir = self.paths.dataset_preprocess_dir(
            dataset_name=self.dataset_name,
            data_type=self.data_type,
        )
        log_path = self.paths.preprocess_log_path(
            dataset_name=self.dataset_name,
            data_type=self.data_type,
        )
        has_log = log_path.exists()
        has_data = preprocess_dir.exists() and any(preprocess_dir.iterdir())
        return ExistingPreprocessOutput(
            preprocess_dir=preprocess_dir,
            log_path=log_path,
            has_log=has_log,
            has_data=has_data,
        )

    def _build_pipeline(self):
        try:
            pipeline_cls = PIPELINES_REGISTRY[self.pipeline_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown pipeline '{self.pipeline_name}'. Available: {sorted(PIPELINES_REGISTRY)}"
            ) from exc

        supported_data_types = getattr(
            pipeline_cls,
            "supported_data_types",
            {getattr(pipeline_cls, "data_type", None)},
        )
        supported_fmts = getattr(
            pipeline_cls,
            "supported_fmts",
            {getattr(pipeline_cls, "fmt", None)},
        )

        if self.data_type not in supported_data_types:
            raise ValueError(
                f"Pipeline '{self.pipeline_name}' does not support data_type='{self.data_type}'. "
                f"Supported: {sorted(supported_data_types)}"
            )
        if self.fmt not in supported_fmts:
            raise ValueError(
                f"Pipeline '{self.pipeline_name}' does not support fmt='{self.fmt}'. "
                f"Supported: {sorted(supported_fmts)}"
            )

        cfg_obj = pipeline_cls.parse_cfg(self.pipeline_cfg)
        return pipeline_cls(cfg=cfg_obj)

    def _build_run_log(self) -> PreprocessRunLog | None:
        if not self.log_cfg.enabled:
            return None
        return PreprocessRunLog.start(
            path=self.paths.preprocess_log_path(
                dataset_name=self.dataset_name,
                data_type=self.data_type,
            ),
            dataset_name=self.dataset_name,
            pipeline_name=self.pipeline_name,
            data_type=self.data_type,
            fmt=self.fmt,
            usage=self.usage,
            pipeline_cfg=self.pipeline_cfg,
            log_cfg=self.log_cfg.model_dump(exclude_none=True),
            split_cfg=self.split_cfg.model_dump(exclude_none=True),
            preprocess_dir=self.paths.dataset_preprocess_dir(
                dataset_name=self.dataset_name,
                data_type=self.data_type,
            ),
        )

    def _report_start(self, reporter: PreprocessReporter, *, source, preprocess_dir: Path) -> None:
        if isinstance(source, FolderSource):
            report = PreprocessStartReport(
                source=str(source.root.resolve()),
                target=str(preprocess_dir.resolve()),
                combine_subfolders=source.combine_subfolders,
            )
        else:
            report = PreprocessStartReport(
                source=source.__class__.__name__,
                target=str(preprocess_dir.resolve()),
                combine_subfolders=None,
            )
        reporter.report_start(report)

    def _source_name(self, item: Item) -> str:
        if item.source_name is not None:
            return item.source_name
        return item.paths[0].name

    def _progress_output_name(self, saved_outputs: dict[str, str]) -> str:
        output_names = [Path(relative_path).name for relative_path in saved_outputs.values()]
        if not output_names:
            return ""
        if len(output_names) == 1:
            return output_names[0]
        return ", ".join(output_names)

    def _finish_report(
        self,
        *,
        status: str,
        preprocess_dir: Path,
        n_files_written: int,
        split_summary: dict[str, Any],
        error: Exception | None = None,
    ) -> PreprocessFinishReport:
        empty_bucket = {"count": 0, "source_names": [], "output_names": []}
        val = split_summary.get("val", empty_bucket)
        test = split_summary.get("test", empty_bucket)
        return PreprocessFinishReport(
            status=status,
            preprocess_dir=str(preprocess_dir.resolve()),
            n_files_written=n_files_written,
            n_files_moved=int(val["count"]) + int(test["count"]),
            split_enabled=bool(split_summary.get("enabled", False)),
            val=val,
            test=test,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
        )

    def execute(
        self,
        *,
        overwrite: bool = False,
        reporter: PreprocessReporter | None = None,
    ) -> PipelineResult:
        reporter = NoOpPreprocessReporter() if reporter is None else reporter

        existing_output = self.existing_output()
        if existing_output.exists and not overwrite:
            raise FileExistsError(existing_output.describe())
        if existing_output.exists and overwrite and existing_output.preprocess_dir.exists():
            shutil.rmtree(existing_output.preprocess_dir)

        pipeline = self._build_pipeline()
        spec = pipeline.output_spec()
        source = pipeline.build_source(run=self)
        preprocess_dir = self.paths.dataset_preprocess_dir(
            dataset_name=self.dataset_name,
            data_type=self.data_type,
        )
        self._report_start(reporter, source=source, preprocess_dir=preprocess_dir)

        split_plan = None
        run_log: PreprocessRunLog | None = None
        total_items: int | None = None
        n_files = 0
        stats = pipeline.init_stats()
        processed_items: list[dict[str, Any]] = []

        try:
            saver = PreprocessSaver(
                paths=self.paths,
                dataset_name=self.dataset_name,
                data_type=self.data_type,
                fmt=self.fmt,
                output_spec=spec,
            )
            run_log = self._build_run_log()

            if self.split_cfg.enabled:
                items = list(source.iter_items())
                split_plan = plan_split(
                    items=items,
                    split_cfg=self.split_cfg,
                    sample_id_fn=saver.sample_id,
                    dataset_name=self.dataset_name,
                    data_type=self.data_type,
                    paths=self.paths,
                )
                item_iterable = enumerate(items)
                total_items = len(items)
            else:
                item_iterable = enumerate(source.iter_items())

            for index, item in item_iterable:
                sample_id = saver.sample_id(index)
                save_split = split_plan.split_for(index) if split_plan is not None else None
                if split_plan is not None:
                    recorded_split = save_split
                elif self.usage == "training":
                    # Preserve the historical meaning of an unsplit training dataset: all
                    # samples are available to the training loader as the training pool.
                    recorded_split = "train"
                else:
                    # Evaluation-only datasets are whole-dataset resources, not a synthetic
                    # train split. Their files remain at the output root.
                    recorded_split = None
                outputs = pipeline.process_item(item=item)
                template_kwargs = pipeline.template_kwargs(item=item, outputs=outputs)

                saved_outputs: dict[str, str] = {}
                for key, array in outputs.items():
                    output_path = saver.save(
                        key=key,
                        array=array,
                        sample_id=sample_id,
                        split=save_split,
                        **template_kwargs,
                    )
                    saved_outputs[key] = output_path.relative_to(preprocess_dir).as_posix()

                if run_log is not None:
                    run_log.record_item(
                        index=index,
                        sample_id=sample_id,
                        split=recorded_split,
                        item=item,
                        saved_outputs=saved_outputs,
                    )

                source_name = self._source_name(item)
                output_name = self._progress_output_name(saved_outputs)
                processed_items.append(
                    {
                        "source_name": source_name,
                        "sample_id": sample_id,
                        "output_name": output_name,
                        "split": recorded_split,
                    }
                )
                reporter.report_item(
                    PreprocessItemReport(
                        index=index + 1,
                        total=total_items,
                        source_name=source_name,
                        output_name=output_name,
                        split=recorded_split,
                    )
                )

                stats = pipeline.update_stats(stats=stats, item=item, outputs=outputs)
                n_files += 1

            result = pipeline.make_result(n_files=n_files, stats=stats)
            registry_split_summary = summarize_processed_splits(
                processed_items=processed_items,
                split_plan=split_plan,
                include_names=False,
            )
            manifest_split_summary = summarize_processed_splits(
                processed_items=processed_items,
                split_plan=split_plan,
                include_names=True,
            )

            self.registry.update_after_preprocess(
                dataset_name=self.dataset_name,
                data_type=self.data_type,
                data_format=self.fmt,
                structure=spec.structure_keys(),
                outputs=spec.output_entries(),
                result=result,
                usage=self.usage,
                default_overrides=self.registry_defaults,
                split_summary=registry_split_summary,
            )
            self.registry.save()

            if run_log is not None:
                run_log.finalize_success(
                    result=result,
                    structure=spec.structure_keys(),
                    split_summary=manifest_split_summary,
                )

            reporter.report_finish(
                self._finish_report(
                    status="success",
                    preprocess_dir=preprocess_dir,
                    n_files_written=result.n_files,
                    split_summary=manifest_split_summary,
                )
            )
            return result
        except Exception as exc:
            manifest_split_summary = summarize_processed_splits(
                processed_items=processed_items,
                split_plan=split_plan,
                include_names=True,
            )
            if run_log is not None:
                run_log.finalize_failure(
                    error=exc,
                    structure=spec.structure_keys(),
                    n_files=n_files,
                    stats=stats,
                    split_summary=manifest_split_summary,
                )
            reporter.report_finish(
                self._finish_report(
                    status="failed",
                    preprocess_dir=preprocess_dir,
                    n_files_written=n_files,
                    split_summary=manifest_split_summary,
                    error=exc,
                )
            )
            raise
