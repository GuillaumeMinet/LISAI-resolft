from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, TextIO


@dataclass(frozen=True)
class PreprocessStartReport:
    source: str
    target: str
    combine_subfolders: bool | None = None


@dataclass(frozen=True)
class PreprocessItemReport:
    index: int
    total: int | None
    source_name: str
    output_name: str
    split: str | None


@dataclass(frozen=True)
class PreprocessFinishReport:
    status: str
    preprocess_dir: str
    n_files_written: int
    n_files_moved: int
    split_enabled: bool
    val: Mapping[str, Any]
    test: Mapping[str, Any]
    auxiliary_matches: Mapping[str, Mapping[str, int]]
    readme_path: str | None = None
    readme_created: bool = False
    readme_error: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class PreprocessReporter(Protocol):
    def report_start(self, report: PreprocessStartReport) -> None:
        ...

    def report_item(self, report: PreprocessItemReport) -> None:
        ...

    def report_finish(self, report: PreprocessFinishReport) -> None:
        ...


class NoOpPreprocessReporter:
    def report_start(self, report: PreprocessStartReport) -> None:
        return None

    def report_item(self, report: PreprocessItemReport) -> None:
        return None

    def report_finish(self, report: PreprocessFinishReport) -> None:
        return None


class ConsolePreprocessReporter:
    def __init__(self, *, stream: TextIO | None = None):
        self.stream = sys.stdout if stream is None else stream

    def report_start(self, report: PreprocessStartReport) -> None:
        print(f"Preprocess source: {report.source}", file=self.stream)
        if report.combine_subfolders is not None:
            combine = "yes" if report.combine_subfolders else "no"
            print(f"Combine subfolders: {combine}", file=self.stream)
        print(f"Preprocess target: {report.target}", file=self.stream)

    def report_item(self, report: PreprocessItemReport) -> None:
        if report.total is None:
            progress = f"[{report.index}]"
        else:
            progress = f"[{report.index}/{report.total}]"
        suffix = f" (split={report.split})" if report.split is not None else ""
        print(
            f"{progress} {report.source_name} -> {report.output_name}{suffix}",
            file=self.stream,
        )

    def report_finish(self, report: PreprocessFinishReport) -> None:
        print("Preprocess summary:", file=self.stream)
        print(f"  status: {report.status}", file=self.stream)
        print(f"  files written: {report.n_files_written}", file=self.stream)
        if report.split_enabled:
            print(f"  files moved to validation/test: {report.n_files_moved}", file=self.stream)
        print(f"  preprocess location: {report.preprocess_dir}", file=self.stream)
        if report.split_enabled:
            print(
                f"  validation images ({report.val.get('count', 0)}): {self._format_names(report.val)}",
                file=self.stream,
            )
            print(
                f"  test images ({report.test.get('count', 0)}): {self._format_names(report.test)}",
                file=self.stream,
            )
        if report.status == "success":
            if report.readme_created and report.readme_path is not None:
                print(
                    f"  dataset README: {report.readme_path} (created; add notes when useful)",
                    file=self.stream,
                )
            elif report.readme_error is not None:
                print(
                    f"  dataset README: could not create ({report.readme_error})",
                    file=self.stream,
                )
            for name, summary in report.auxiliary_matches.items():
                print(
                    f"  Auxiliary '{name}': matched {summary.get('matched', 0)}/{summary.get('total', 0)}",
                    file=self.stream,
                )
        if report.error_type is not None or report.error_message is not None:
            print(
                f"  error: {report.error_type or 'Error'}: {report.error_message or ''}".rstrip(),
                file=self.stream,
            )

    @staticmethod
    def _format_names(bucket: Mapping[str, Any]) -> str:
        source_names = list(bucket.get("source_names", []))
        output_names = list(bucket.get("output_names", bucket.get("sample_ids", [])))
        if not source_names and not output_names:
            return "none"

        pairs: list[str] = []
        length = max(len(source_names), len(output_names))
        for index in range(length):
            source_name = source_names[index] if index < len(source_names) else ""
            output_name = output_names[index] if index < len(output_names) else ""
            if source_name and output_name:
                pairs.append(f"{source_name} -> {output_name}")
            elif source_name:
                pairs.append(source_name)
            elif output_name:
                pairs.append(output_name)
        return ", ".join(pairs) if pairs else "none"

