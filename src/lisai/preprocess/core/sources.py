# lisai/data/preprocess/sources/folder.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol


@dataclass(frozen=True)
class Item:
    """
    Represents one logical dataset element to be processed by a pipeline.

    An Item contains:
    - key: a stable identifier used for alignment and naming.
    - paths: one or more file paths associated with this element.
    - source_name: canonical filename of the primary raw source.
    - source_relpaths: source-relative paths useful for split reuse and logging.
    """

    key: str
    paths: tuple[Path, ...]
    source_name: str | None = None
    source_relpaths: tuple[str, ...] = ()
    auxiliary_paths: dict[str, Path] = field(default_factory=dict)


class Source(Protocol):
    """
    Abstract source interface for dataset discovery.

    A Source is responsible for enumerating the raw input data and yielding
    `Item` objects. Each Item represents one logical unit to be processed
    by a pipeline.
    """

    def iter_items(self) -> Iterable[Item]:
        ...


@dataclass(frozen=True)
class FolderSource(Source):
    """
    Simple source that reads files from a folder.

    It scans `root` for files matching the given extensions and yields one
    Item per file. If `combine_subfolders` is True, all immediate subfolders
    of `root` are scanned and combined into a single stream.
    """

    root: Path
    exts: tuple[str, ...]
    combine_subfolders: bool = False

    def iter_items(self) -> Iterable[Item]:
        roots = [self.root]
        if self.combine_subfolders:
            roots = sorted((path for path in self.root.iterdir() if path.is_dir()), key=lambda path: path.name)

        files: list[Path] = []
        for root in roots:
            for path in sorted(root.iterdir()):
                if path.is_file() and path.suffix.lower() in self.exts:
                    files.append(path)

        for path in files:
            relpath = path.relative_to(self.root).as_posix()
            yield Item(
                key=path.stem,
                paths=(path,),
                source_name=path.name,
                source_relpaths=(relpath,),
            )
