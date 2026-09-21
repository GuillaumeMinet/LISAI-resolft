from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile


@dataclass(frozen=True)
class TimelapseSpec:
    name: str
    total_frames: int
    expected_frames: int


def _natural_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _iter_tiff_files(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
            yield path


def _expected_frames_from_cl(total_frames: int, cl: int) -> int:
    # For CL=5, this removes 4 frames total (2 at each side): 20 -> 16.
    frames_to_remove = 2 * (cl // 2)
    expected = total_frames - frames_to_remove
    if expected <= 0:
        raise ValueError(
            f"Invalid expected frames: total={total_frames}, cl={cl}, "
            f"frames_to_remove={frames_to_remove}"
        )
    return expected


def _load_timelapse_specs(test_dataset: Path, cl: int) -> list[TimelapseSpec]:
    if not test_dataset.exists():
        raise FileNotFoundError(f"Test dataset folder does not exist: {test_dataset}")

    pattern = re.compile(r"^(?P<name>.+)_t(?P<frames>\d+)$", flags=re.IGNORECASE)
    seen_stems: set[str] = set()
    specs: list[TimelapseSpec] = []

    for file_path in sorted(_iter_tiff_files(test_dataset), key=lambda p: _natural_key(p.name)):
        stem = file_path.stem
        if stem in seen_stems:
            continue
        seen_stems.add(stem)

        match = pattern.match(stem)
        if match is None:
            continue

        name = match.group("name")
        total_frames = int(match.group("frames"))
        expected_frames = _expected_frames_from_cl(total_frames, cl)
        specs.append(
            TimelapseSpec(
                name=f"{name}_t{total_frames}",
                total_frames=total_frames,
                expected_frames=expected_frames,
            )
        )

    if not specs:
        raise ValueError(
            f"No matching tif/tiff files found in {test_dataset} with pattern '<name>_t<frames>.tif(f)'."
        )

    return specs


def _collect_eval_files(source_folder: Path) -> dict[str, list[Path]]:
    if not source_folder.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_folder}")

    pattern = re.compile(r"^img_(?P<idx>\d+)_(?P<kind>pred|gt|inp)\.tiff?$", flags=re.IGNORECASE)
    by_kind: dict[str, dict[int, Path]] = {"pred": {}, "gt": {}, "inp": {}}

    for file_path in source_folder.iterdir():
        if not file_path.is_file():
            continue
        match = pattern.match(file_path.name)
        if match is None:
            continue

        idx = int(match.group("idx"))
        kind = match.group("kind").lower()
        if idx in by_kind[kind]:
            raise ValueError(f"Duplicate index for {kind}: img_{idx}_*.tif")
        by_kind[kind][idx] = file_path

    files: dict[str, list[Path]] = {}
    for kind in ("pred", "gt", "inp"):
        kind_files = by_kind[kind]
        if not kind_files:
            raise ValueError(f"No files found for '{kind}' in {source_folder}")
        ordered_indices = sorted(kind_files)
        files[kind] = [kind_files[i] for i in ordered_indices]

    return files


def _to_2d_frame(array: np.ndarray, *, source: Path) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    raise ValueError(f"Expected 2D frame (or 1xYX) in {source}, got shape {array.shape}")


def _to_context_stack(array: np.ndarray, *, source: Path, cl: int) -> np.ndarray:
    """
    Parse one input sample when CL > 1.
    Expected shape is [CL, Y, X].
    """
    if array.ndim != 3:
        raise ValueError(f"Expected 3D input stack [CL,Y,X] in {source}, got shape {array.shape}")
    if array.shape[0] != cl:
        raise ValueError(
            f"Unexpected CL dimension in {source}: found {array.shape[0]}, expected {cl}"
        )
    return array


def _save_stack(path: Path, stack: np.ndarray) -> None:
    data = stack.astype(np.float32) if stack.dtype == np.float64 else stack
    if data.ndim == 3:
        # [T, Y, X]
        tifffile.imwrite(path, data, imagej=True, metadata={"axes": "TYX"})
        return
    if data.ndim == 4:
        # [T, CL, Y, X]
        tifffile.imwrite(path, data, imagej=True, metadata={"axes": "TZYX"})
        return
    raise ValueError(f"Expected 3D or 4D stack for saving, got shape {data.shape}")


def gather_evaluate_timelapse(
    *,
    test_dataset: Path | str,
    source_folder: Path | str,
    target_folder: str,
    cl: int,
) -> Path:
    test_dataset = Path(test_dataset)
    source_folder = Path(source_folder)
    target_path = source_folder.parent / target_folder
    target_path.mkdir(parents=True, exist_ok=True)

    specs = _load_timelapse_specs(test_dataset=test_dataset, cl=cl)
    eval_files = _collect_eval_files(source_folder=source_folder)

    expected_total_frames = sum(spec.expected_frames for spec in specs)
    for kind in ("pred", "gt", "inp"):
        found = len(eval_files[kind])
        if found != expected_total_frames:
            raise ValueError(
                f"Unexpected number of '{kind}' files: found={found}, expected={expected_total_frames} "
                f"(sum of timelapse expected frames from test dataset)."
            )

    offset = 0
    for stack_idx, spec in enumerate(specs):
        count = spec.expected_frames
        for kind in ("pred", "gt", "inp"):
            selected = eval_files[kind][offset : offset + count]
            if kind == "inp" and cl > 1:
                # Each input sample is already [CL,Y,X], gather as [T,CL,Y,X].
                contexts = [_to_context_stack(tifffile.imread(p), source=p, cl=cl) for p in selected]
                stack = np.stack(contexts, axis=0)
            else:
                frames = [_to_2d_frame(tifffile.imread(p), source=p) for p in selected]
                stack = np.stack(frames, axis=0)
            # Keep source naming convention in gathered output.
            out_path = target_path / f"img_{stack_idx}_{kind}.tif"
            _save_stack(out_path, stack)

        print(
            f"[{stack_idx:03d}] {spec.name}: total={spec.total_frames}, "
            f"expected={spec.expected_frames}, saved pred/gt/inp stacks."
        )
        offset += count

    print(f"Done. Saved gathered stacks to: {target_path}")
    return target_path


if __name__ == "__main__":
    source_folder = Path(
    r"E:\lisai\datasets\vim_live\models\Upsamp_selected\Fulldataset_CL1_1FramesMax_Upsamp2_smallerNet_clip_modifUpsamp_03\evaluation_last_epoch_120"
    )
    target_folder = "eval_gathered"
    test_dataset = Path(r"E:\lisai\datasets\vim_live\preprocess\recon\test")
    CL = 1

    gather_evaluate_timelapse(
        test_dataset=test_dataset,
        source_folder=source_folder,
        target_folder=target_folder,
        cl=CL,
    )
