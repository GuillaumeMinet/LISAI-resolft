#!/usr/bin/env python
"""M3-D installed promoted-model smoke checks, run from an installed LISAI wheel.

Example (Windows PowerShell):
    python scripts/validate_m3d_models.py --workspace E:/lisai --download --device cuda

This script is deliberately outside the LISAI package API. It performs inference
on synthetic *input images* using weights from installed promoted-model bundles.
For the default published model names, this validates the published weights
only when they were genuinely downloaded/installed from the official catalog.
It does not assess restoration quality on microscopy data.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile
import torch

from lisai import (
    ApplyInferenceOverrides,
    ApplyInputOverrides,
    ApplyOverrides,
    InputNormalization,
    predict,
)
from lisai.config.settings import Settings
from lisai.infra.paths import Paths
from lisai.promoted_models.download import download_model
from lisai.promoted_models.install import install_model_archive
from lisai.promoted_models.package import load_promoted_model
from lisai.promoted_models.registry import load_promoted_model_registry

DEFAULT_MODELS = ("vim_denoise_hdnsup", "vim_upsamp_5frames")


def _reset_seed() -> None:
    torch.manual_seed(913)
    np.random.seed(913)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(913)


def _test_model(name: str, settings: Settings, *, device: str, full: bool) -> dict:
    promoted = load_promoted_model(name, paths=Paths(settings))
    saved_run = promoted.saved_run
    n_context = saved_run.context_length
    side = 64  # divisible by common encoder downscaling factors
    rng = np.random.default_rng(21)
    frames = max(7, int(n_context or 1) + 2)
    shape = (frames, side, side) if n_context is not None else (side, side)
    array = rng.uniform(6, 15, size=shape).astype(np.float32)
    original = array.copy()
    modes = ("drop", "black_output", "black_input") if n_context is not None and full else ("drop",)
    normalizations = ("training", "image", "manual") if full else ("training",)
    tested = []
    with tempfile.TemporaryDirectory(prefix="lisai-m3d-validation-") as tmp:
        temporary = Path(tmp)
        input_path = temporary / "input.tif"
        tifffile.imwrite(input_path, array)
        for norm in normalizations:
            for mode in modes:
                options = ApplyOverrides(
                    input=ApplyInputOverrides(normalization=InputNormalization(
                        mode=norm, gain=1.2, offset=0.5,
                    )),
                    inference=ApplyInferenceOverrides(
                        temporal_edge_mode=mode,
                        tiling_size="off",
                        lvae_num_samples=2,
                    ),
                    saving={"lvae_save_samples": False, "save_input_mode": "never"},
                )
                _reset_seed()
                prediction, info = predict(
                    name, array, settings=settings, device=device,
                    overrides=options, return_info=True,
                )
                if not np.isfinite(prediction).all():
                    raise AssertionError(f"{name}: non-finite array prediction for {norm}/{mode}")
                np.testing.assert_array_equal(array, original)
                if n_context is not None:
                    expected_t = frames - (n_context - 1) if mode == "drop" else frames
                    expected_shape = (expected_t, side * saved_run.upsampling_factor, side * saved_run.upsampling_factor)
                    if prediction.shape != expected_shape:
                        raise AssertionError(f"{name}: {norm}/{mode}, expected {expected_shape}, got {prediction.shape}")
                    if mode == "black_output":
                        half = n_context // 2
                        if np.any(prediction[:half]) or np.any(prediction[-half:]):
                            raise AssertionError(f"{name}: black output frames are nonzero")
                else:
                    expected_shape = (side * saved_run.upsampling_factor, side * saved_run.upsampling_factor)
                    if prediction.shape != expected_shape:
                        raise AssertionError(f"{name}: unexpected shape {prediction.shape}")
                _reset_seed()
                files = predict(
                    name, input_path, settings=settings, device=device, overrides=options,
                    output_dir=temporary / f"{norm}-{mode}",
                )
                if len(files.prediction_paths) != 1 or not files.prediction_paths[0].is_file():
                    raise AssertionError(f"{name}: file prediction was not written")
                if files.info_by_path[files.prediction_paths[0]] != info:
                    raise AssertionError(f"{name}: array/file alignment info differs")
                observed = tifffile.imread(files.prediction_paths[0])
                np.testing.assert_allclose(observed, prediction, rtol=1e-4, atol=1e-3)
                tested.append(f"{norm}/{mode}")
    return {
        "model": name,
        "architecture": saved_run.model_architecture,
        "context_length": n_context,
        "upsampling_factor": saved_run.upsampling_factor,
        "checks": tested,
        "status": "passed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Initialized M2 workspace root")
    parser.add_argument("--model", action="append", dest="models", help="Installed promoted-model name (repeatable)")
    parser.add_argument("--download", action="store_true", help="Download and install specified catalog models first")
    parser.add_argument("--archive", type=Path, action="append", default=[], help="Install a local .lisai.zip archive first (repeatable)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--full", action="store_true", help="Test all normalization and temporal-edge modes")
    parser.add_argument("--report", type=Path, help="Optional JSON report output path")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires a CUDA-enabled PyTorch with an available NVIDIA GPU")
    settings = Settings.from_workspace(args.workspace)
    paths = Paths(settings)
    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "requested_device": args.device,
        "workspace": str(args.workspace.resolve()),
        "weight_origin": "installed bundles (catalog authenticity not inferred)",
        "models": [],
    }
    for archive in args.archive:
        install_model_archive(archive, paths=paths)
    models = args.models or list(DEFAULT_MODELS)
    try:
        for name in models:
            if args.download:
                item = download_model(name, paths=paths)
                registry = load_promoted_model_registry(paths=paths)
                if name not in registry.models:
                    install_model_archive(item.archive_path, paths=paths)
            report["models"].append(_test_model(name, settings, device=args.device, full=args.full))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
        if not args.download:
            print("Install models first with `lisai models download NAME --install`, or use --archive PATH.", file=sys.stderr)
        status = 1
    else:
        status = 0
    serialized = json.dumps(report, indent=2, default=str)
    print(serialized)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
