from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from lisai.infra.paths import Paths
from lisai.config.settings import settings as _legacy_settings  # M1: replace with injected paths


def read_loss_table(loss_file: Path, *, stderr: TextIO | None = None):
    if not loss_file.exists():
        _emit(stderr, f"Loss file not found: {loss_file}")
        return None

    try:
        import numpy as np
    except Exception as exc:
        _emit(stderr, f"Failed to import numpy for loss plotting: {exc}")
        return None

    try:
        with loss_file.open("r", encoding="utf-8") as handle:
            header_line = handle.readline().strip()
        headers = [token for token in header_line.split() if token]
        data = np.loadtxt(loss_file, skiprows=1)
    except OSError as exc:
        _emit(stderr, f"Failed to read loss file '{loss_file}': {exc}")
        return None
    except ValueError as exc:
        _emit(stderr, f"Failed to parse loss file '{loss_file}': {exc}")
        return None

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        _emit(
            stderr,
            f"Loss file '{loss_file}' must have at least 3 columns: epoch/train/val.",
        )
        return None

    return headers, data


def is_hdn_layout(
    *,
    architecture: str | None,
    headers: Sequence[str],
    n_columns: int,
) -> bool:
    normalized_architecture = (architecture or "").strip().lower()
    if "hdn" in normalized_architecture or "lvae" in normalized_architecture:
        return True

    normalized_headers = {token.strip().lower() for token in headers}
    if {"recons_loss", "kl_loss"}.issubset(normalized_headers):
        return True
    return n_columns >= 5


def show_loss_plot_for_run(
    *,
    run_dir: str | Path,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    architecture: str | None = None,
    paths: Paths | None = None,
    stderr: TextIO | None = None,
    open_saved_plot: Callable[[Path], bool] | None = None,
) -> int:
    resolved_run_dir = Path(run_dir).resolve()
    resolved_paths = paths or Paths(_legacy_settings)
    loss_file = resolved_paths.loss_file_path(run_dir=resolved_run_dir)
    loaded = read_loss_table(loss_file, stderr=stderr)
    if loaded is None:
        return 1
    headers, data = loaded
    hdn_layout = is_hdn_layout(
        architecture=architecture,
        headers=headers,
        n_columns=int(data.shape[1]),
    )

    prepared = _prepare_matplotlib_for_plot(prefer_interactive=True, stderr=stderr)
    if prepared is None:
        return 1
    plt, backend_name, interactive_backend = prepared
    fig = _build_loss_figure(
        plt=plt,
        run_label=_format_run_label(
            run_dir=resolved_run_dir,
            dataset=dataset,
            model_subfolder=model_subfolder,
        ),
        loss_file=loss_file,
        data=data,
        hdn_layout=hdn_layout,
        stderr=stderr,
    )
    if fig is None:
        return 1

    if interactive_backend:
        plt.show()
        return 0

    save_path = resolved_paths.loss_plot_path(run_dir=resolved_run_dir)
    saved = _save_figure(
        save_path=save_path,
        fig=fig,
        backend_name=backend_name,
        stderr=stderr,
    )
    plt.close(fig)
    if not saved:
        return 1
    _emit(
        stderr,
        f"warning: matplotlib backend '{backend_name}' is non-interactive; saved plot to {save_path}",
    )
    if open_saved_plot is not None and open_saved_plot(save_path):
        _emit(stderr, f"Opened fallback plot image: {save_path}")
    return 0


def save_loss_plot_for_run(
    *,
    run_dir: str | Path,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    architecture: str | None = None,
    paths: Paths | None = None,
    stderr: TextIO | None = None,
) -> Path | None:
    resolved_run_dir = Path(run_dir).resolve()
    resolved_paths = paths or Paths(_legacy_settings)
    loss_file = resolved_paths.loss_file_path(run_dir=resolved_run_dir)
    loaded = read_loss_table(loss_file, stderr=stderr)
    if loaded is None:
        return None
    headers, data = loaded
    hdn_layout = is_hdn_layout(
        architecture=architecture,
        headers=headers,
        n_columns=int(data.shape[1]),
    )

    prepared = _prepare_matplotlib_for_plot(prefer_interactive=False, stderr=stderr)
    if prepared is None:
        return None
    plt, backend_name, _interactive_backend = prepared
    fig = _build_loss_figure(
        plt=plt,
        run_label=_format_run_label(
            run_dir=resolved_run_dir,
            dataset=dataset,
            model_subfolder=model_subfolder,
        ),
        loss_file=loss_file,
        data=data,
        hdn_layout=hdn_layout,
        stderr=stderr,
    )
    if fig is None:
        return None

    save_path = resolved_paths.loss_plot_path(run_dir=resolved_run_dir)
    saved = _save_figure(
        save_path=save_path,
        fig=fig,
        backend_name=backend_name,
        stderr=stderr,
    )
    plt.close(fig)
    if not saved:
        return None
    return save_path


def _build_loss_figure(
    *,
    plt,
    run_label: str,
    loss_file: Path,
    data,
    hdn_layout: bool,
    stderr: TextIO | None = None,
):
    epochs = data[:, 0]
    train_loss = data[:, 1]
    val_loss = data[:, 2]

    if hdn_layout and data.shape[1] >= 5:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_left, ax_right = axes

        ax_left.plot(epochs, train_loss, "r", label="train")
        ax_left.plot(epochs, val_loss, "b", label="val")
        ax_left.set_title("Train/Val Recon Loss")
        ax_left.set_xlabel("epoch")
        ax_left.set_ylabel("loss")
        ax_left.grid(True, alpha=0.3)
        ax_left.legend()

        ax_right.plot(epochs, data[:, 3], "g", label="recons_loss")
        ax_right.plot(epochs, data[:, 4], "k", label="kl_loss")
        ax_right.set_title("Training Recon/KL Loss")
        ax_right.set_xlabel("epoch")
        ax_right.set_ylabel("loss")
        ax_right.grid(True, alpha=0.3)
        ax_right.legend()
    else:
        if hdn_layout:
            _emit(
                stderr,
                f"warning: HDN-like run detected, but '{loss_file}' has fewer than 5 columns. "
                "Plotting train/val only.",
            )
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        ax.plot(epochs, train_loss, "r", label="train")
        ax.plot(epochs, val_loss, "b", label="val")
        ax.set_title("Train/Val Loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(run_label)
    fig.tight_layout()
    return fig


def _save_figure(*, save_path: Path, fig, backend_name: str, stderr: TextIO | None = None) -> bool:
    try:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    except OSError as exc:
        _emit(
            stderr,
            f"Failed to save loss plot to '{save_path}' (backend='{backend_name}'): {exc}",
        )
        return False
    return True


def _prepare_matplotlib_for_plot(*, prefer_interactive: bool, stderr: TextIO | None = None):
    _ensure_writable_mplconfigdir()
    try:
        import matplotlib
    except Exception as exc:
        _emit(stderr, f"Failed to import matplotlib for loss plotting: {exc}")
        return None

    backend_name = str(matplotlib.get_backend())
    if prefer_interactive and _is_non_interactive_backend(backend_name):
        for candidate in _preferred_interactive_backends():
            try:
                matplotlib.use(candidate, force=True)
            except Exception:
                continue
            backend_name = str(matplotlib.get_backend())
            if not _is_non_interactive_backend(backend_name):
                break

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        _emit(stderr, f"Failed to import matplotlib pyplot for loss plotting: {exc}")
        return None

    backend_name = str(matplotlib.get_backend())
    return plt, backend_name, (not _is_non_interactive_backend(backend_name))


def _preferred_interactive_backends() -> tuple[str, ...]:
    if os.name == "nt":
        return ("QtAgg", "TkAgg")
    return ("QtAgg", "TkAgg", "GTK3Agg", "WXAgg")


def _is_non_interactive_backend(name: str) -> bool:
    lowered = name.strip().lower()
    if lowered in {"qtagg", "tkagg", "wxagg", "macosx"}:
        return False
    if lowered.startswith("gtk3agg") or lowered.startswith("gtk4agg"):
        return False
    return (
        "agg" in lowered
        or "inline" in lowered
        or lowered.startswith("module://matplotlib_inline")
    )


def _ensure_writable_mplconfigdir() -> None:
    if os.getenv("MPLCONFIGDIR"):
        return

    default_dir = Path.home() / ".config" / "matplotlib"
    if _is_writable_dir(default_dir):
        return

    fallback_dir = Path("/tmp") / "lisai-mplconfig"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(fallback_dir)


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        marker = path / ".write_test"
        with marker.open("w", encoding="utf-8"):
            pass
        marker.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _format_run_label(
    *,
    run_dir: Path,
    dataset: str | None,
    model_subfolder: str | None,
) -> str:
    if dataset and model_subfolder:
        return f"{dataset}/{model_subfolder}/{run_dir.name}"
    if dataset:
        return f"{dataset}/{run_dir.name}"
    return run_dir.name


def _emit(stream: TextIO | None, message: str) -> None:
    if stream is None:
        return
    print(message, file=stream)


__all__ = [
    "is_hdn_layout",
    "read_loss_table",
    "save_loss_plot_for_run",
    "show_loss_plot_for_run",
]
