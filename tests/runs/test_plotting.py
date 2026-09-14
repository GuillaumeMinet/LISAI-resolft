from __future__ import annotations

import io
from pathlib import Path

import lisai.runs.plotting as plotting


def test_read_loss_table_parses_standard_layout(tmp_path):
    loss_file = tmp_path / "loss.txt"
    loss_file.write_text(
        "Epoch Train_loss Val_loss\n"
        "0 1.2 1.5\n"
        "1 0.9 1.1\n",
        encoding="utf-8",
    )

    loaded = plotting.read_loss_table(loss_file, stderr=io.StringIO())

    assert loaded is not None
    headers, data = loaded
    assert tuple(headers) == ("Epoch", "Train_loss", "Val_loss")
    assert data.shape == (2, 3)


def test_is_hdn_layout_supports_architecture_headers_and_columns():
    assert plotting.is_hdn_layout(
        architecture="hdn",
        headers=("Epoch", "Train_loss", "Val_loss"),
        n_columns=3,
    )
    assert plotting.is_hdn_layout(
        architecture=None,
        headers=("Epoch", "Train_loss", "Val_loss", "Recons_Loss", "KL_Loss"),
        n_columns=5,
    )
    assert plotting.is_hdn_layout(
        architecture=None,
        headers=("Epoch", "Train_loss", "Val_loss"),
        n_columns=5,
    )


def test_show_loss_plot_fallback_saves_and_opens_when_backend_non_interactive(tmp_path, monkeypatch):
    run_dir = tmp_path / "dataset" / "runs" / "Upsamp" / "run_00"
    run_dir.mkdir(parents=True, exist_ok=True)
    loss_file = run_dir / "custom_loss.dat"
    loss_file.write_text(
        "Epoch Train_loss Val_loss\n"
        "0 1.0 1.2\n",
        encoding="utf-8",
    )
    save_path = run_dir / "custom_plot.png"

    class _FakePaths:
        def loss_file_path(self, *, run_dir):
            return Path(run_dir) / "custom_loss.dat"

        def loss_plot_path(self, *, run_dir):
            return Path(run_dir) / "custom_plot.png"

    class _FakeAxes:
        def plot(self, *_args, **_kwargs):
            return None

        def set_title(self, *_args, **_kwargs):
            return None

        def set_xlabel(self, *_args, **_kwargs):
            return None

        def set_ylabel(self, *_args, **_kwargs):
            return None

        def grid(self, *_args, **_kwargs):
            return None

        def legend(self, *_args, **_kwargs):
            return None

    class _FakeFigure:
        def __init__(self):
            self.saved_path = None

        def suptitle(self, *_args, **_kwargs):
            return None

        def tight_layout(self):
            return None

        def savefig(self, path, **_kwargs):
            self.saved_path = Path(path)
            self.saved_path.write_text("fake-png", encoding="utf-8")

    class _FakePyplot:
        def __init__(self):
            self.show_called = False
            self.closed = []
            self.figure = _FakeFigure()

        def subplots(self, *_args, **_kwargs):
            return self.figure, _FakeAxes()

        def show(self):
            self.show_called = True

        def close(self, fig):
            self.closed.append(fig)

    fake_plt = _FakePyplot()
    monkeypatch.setattr(
        plotting,
        "_prepare_matplotlib_for_plot",
        lambda prefer_interactive, stderr=None: (fake_plt, "Agg", False),
    )

    opened = {"value": False}
    stderr = io.StringIO()
    exit_code = plotting.show_loss_plot_for_run(
        run_dir=run_dir,
        dataset="Gag",
        model_subfolder="Upsamp",
        architecture="upsamp",
        paths=_FakePaths(),
        stderr=stderr,
        open_saved_plot=lambda _path: opened.update({"value": True}) or True,
    )

    assert exit_code == 0
    assert fake_plt.show_called is False
    assert fake_plt.figure.saved_path == save_path
    assert save_path.exists()
    assert fake_plt.closed == [fake_plt.figure]
    assert "non-interactive" in stderr.getvalue().lower()
    assert "opened fallback plot image" in stderr.getvalue().lower()
    assert opened["value"] is True


def test_show_loss_plot_uses_show_with_interactive_backend(tmp_path, monkeypatch):
    run_dir = tmp_path / "dataset" / "runs" / "Upsamp" / "run_00"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "loss.txt").write_text(
        "Epoch Train_loss Val_loss\n"
        "0 1.0 1.2\n",
        encoding="utf-8",
    )

    class _FakeAxes:
        def plot(self, *_args, **_kwargs):
            return None

        def set_title(self, *_args, **_kwargs):
            return None

        def set_xlabel(self, *_args, **_kwargs):
            return None

        def set_ylabel(self, *_args, **_kwargs):
            return None

        def grid(self, *_args, **_kwargs):
            return None

        def legend(self, *_args, **_kwargs):
            return None

    class _FakeFigure:
        def suptitle(self, *_args, **_kwargs):
            return None

        def tight_layout(self):
            return None

        def savefig(self, *_args, **_kwargs):
            return None

    class _FakePyplot:
        def __init__(self):
            self.show_called = False
            self.closed = []

        def subplots(self, *_args, **_kwargs):
            return _FakeFigure(), _FakeAxes()

        def show(self):
            self.show_called = True

        def close(self, fig):
            self.closed.append(fig)

    fake_plt = _FakePyplot()
    monkeypatch.setattr(
        plotting,
        "_prepare_matplotlib_for_plot",
        lambda prefer_interactive, stderr=None: (fake_plt, "QtAgg", True),
    )

    stderr = io.StringIO()
    exit_code = plotting.show_loss_plot_for_run(
        run_dir=run_dir,
        architecture="upsamp",
        stderr=stderr,
    )

    assert exit_code == 0
    assert fake_plt.show_called is True
    assert "non-interactive" not in stderr.getvalue().lower()


def test_save_loss_plot_for_run_uses_configured_artifact_path(tmp_path, monkeypatch):
    run_dir = tmp_path / "dataset" / "runs" / "HDN" / "run_00"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "loss_history.txt").write_text(
        "Epoch Train_loss Val_loss Recons_Loss KL_Loss\n"
        "0 1.2 1.5 1.1 0.1\n"
        "1 0.9 1.1 0.8 0.1\n",
        encoding="utf-8",
    )
    expected_plot_path = run_dir / "loss_history.png"

    class _FakePaths:
        def loss_file_path(self, *, run_dir):
            return Path(run_dir) / "loss_history.txt"

        def loss_plot_path(self, *, run_dir):
            return Path(run_dir) / "loss_history.png"

    class _FakeAxes:
        def plot(self, *_args, **_kwargs):
            return None

        def set_title(self, *_args, **_kwargs):
            return None

        def set_xlabel(self, *_args, **_kwargs):
            return None

        def set_ylabel(self, *_args, **_kwargs):
            return None

        def grid(self, *_args, **_kwargs):
            return None

        def legend(self, *_args, **_kwargs):
            return None

    class _FakeFigure:
        def __init__(self):
            self.saved_path = None

        def suptitle(self, *_args, **_kwargs):
            return None

        def tight_layout(self):
            return None

        def savefig(self, path, **_kwargs):
            self.saved_path = Path(path)
            self.saved_path.write_text("fake-png", encoding="utf-8")

    class _FakePyplot:
        def __init__(self):
            self.closed = []
            self.subplots_calls = []
            self.figure = _FakeFigure()

        def subplots(self, *args, **kwargs):
            self.subplots_calls.append((args, kwargs))
            if args[:2] == (1, 2):
                return self.figure, (_FakeAxes(), _FakeAxes())
            return self.figure, _FakeAxes()

        def close(self, fig):
            self.closed.append(fig)

    fake_plt = _FakePyplot()
    monkeypatch.setattr(
        plotting,
        "_prepare_matplotlib_for_plot",
        lambda prefer_interactive, stderr=None: (fake_plt, "Agg", False),
    )

    saved_path = plotting.save_loss_plot_for_run(
        run_dir=run_dir,
        architecture="hdn",
        paths=_FakePaths(),
        stderr=io.StringIO(),
    )

    assert saved_path == expected_plot_path
    assert expected_plot_path.exists()
    assert fake_plt.figure.saved_path == expected_plot_path
    assert fake_plt.closed == [fake_plt.figure]
    assert fake_plt.subplots_calls[0][0][:2] == (1, 2)
