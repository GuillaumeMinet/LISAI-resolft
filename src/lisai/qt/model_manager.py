"""Embeddable promoted-model catalog and installer, with no ImProcess dependency.

Catalog/download/install operations use the existing LISAI backend. QThread
owns only GUI execution and signals; the scientific and registry operations
stay synchronous and Qt-free. Jobs are retained until finished, even if their
owning page is hidden or closed.
"""
from __future__ import annotations

from threading import Event
from weakref import ref

from qtpy import QtCore, QtWidgets

from lisai.application.catalogs import get_installed_model_card, get_installed_model_names
from lisai.config.settings import Settings

# QThread instances must not be destroyed while running. This registry also
# prevents two GUI pages from downloading into the same workspace at once.
_ACTIVE_JOBS: set["ModelCatalogJob"] = set()


class ModelCatalogJob(QtCore.QThread):
    """One cancellable catalog/download request bound to explicit Settings."""

    stage = QtCore.Signal(str)
    progress = QtCore.Signal(int, object)
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, *, settings: Settings, action: str, model_name: str | None = None):
        super().__init__()  # intentionally parentless; may outlive its view
        if action not in {"catalog", "install"}:
            raise ValueError(f"Unsupported model job action {action!r}")
        self.settings = settings
        self.action = action
        self.model_name = model_name
        self.workspace_key = str(settings.WORKSPACE_ROOT)
        self._cancel = Event()

    def request_cancel(self):
        self._cancel.set()

    def run(self):
        try:
            from lisai.promoted_models.download import DownloadCancelled

            if self.action == "catalog":
                from lisai.promoted_models.catalog import list_models

                self.stage.emit("Loading online model catalog…")
                models = list_models()
                if self._cancel.is_set():
                    raise DownloadCancelled("Catalog request cancelled.")
                self.succeeded.emit(models)
            else:
                from lisai.infra.paths import Paths
                from lisai.promoted_models.download import download_model
                from lisai.promoted_models.install import install_model_archive

                if not self.model_name:
                    raise ValueError("Choose a model to install first.")
                self.stage.emit("Downloading and verifying…")
                downloaded = download_model(
                    self.model_name, paths=Paths(self.settings), overwrite=True,
                    progress_callback=lambda got, total: self.progress.emit(got, total),
                    cancel_callback=self._cancel.is_set,
                )
                if self._cancel.is_set():
                    raise DownloadCancelled("Download cancelled before installation.")
                # Installation is transactional but not interruptible. Cancellation
                # is disabled by the manager as soon as this stage begins.
                self.stage.emit("Installing (do not close the application)…")
                install_model_archive(downloaded.archive_path, paths=Paths(self.settings))
                self.succeeded.emit(self.model_name)
        except Exception as exc:  # deliver worker failures to Qt, never crash the thread
            from lisai.promoted_models.download import DownloadCancelled

            if isinstance(exc, DownloadCancelled):
                self.cancelled.emit()
            else:
                self.failed.emit(f"{type(exc).__name__}: {exc}")


def start_model_job(*, settings: Settings, action: str, model_name: str | None = None):
    """Return an unstarted job; callers connect signals before calling start()."""
    key = str(settings.WORKSPACE_ROOT)
    if action == "install" and any(
        job.action == "install" and job.workspace_key == key for job in _ACTIVE_JOBS
    ):
        raise RuntimeError("Another model installation is already active in this workspace.")
    job = ModelCatalogJob(settings=settings, action=action, model_name=model_name)
    _ACTIVE_JOBS.add(job)

    job_ref = ref(job)

    def release():
        finished_job = job_ref()
        if finished_job is not None:
            _ACTIVE_JOBS.discard(finished_job)

    job.finished.connect(release)
    return job


class ModelManagerWidget(QtWidgets.QWidget):
    """Full-width, embeddable model manager with local-first/offline operation.

    Do not require a network request to display installed models. A hosting
    application decides whether to expose this panel via internal navigation.
    """

    modelsChanged = QtCore.Signal()
    modelInstalled = QtCore.Signal(str)
    backRequested = QtCore.Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._catalog: dict[str, object] = {}
        self._installed: set[str] = set()
        self._job: ModelCatalogJob | None = None

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        self.back_button = QtWidgets.QPushButton("← Back", self)
        self.back_button.clicked.connect(lambda *_: self.backRequested.emit())
        header.addWidget(self.back_button)
        title = QtWidgets.QLabel("LISAI models", self)
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        header.addWidget(title)
        header.addStretch(1)
        self.workspace_label = QtWidgets.QLabel(str(settings.WORKSPACE_ROOT), self)
        self.workspace_label.setToolTip(str(settings.WORKSPACE_ROOT))
        header.addWidget(self.workspace_label)
        layout.addLayout(header)

        content = QtWidgets.QHBoxLayout()
        self.model_list = QtWidgets.QListWidget(self)
        content.addWidget(self.model_list, 2)
        self.details = QtWidgets.QTextBrowser(self)
        self.details.setOpenExternalLinks(False)
        content.addWidget(self.details, 3)
        layout.addLayout(content, 1)

        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.status = QtWidgets.QLabel(
            "Installed models are available even when the online catalog is offline.", self
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        actions = QtWidgets.QHBoxLayout()
        self.local_button = QtWidgets.QPushButton("Refresh installed", self)
        self.catalog_button = QtWidgets.QPushButton("Refresh online catalog", self)
        self.install_button = QtWidgets.QPushButton("Download & install", self)
        self.cancel_button = QtWidgets.QPushButton("Cancel download", self)
        actions.addWidget(self.local_button)
        actions.addWidget(self.catalog_button)
        actions.addWidget(self.install_button)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)

        self.model_list.currentItemChanged.connect(self._show_details)
        self.local_button.clicked.connect(self.refresh_local)
        self.catalog_button.clicked.connect(self.refresh_online)
        self.install_button.clicked.connect(self.install_selected)
        self.cancel_button.clicked.connect(self.cancel_action)
        self.refresh_local()

    def selected_model_name(self) -> str:
        current = self.model_list.currentItem()
        return str(current.data(QtCore.Qt.UserRole)) if current is not None else ""

    def is_busy(self) -> bool:
        """Whether a catalog or installation job is still owned by this page."""
        return self._job is not None

    def is_installed(self, name: str) -> bool:
        """Check the currently refreshed installed-model catalog."""
        return bool(name) and name in self._installed

    def set_settings(self, settings: Settings):
        """Change workspace only when no catalog/download job is using the old one."""
        if self._job is not None:
            raise RuntimeError("Finish or cancel the active model job before changing workspace.")
        if not isinstance(settings, Settings):
            raise TypeError("Explicit LISAI Settings are required.")
        self.settings = settings
        self._catalog.clear()
        self.workspace_label.setText(str(settings.WORKSPACE_ROOT))
        self.workspace_label.setToolTip(str(settings.WORKSPACE_ROOT))
        self.refresh_local()

    def refresh_local(self, *_):
        """Refresh only disk state; never touch the network."""
        try:
            self._installed = set(get_installed_model_names(self.settings))
        except Exception as exc:
            self._installed.clear()
            self.status.setText(f"Cannot read installed models: {exc}")
        self._refresh_items()

    def _refresh_items(self):
        selected = self.selected_model_name()
        self.model_list.clear()
        for name in sorted(self._installed | self._catalog.keys()):
            online = self._catalog.get(name)
            task = f" · {online.task}" if online is not None else ""
            state = "Installed" if name in self._installed else "Available"
            item = QtWidgets.QListWidgetItem(f"{name}{task} — {state}", self.model_list)
            item.setData(QtCore.Qt.UserRole, name)
        if selected:
            for i in range(self.model_list.count()):
                if self.model_list.item(i).data(QtCore.Qt.UserRole) == selected:
                    self.model_list.setCurrentRow(i)
                    break
        if self.model_list.currentItem() is None and self.model_list.count():
            self.model_list.setCurrentRow(0)
        elif not self.model_list.count():
            self.details.setPlainText("No local models. Refresh the online catalog to discover models.")
        self._show_details()

    def _show_details(self, *_):
        name = self.selected_model_name()
        info = self._catalog.get(name)
        if not name:
            self.details.setPlainText("Select a model to inspect its details.")
            self._sync_actions()
            return
        content = f"# {name}\n\n"
        if info is not None:
            content += f"**Task:** {info.task}\n\n{info.description}\n\n"
        if name in self._installed:
            content += "**Status:** Installed\n\n"
            try:
                card = get_installed_model_card(self.settings, name)
            except Exception as exc:
                content += f"Could not read local model card: {exc}"
            else:
                content += card if card else "No local model card is included."
        else:
            content += "**Status:** Available for download\n\n"
            content += "The full local model card is available after installation."
        if hasattr(self.details, "setMarkdown"):
            self.details.setMarkdown(content)
        else:
            self.details.setPlainText(content)
        self._sync_actions()

    def _sync_actions(self):
        busy = self._job is not None
        self.model_list.setEnabled(not busy)
        self.catalog_button.setEnabled(not busy)
        self.local_button.setEnabled(not busy)
        name = self.selected_model_name()
        self.install_button.setEnabled(not busy and name in self._catalog
                                       and name not in self._installed)
        self.cancel_button.setEnabled(
            self._job is not None and self._job.action == "install"
        )

    def refresh_online(self, *_):
        self._begin_job("catalog")

    def install_selected(self, *_):
        name = self.selected_model_name()
        if not name or name not in self._catalog or name in self._installed:
            self.status.setText("Select a downloadable model that is not installed.")
            return
        self._begin_job("install", model_name=name)

    def _begin_job(self, action: str, model_name: str | None = None):
        if self._job is not None:
            return
        try:
            job = start_model_job(settings=self.settings, action=action, model_name=model_name)
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self._job = job
        job.stage.connect(self._on_stage)
        job.progress.connect(self._on_progress)
        job.succeeded.connect(self._on_success)
        job.failed.connect(self._on_failure)
        job.cancelled.connect(self._on_cancelled)
        job.finished.connect(self._on_finished)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self._sync_actions()
        job.start()

    def _on_stage(self, stage):
        self.status.setText(stage)
        if stage.startswith("Installing"):
            self.cancel_button.setEnabled(False)
            self.progress_bar.setRange(0, 0)

    def _on_progress(self, received, total):
        if total is not None and total > 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(min(100, int(100 * received / total)))
            self.status.setText(f"Downloading: {received:,} / {total:,} bytes")
        else:
            self.progress_bar.setRange(0, 0)
            self.status.setText(f"Downloading: {received:,} bytes")

    def _on_success(self, result):
        if self._job is None:
            return
        if self._job.action == "catalog":
            self._catalog = {item.name: item for item in result}
            self.status.setText(f"Online catalog: {len(result)} models available.")
        else:
            self.status.setText(f"Installed {result} in {self.settings.WORKSPACE_ROOT}")
            self._installed.add(str(result))
            self.modelInstalled.emit(str(result))
            self.modelsChanged.emit()
        self._refresh_items()

    def _on_failure(self, message):
        self.status.setText(f"{message}. Installed models remain accessible.")

    def _on_cancelled(self):
        self.status.setText("Download cancelled; no partial archive was kept.")

    def _on_finished(self):
        self._job = None
        self.progress_bar.hide()
        self._sync_actions()

    def cancel_action(self, *_):
        if self._job is not None and self.cancel_button.isEnabled():
            self._job.request_cancel()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancellation requested…")

    def closeEvent(self, event):
        # A hidden internal page keeps working; an actual closed view requests
        # cooperative cancellation, while the global registry retains the QThread.
        if self._job is not None:
            self._job.request_cancel()
        super().closeEvent(event)


__all__ = ["ModelCatalogJob", "ModelManagerWidget", "start_model_job"]
