from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QMainWindow, QSplitter, QStackedWidget

from .pages.logs import LogsPage
from .pages.project import ProjectPage
from .pages.step1 import Step1Page
from .pages.step2 import Step2Page
from .pages.step3 import Step3Page
from .pages.step4 import Step4Page
from .worker import StepWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gelai Translate Desktop")
        self.resize(1200, 760)

        self.current_config_path: Path | None = None
        self.current_workdir: Path | None = None
        self.current_worker: StepWorker | None = None

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.nav_list = QListWidget(splitter)
        self.stack = QStackedWidget(splitter)
        self.nav_list.setMaximumWidth(220)
        self.nav_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.setCentralWidget(splitter)

        self.logs_page = LogsPage(self)
        self.project_page = ProjectPage(self)
        self.project_page.config_path_changed.connect(self.set_config_path)
        self.project_page.workdir_changed.connect(self.set_workdir)

        self._add_page("Project", self.project_page)
        self._add_page("Step1 Download", Step1Page(self))
        self._add_page("Step2 ASR", Step2Page(self))
        self._add_page("Step3 Translate", Step3Page(self))
        self._add_page("Step4 Render", Step4Page(self))
        self._add_page("Logs", self.logs_page)
        self.nav_list.setCurrentRow(0)

    def _add_page(self, label: str, widget) -> None:
        self.stack.addWidget(widget)
        self.nav_list.addItem(QListWidgetItem(label))

    def append_log(self, message: str) -> None:
        self.logs_page.append_log(message)

    def set_config_path(self, config_path: str) -> None:
        self.current_config_path = Path(config_path) if config_path else None
        self.append_log(f"Config path set to: {self.current_config_path or '-'}")

    def set_workdir(self, workdir: str) -> None:
        self.current_workdir = Path(workdir) if workdir else None
        self.append_log(f"Workdir set to: {self.current_workdir or '-'}")

    def is_task_running(self) -> bool:
        return self.current_worker is not None and self.current_worker.isRunning()

    def start_step(self, name: str, step_fn, **kwargs) -> bool:
        if self.is_task_running():
            self.append_log("Another task is already running.")
            return False
        worker = StepWorker(step_fn, **kwargs)
        worker.log_line.connect(self.append_log)
        worker.failed.connect(self._handle_worker_failure)
        worker.finished_with_code.connect(lambda code, step_name=name: self._handle_worker_finished(step_name, code))
        self.current_worker = worker
        self.append_log(f"Starting {name} ...")
        worker.start()
        return True

    def _handle_worker_failure(self, traceback_text: str) -> None:
        self.append_log(traceback_text)

    def _handle_worker_finished(self, step_name: str, exit_code: int) -> None:
        self.append_log(f"{step_name} finished with exit code {exit_code}.")
        self.current_worker = None
