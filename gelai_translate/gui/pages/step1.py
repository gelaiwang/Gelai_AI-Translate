from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class Step1Page(QWidget):
    def __init__(self, main_window) -> None:
        super().__init__(main_window)
        self.main_window = main_window
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.source_input = QLineEdit(self)
        self.source_input.setPlaceholderText("https://www.youtube.com/playlist?list=...")
        self.format_input = QLineEdit(self)
        self.format_input.setPlaceholderText("optional yt-dlp format override")
        form.addRow("Source URL", self.source_input)
        form.addRow("Format Override", self.format_input)
        layout.addLayout(form)

        actions = QHBoxLayout()
        self.run_button = QPushButton("Run Step1", self)
        self.run_button.clicked.connect(self._run_step1)
        actions.addWidget(self.run_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        layout.addWidget(QLabel("Step1 Notes", self))
        self.notes = QTextEdit(self)
        self.notes.setReadOnly(True)
        self.notes.setPlainText(
            "Runs pipeline.step1_download.run(config_path=..., workdir=..., source=...).\n"
            "Logs are streamed to the Logs page.\n"
            "If YouTube limits access, advanced settings such as cookies/bgutil still come from config.yaml."
        )
        layout.addWidget(self.notes)
        layout.addStretch(1)

    def _run_step1(self) -> None:
        source = self.source_input.text().strip()
        if not source:
            QMessageBox.warning(self, "Missing Source", "Source URL is required.")
            return

        config_path = self.main_window.current_config_path
        workdir = self.main_window.current_workdir
        if workdir is None:
            QMessageBox.warning(self, "Missing Workdir", "Set workdir in the Project page first.")
            return

        from pipeline.step1_download import run as step1_run

        started = self.main_window.start_step(
            "step1",
            step1_run,
            config_path=config_path,
            workdir=Path(workdir),
            source=source,
            fmt=self.format_input.text().strip() or None,
        )
        if not started:
            QMessageBox.information(self, "Busy", "Another task is already running.")
