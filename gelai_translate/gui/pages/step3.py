from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..configview import load_config_snapshot


class Step3Page(QWidget):
    def __init__(self, main_window) -> None:
        super().__init__(main_window)
        self.main_window = main_window
        layout = QVBoxLayout(self)
        actions = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Summary", self)
        self.refresh_button.clicked.connect(self.refresh_summary)
        self.run_button = QPushButton("Run Step3", self)
        self.run_button.clicked.connect(self._run_step3)
        actions.addWidget(self.refresh_button)
        actions.addWidget(self.run_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        layout.addWidget(QLabel("Translation Summary", self))
        self.summary_text = QTextEdit(self)
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)
        layout.addStretch(1)
        self.refresh_summary()

    def refresh_summary(self) -> None:
        snapshot = load_config_snapshot(self.main_window.current_config_path)
        self.summary_text.setPlainText(
            "\n".join(
                [
                    f"Translation provider: {snapshot.get('translation_service', '-')}",
                    f"Gemini model: {snapshot.get('gemini_model', '-')}",
                    f"Workdir: {snapshot.get('workdir', '-')}",
                ]
            )
        )

    def _run_step3(self) -> None:
        workdir = self.main_window.current_workdir
        if workdir is None:
            QMessageBox.warning(self, "Missing Workdir", "Set workdir in the Project page first.")
            return

        from pipeline.step3_translate import run as step3_run

        started = self.main_window.start_step(
            "step3",
            step3_run,
            config_path=self.main_window.current_config_path,
            workdir=Path(workdir),
        )
        if not started:
            QMessageBox.information(self, "Busy", "Another task is already running.")
