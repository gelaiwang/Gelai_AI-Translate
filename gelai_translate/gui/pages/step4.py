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


class Step4Page(QWidget):
    def __init__(self, main_window) -> None:
        super().__init__(main_window)
        self.main_window = main_window
        layout = QVBoxLayout(self)
        actions = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Summary", self)
        self.refresh_button.clicked.connect(self.refresh_summary)
        self.run_button = QPushButton("Run Step4", self)
        self.run_button.clicked.connect(self._run_step4)
        actions.addWidget(self.refresh_button)
        actions.addWidget(self.run_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        layout.addWidget(QLabel("Render Summary", self))
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
                    f"Render codec: {snapshot.get('render_codec', '-')}",
                    f"Output suffix: {snapshot.get('render_output_suffix', '-')}",
                    f"Font EN: {snapshot.get('render_font_en', '-')}",
                    f"Font CN: {snapshot.get('render_font_cn', '-')}",
                ]
            )
        )

    def _run_step4(self) -> None:
        workdir = self.main_window.current_workdir
        if workdir is None:
            QMessageBox.warning(self, "Missing Workdir", "Set workdir in the Project page first.")
            return

        from pipeline.step4_render import run as step4_run

        started = self.main_window.start_step(
            "step4",
            step4_run,
            config_path=self.main_window.current_config_path,
            workdir=Path(workdir),
        )
        if not started:
            QMessageBox.information(self, "Busy", "Another task is already running.")
