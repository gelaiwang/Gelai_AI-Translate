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


class Step2Page(QWidget):
    def __init__(self, main_window) -> None:
        super().__init__(main_window)
        self.main_window = main_window
        layout = QVBoxLayout(self)
        actions = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Summary", self)
        self.refresh_button.clicked.connect(self.refresh_summary)
        self.run_button = QPushButton("Run Step2", self)
        self.run_button.clicked.connect(self._run_step2)
        actions.addWidget(self.refresh_button)
        actions.addWidget(self.run_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        layout.addWidget(QLabel("ASR Summary", self))
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
                    f"ASR model: {snapshot.get('asr_model', '-')}",
                    f"Device: {snapshot.get('asr_device', '-')}",
                    f"Compute type: {snapshot.get('asr_compute_type', '-')}",
                    f"Diarization: {snapshot.get('speaker_diarization', False)}",
                    f"Vocal separation: {snapshot.get('use_vocal_separation', False)}",
                ]
            )
        )

    def _run_step2(self) -> None:
        workdir = self.main_window.current_workdir
        if workdir is None:
            QMessageBox.warning(self, "Missing Workdir", "Set workdir in the Project page first.")
            return

        from pipeline.step2_ingest import run as step2_run

        started = self.main_window.start_step(
            "step2",
            step2_run,
            config_path=self.main_window.current_config_path,
            workdir=Path(workdir),
        )
        if not started:
            QMessageBox.information(self, "Busy", "Another task is already running.")
