from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..configview import build_config_summary, load_config_snapshot
from ..envcheck import format_environment_checks, run_environment_checks


class ProjectPage(QWidget):
    config_path_changed = Signal(str)
    workdir_changed = Signal(str)

    def __init__(self, main_window) -> None:
        super().__init__(main_window)
        self.main_window = main_window
        self.current_snapshot: dict[str, object] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_selector_group())
        layout.addWidget(self._build_summary_group())
        layout.addWidget(self._build_environment_group())
        layout.addStretch(1)
        self.reload_config()

    def _build_selector_group(self) -> QGroupBox:
        group = QGroupBox("Project Context", self)
        layout = QGridLayout(group)

        self.config_input = QLineEdit(group)
        self.workdir_input = QLineEdit(group)

        config_button = QPushButton("Browse...", group)
        config_button.clicked.connect(self._choose_config)
        workdir_button = QPushButton("Browse...", group)
        workdir_button.clicked.connect(self._choose_workdir)

        reload_button = QPushButton("Reload Config", group)
        reload_button.clicked.connect(self.reload_config)
        env_button = QPushButton("Check Environment", group)
        env_button.clicked.connect(self.check_environment)

        layout.addWidget(QLabel("Config File", group), 0, 0)
        layout.addWidget(self.config_input, 0, 1)
        layout.addWidget(config_button, 0, 2)
        layout.addWidget(QLabel("Workdir", group), 1, 0)
        layout.addWidget(self.workdir_input, 1, 1)
        layout.addWidget(workdir_button, 1, 2)

        actions = QHBoxLayout()
        actions.addWidget(reload_button)
        actions.addWidget(env_button)
        actions.addStretch(1)
        layout.addLayout(actions, 2, 0, 1, 3)
        return group

    def _build_summary_group(self) -> QGroupBox:
        group = QGroupBox("Config Summary", self)
        layout = QVBoxLayout(group)
        self.summary_text = QTextEdit(group)
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)
        return group

    def _build_environment_group(self) -> QGroupBox:
        group = QGroupBox("Environment Check", self)
        layout = QVBoxLayout(group)
        self.environment_text = QTextEdit(group)
        self.environment_text.setReadOnly(True)
        layout.addWidget(self.environment_text)
        return group

    def _choose_config(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select config.yaml",
            str(Path.cwd()),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if selected:
            self.config_input.setText(selected)
            self.reload_config()

    def _choose_workdir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select workdir",
            self.workdir_input.text() or str(Path.cwd()),
        )
        if selected:
            self.workdir_input.setText(selected)
            self.workdir_changed.emit(selected)

    def reload_config(self) -> None:
        config_path = self.config_input.text().strip() or None
        self.current_snapshot = load_config_snapshot(config_path)
        self.summary_text.setPlainText(build_config_summary(self.current_snapshot))

        resolved_config = str(self.current_snapshot.get("config_path", "") or "")
        resolved_workdir = str(self.current_snapshot.get("workdir", "") or "")
        if resolved_config and not self.config_input.text().strip():
            self.config_input.setText(resolved_config)
        if resolved_workdir and not self.workdir_input.text().strip():
            self.workdir_input.setText(resolved_workdir)

        self.config_path_changed.emit(self.config_input.text().strip() or resolved_config)
        self.workdir_changed.emit(self.workdir_input.text().strip() or resolved_workdir)
        self.main_window.append_log("Reloaded config snapshot.")

    def check_environment(self) -> None:
        config_path = self.config_input.text().strip() or None
        results = run_environment_checks(config_path)
        self.environment_text.setPlainText(format_environment_checks(results))
        self.main_window.append_log("Environment check completed.")
