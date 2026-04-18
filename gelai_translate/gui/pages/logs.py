from __future__ import annotations

from PySide6.QtWidgets import QPushButton, QVBoxLayout, QTextEdit, QWidget


class LogsPage(QWidget):
    def __init__(self, main_window) -> None:
        super().__init__(main_window)
        layout = QVBoxLayout(self)
        clear_button = QPushButton("Clear Logs", self)
        clear_button.clicked.connect(self._clear)
        layout.addWidget(clear_button)
        self.text = QTextEdit(self)
        self.text.setReadOnly(True)
        layout.addWidget(self.text)

    def append_log(self, message: str) -> None:
        self.text.append(message)

    def _clear(self) -> None:
        self.text.clear()
