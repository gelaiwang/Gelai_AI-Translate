from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .window import MainWindow


def _bootstrap_bundled_tools() -> None:
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        meipass = Path(getattr(sys, "_MEIPASS", exe_dir)).resolve()
        candidates.extend(
            [
                exe_dir / "vendor" / "ffmpeg" / "windows-x64",
                meipass / "vendor" / "ffmpeg" / "windows-x64",
            ]
        )
    else:
        project_root = Path(__file__).resolve().parents[2]
        candidates.append(project_root / "vendor" / "ffmpeg" / "windows-x64")

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        current_path = os.environ.get("PATH", "")
        candidate_str = str(candidate)
        if candidate_str not in current_path.split(os.pathsep):
            os.environ["PATH"] = candidate_str + os.pathsep + current_path
        break


def main() -> int:
    _bootstrap_bundled_tools()
    app = QApplication(sys.argv)
    app.setApplicationName("Gelai Translate")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
