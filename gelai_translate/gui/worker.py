from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import traceback
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal


class _SignalStream(io.TextIOBase):
    def __init__(self, emit: Callable[[str], None]) -> None:
        super().__init__()
        self._emit = emit

    def write(self, s: str) -> int:
        text = str(s)
        if text:
            self._emit(text.rstrip("\n"))
        return len(text)

    def flush(self) -> None:
        return None


class _SignalConsole:
    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit

    def print(self, *args: Any, **kwargs: Any) -> None:
        sep = kwargs.get("sep", " ")
        message = sep.join(str(arg) for arg in args)
        self._emit(message)


class StepWorker(QThread):
    log_line = Signal(str)
    finished_with_code = Signal(int)
    failed = Signal(str)

    def __init__(self, step_fn: Callable[..., int], **kwargs: Any) -> None:
        super().__init__()
        self.step_fn = step_fn
        self.kwargs = kwargs

    def _emit_log(self, message: str) -> None:
        if message.strip():
            self.log_line.emit(message)

    def _patch_console(self) -> tuple[object | None, object | None]:
        module = inspect.getmodule(self.step_fn)
        if module is None or not hasattr(module, "console"):
            return None, None
        original = getattr(module, "console")
        proxy = _SignalConsole(self._emit_log)
        setattr(module, "console", proxy)
        return module, original

    def run(self) -> None:
        stdout_proxy = _SignalStream(self._emit_log)
        stderr_proxy = _SignalStream(self._emit_log)
        patched_module, original_console = self._patch_console()
        try:
            with contextlib.redirect_stdout(stdout_proxy), contextlib.redirect_stderr(stderr_proxy):
                exit_code = int(self.step_fn(**self.kwargs))
            self.finished_with_code.emit(exit_code)
        except Exception:
            self.failed.emit(traceback.format_exc())
            self.finished_with_code.emit(1)
        finally:
            if patched_module is not None:
                setattr(patched_module, "console", original_console)

