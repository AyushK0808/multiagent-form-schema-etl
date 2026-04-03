from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, List


_LOG_BUFFER: Deque[str] = deque(maxlen=500)
_HANDLER_NAME = "streamlit_ui_buffer_handler"


class UILogBufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        _LOG_BUFFER.append(message)


def setup_ui_log_capture() -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if getattr(handler, "name", "") == _HANDLER_NAME:
            return

    handler = UILogBufferHandler()
    handler.name = _HANDLER_NAME
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s - %(message)s")
    )
    root_logger.addHandler(handler)

    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    logging.getLogger(__name__).info("[UI] Streamlit log capture attached to root logger")


def get_terminal_logs(limit: int = 200) -> List[str]:
    if limit <= 0:
        return []
    return list(_LOG_BUFFER)[-limit:]


class StreamlitLiveLogHandler(logging.Handler):
    def __init__(self, max_lines: int = 200) -> None:
        super().__init__(level=logging.INFO)
        self.lines: Deque[str] = deque(maxlen=max_lines)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        with self._lock:
            self.lines.append(message)

    def snapshot(self) -> List[str]:
        with self._lock:
            return list(self.lines)


def run_with_live_logs(
    fn: Callable[..., Any],
    *args: Any,
    log_placeholder,
    status_placeholder,
    max_lines: int = 200,
    poll_interval: float = 0.1,
    **kwargs: Any,
) -> Any:
    root_logger = logging.getLogger()
    handler = StreamlitLiveLogHandler(max_lines=max_lines)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s - %(message)s")
    )
    root_logger.addHandler(handler)
    result: dict[str, Any] = {"value": None, "error": None}

    def _target() -> None:
        try:
            result["value"] = fn(*args, **kwargs)
        except Exception as exc:
            result["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    try:
        status_placeholder.caption("Live execution trace: waiting for logs...")
        while worker.is_alive():
            lines = handler.snapshot()
            status_placeholder.caption(f"Live execution trace: {len(lines)} log lines captured")
            if lines:
                log_placeholder.code("\n".join(lines), language="text")
            time.sleep(poll_interval)

        worker.join()
        lines = handler.snapshot()
        status_placeholder.caption(f"Live execution trace: {len(lines)} log lines captured")
        if lines:
            log_placeholder.code("\n".join(lines), language="text")
    finally:
        root_logger.removeHandler(handler)

    if result["error"] is not None:
        raise result["error"]
    return result["value"]
