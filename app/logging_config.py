"""Structured JSON logging. Every log line carries a trace_id so a single
request can be followed end-to-end in log files and cross-referenced with
LangSmith traces.
"""
import json
import logging
import sys
import time
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_id = getattr(record, "trace_id", None)
        if trace_id:
            payload["trace_id"] = trace_id
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            payload.update(extra_fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger("stock_analyzer")
    root.setLevel(level)
    root.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JsonFormatter())
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_DIR / "app.log")
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"stock_analyzer.{name}")


def log_event(logger: logging.Logger, level: str, message: str, trace_id: str = "", **fields) -> None:
    """One-line structured log call — used everywhere instead of ad-hoc print()."""
    getattr(logger, level.lower())(message, extra={"trace_id": trace_id, "extra_fields": fields})
