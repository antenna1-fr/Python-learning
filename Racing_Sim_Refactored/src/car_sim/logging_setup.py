import logging
import logging.config
from typing import Dict


class SafeFormatter(logging.Formatter):
    """Formatter that fills in missing attributes with defaults."""

    def format(self, record: logging.LogRecord) -> str:
        # ensure run_id exists
        if not hasattr(record, "run_id"):
            record.run_id = "-"
        return super().format(record)


class KeyValueFormatter(logging.Formatter):
    """Formats standard fields and then any extra attributes as key=value."""

    def format(self, record):
        # Base fields
        base = (
            f"time={self.formatTime(record, self.datefmt)} "
            f"level={record.levelname} "
            f"run_id={getattr(record, 'run_id', '-')} "
            f"module={record.name} "
            f'msg="{record.getMessage()}"'
        )

        # Add any extra attributes that aren't standard LogRecord fields
        standard = set(vars(logging.LogRecord("", 0, "", 0, "", (), None)))
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in standard and k not in {"asctime", "message", "run_id"}
        }
        if extras:
            extras_str = " " + " ".join(f"{k}={v!r}" for k, v in extras.items())
            return base + extras_str
        return base


class ContextFilter:
    def __init__(self, **ctx):
        super().__init__()
        self.ctx = ctx

    def filter(self, record):
        # copy provided context if not already set on the record
        for k, v in self.ctx.items():
            if not hasattr(record, k):
                setattr(record, k, v)
        return True


def configure_logging(log_path: str, console_level: str, **context):
    config = build_logging_config(log_path, console_level)
    logging.config.dictConfig(config)

    # attach context filter to root and its handlers
    f = ContextFilter(**context)
    root = logging.getLogger()
    root.addFilter(f)
    for h in root.handlers:
        h.addFilter(f)


def build_logging_config(log_path: str, console_level: str) -> Dict:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": "%(levelname)s %(name)s %(message)s",
            },
            "kv": {
                "()": "car_sim.logging_setup.KeyValueFormatter",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": console_level,
                "formatter": "console",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "kv",
                "filename": log_path,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        },
    }
