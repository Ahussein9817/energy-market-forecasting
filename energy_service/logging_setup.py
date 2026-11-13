"""Structured JSON logging: one JSON object per line on stderr, with secrets redacted."""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

REDACT = ("api_key", "x-api-key", "authorization", "password", "token", "webhook")


def redact(value, secrets: tuple[str, ...] = ()):
    """Mask secret values and secret-looking keys in a (nested) structure."""
    if isinstance(value, dict):
        return {k: "***" if any(r in str(k).lower() for r in REDACT) else redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        for sec in secrets:
            if sec and sec in value:
                value = value.replace(sec, "***")
    return value


class JsonFormatter(logging.Formatter):
    def __init__(self, secrets: tuple[str, ...] = ()):
        super().__init__()
        self.secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        body = {"ts": datetime.fromtimestamp(record.created, UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "level": record.levelname,
                "logger": record.name, "message": redact(record.getMessage(), self.secrets)}
        fields = getattr(record, "fields", None)
        if fields:
            body.update(redact(fields, self.secrets))
        if record.exc_info:
            body["exception"] = redact(self.formatException(record.exc_info), self.secrets)
        return json.dumps(body, default=str)


def configure(level: str = "INFO", stream=None, secrets: tuple[str, ...] = ()) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    h = logging.StreamHandler(stream or sys.stderr)
    h.setFormatter(JsonFormatter(secrets))
    root.addHandler(h)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
