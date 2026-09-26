from __future__ import annotations

import re
from typing import Any

_SECRET = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|ya29\.[A-Za-z0-9_\-]+|Bearer\s+[A-Za-z0-9._\-]{8,})",
    re.IGNORECASE,
)


def redact(value: str | None) -> str:
    return _SECRET.sub("[redacted]", value or "")


def redact_obj(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [redact_obj(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_obj(item) for key, item in value.items()}
    return value
