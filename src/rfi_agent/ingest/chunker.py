from __future__ import annotations

import re

HEADING = re.compile(r"(?m)^(#{1,6} .+)$")


def chunk_text(text: str, *, target_chars: int = 2400, overlap: int = 240) -> list[str]:
    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []
    parts = _split_headings(cleaned)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        if buf and len(buf) + len(part) > target_chars:
            chunks.append(buf.strip())
            buf = buf[-overlap:] + part
        else:
            buf = f"{buf}\n\n{part}" if buf else part
    if buf.strip():
        chunks.append(buf.strip())
    out: list[str] = []
    for chunk in chunks:
        if len(chunk) <= target_chars * 2:
            out.append(chunk)
            continue
        for i in range(0, len(chunk), target_chars - overlap):
            piece = chunk[i : i + target_chars]
            if piece.strip():
                out.append(piece.strip())
    return out


def _split_headings(text: str) -> list[str]:
    matches = list(HEADING.finditer(text))
    if not matches:
        return [text]
    parts: list[str] = []
    if matches[0].start() > 0:
        parts.append(text[: matches[0].start()].strip())
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append(text[match.start() : end].strip())
    return [p for p in parts if p]
