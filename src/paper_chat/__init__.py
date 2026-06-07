"""paper-chat: ask questions across a library of PDFs, with cited answers."""

from __future__ import annotations

from dataclasses import dataclass

__version__ = "0.1.0"


@dataclass(frozen=True)
class Chunk:
    """A retrievable span of text plus where it came from (for citations)."""

    text: str
    source: str  # paper title or filename
    page: int  # 1-indexed page number
