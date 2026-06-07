"""Turn PDFs (local or arXiv) into citation-carrying chunks.

The pure functions here (arXiv-id parsing, chunking, page->chunk) are unit
tested directly. The two functions that touch the filesystem/network
(``load_pdf`` via PyMuPDF, ``fetch_arxiv`` via urllib) import their heavy deps
lazily so the rest of the package stays importable without them.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import Chunk
from .config import CHUNK_OVERLAP, CHUNK_SIZE

# Matches bare ids (2401.01234, optionally with version) and full abs/pdf URLs.
_ARXIV_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE
)


def parse_arxiv_id(text: str) -> str | None:
    """Pull an arXiv id out of a bare id or an abs/pdf URL; else None."""
    m = _ARXIV_RE.search(text.strip())
    if not m:
        return None
    return m.group(1) + (m.group(2) or "")


def chunk_text(text: str, *, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character windows, skipping empties."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")

    text = text.strip()
    if not text:
        return []

    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


def pages_to_chunks(pages: list[tuple[int, str]], source: str) -> list[Chunk]:
    """Chunk each (page_no, page_text) pair, keeping the page for citations."""
    out: list[Chunk] = []
    for page_no, page_text in pages:
        for piece in chunk_text(page_text):
            out.append(Chunk(text=piece, source=source, page=page_no))
    return out


def load_pdf(path: Path, source: str | None = None) -> list[Chunk]:
    """Extract per-page text from a PDF and chunk it. Requires PyMuPDF."""
    import fitz  # type: ignore  # PyMuPDF, imported lazily

    src = source or Path(path).stem
    pages: list[tuple[int, str]] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append((i, page.get_text()))
    return pages_to_chunks(pages, src)


def fetch_arxiv(arxiv_id: str, dest_dir: Path) -> Path:
    """Download an arXiv PDF to ``dest_dir`` and return the path."""
    import urllib.request

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{arxiv_id}.pdf"
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    urllib.request.urlretrieve(url, out)  # noqa: S310 (trusted arxiv host)
    return out
