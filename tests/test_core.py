"""Tests for the core logic — no models, network, or API key required."""

from __future__ import annotations

import numpy as np
import pytest

from paper_chat import Chunk
from paper_chat.chat import answer, build_prompt, extract_cited, format_sources
from paper_chat.ingest import chunk_text, pages_to_chunks, parse_arxiv_id
from paper_chat.store import VectorStore


# --- fakes ---------------------------------------------------------------

class FakeEmbedder:
    """Deterministic bag-of-chars embedding so cosine ranking is meaningful."""

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 26), dtype=np.float32)
        for r, t in enumerate(texts):
            for ch in t.lower():
                if "a" <= ch <= "z":
                    out[r, ord(ch) - 97] += 1.0
        return out


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_prompt = ""

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.reply


# --- arxiv parsing -------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("2401.01234", "2401.01234"),
        ("https://arxiv.org/abs/2401.01234", "2401.01234"),
        ("https://arxiv.org/pdf/2401.01234v2", "2401.01234v2"),
        ("see arxiv.org/abs/2310.12345 for details", "2310.12345"),
        ("no id here", None),
    ],
)
def test_parse_arxiv_id(text, expected):
    assert parse_arxiv_id(text) == expected


# --- chunking ------------------------------------------------------------

def test_chunk_overlap_and_coverage():
    text = "abcdefghij" * 20  # 200 chars
    chunks = chunk_text(text, size=90, overlap=20)
    assert len(chunks) >= 2
    assert all(len(c) <= 90 for c in chunks)
    # step = size - overlap = 70, so chunk[1] starts 70 in and its first 20
    # chars are the last 20 of chunk[0].
    assert chunks[1][:20] == chunks[0][-20:]


def test_chunk_empty():
    assert chunk_text("   ") == []


def test_chunk_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_pages_to_chunks_keeps_page():
    pages = [(1, "x" * 1000), (7, "y" * 50)]
    chunks = pages_to_chunks(pages, "MyPaper")
    assert {c.page for c in chunks} == {1, 7}
    assert all(c.source == "MyPaper" for c in chunks)


# --- retrieval -----------------------------------------------------------

def test_vector_store_ranks_relevant_first():
    emb = FakeEmbedder()
    store = VectorStore()
    store.add(
        [
            Chunk("gaussian splatting compression", "A", 1),
            Chunk("banana bread recipe", "B", 2),
        ],
        emb,
    )
    hits = store.search("splatting compression method", emb, k=2)
    assert hits[0][0].source == "A"
    assert len(store) == 2


def test_empty_store_search():
    assert VectorStore().search("q", FakeEmbedder()) == []


# --- prompt + citations --------------------------------------------------

def test_build_prompt_tags_sources():
    hits = [(Chunk("text one", "Paper", 3), 0.9)]
    p = build_prompt("what?", hits)
    assert "[S1]" in p and "Paper, p.3" in p and "what?" in p


def test_format_sources():
    hits = [(Chunk("t", "FlexGaussian", 4), 0.5)]
    assert format_sources(hits) == ["[S1] FlexGaussian, p.4"]


def test_extract_cited_in_range_dedup_ordered():
    assert extract_cited("Per [S2] and [S2], not [S9].", n_sources=3) == ["[S2]"]


def test_answer_end_to_end_with_fakes():
    emb = FakeEmbedder()
    store = VectorStore()
    store.add([Chunk("memory efficient gaussian splatting", "Thesis", 5)], emb)
    llm = FakeLLM("It reduces memory [S1].")
    a = answer("how does it save memory?", store, emb, llm, k=1)
    assert a.cited == ["[S1]"]
    assert "Thesis, p.5" in a.sources[0]
    assert "Sources:" in llm.last_prompt
