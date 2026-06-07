"""A tiny in-memory vector store: embed chunks, retrieve by cosine similarity.

numpy is plenty for paper-scale corpora (hundreds to low-thousands of chunks),
and it has no native build headaches. The embedder is an injectable protocol so
tests can run without downloading a model.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from . import Chunk


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float array of L2-normalisable embeddings."""
        ...


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class VectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._emb: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], embedder: Embedder) -> None:
        if not chunks:
            return
        vecs = _normalize(np.asarray(embedder.embed([c.text for c in chunks]), dtype=np.float32))
        self.chunks.extend(chunks)
        self._emb = vecs if self._emb is None else np.vstack([self._emb, vecs])

    def search(self, query: str, embedder: Embedder, k: int = 5) -> list[tuple[Chunk, float]]:
        if self._emb is None or not self.chunks:
            return []
        q = _normalize(np.asarray(embedder.embed([query]), dtype=np.float32))[0]
        scores = self._emb @ q
        top = np.argsort(-scores)[: min(k, len(self.chunks))]
        return [(self.chunks[i], float(scores[i])) for i in top]


class SentenceTransformerEmbedder:
    """Real embedder backed by sentence-transformers (lazy import)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=False))
