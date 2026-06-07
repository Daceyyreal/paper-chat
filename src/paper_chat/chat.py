"""Assemble a grounded prompt, call the LLM, return an answer with its sources.

The whole point of the tool is *cited* answers: every retrieved chunk is tagged
[S1], [S2], ... in the prompt, the model is told to cite those tags inline, and
we hand back the mapping so the UI can show exactly where each claim came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from . import Chunk
from .config import GEMINI_MODEL, GROQ_MODEL, LLM_PROVIDER, TOP_K
from .store import Embedder, VectorStore


class LLM(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@dataclass
class Answer:
    text: str
    sources: list[str]  # human-readable, e.g. "[S1] FlexGaussian, p.4"
    cited: list[str]  # the subset of source tags the answer actually used


SYSTEM = (
    "You answer questions using ONLY the numbered sources below. "
    "Cite the sources you use inline with their tags, e.g. [S2]. "
    "If the sources do not contain the answer, say so plainly. Do not invent facts."
)


def format_sources(hits: list[tuple[Chunk, float]]) -> list[str]:
    return [f"[S{i}] {c.source}, p.{c.page}" for i, (c, _) in enumerate(hits, start=1)]


def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    blocks = [f"[S{i}] ({c.source}, p.{c.page})\n{c.text}" for i, (c, _) in enumerate(hits, 1)]
    context = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
    return f"{SYSTEM}\n\nSources:\n{context}\n\nQuestion: {question}\n\nAnswer (with [S#] citations):"


def extract_cited(answer_text: str, n_sources: int) -> list[str]:
    """Which [S#] tags appear in the answer, in order, within range."""
    seen: list[str] = []
    for m in re.findall(r"\[S(\d+)\]", answer_text):
        tag = f"[S{m}]"
        if 1 <= int(m) <= n_sources and tag not in seen:
            seen.append(tag)
    return seen


def answer(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    llm: LLM,
    k: int = TOP_K,
) -> Answer:
    hits = store.search(question, embedder, k=k)
    prompt = build_prompt(question, hits)
    text = llm.complete(prompt).strip()
    sources = format_sources(hits)
    return Answer(text=text, sources=sources, cited=extract_cited(text, len(hits)))


class GroqLLM:
    """Real LLM backed by the Groq API (lazy import; reads GROQ_API_KEY)."""

    def __init__(self, model: str = GROQ_MODEL) -> None:
        from groq import Groq  # lazy

        self._client = Groq()
        self._model = model

    def complete(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


class GeminiLLM:
    """Real LLM backed by the Google Gemini API (lazy import; reads GEMINI_API_KEY)."""

    def __init__(self, model: str = GEMINI_MODEL) -> None:
        import os

        import google.generativeai as genai  # lazy

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use the Gemini backend.")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def complete(self, prompt: str) -> str:
        resp = self._model.generate_content(
            prompt,
            generation_config={"temperature": 0.2},
        )
        return resp.text or ""


def make_llm(provider: str = LLM_PROVIDER) -> LLM:
    """Construct the configured chat backend. Override with PC_LLM_PROVIDER."""
    provider = provider.lower()
    if provider == "gemini":
        return GeminiLLM()
    if provider == "groq":
        return GroqLLM()
    raise ValueError(f"Unknown PC_LLM_PROVIDER {provider!r} (expected 'groq' or 'gemini').")
