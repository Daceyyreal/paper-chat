"""Tunables for paper-chat. Override via environment variables if you like."""

from __future__ import annotations

import os

# Local embedding model (downloaded once from Hugging Face on first run).
EMBED_MODEL = os.getenv("PC_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Which chat backend to use: "groq" (default) or "gemini".
LLM_PROVIDER = os.getenv("PC_LLM_PROVIDER", "groq").lower()

# Groq chat model. Check https://console.groq.com/docs/models for the current
# list; names change. llama-3.3-70b-versatile is a solid default.
GROQ_MODEL = os.getenv("PC_GROQ_MODEL", "llama-3.3-70b-versatile")

# Gemini chat model. See https://ai.google.dev/gemini-api/docs/models for the
# current list. gemini-2.5-flash is current and works on the free tier.
GEMINI_MODEL = os.getenv("PC_GEMINI_MODEL", "gemini-2.5-flash")

# Retrieval + chunking. Larger TOP_K / chunks give the model more context to
# write a detailed, well-grounded answer (at a small latency/token cost).
TOP_K = int(os.getenv("PC_TOP_K", "8"))
CHUNK_SIZE = int(os.getenv("PC_CHUNK_SIZE", "1200"))  # characters
CHUNK_OVERLAP = int(os.getenv("PC_CHUNK_OVERLAP", "200"))
