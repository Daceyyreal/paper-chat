"""Tunables for paper-chat. Override via environment variables if you like."""

from __future__ import annotations

import os

# Local embedding model (downloaded once from Hugging Face on first run).
EMBED_MODEL = os.getenv("PC_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Groq chat model. Check https://console.groq.com/docs/models for the current
# list; names change. llama-3.3-70b-versatile is a solid default.
GROQ_MODEL = os.getenv("PC_GROQ_MODEL", "llama-3.3-70b-versatile")

# Retrieval + chunking.
TOP_K = int(os.getenv("PC_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("PC_CHUNK_SIZE", "900"))  # characters
CHUNK_OVERLAP = int(os.getenv("PC_CHUNK_OVERLAP", "150"))
