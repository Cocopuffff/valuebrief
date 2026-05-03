"""
embeddings.py
~~~~~~~~~~~~~
Async wrapper around OpenAI's Embeddings API for the Hybrid RAG vector layer.

Default model: text-embedding-3-small (1536 dimensions).
Override via the EMBEDDING_MODEL environment variable.
"""

import os
from typing import Optional

from openai import AsyncOpenAI
from utils.logger import get_logger

logger = get_logger(__name__)

_client: Optional[AsyncOpenAI] = None

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = 1536


def _get_client() -> AsyncOpenAI:
    """Lazy-initialise the async OpenAI client."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for embeddings. "
                "Set it in your .env file."
            )
        _client = AsyncOpenAI(api_key=api_key)
    return _client


async def get_embedding(text: str) -> list[float]:
    """Return a 1536-dim embedding vector for a single text string."""
    client = _get_client()
    # Replace newlines with spaces for cleaner embedding
    clean = text.replace("\n", " ").strip()
    if not clean:
        return [0.0] * EMBEDDING_DIMENSIONS

    response = await client.embeddings.create(
        input=[clean],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return response.data[0].embedding


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for a batch of texts.

    OpenAI supports up to 2048 inputs per request.  We chunk to stay
    under that limit, though typical usage will be well below it.
    """
    if not texts:
        return []

    client = _get_client()
    cleaned = [t.replace("\n", " ").strip() for t in texts]

    # Replace empty strings with a placeholder to avoid API errors
    for i, t in enumerate(cleaned):
        if not t:
            cleaned[i] = " "

    BATCH_SIZE = 2048
    all_embeddings: list[list[float]] = []

    for start in range(0, len(cleaned), BATCH_SIZE):
        batch = cleaned[start : start + BATCH_SIZE]
        response = await client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        # OpenAI returns embeddings in the same order as the input
        all_embeddings.extend([d.embedding for d in response.data])

    return all_embeddings
