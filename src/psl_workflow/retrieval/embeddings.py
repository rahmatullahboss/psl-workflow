from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9$%.-]+")


class LocalHashEmbeddings(Embeddings):
    """Deterministic embeddings for offline demos and tests.

    This keeps the take-home project runnable without downloading a model. The retrieval layer
    still uses LangChain and ChromaDB, and this embedding adapter can be replaced with OpenAI or
    another production embedding provider without changing the store interface.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + min(len(token), 12) / 12.0
            vector[index] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
