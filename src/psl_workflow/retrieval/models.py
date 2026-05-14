from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RetrievedSnippet:
    text: str
    source_path: str
    page_number: int
    chunk_id: str
    score: float
    metadata: dict[str, Any] | None = None

    @property
    def citation(self) -> str:
        return f"{Path(self.source_path).name} p.{self.page_number}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_path": self.source_path,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "citation": self.citation,
            "metadata": self.metadata or {},
        }
