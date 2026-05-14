from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DocumentMetadata:
    source_path: str
    document_type: str
    page_count: int
    title: str | None = None
    author: str | None = None
    created_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "document_type": self.document_type,
            "page_count": self.page_count,
            "title": self.title,
            "author": self.author,
            "created_at": self.created_at,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DocumentMetadata:
        return cls(
            source_path=payload["source_path"],
            document_type=payload["document_type"],
            page_count=int(payload["page_count"]),
            title=payload.get("title"),
            author=payload.get("author"),
            created_at=payload.get("created_at"),
            extra=dict(payload.get("extra") or {}),
        )


@dataclass(slots=True)
class PageText:
    page_number: int
    text: str
    extraction_method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "extraction_method": self.extraction_method,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PageText:
        return cls(
            page_number=int(payload["page_number"]),
            text=payload.get("text", ""),
            extraction_method=payload.get("extraction_method", "unknown"),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True)
class IngestedDocument:
    metadata: DocumentMetadata
    pages: list[PageText]

    @property
    def raw_text(self) -> str:
        return "\n\n".join(page.text.strip() for page in self.pages if page.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "pages": [page.to_dict() for page in self.pages],
            "raw_text": self.raw_text,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IngestedDocument:
        return cls(
            metadata=DocumentMetadata.from_dict(payload["metadata"]),
            pages=[PageText.from_dict(page) for page in payload["pages"]],
        )

    def save_json(self, output_path: Path) -> Path:
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def load_json(cls, input_path: Path) -> IngestedDocument:
        import json

        return cls.from_dict(json.loads(input_path.read_text(encoding="utf-8")))
