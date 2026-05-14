from __future__ import annotations

import re
from dataclasses import dataclass

from psl_workflow.processing.models import IngestedDocument


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source_path: str
    page_number: int


def chunk_document(
    document: IngestedDocument,
    max_words: int = 120,
    overlap_words: int = 25,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    source_path = document.metadata.source_path
    for page in document.pages:
        words = _tokenize_preserving_text(page.text)
        if not words:
            continue
        start = 0
        chunk_index = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            text = " ".join(words[start:end]).strip()
            if text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{source_path}:p{page.page_number}:c{chunk_index}",
                        text=text,
                        source_path=source_path,
                        page_number=page.page_number,
                    )
                )
            if end == len(words):
                break
            start = max(end - overlap_words, start + 1)
            chunk_index += 1
    return chunks


def _tokenize_preserving_text(text: str) -> list[str]:
    return re.findall(r"\S+", text)
