from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class FeedbackRecord:
    id: int | None
    matter_id: str
    query: str
    draft_text: str
    final_text: str
    diff_summary: str
    extracted_rules: list[str]
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "matter_id": self.matter_id,
            "query": self.query,
            "draft_text": self.draft_text,
            "final_text": self.final_text,
            "diff_summary": self.diff_summary,
            "extracted_rules": self.extracted_rules,
            "created_at": self.created_at,
        }
