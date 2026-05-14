from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from psl_workflow.learning.models import FeedbackRecord


class StyleMemory:
    """SQLite-backed reusable memory derived from operator edits."""

    def __init__(self, db_path: str | Path = "data/feedback.sqlite3") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def save_feedback(
        self,
        matter_id: str,
        query: str,
        draft_text: str,
        final_text: str,
        extracted_rules: list[str],
        diff_summary: str = "",
    ) -> FeedbackRecord:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback
                    (
                        matter_id,
                        query,
                        draft_text,
                        final_text,
                        diff_summary,
                        extracted_rules,
                        created_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    matter_id,
                    query,
                    draft_text,
                    final_text,
                    diff_summary,
                    json.dumps(extracted_rules),
                    created_at,
                ),
            )
            record_id = int(cursor.lastrowid)
        return FeedbackRecord(
            id=record_id,
            matter_id=matter_id,
            query=query,
            draft_text=draft_text,
            final_text=final_text,
            diff_summary=diff_summary,
            extracted_rules=extracted_rules,
            created_at=created_at,
        )

    def records(self) -> list[FeedbackRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    matter_id,
                    query,
                    draft_text,
                    final_text,
                    diff_summary,
                    extracted_rules,
                    created_at
                FROM feedback
                ORDER BY id DESC
                """
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def build_prompt_context(self, query: str, limit: int = 3) -> str:
        candidates = self._rank_records(query, self.records())[:limit]
        if not candidates:
            return ""

        lines = ["Style and Correction Memory:"]
        seen_rules: set[str] = set()
        for record in candidates:
            for rule in record.extracted_rules:
                if rule not in seen_rules:
                    lines.append(f"- Rule: {rule}")
                    seen_rules.add(rule)
            lines.append(f"- Example from {record.matter_id}:")
            lines.append(f"  Before: {self._truncate(record.draft_text, 260)}")
            lines.append(f"  After: {self._truncate(record.final_text, 260)}")
        return "\n".join(lines)

    def _rank_records(
        self, query: str, records: Iterable[FeedbackRecord]
    ) -> list[FeedbackRecord]:
        query_terms = set(_terms(query))

        def score(record: FeedbackRecord) -> tuple[int, int]:
            haystack = set(
                _terms(" ".join([record.query, *record.extracted_rules, record.final_text]))
            )
            return (len(query_terms & haystack), int(record.id or 0))

        return sorted(records, key=score, reverse=True)

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    matter_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    draft_text TEXT NOT NULL,
                    final_text TEXT NOT NULL,
                    diff_summary TEXT NOT NULL,
                    extracted_rules TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _record_from_row(row: sqlite3.Row | tuple) -> FeedbackRecord:
        return FeedbackRecord(
            id=int(row[0]),
            matter_id=str(row[1]),
            query=str(row[2]),
            draft_text=str(row[3]),
            final_text=str(row[4]),
            diff_summary=str(row[5]),
            extracted_rules=json.loads(str(row[6])),
            created_at=str(row[7]),
        )

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + "..."


def _terms(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(term) > 2]
