from __future__ import annotations

import difflib
import re

from psl_workflow.learning.memory import StyleMemory
from psl_workflow.learning.models import FeedbackRecord


class FeedbackLearner:
    """Convert operator edits into prompt rules and reusable before/after examples."""

    def __init__(self, memory: StyleMemory) -> None:
        self.memory = memory

    def capture_operator_edit(
        self,
        matter_id: str,
        query: str,
        draft_text: str,
        final_text: str,
    ) -> FeedbackRecord:
        if not draft_text.strip():
            raise ValueError("draft_text is required to learn from an operator edit")
        if not final_text.strip():
            raise ValueError("final_text is required to learn from an operator edit")

        diff_summary = _summarize_diff(draft_text, final_text)
        extracted_rules = _extract_rules(draft_text, final_text)
        return self.memory.save_feedback(
            matter_id=matter_id,
            query=query,
            draft_text=draft_text,
            final_text=final_text,
            diff_summary=diff_summary,
            extracted_rules=extracted_rules,
        )


def _summarize_diff(draft_text: str, final_text: str) -> str:
    draft_lines = draft_text.splitlines()
    final_lines = final_text.splitlines()
    diff = difflib.unified_diff(
        draft_lines,
        final_lines,
        fromfile="draft",
        tofile="final",
        lineterm="",
    )
    lines = list(diff)
    if len(lines) > 30:
        lines = lines[:30] + ["... diff truncated ..."]
    return "\n".join(lines)


def _extract_rules(draft_text: str, final_text: str) -> list[str]:
    rules: list[str] = []

    section_rules = {
        "risk level": (
            "Include a clear Risk Level label when the operator adds risk classification."
        ),
        "recommendation": "Prefer a Recommendation section for action-oriented conclusions.",
        "rationale": "Add a Rationale section that separates reasoning from the recommendation.",
        "next steps": "Include concrete Next Steps when the operator adds follow-up actions.",
    }
    for marker, rule in section_rules.items():
        if _has_section(final_text, marker) and not _has_section(draft_text, marker):
            rules.append(rule)

    if final_text.count("[") > draft_text.count("["):
        rules.append("Preserve or add source citations for legally significant claims.")
    if _word_count(final_text) < _word_count(draft_text) * 0.75:
        rules.append("Be more concise when the operator removes unsupported or repetitive prose.")

    replacement_rule = _extract_replacement_rule(draft_text, final_text)
    if replacement_rule:
        rules.append(replacement_rule)

    if not rules:
        rules.append("Use the operator-approved final version as a few-shot style example.")
    return rules


def _extract_replacement_rule(draft_text: str, final_text: str) -> str | None:
    draft_phrases = _candidate_phrases(draft_text)
    final_phrases = _candidate_phrases(final_text)
    for draft_phrase in draft_phrases:
        for final_phrase in final_phrases:
            if draft_phrase != final_phrase and _token_overlap(draft_phrase, final_phrase) >= 1:
                return (
                    f"Prefer wording like '{final_phrase}' over looser phrasing "
                    f"such as '{draft_phrase}'."
                )
    return None


def _candidate_phrases(text: str) -> list[str]:
    phrases = []
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line.strip(" -:\t"))
        if 20 <= len(cleaned) <= 120:
            phrases.append(cleaned)
    if not phrases:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        phrases = [
            sentence.strip()
            for sentence in sentences
            if 20 <= len(sentence.strip()) <= 120
        ]
    return phrases[:3]


def _has_section(text: str, marker: str) -> bool:
    pattern = rf"(?im)^\s*(?:#+\s*)?{re.escape(marker)}\s*[:#-]?"
    return re.search(pattern, text) is not None


def _token_overlap(left: str, right: str) -> int:
    left_tokens = set(re.findall(r"[a-zA-Z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-zA-Z0-9]+", right.lower()))
    return len(left_tokens & right_tokens)


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))
