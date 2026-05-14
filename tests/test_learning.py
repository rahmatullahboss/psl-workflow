from pathlib import Path

from psl_workflow.learning.feedback import FeedbackLearner
from psl_workflow.learning.memory import StyleMemory


def test_feedback_learning_extracts_rules_and_future_prompt_context(tmp_path: Path) -> None:
    memory = StyleMemory(tmp_path / "feedback.sqlite3")
    learner = FeedbackLearner(memory)

    record = learner.capture_operator_edit(
        matter_id="matter-1",
        query="termination risk",
        draft_text=(
            "The party can terminate. This seems risky. "
            "The memo does not include a recommendation."
        ),
        final_text=(
            "Risk Level: High.\n"
            "Recommendation: Do not terminate until written notice is served.\n"
            "Rationale: The record supports notice as a condition precedent."
        ),
    )

    assert record.diff_summary
    assert record.extracted_rules
    assert any("Recommendation" in rule for rule in record.extracted_rules)

    context = memory.build_prompt_context("termination risk")

    assert "Style and Correction Memory" in context
    assert "Recommendation" in context
    assert "Before:" in context and "After:" in context
