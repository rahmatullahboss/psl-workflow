from psl_workflow.generation.engine import GroundedDraftingEngine, LocalDraftLLM
from psl_workflow.learning.memory import StyleMemory
from psl_workflow.retrieval.models import RetrievedSnippet


def test_grounded_engine_builds_memo_with_citations_and_memory(tmp_path) -> None:
    memory = StyleMemory(tmp_path / "feedback.sqlite3")
    memory.save_feedback(
        matter_id="matter-1",
        query="termination",
        draft_text="Draft: client may terminate.",
        final_text="Recommendation: client may terminate only after written notice.",
        extracted_rules=["Prefer a Recommendation section for action-oriented conclusions."],
    )
    engine = GroundedDraftingEngine(llm=LocalDraftLLM(), style_memory=memory)
    snippets = [
        RetrievedSnippet(
            text="The agreement requires written notice before termination.",
            source_path="contracts/wayne-psl.pdf",
            page_number=7,
            chunk_id="abc",
            score=0.91,
        )
    ]

    memo = engine.generate_memo(
        matter_id="matter-2",
        question="Can the client terminate immediately?",
        snippets=snippets,
    )

    assert "Internal Legal Memo" in memo
    assert "[wayne-psl.pdf p.7]" in memo
    assert "Recommendation" in memo


def test_grounded_engine_refuses_without_evidence(tmp_path) -> None:
    engine = GroundedDraftingEngine(
        llm=LocalDraftLLM(),
        style_memory=StyleMemory(tmp_path / "db.sqlite3"),
    )

    memo = engine.generate_memo(
        matter_id="matter-3",
        question="Can we sue for fraud?",
        snippets=[],
    )

    assert "Insufficient sourced evidence" in memo
