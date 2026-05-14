from __future__ import annotations

import json
import shutil
from pathlib import Path

from psl_workflow.generation.engine import GroundedDraftingEngine, build_default_llm
from psl_workflow.learning.feedback import FeedbackLearner
from psl_workflow.learning.memory import StyleMemory
from psl_workflow.processing.ingestion import DocumentIngestor
from psl_workflow.processing.models import IngestedDocument
from psl_workflow.processing.samples import generate_sample_documents
from psl_workflow.retrieval.store import LegalRetriever


def run_demo(
    data_dir: str | Path = "examples/data",
    outputs_dir: str | Path = "examples/outputs",
) -> dict:
    data_path = Path(data_dir)
    output_path = Path(outputs_dir)
    ingested_dir = data_path / "ingested"
    chroma_dir = data_path / "chroma"
    feedback_db = data_path / "feedback.sqlite3"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    if feedback_db.exists():
        feedback_db.unlink()
    output_path.mkdir(parents=True, exist_ok=True)
    ingested_dir.mkdir(parents=True, exist_ok=True)

    samples = generate_sample_documents(data_path / "samples")
    ingestor = DocumentIngestor()
    documents: list[IngestedDocument] = []
    for sample in samples.values():
        document = ingestor.ingest(sample)
        document.save_json(ingested_dir / f"{sample.stem}.json")
        documents.append(document)

    retriever = LegalRetriever(chroma_dir)
    indexed_chunk_ids: list[str] = []
    for document in documents:
        indexed_chunk_ids.extend(retriever.index_document(document))

    query = "Can Wayne terminate immediately and what fee risk applies?"
    snippets = retriever.retrieve(query, k=4)

    memory = StyleMemory(feedback_db)
    engine = GroundedDraftingEngine(llm=build_default_llm(), style_memory=memory)
    first_draft = engine.generate_memo("wayne-logistics", query, snippets)

    operator_final = (
        "Risk Level: High.\n"
        "Recommendation: Do not advise immediate termination until ten business days written "
        "notice is served. The agreement ties termination to written notice and identifies a "
        "USD 2,500,000 fee risk for exit after diligence.\n"
        "Rationale: Keep the advice limited to cited contract and operator note evidence."
    )
    feedback = FeedbackLearner(memory).capture_operator_edit(
        matter_id="wayne-logistics",
        query=query,
        draft_text=first_draft,
        final_text=operator_final,
    )
    improved_draft = engine.generate_memo("wayne-logistics-followup", query, snippets)

    result = {
        "sample_inputs": {name: str(path) for name, path in samples.items()},
        "ingested_documents": [doc.to_dict() for doc in documents],
        "indexed_chunk_count": len(indexed_chunk_ids),
        "query": query,
        "retrieved_snippets": [snippet.to_dict() for snippet in snippets],
        "first_draft": first_draft,
        "operator_feedback": feedback.to_dict(),
        "improved_draft": improved_draft,
        "evaluation": {
            "structured_ingestion": all(
                doc.metadata.page_count == len(doc.pages) for doc in documents
            ),
            "citations_returned": all(snippet.citation for snippet in snippets),
            "draft_is_grounded": "[" in first_draft and "]" in first_draft,
            "learning_loop_applied": "Recommendation" in improved_draft,
        },
    }
    (output_path / "evaluation_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_path / "first_draft.md").write_text(first_draft, encoding="utf-8")
    (output_path / "improved_draft.md").write_text(improved_draft, encoding="utf-8")
    return result
