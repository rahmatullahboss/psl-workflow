from pathlib import Path

from psl_workflow.processing.models import DocumentMetadata, IngestedDocument, PageText
from psl_workflow.retrieval.store import LegalRetriever


def test_retrieval_returns_ranked_snippets_with_source_citations(tmp_path: Path) -> None:
    document = IngestedDocument(
        metadata=DocumentMetadata(
            source_path="contracts/wayne-psl.pdf",
            document_type="pdf",
            page_count=2,
            title="Wayne Contract",
        ),
        pages=[
            PageText(
                page_number=1,
                text="The acquisition agreement requires notice before termination.",
                extraction_method="pymupdf",
            ),
            PageText(
                page_number=2,
                text=(
                    "A termination fee of USD 2,500,000 is payable "
                    "if Wayne exits after diligence."
                ),
                extraction_method="pymupdf",
            ),
        ],
    )
    retriever = LegalRetriever(persist_directory=tmp_path / "chroma")
    retriever.index_document(document)

    results = retriever.retrieve("What is the termination fee?", k=2)

    assert results
    assert "termination fee" in results[0].text.lower()
    assert results[0].citation == "wayne-psl.pdf p.2"
    assert results[0].source_path == "contracts/wayne-psl.pdf"
