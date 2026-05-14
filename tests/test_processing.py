from pathlib import Path

from psl_workflow.processing.ingestion import DocumentIngestor
from psl_workflow.processing.samples import generate_sample_documents


def test_ingests_generated_pdf_as_structured_json(tmp_path: Path) -> None:
    samples = generate_sample_documents(tmp_path / "samples")

    ingested = DocumentIngestor().ingest(samples["clean_pdf"])
    payload = ingested.to_dict()

    assert payload["metadata"]["source_path"].endswith("clean_contract.pdf")
    assert payload["metadata"]["document_type"] == "pdf"
    assert payload["pages"][0]["page_number"] == 1
    assert "Pearson Specter Litt" in payload["raw_text"]
    assert payload["pages"][0]["extraction_method"] in {"pymupdf", "ocr"}


def test_ingests_generated_messy_image_with_ocr_metadata(tmp_path: Path) -> None:
    samples = generate_sample_documents(tmp_path / "samples")

    ingested = DocumentIngestor().ingest(samples["messy_image"])

    assert ingested.metadata.document_type == "image"
    assert ingested.pages[0].page_number == 1
    assert ingested.pages[0].extraction_method in {"ocr", "ocr-unavailable"}
    assert ingested.pages[0].metadata["ocr_engine"] in {"tesseract", "easyocr", "unavailable"}
