from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from psl_workflow.evaluation import run_demo
from psl_workflow.generation.engine import GroundedDraftingEngine, build_default_llm
from psl_workflow.learning.feedback import FeedbackLearner
from psl_workflow.learning.memory import StyleMemory
from psl_workflow.processing.ingestion import DocumentIngestor, IngestionError
from psl_workflow.processing.models import IngestedDocument
from psl_workflow.retrieval.store import LegalRetriever


class RetrieveRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)


class DraftRequest(BaseModel):
    matter_id: str
    query: str
    k: int = Field(default=5, ge=1, le=20)


class FeedbackRequest(BaseModel):
    matter_id: str
    query: str
    draft_text: str
    final_text: str


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Pearson Specter Litt Legal Workflow", version="0.1.0")
    resolved_data_dir = Path(data_dir or os.getenv("PSL_DATA_DIR", "data"))
    uploads_dir = resolved_data_dir / "uploads"
    ingested_dir = resolved_data_dir / "ingested"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    ingested_dir.mkdir(parents=True, exist_ok=True)

    ingestor = DocumentIngestor()
    retriever = LegalRetriever(
        os.getenv("PSL_CHROMA_DIR", str(resolved_data_dir / "chroma"))
    )
    memory = StyleMemory(
        os.getenv("PSL_FEEDBACK_DB", str(resolved_data_dir / "feedback.sqlite3"))
    )
    engine = GroundedDraftingEngine(llm=build_default_llm(), style_memory=memory)
    learner = FeedbackLearner(memory)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest")
    async def ingest(file: Annotated[UploadFile, File(...)]) -> dict:
        destination = uploads_dir / Path(file.filename or "upload.bin").name
        destination.write_bytes(await file.read())
        try:
            document = ingestor.ingest(destination)
        except IngestionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        document.save_json(ingested_dir / f"{destination.stem}.json")
        chunk_ids = retriever.index_document(document)
        return {"document": document.to_dict(), "indexed_chunk_ids": chunk_ids}

    @app.post("/retrieve")
    def retrieve(request: RetrieveRequest) -> dict:
        snippets = retriever.retrieve(request.query, request.k)
        return {"snippets": [snippet.to_dict() for snippet in snippets]}

    @app.post("/draft")
    def draft(request: DraftRequest) -> dict:
        snippets = retriever.retrieve(request.query, request.k)
        memo = engine.generate_memo(request.matter_id, request.query, snippets)
        return {"memo": memo, "snippets": [snippet.to_dict() for snippet in snippets]}

    @app.post("/feedback")
    def feedback(request: FeedbackRequest) -> dict:
        record = learner.capture_operator_edit(
            matter_id=request.matter_id,
            query=request.query,
            draft_text=request.draft_text,
            final_text=request.final_text,
        )
        return {
            "feedback": record.to_dict(),
            "prompt_memory": memory.build_prompt_context(request.query),
        }

    @app.post("/demo")
    def demo() -> dict:
        return run_demo(data_dir=resolved_data_dir / "demo")

    @app.post("/index-json")
    def index_json(path: str) -> dict:
        document_path = Path(path)
        if not document_path.exists():
            raise HTTPException(status_code=404, detail=f"Not found: {path}")
        document = IngestedDocument.load_json(document_path)
        return {"indexed_chunk_ids": retriever.index_document(document)}

    return app
