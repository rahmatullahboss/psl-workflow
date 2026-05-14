from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from psl_workflow.evaluation import run_demo
from psl_workflow.generation.engine import GroundedDraftingEngine, build_default_llm
from psl_workflow.learning.feedback import FeedbackLearner
from psl_workflow.learning.memory import StyleMemory
from psl_workflow.processing.ingestion import DocumentIngestor
from psl_workflow.processing.models import IngestedDocument
from psl_workflow.processing.samples import generate_sample_documents
from psl_workflow.retrieval.store import LegalRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Pearson Specter Litt AI workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("generate-samples", help="Create synthetic legal samples")
    sample_parser.add_argument("--output", default="data/samples")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF or image to structured JSON")
    ingest_parser.add_argument("path")
    ingest_parser.add_argument("--output-dir", default="data/ingested")

    index_parser = subparsers.add_parser("index", help="Index an ingested JSON file into Chroma")
    index_parser.add_argument("json_path")
    index_parser.add_argument("--chroma-dir", default=os.getenv("PSL_CHROMA_DIR", "data/chroma"))

    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve cited snippets")
    retrieve_parser.add_argument("query")
    retrieve_parser.add_argument("--k", type=int, default=5)
    retrieve_parser.add_argument("--chroma-dir", default=os.getenv("PSL_CHROMA_DIR", "data/chroma"))

    draft_parser = subparsers.add_parser("draft", help="Generate grounded internal memo")
    draft_parser.add_argument("query")
    draft_parser.add_argument("--matter-id", default="demo-matter")
    draft_parser.add_argument("--k", type=int, default=5)
    draft_parser.add_argument("--chroma-dir", default=os.getenv("PSL_CHROMA_DIR", "data/chroma"))

    feedback_parser = subparsers.add_parser("feedback", help="Save operator edit feedback")
    feedback_parser.add_argument("--matter-id", required=True)
    feedback_parser.add_argument("--query", required=True)
    feedback_parser.add_argument("--draft-file", required=True)
    feedback_parser.add_argument("--final-file", required=True)

    subparsers.add_parser("demo", help="Run complete ingest -> retrieve -> draft -> feedback demo")

    api_parser = subparsers.add_parser("api", help="Start FastAPI server")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    _dispatch(args)


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "generate-samples":
        samples = generate_sample_documents(args.output)
        print(json.dumps({key: str(value) for key, value in samples.items()}, indent=2))
    elif args.command == "ingest":
        document = DocumentIngestor().ingest(args.path)
        output_path = Path(args.output_dir) / f"{Path(args.path).stem}.json"
        document.save_json(output_path)
        print(
            json.dumps(
                {"output_path": str(output_path), "document": document.to_dict()},
                indent=2,
            )
        )
    elif args.command == "index":
        document = IngestedDocument.load_json(Path(args.json_path))
        chunk_ids = LegalRetriever(args.chroma_dir).index_document(document)
        print(json.dumps({"indexed_chunk_ids": chunk_ids, "count": len(chunk_ids)}, indent=2))
    elif args.command == "retrieve":
        snippets = LegalRetriever(args.chroma_dir).retrieve(args.query, args.k)
        print(json.dumps({"snippets": [snippet.to_dict() for snippet in snippets]}, indent=2))
    elif args.command == "draft":
        retriever = LegalRetriever(args.chroma_dir)
        snippets = retriever.retrieve(args.query, args.k)
        memory = StyleMemory(os.getenv("PSL_FEEDBACK_DB", "data/feedback.sqlite3"))
        engine = GroundedDraftingEngine(llm=build_default_llm(), style_memory=memory)
        print(engine.generate_memo(args.matter_id, args.query, snippets))
    elif args.command == "feedback":
        memory = StyleMemory(os.getenv("PSL_FEEDBACK_DB", "data/feedback.sqlite3"))
        record = FeedbackLearner(memory).capture_operator_edit(
            matter_id=args.matter_id,
            query=args.query,
            draft_text=Path(args.draft_file).read_text(encoding="utf-8"),
            final_text=Path(args.final_file).read_text(encoding="utf-8"),
        )
        print(json.dumps(record.to_dict(), indent=2))
    elif args.command == "demo":
        print(json.dumps(run_demo(), indent=2))
    elif args.command == "api":
        import uvicorn

        from psl_workflow.api import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)
