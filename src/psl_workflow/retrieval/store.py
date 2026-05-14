from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from psl_workflow.processing.models import IngestedDocument
from psl_workflow.retrieval.chunking import chunk_document
from psl_workflow.retrieval.embeddings import LocalHashEmbeddings
from psl_workflow.retrieval.models import RetrievedSnippet


class LegalRetriever:
    """Index and retrieve legal evidence snippets from ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma",
        collection_name: str = "psl_legal_evidence",
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_function = LocalHashEmbeddings()
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_function,
            persist_directory=str(self.persist_directory),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def index_document(self, document: IngestedDocument) -> list[str]:
        chunks = chunk_document(document)
        if not chunks:
            return []

        docs = [
            Document(
                page_content=chunk.text,
                metadata={
                    "source_path": chunk.source_path,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                    "citation": f"{Path(chunk.source_path).name} p.{chunk.page_number}",
                },
            )
            for chunk in chunks
        ]
        ids = [chunk.chunk_id for chunk in chunks]
        self.vector_store.add_documents(docs, ids=ids)
        return ids

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedSnippet]:
        if not query.strip():
            return []
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        snippets: list[RetrievedSnippet] = []
        for document, score in results:
            metadata = dict(document.metadata)
            snippets.append(
                RetrievedSnippet(
                    text=document.page_content,
                    source_path=str(metadata.get("source_path", "")),
                    page_number=int(metadata.get("page_number", 0)),
                    chunk_id=str(metadata.get("chunk_id", "")),
                    score=float(score),
                    metadata=metadata,
                )
            )
        return snippets
