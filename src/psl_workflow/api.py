from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
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

    @app.get("/", response_class=HTMLResponse)
    def web_ui() -> str:
        return WEB_UI_HTML

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


WEB_UI_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pearson Specter Litt Workflow</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d9dee7;
      --text: #17202a;
      --muted: #667085;
      --accent: #1f6feb;
      --accent-dark: #174ea6;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: #101828;
      color: #fff;
      padding: 16px 24px;
    }
    header h1 { margin: 0; font-size: 20px; font-weight: 700; }
    header p { margin: 4px 0 0; color: #d0d5dd; font-size: 13px; }
    main {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 16px;
      padding: 16px;
      max-width: 1440px;
      margin: 0 auto;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    h2 { margin: 0 0 10px; font-size: 15px; }
    label { display: block; font-size: 12px; color: var(--muted); margin: 12px 0 6px; }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      background: #fff;
      color: var(--text);
    }
    textarea { min-height: 120px; resize: vertical; }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      padding: 9px 12px;
      font-weight: 700;
      cursor: pointer;
      margin-top: 10px;
    }
    button.secondary { background: #475467; }
    button:hover { background: var(--accent-dark); }
    button.secondary:hover { background: #344054; }
    .stack { display: grid; gap: 12px; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; }
    .status {
      min-height: 22px;
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
      white-space: pre-wrap;
    }
    .status.error { color: var(--danger); }
    pre {
      margin: 0;
      padding: 12px;
      background: #0b1220;
      color: #e6edf7;
      border-radius: 8px;
      overflow: auto;
      min-height: 220px;
      white-space: pre-wrap;
      line-height: 1.45;
      font-size: 13px;
    }
    .snippet {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      margin-bottom: 8px;
      background: #fbfcfe;
    }
    .snippet strong { display: block; font-size: 12px; color: var(--accent-dark); }
    .snippet p { margin: 6px 0 0; font-size: 13px; line-height: 1.4; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Pearson Specter Litt Legal Workflow</h1>
    <p>Ingest documents, retrieve cited evidence, draft grounded memos, and learn from edits.</p>
  </header>
  <main>
    <div class="stack">
      <section>
        <h2>Document Intake</h2>
        <input id="file" type="file" accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp" />
        <div class="row">
          <button onclick="ingest()">Upload & Index</button>
          <button class="secondary" onclick="runDemo()">Run Demo</button>
        </div>
        <div id="intakeStatus" class="status"></div>
      </section>
      <section>
        <h2>Retrieve & Draft</h2>
        <label for="matter">Matter ID</label>
        <input id="matter" value="wayne-logistics" />
        <label for="query">Question</label>
        <textarea id="query">Can Wayne terminate immediately and what fee risk applies?</textarea>
        <div class="row">
          <button onclick="retrieve()">Retrieve Evidence</button>
          <button onclick="draft()">Draft Memo</button>
        </div>
        <div id="draftStatus" class="status"></div>
      </section>
      <section>
        <h2>Operator Feedback</h2>
        <label for="finalText">Final edited version</label>
        <textarea id="finalText">Risk Level: High.
Recommendation: Do not advise immediate termination until written notice is served.
Rationale: Keep the advice limited to cited source evidence.</textarea>
        <button onclick="saveFeedback()">Save Feedback Memory</button>
        <div id="feedbackStatus" class="status"></div>
      </section>
    </div>
    <div class="stack">
      <section>
        <h2>Retrieved Evidence</h2>
        <div id="snippets"></div>
      </section>
      <section>
        <h2>Memo Draft</h2>
        <pre id="memo">No memo generated yet.</pre>
      </section>
      <section>
        <h2>Raw Response</h2>
        <pre id="raw">Ready.</pre>
      </section>
    </div>
  </main>
  <script>
    let lastDraft = "";

    function setStatus(id, message, isError = false) {
      const el = document.getElementById(id);
      el.textContent = message;
      el.className = isError ? "status error" : "status";
    }

    function setRaw(payload) {
      document.getElementById("raw").textContent = JSON.stringify(payload, null, 2);
    }

    function renderSnippets(snippets) {
      const container = document.getElementById("snippets");
      container.innerHTML = "";
      if (!snippets || snippets.length === 0) {
        container.textContent = "No evidence retrieved yet.";
        return;
      }
      for (const snippet of snippets) {
        const div = document.createElement("div");
        div.className = "snippet";
        div.innerHTML = `<strong>${snippet.citation}</strong><p>${snippet.text}</p>`;
        container.appendChild(div);
      }
    }

    async function ingest() {
      const file = document.getElementById("file").files[0];
      if (!file) {
        setStatus("intakeStatus", "Choose a PDF or image first.", true);
        return;
      }
      const form = new FormData();
      form.append("file", file);
      setStatus("intakeStatus", "Uploading and indexing...");
      const response = await fetch("/ingest", { method: "POST", body: form });
      const payload = await response.json();
      setRaw(payload);
      setStatus("intakeStatus", response.ok ? "Indexed document." : payload.detail, !response.ok);
    }

    async function retrieve() {
      const query = document.getElementById("query").value;
      setStatus("draftStatus", "Retrieving evidence...");
      const response = await fetch("/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, k: 5 })
      });
      const payload = await response.json();
      renderSnippets(payload.snippets);
      setRaw(payload);
      setStatus(
        "draftStatus",
        response.ok ? "Evidence retrieved." : "Retrieve failed.",
        !response.ok
      );
    }

    async function draft() {
      const query = document.getElementById("query").value;
      const matter_id = document.getElementById("matter").value;
      setStatus("draftStatus", "Generating grounded memo...");
      const response = await fetch("/draft", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ matter_id, query, k: 5 })
      });
      const payload = await response.json();
      lastDraft = payload.memo || "";
      document.getElementById("memo").textContent = lastDraft || "No memo returned.";
      renderSnippets(payload.snippets);
      setRaw(payload);
      setStatus("draftStatus", response.ok ? "Memo generated." : "Draft failed.", !response.ok);
    }

    async function saveFeedback() {
      if (!lastDraft) {
        setStatus("feedbackStatus", "Generate a draft before saving feedback.", true);
        return;
      }
      const payload = {
        matter_id: document.getElementById("matter").value,
        query: document.getElementById("query").value,
        draft_text: lastDraft,
        final_text: document.getElementById("finalText").value
      };
      const response = await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const body = await response.json();
      setRaw(body);
      setStatus(
        "feedbackStatus",
        response.ok ? "Feedback memory saved." : "Feedback failed.",
        !response.ok
      );
    }

    async function runDemo() {
      setStatus("intakeStatus", "Running full demo...");
      const response = await fetch("/demo", { method: "POST" });
      const payload = await response.json();
      lastDraft = payload.improved_draft || payload.first_draft || "";
      document.getElementById("memo").textContent = lastDraft || "No memo returned.";
      renderSnippets(payload.retrieved_snippets);
      setRaw(payload.evaluation);
      setStatus("intakeStatus", response.ok ? "Demo complete." : "Demo failed.", !response.ok);
    }

    renderSnippets([]);
  </script>
</body>
</html>
"""
