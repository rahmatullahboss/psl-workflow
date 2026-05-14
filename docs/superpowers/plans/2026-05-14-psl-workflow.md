# Pearson Specter Litt Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a ready-to-ship internal legal workflow that ingests messy documents, retrieves cited context, drafts grounded internal memos, and learns from operator edits.

**Architecture:** The system is a Python package with four bounded modules: `processing`, `retrieval`, `generation`, and `learning`. A CLI in `main.py` and a FastAPI app expose the same services, while sample generation and evaluation scripts prove the end-to-end flow without requiring an OpenAI key.

**Tech Stack:** Python 3.11/3.12, PyMuPDF, Tesseract/EasyOCR hooks, LangChain with ChromaDB, OpenAI GPT-4-class chat model, FastAPI, pytest.

---

## File Structure

- `src/psl_workflow/processing/`: PDF/image ingestion, OCR fallback, normalized document models, synthetic sample generation.
- `src/psl_workflow/retrieval/`: legal text chunking, deterministic local embeddings, Chroma-backed retrieval with citations.
- `src/psl_workflow/generation/`: grounded memo prompt construction, OpenAI client adapter, deterministic local fallback for demos/tests.
- `src/psl_workflow/learning/`: operator edit capture, correction/style memory extraction, SQLite-backed future prompt context.
- `src/psl_workflow/api.py`: FastAPI demo endpoints.
- `main.py`: CLI entry point for generate, ingest, retrieve, draft, feedback, and demo.
- `tests/`: behavior-first tests for each required module.

## Tasks

- [ ] Create tests for the four architecture requirements and run them red.
- [ ] Implement processing models, PyMuPDF extraction, OCR fallback hooks, JSON output, and sample document generation.
- [ ] Implement LangChain/Chroma retrieval with source snippets and deterministic embeddings.
- [ ] Implement grounded memo generation with strict citation prompt and no-key local fallback.
- [ ] Implement feedback learning that extracts correction/style rules from edits and injects them into future prompts.
- [ ] Add CLI and FastAPI surfaces for the full workflow.
- [ ] Generate sample inputs and evaluation outputs.
- [ ] Write README, run tests, run a demo, and initialize a clean git repo if needed.
