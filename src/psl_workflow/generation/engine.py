import json
import os
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol
from urllib import request
from urllib.error import URLError

from psl_workflow.learning.memory import StyleMemory
from psl_workflow.retrieval.models import RetrievedSnippet


class DraftLLM(Protocol):
    def generate_memo(
        self,
        *,
        question: str,
        snippets: list[RetrievedSnippet],
        memory_context: str,
        prompt: str,
    ) -> str: ...


@dataclass(slots=True)
class GroundedDraftingEngine:
    llm: DraftLLM
    style_memory: StyleMemory

    def generate_memo(
        self,
        matter_id: str,
        question: str,
        snippets: list[RetrievedSnippet],
    ) -> str:
        if not snippets:
            return (
                "# Internal Legal Memo\n\n"
                f"**Matter:** {matter_id}\n\n"
                "Insufficient sourced evidence to draft a grounded memo. "
                "Add or retrieve source passages before generating legal analysis."
            )

        memory_context = self.style_memory.build_prompt_context(question)
        prompt = build_grounded_prompt(question, snippets, memory_context)
        return self.llm.generate_memo(
            question=question,
            snippets=snippets,
            memory_context=memory_context,
            prompt=prompt,
        )


class LocalDraftLLM:
    """Deterministic fallback used for tests and no-key demos."""

    def generate_memo(
        self,
        *,
        question: str,
        snippets: list[RetrievedSnippet],
        memory_context: str,
        prompt: str,
    ) -> str:
        del prompt
        facts = "\n".join(
            f"- {snippet.text} [{snippet.citation}]" for snippet in snippets[:4]
        )
        recommendation_heading = (
            "Recommendation" if "Recommendation" in memory_context else "Conclusion"
        )
        cited_basis = "; ".join(f"[{snippet.citation}]" for snippet in snippets[:3])
        return (
            "# Internal Legal Memo\n\n"
            f"## Issue\n{question}\n\n"
            f"## Relevant Source Evidence\n{facts}\n\n"
            "## Analysis\n"
            "The available source passages support only the facts listed above. "
            f"Any legal conclusion should stay limited to those cited passages {cited_basis}.\n\n"
            f"## {recommendation_heading}\n"
            "Proceed on a notice-first basis and flag unsupported questions for attorney review "
            f"unless additional evidence is retrieved {cited_basis}."
        )


class OpenAIDraftLLM:
    """OpenAI GPT-4-class adapter. Requires OPENAI_API_KEY."""

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_memo(
        self,
        *,
        question: str,
        snippets: list[RetrievedSnippet],
        memory_context: str,
        prompt: str,
    ) -> str:
        del question, snippets, memory_context
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You draft internal legal memos for Pearson Specter Litt. "
                        "You must be precise, conservative, and grounded in cited evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        return content or "Insufficient sourced evidence to draft a grounded memo."


class OllamaDraftLLM:
    """Ollama adapter for local Ollama or Ollama Cloud."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
        ).rstrip("/")
        self.api_key = os.getenv("OLLAMA_API_KEY")

    def generate_memo(
        self,
        *,
        question: str,
        snippets: list[RetrievedSnippet],
        memory_context: str,
        prompt: str,
    ) -> str:
        del question, snippets, memory_context
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": (
                    "You draft internal legal memos for Pearson Specter Litt. "
                    "Only use cited evidence.\n\n"
                    f"{prompt}"
                ),
                "stream": False,
                "options": {"temperature": 0.2},
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        http_request = request.Request(
            self._endpoint("generate"),
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        return body.get("response") or "Insufficient sourced evidence to draft a grounded memo."

    def _endpoint(self, endpoint: str) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/api"):
            return f"{base}/{endpoint.lstrip('/')}"
        path = PurePosixPath("/api") / endpoint.lstrip("/")
        return f"{base}{path}"


def build_default_llm() -> DraftLLM:
    provider = os.getenv("LLM_PROVIDER", "").lower().strip()
    if provider == "ollama":
        return OllamaDraftLLM()
    if provider == "openai" or os.getenv("OPENAI_API_KEY"):
        return OpenAIDraftLLM()
    return LocalDraftLLM()


def build_grounded_prompt(
    question: str,
    snippets: list[RetrievedSnippet],
    memory_context: str = "",
) -> str:
    evidence = "\n\n".join(
        f"[{snippet.citation}]\n{snippet.text}" for snippet in snippets
    )
    memory = f"\n\n{memory_context}" if memory_context else ""
    return f"""Draft an Internal Legal Memo for Pearson Specter Litt.

Question:
{question}

Evidence:
{evidence}{memory}

Grounding rules:
- Use only the Evidence passages above for factual claims.
- Every factual claim must include a source citation in square brackets, e.g. [contract.pdf p.2].
- If a requested conclusion is not supported by the evidence, say
  "Insufficient sourced evidence" and identify what evidence is missing.
- Do not invent parties, dates, duties, damages, statutes, or procedural posture.
- Use a professional internal memo structure: Issue, Relevant Source Evidence,
  Analysis, Recommendation, Open Questions.
- Apply the Style and Correction Memory when it does not conflict with the evidence.
"""
