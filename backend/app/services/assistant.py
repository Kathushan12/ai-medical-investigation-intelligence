from openai import OpenAI
from sqlalchemy.orm import Session

from app.config import settings
from app.schemas import AssistantResponse, Citation
from app.services.search import hybrid_search


class InvestigationAssistantService:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def answer(
        self,
        db: Session,
        question: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> AssistantResponse:
        results = hybrid_search(db=db, query_text=question, limit=top_k, filters=filters)
        if not results:
            return AssistantResponse(answer="No supporting evidence found.", citations=[], retrieved_results=[])

        citations = [
            Citation(
                source_id=f"S{index}",
                investigation_id=result.investigation_id,
                chunk_id=result.chunk_id,
                original_filename=result.original_filename,
                chunk_index=result.chunk_index,
                source_page=result.source_page,
                excerpt=result.content[:350],
            )
            for index, result in enumerate(results, start=1)
        ]

        if not self.client:
            answer = self._fallback_answer(question, citations)
            return AssistantResponse(answer=answer, citations=citations, retrieved_results=results)

        context = "\n\n".join(
            f"[{citation.source_id}] File: {citation.original_filename} | "
            f"Investigation: {citation.investigation_id} | Chunk: {citation.chunk_index} | "
            f"Page: {citation.source_page or 'unknown'}\n{result.content}"
            for citation, result in zip(citations, results)
        )
        prompt = f"""
You are an AI investigation assistant for medical investigation evidence.
Answer the user's question using ONLY the retrieved evidence below.
Rules:
1. If the answer is not supported, return exactly: No supporting evidence found.
2. Do not use outside knowledge.
3. Cite every factual claim with source IDs like [S1] or [S2].
4. Be concise and investigation-focused.

Question: {question}

Retrieved evidence:
{context}
""".strip()
        response = self.client.responses.create(model=settings.openai_text_model, input=prompt)
        answer = getattr(response, "output_text", "").strip()
        if not answer:
            answer = "No supporting evidence found."
        return AssistantResponse(answer=answer, citations=citations, retrieved_results=results)

    @staticmethod
    def _fallback_answer(question: str, citations: list[Citation]) -> str:
        if not citations:
            return "No supporting evidence found."
        joined = "\n".join(f"[{c.source_id}] {c.excerpt}" for c in citations)
        return (
            "OPENAI_API_KEY is not configured, so I cannot generate a final AI response. "
            "Retrieved supporting evidence is shown below.\n\n"
            f"Question: {question}\n\n{joined}"
        )


assistant_service = InvestigationAssistantService()
