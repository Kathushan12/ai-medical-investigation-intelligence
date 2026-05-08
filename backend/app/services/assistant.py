from openai import OpenAI
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Investigation
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
        normalized_question = question.lower().strip()

        if self._is_list_all_patients_question(normalized_question):
            return self._list_all_patients(db)

        if self._is_list_all_investigations_question(normalized_question):
            return self._list_all_investigations(db)

        results = hybrid_search(
            db=db,
            query_text=question,
            limit=top_k,
            filters=filters,
        )

        if not results:
            return AssistantResponse(
                answer="No supporting evidence found.",
                citations=[],
                retrieved_results=[],
            )

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
            return AssistantResponse(
                answer=answer,
                citations=citations,
                retrieved_results=results,
            )

        context = "\n\n".join(
            f"[{citation.source_id}] "
            f"File: {citation.original_filename} | "
            f"Investigation: {citation.investigation_id} | "
            f"Chunk: {citation.chunk_index} | "
            f"Page: {citation.source_page or 'unknown'}\n"
            f"{result.content}"
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
5. Never invent patient names, hospitals, dates, severity levels, or diagnoses.

Question:
{question}

Retrieved evidence:
{context}
""".strip()

        response = self.client.chat.completions.create(
            model=settings.openai_text_model,
            messages=[
                {
                    "role": "system",
                    "content": "You answer only from retrieved evidence and always provide citations.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content or ""
        answer = answer.strip()

        if not answer:
            answer = "No supporting evidence found."

        return AssistantResponse(
            answer=answer,
            citations=citations,
            retrieved_results=results,
        )

    @staticmethod
    def _is_list_all_patients_question(question: str) -> bool:
        patterns = [
            "list all patients",
            "show all patients",
            "all patients",
            "patient list",
            "list patients",
            "show patients",
            "what are the patients",
            "who are the patients",
        ]

        return any(pattern in question for pattern in patterns)

    @staticmethod
    def _is_list_all_investigations_question(question: str) -> bool:
        patterns = [
            "list all investigations",
            "show all investigations",
            "all investigations",
            "list all reports",
            "show all reports",
            "all reports",
            "list all records",
            "show all records",
        ]

        return any(pattern in question for pattern in patterns)

    @staticmethod
    def _list_all_patients(db: Session) -> AssistantResponse:
        investigations = (
            db.query(Investigation)
            .order_by(Investigation.created_at.desc())
            .all()
        )

        if not investigations:
            return AssistantResponse(
                answer="No patient records found.",
                citations=[],
                retrieved_results=[],
            )

        lines = ["Patient records found in the system:\n"]

        for index, investigation in enumerate(investigations, start=1):
            patient_name = investigation.patient_name or "Unknown patient"
            condition = investigation.medical_condition or "Condition not available"
            location = investigation.hospital_location or "Location not available"
            severity = investigation.severity_level or "Unknown"
            filename = investigation.original_filename or "Unknown file"
            status = investigation.processing_status or "Unknown"

            lines.append(
                f"{index}. {patient_name} | "
                f"Condition: {condition} | "
                f"Location: {location} | "
                f"Severity: {severity} | "
                f"Status: {status} | "
                f"Source file: {filename}"
            )

        answer = "\n".join(lines)

        return AssistantResponse(
            answer=answer,
            citations=[],
            retrieved_results=[],
        )

    @staticmethod
    def _list_all_investigations(db: Session) -> AssistantResponse:
        investigations = (
            db.query(Investigation)
            .order_by(Investigation.created_at.desc())
            .all()
        )

        if not investigations:
            return AssistantResponse(
                answer="No investigation records found.",
                citations=[],
                retrieved_results=[],
            )

        lines = ["Investigation records found in the system:\n"]

        for index, investigation in enumerate(investigations, start=1):
            filename = investigation.original_filename or "Unknown file"
            patient_name = investigation.patient_name or "Unknown patient"
            condition = investigation.medical_condition or "Condition not available"
            location = investigation.hospital_location or "Location not available"
            severity = investigation.severity_level or "Unknown"
            status = investigation.processing_status or "Unknown"

            lines.append(
                f"{index}. {filename} | "
                f"Patient: {patient_name} | "
                f"Condition: {condition} | "
                f"Location: {location} | "
                f"Severity: {severity} | "
                f"Status: {status}"
            )

        answer = "\n".join(lines)

        return AssistantResponse(
            answer=answer,
            citations=[],
            retrieved_results=[],
        )

    @staticmethod
    def _fallback_answer(question: str, citations: list[Citation]) -> str:
        if not citations:
            return "No supporting evidence found."

        joined = "\n".join(
            f"[{citation.source_id}] {citation.excerpt}"
            for citation in citations
        )

        return (
            "OPENAI_API_KEY is not configured, so I cannot generate a final AI response. "
            "Retrieved supporting evidence is shown below.\n\n"
            f"Question: {question}\n\n"
            f"{joined}"
        )


assistant_service = InvestigationAssistantService()