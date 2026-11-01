"""The RAG pipeline: rewrite, retrieve, answer, cite."""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError

from app.config import Settings
from app.llm import ChatMessage, LLMClient, LLMError, StreamResult
from app.logging import get_logger
from app.rag.cache import AnswerCache
from app.rag.guardrails import allowed_contacts, limit_context, redact_stream
from app.rag.prompts import FOLLOWUP_SYSTEM, REWRITE_SYSTEM, SYSTEM_PROMPT, build_prompt
from app.rag.schemas import (
    NOT_FOUND,
    Citation,
    Confidence,
    FinalEvent,
    RagAnswer,
    RagRequest,
    TokenEvent,
    UserProfile,
)
from app.vectorstore import HybridRetriever, RetrievalResult, ScoredChunk, SearchFilters

log = get_logger(__name__)

HISTORY_MESSAGES = 6
SNIPPET_CHARS = 280
HIGH_CONFIDENCE_SIMILARITY = 0.7

_FOLLOW_UP = re.compile(
    r"\b(it|its|that|this|those|these|they|them|there|he|she)\b"
    r"|\bthe\s+(first|second|third|last|former|latter|other|same)\b"
    r"|^\s*(what|how)\s+about\b|^\s*and\b|\bone\b",
    re.IGNORECASE,
)
_CITATION = re.compile(r"\[(\d+)\]")


class _Followups(BaseModel):
    """The shape of the follow-up suggestions reply."""

    questions: list[str] = Field(default_factory=list)


@dataclass
class _Retrieved:
    """Chunks to answer from, and how well they match."""

    chunks: list[ScoredChunk]
    relevant_count: int
    best_similarity: float


def depends_on_history(question: str) -> bool:
    """Return True if a question cannot be understood without the conversation."""
    return len(question.split()) <= 3 or _FOLLOW_UP.search(question) is not None


class RagPipeline:
    """Answers questions from retrieved documents."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: HybridRetriever,
        cache: AnswerCache,
        settings: Settings,
    ) -> None:
        """Keep the collaborators."""
        self._llm = llm
        self._retriever = retriever
        self._cache = cache
        self._settings = settings

    def rewrite(self, question: str, history: tuple[ChatMessage, ...]) -> str:
        """Make a follow-up standalone. If the LLM fails, use the question as it is."""
        recent = history[-HISTORY_MESSAGES:]
        if not recent or not depends_on_history(question):
            return question
        transcript = "\n".join(f"{message.role}: {message.content}" for message in recent)
        prompt = f"Conversation:\n{transcript}\n\nLatest question: {question}"
        try:
            reply = self._llm.complete(
                [ChatMessage("user", prompt)], system=REWRITE_SYSTEM, max_tokens=120
            )
        except LLMError as exc:
            log.warning("query_rewrite_failed", error=str(exc))
            return question
        return reply.text.strip().strip('"') or question

    def _summarize(self, result: RetrievalResult) -> _Retrieved:
        """Count the chunks whose similarity clears the threshold."""
        scores = [result.similarities.get(chunk.chunk_id, 0.0) for chunk in result.chunks]
        threshold = self._settings.rag_similarity_threshold
        return _Retrieved(
            chunks=result.chunks,
            relevant_count=sum(score >= threshold for score in scores),
            best_similarity=max(scores, default=0.0),
        )

    def retrieve(self, question: str, filters: SearchFilters) -> _Retrieved:
        """Search with the user's filters, then widen if too few good results came back."""
        top_k = self._settings.rag_top_k
        found = self._summarize(self._retriever.retrieve(question, k=top_k, filters=filters))
        if found.relevant_count >= self._settings.rag_min_results:
            return found
        if filters == SearchFilters():
            return found

        widened = self._summarize(self._retriever.retrieve(question, k=top_k))
        log.info(
            "retrieval_filters_dropped",
            filtered=found.relevant_count,
            unfiltered=widened.relevant_count,
        )
        return widened if widened.relevant_count > found.relevant_count else found

    @staticmethod
    def _citations(answer: str, chunks: list[ScoredChunk]) -> list[Citation]:
        """Turn the [n] markers in an answer into citations. Unknown numbers are ignored."""
        citations = []
        seen: set[int] = set()
        for marker in _CITATION.findall(answer):
            number = int(marker)
            if number in seen or not 1 <= number <= len(chunks):
                continue
            seen.add(number)
            chunk = chunks[number - 1]
            citations.append(
                Citation(
                    number=number,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    page=chunk.page_number,
                    snippet=chunk.text[:SNIPPET_CHARS],
                )
            )
        return citations

    @staticmethod
    def _confidence(citations: list[Citation], found: _Retrieved) -> Confidence:
        """Grade an answer from its citations and retrieval strength."""
        if not citations:
            return "low"
        strong = found.best_similarity >= HIGH_CONFIDENCE_SIMILARITY
        return "high" if strong and found.relevant_count >= 2 else "medium"

    def _followups(self, question: str, answer: str) -> list[str]:
        """Suggest next questions. This is a nicety, so failures just return nothing."""
        if not self._settings.rag_followups_enabled:
            return []
        prompt = f"Question: {question}\n\nAnswer: {answer}"
        try:
            parsed, _ = self._llm.complete_json(
                [ChatMessage("user", prompt)], _Followups, system=FOLLOWUP_SYSTEM, max_tokens=150
            )
        except (LLMError, ValidationError) as exc:
            log.info("followups_skipped", error=str(exc))
            return []
        return [item.strip() for item in parsed.questions if item.strip()][:3]

    def _not_found(self, standalone: str) -> Iterator[TokenEvent | FinalEvent]:
        """Reply that nothing was found, without spending an LLM call."""
        yield TokenEvent(NOT_FOUND)
        yield FinalEvent(RagAnswer(answer=NOT_FOUND, standalone_question=standalone))

    def stream(self, request: RagRequest) -> Iterator[TokenEvent | FinalEvent]:
        """Yield answer tokens, then one FinalEvent."""
        standalone = self.rewrite(request.question, request.history)
        filters = _filters_for(request.profile)

        cached = self._cache.get(standalone, filters)
        if cached is not None:
            yield TokenEvent(cached.answer)
            yield FinalEvent(cached)
            return

        found = self.retrieve(standalone, filters)
        chunks = limit_context(found.chunks, self._settings.rag_context_tokens)
        if found.relevant_count == 0 or not chunks:
            yield from self._not_found(standalone)
            return

        usage = StreamResult()
        fragments = self._llm.stream(
            [ChatMessage("user", build_prompt(standalone, chunks))],
            system=SYSTEM_PROMPT,
            result=usage,
        )
        parts = []
        for text in redact_stream(fragments, allowed_contacts(chunks)):
            parts.append(text)
            yield TokenEvent(text)
        answer_text = "".join(parts).strip()

        if NOT_FOUND.lower() in answer_text.lower():
            answer = RagAnswer(
                answer=NOT_FOUND,
                standalone_question=standalone,
                model=usage.model,
                tokens_in=usage.tokens_in,
                tokens_out=usage.tokens_out,
            )
            yield FinalEvent(answer)
            return

        citations = self._citations(answer_text, chunks)
        answer = RagAnswer(
            answer=answer_text,
            citations=citations,
            confidence=self._confidence(citations, found),
            suggested_followups=self._followups(standalone, answer_text),
            answered=True,
            standalone_question=standalone,
            model=usage.model,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
        )
        self._cache.set(standalone, filters, answer)
        yield FinalEvent(answer)

    def answer(self, request: RagRequest) -> RagAnswer:
        """Run the pipeline to completion."""
        final: RagAnswer | None = None
        for event in self.stream(request):
            if isinstance(event, FinalEvent):
                final = event.answer
        if final is None:
            raise LLMError("The RAG pipeline ended without an answer")
        return final


def _filters_for(profile: UserProfile) -> SearchFilters:
    """Build search filters from a user's profile."""
    return SearchFilters(department=profile.department, semester=profile.semester)
