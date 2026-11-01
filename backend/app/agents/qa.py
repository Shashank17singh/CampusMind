"""The question-answering agent, a thin wrapper around the RAG pipeline."""

from app.agents.state import Handler, HandlerInput, HandlerResult
from app.rag.pipeline import RagPipeline
from app.rag.schemas import FinalEvent, RagAnswer, RagRequest, TokenEvent, UserProfile


def make_qa_handler(pipeline: RagPipeline) -> Handler:
    """Build a handler that streams RAG tokens and returns citations as data."""

    def handle(request: HandlerInput) -> HandlerResult:
        rag_request = RagRequest(
            question=request.message,
            profile=UserProfile(request.user.department, request.user.semester),
            history=request.history,
        )
        final: RagAnswer | None = None
        for event in pipeline.stream(rag_request):
            if isinstance(event, TokenEvent):
                request.emit(event.text)
            elif isinstance(event, FinalEvent):
                final = event.answer
        if final is None:
            raise RuntimeError("The RAG pipeline ended without an answer")
        return HandlerResult(
            text=final.answer,
            data={"rag": final.model_dump(mode="json")},
            tokens_in=final.tokens_in,
            tokens_out=final.tokens_out,
        )

    return handle
