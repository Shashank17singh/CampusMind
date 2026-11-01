"""The agent graph.

    router -> (clarify | agent -> agent ...) -> finalize

The router queues one or more intents. Each agent node takes the next intent off the
queue, sees earlier steps' results in `tool_results`, and hands on to the next agent or to
`finalize`. Every node runs inside a wrapper that enforces the step limit and timeout,
records a trace row, and turns any exception into a friendly message.
"""

import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from app.agents.router import CONFIDENCE_THRESHOLD, classify
from app.agents.state import INTENTS, AgentState, Handler, HandlerInput, HandlerResult, UserContext
from app.agents.tracing import TraceRecorder
from app.llm import ChatMessage, LLMClient
from app.logging import get_logger

log = get_logger(__name__)

MAX_GRAPH_STEPS = 6
NODE_TIMEOUT_SECONDS = 90.0
CLARIFY_MESSAGE = (
    "I want to be sure I help with the right thing. Could you say a bit more? I can answer "
    "questions from college documents, build a study plan, set reminders, or summarize notes."
)
SMALLTALK_REPLY = (
    "Hello! I'm CampusMind. I can answer questions from official college documents, plan "
    "your studies, set reminders and summarize notes. What would you like to do?"
)
OUT_OF_SCOPE_REPLY = (
    "I can only help with college matters: official documents, study plans, reminders and "
    "summaries."
)
STEP_LIMIT_MESSAGE = "That request needed too many steps, so I stopped. Try one thing at a time."

_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="agent-node")


def _friendly_error(trace_id: uuid.UUID) -> str:
    """Build the message shown when something fails. The id lets support find the logs."""
    return f"Sorry, something went wrong. Please try again. (Reference: {trace_id})"


@dataclass
class RunContext:
    """Per-request values, passed to nodes through the LangGraph config."""

    trace_id: uuid.UUID
    emit: Callable[[str], None] = lambda _: None
    step: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    timeout: float = NODE_TIMEOUT_SECONDS
    max_steps: int = MAX_GRAPH_STEPS


@dataclass
class NodeResult:
    """A node's state update and its token usage."""

    update: dict[str, Any]
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class AgentOutcome:
    """The result of a whole request."""

    text: str
    trace_id: uuid.UUID
    intent: str | None = None
    confidence: float = 0.0
    error: str | None = None
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0


NodeFunction = Callable[[AgentState, RunContext], NodeResult]


def _smalltalk(_: HandlerInput) -> HandlerResult:
    """Reply to greetings without spending an LLM call."""
    return HandlerResult(SMALLTALK_REPLY)


def _out_of_scope(_: HandlerInput) -> HandlerResult:
    """Decline politely without spending an LLM call."""
    return HandlerResult(OUT_OF_SCOPE_REPLY)


def _next_after_agent(state: AgentState) -> str:
    """Go to the next queued agent, or finish."""
    if state.get("error") or not state.get("queue"):
        return "finalize"
    return state["queue"][0]


def _after_router(state: AgentState) -> str:
    """Send a confident decision to its agent, and anything else to `clarify`."""
    if state.get("error"):
        return "finalize"
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        return "clarify"
    return state["queue"][0]


class AgentRunner:
    """Builds the graph once, and runs it per request."""

    def __init__(
        self, llm: LLMClient, handlers: dict[str, Handler], recorder: TraceRecorder
    ) -> None:
        """Compile the graph. Any intent without a handler reports an error when reached."""
        self._llm = llm
        self._handlers = {"smalltalk": _smalltalk, "out_of_scope": _out_of_scope, **handlers}
        self._recorder = recorder
        self._graph = self._build()

    def _wrap(self, name: str, function: NodeFunction) -> Callable[..., dict[str, Any]]:
        """Add the step limit, timeout, tracing and error handling to a node."""

        def node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
            context: RunContext = config["configurable"]["context"]
            context.step += 1
            step = context.step
            if step > context.max_steps and name != "finalize":
                log.warning("step_limit_reached", trace_id=str(context.trace_id), node=name)
                return {"error": STEP_LIMIT_MESSAGE, "queue": []}

            started = time.perf_counter()
            error: str | None = None
            try:
                future = _executor.submit(function, state, context)
                result = future.result(timeout=context.timeout)
            except FutureTimeout:
                # The worker thread cannot be killed. It ends when its LLM call times out.
                error = f"Node timed out after {context.timeout}s"
                result = NodeResult({"error": _friendly_error(context.trace_id), "queue": []})
                log.error("node_timeout", trace_id=str(context.trace_id), node=name)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                result = NodeResult({"error": _friendly_error(context.trace_id), "queue": []})
                log.exception("node_failed", trace_id=str(context.trace_id), node=name)

            context.tokens_in += result.tokens_in
            context.tokens_out += result.tokens_out
            self._recorder.record(
                context.trace_id,
                step,
                name,
                {key: value for key, value in state.items() if key != "user"},
                result.update,
                int((time.perf_counter() - started) * 1000),
                result.tokens_in,
                result.tokens_out,
                error,
            )
            return result.update

        return node

    def _router_node(self, state: AgentState, _: RunContext) -> NodeResult:
        """Classify the latest message and queue its intents."""
        route = classify(state["messages"][-1].content, self._llm)
        return NodeResult(
            {
                "intent": route.intents[0],
                "confidence": route.confidence,
                "queue": list(route.intents),
            },
            route.tokens_in,
            route.tokens_out,
        )

    def _agent_node(self, intent: str) -> NodeFunction:
        """Build the node that runs one intent's handler."""

        def run(state: AgentState, context: RunContext) -> NodeResult:
            handler = self._handlers.get(intent)
            if handler is None:
                raise LookupError(f"No handler registered for intent {intent}")
            messages = state["messages"]
            result = handler(
                HandlerInput(
                    message=messages[-1].content,
                    history=tuple(messages[:-1]),
                    user=state["user"],
                    previous=tuple(state.get("tool_results", [])),
                    emit=context.emit,
                )
            )
            return NodeResult(
                {
                    "queue": state["queue"][1:],
                    "tool_results": [{"intent": intent, "text": result.text, **result.data}],
                },
                result.tokens_in,
                result.tokens_out,
            )

        return run

    @staticmethod
    def _clarify_node(_: AgentState, context: RunContext) -> NodeResult:
        """Ask the user what they meant."""
        context.emit(CLARIFY_MESSAGE)
        return NodeResult({"final_answer": CLARIFY_MESSAGE, "queue": []})

    @staticmethod
    def _finalize_node(state: AgentState, _: RunContext) -> NodeResult:
        """Combine every step's text into the final answer."""
        if state.get("final_answer"):
            return NodeResult({})
        parts = [item["text"] for item in state.get("tool_results", [])]
        if state.get("error"):
            parts.append(str(state["error"]))
        return NodeResult({"final_answer": "\n\n".join(parts)})

    def _build(self) -> Any:
        """Assemble and compile the graph."""
        graph = StateGraph(AgentState)
        graph.add_node("router", self._wrap("router", self._router_node))
        graph.add_node("clarify", self._wrap("clarify", self._clarify_node))
        graph.add_node("finalize", self._wrap("finalize", self._finalize_node))
        for intent in INTENTS:
            graph.add_node(intent, self._wrap(intent, self._agent_node(intent)))
            graph.add_conditional_edges(
                intent, _next_after_agent, {**{i: i for i in INTENTS}, "finalize": "finalize"}
            )

        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            _after_router,
            {**{i: i for i in INTENTS}, "clarify": "clarify", "finalize": "finalize"},
        )
        graph.add_edge("clarify", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def run(
        self,
        user: UserContext,
        messages: list[ChatMessage],
        trace_id: uuid.UUID | None = None,
        emit: Callable[[str], None] = lambda _: None,
        timeout: float = NODE_TIMEOUT_SECONDS,
        max_steps: int = MAX_GRAPH_STEPS,
    ) -> AgentOutcome:
        """Run one request to completion. Never raises, because failures become messages."""
        context = RunContext(
            trace_id or uuid.uuid4(), emit=emit, timeout=timeout, max_steps=max_steps
        )
        config: RunnableConfig = {
            "configurable": {"context": context},
            # Allows the limit to be reported nicely by the wrapper, before LangGraph's own.
            "recursion_limit": max_steps + 3,
        }
        try:
            state = self._graph.invoke({"messages": messages, "user": user}, config)
        except GraphRecursionError:
            log.error("graph_recursion_limit", trace_id=str(context.trace_id))
            return AgentOutcome(STEP_LIMIT_MESSAGE, context.trace_id, error=STEP_LIMIT_MESSAGE)
        except Exception:
            log.exception("graph_failed", trace_id=str(context.trace_id))
            message = _friendly_error(context.trace_id)
            return AgentOutcome(message, context.trace_id, error=message)

        return AgentOutcome(
            text=state.get("final_answer", ""),
            trace_id=context.trace_id,
            intent=state.get("intent"),
            confidence=state.get("confidence", 0.0),
            error=state.get("error"),
            tool_results=state.get("tool_results", []),
            tokens_in=context.tokens_in,
            tokens_out=context.tokens_out,
        )

    def stream(
        self,
        user: UserContext,
        messages: list[ChatMessage],
        trace_id: uuid.UUID | None = None,
    ) -> Iterator[str | AgentOutcome]:
        """Yield text fragments as agents emit them, then the AgentOutcome.

        The graph runs on a worker thread, and fragments cross to the caller through a
        queue. This keeps agents free of any knowledge of how they are being consumed.
        """
        fragments: queue.Queue[str | AgentOutcome] = queue.Queue()

        def work() -> None:
            fragments.put(self.run(user, messages, trace_id, emit=fragments.put))

        threading.Thread(target=work, daemon=True, name="agent-run").start()
        while True:
            item = fragments.get()
            yield item
            if isinstance(item, AgentOutcome):
                return
