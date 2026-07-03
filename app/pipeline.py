"""The single analysis pipeline shared by both the synchronous and the
streaming API routes: input guardrails -> agent -> output guardrails.
Keeping this in one place means both endpoints can never drift apart.
"""
from collections.abc import Generator

from langchain_core.messages import AIMessage

from app.agent.graph import get_agent
from app.errors import GuardrailViolation, UpstreamServiceError
from app.guardrails.input_guard import run_input_guardrails
from app.guardrails.output_guard import apply_output_guardrails
from app.guardrails.policies import DISCLAIMER_TEXT, MALFORMED_OUTPUT_MARKERS, MAX_AGENT_RETRIES
from app.logging_config import get_logger, log_event
from app.schemas import AnalyzeResponse

logger = get_logger("pipeline")


def _is_malformed(text: str) -> bool:
    """True if the model leaked internal chat-template markup instead of a
    clean answer — see MALFORMED_OUTPUT_MARKERS for why this can happen."""
    return any(marker in (text or "") for marker in MALFORMED_OUTPUT_MARKERS)


def _invoke_agent_with_retries(query: str, thread_id: str, trace_id: str):
    """Runs the agent, and if the model returns malformed/hallucinated output
    (no real tool_calls, garbage chat-template text as content), retries a
    fresh invocation up to MAX_AGENT_RETRIES times before giving up."""
    agent = get_agent()
    last_messages = None
    for attempt in range(1, MAX_AGENT_RETRIES + 1):
        try:
            result = agent.invoke(
                {"messages": [("user", query)]},
                config={"configurable": {"thread_id": f"{thread_id}-r{attempt}" if attempt > 1 else thread_id}},
            )
        except Exception as exc:
            log_event(logger, "error", "Agent invocation failed", trace_id, error=str(exc), attempt=attempt)
            raise UpstreamServiceError() from exc

        last_messages = result["messages"]
        final_text = last_messages[-1].content or ""
        if not _is_malformed(final_text):
            if attempt > 1:
                log_event(logger, "info", "Malformed output resolved after retry", trace_id, attempt=attempt)
            return last_messages

        log_event(
            logger, "warning", "Model returned malformed/hallucinated output — retrying",
            trace_id, attempt=attempt, preview=final_text[:200],
        )

    log_event(logger, "error", "Model kept returning malformed output after retries", trace_id)
    raise UpstreamServiceError(safe_message=(
        "The analysis engine returned an unexpected response after multiple attempts. "
        "Please try again in a moment."
    ))


def _build_response(trace_id: str, query: str, answer: str, tools_used: list[str],
                     guardrail_events: list[dict], status: str) -> AnalyzeResponse:
    return AnalyzeResponse(
        trace_id=trace_id,
        query=query,
        answer=answer,
        tools_used=tools_used,
        guardrail_events=guardrail_events,
        disclaimer=DISCLAIMER_TEXT,
        status=status,
    )


def run_pipeline(query: str, thread_id: str, trace_id: str) -> AnalyzeResponse:
    try:
        run_input_guardrails(query, trace_id)
    except GuardrailViolation as violation:
        return _build_response(
            trace_id, query, violation.safe_message, [],
            [{"stage": violation.stage, "result": "blocked", "detail": violation.category}],
            status="blocked",
        )

    messages = _invoke_agent_with_retries(query, thread_id, trace_id)
    tools_used = [tc["name"] for m in messages if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
    raw_answer = messages[-1].content or ""

    sanitized, out_events = apply_output_guardrails(raw_answer, trace_id)
    return _build_response(trace_id, query, sanitized, tools_used, out_events, status="ok")


def stream_pipeline(query: str, thread_id: str, trace_id: str) -> Generator[dict, None, None]:
    """Yields small JSON-able dicts describing each step, ending with a
    'final' event carrying the same AnalyzeResponse shape as run_pipeline."""
    yield {"type": "status", "message": "Checking request safety and scope..."}
    try:
        run_input_guardrails(query, trace_id)
    except GuardrailViolation as violation:
        yield {"type": "blocked", "stage": violation.stage, "message": violation.safe_message}
        response = _build_response(
            trace_id, query, violation.safe_message, [],
            [{"stage": violation.stage, "result": "blocked", "detail": violation.category}],
            status="blocked",
        )
        yield {"type": "final", "data": response.model_dump()}
        return

    yield {"type": "status", "message": "Guardrails passed. Starting analysis..."}

    agent = get_agent()
    tools_used: list[str] = []
    final_text = ""

    for attempt in range(1, MAX_AGENT_RETRIES + 1):
        tools_used = []
        final_text = ""
        stream_thread_id = f"{thread_id}-r{attempt}" if attempt > 1 else thread_id
        try:
            for update in agent.stream(
                {"messages": [("user", query)]},
                config={"configurable": {"thread_id": stream_thread_id}},
                stream_mode="updates",
            ):
                for node, data in update.items():
                    for m in data.get("messages", []):
                        if isinstance(m, AIMessage) and m.tool_calls:
                            for tc in m.tool_calls:
                                tools_used.append(tc["name"])
                                yield {"type": "tool_start", "tool": tc["name"], "args": tc["args"]}
                        elif node == "tools":
                            preview = (m.content or "")[:220]
                            yield {"type": "tool_end", "tool": getattr(m, "name", "tool"), "result_preview": preview}
                        elif isinstance(m, AIMessage) and m.content:
                            final_text = m.content
        except Exception as exc:
            log_event(logger, "error", "Agent streaming failed", trace_id, error=str(exc), attempt=attempt)
            yield {"type": "error", "message": "A data provider is temporarily unavailable. Please try again shortly."}
            return

        if not _is_malformed(final_text):
            break

        log_event(
            logger, "warning", "Model returned malformed/hallucinated output while streaming — retrying",
            trace_id, attempt=attempt, preview=final_text[:200],
        )
        if attempt < MAX_AGENT_RETRIES:
            yield {"type": "status", "message": "Got an unexpected response from the model — retrying..."}
        else:
            log_event(logger, "error", "Model kept returning malformed output after retries", trace_id)
            yield {"type": "error", "message": (
                "The analysis engine returned an unexpected response after multiple attempts. "
                "Please try again in a moment."
            )}
            return

    yield {"type": "status", "message": "Applying output guardrails..."}
    sanitized, out_events = apply_output_guardrails(final_text, trace_id)
    for event in out_events:
        yield {"type": "guardrail", **event}

    response = _build_response(trace_id, query, sanitized, tools_used, out_events, status="ok")
    yield {"type": "final", "data": response.model_dump()}
