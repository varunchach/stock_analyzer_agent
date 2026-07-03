"""Input guardrails — two independent layers:

1. Safety  — Llama Guard 3 classifies the raw user text (violence, hate,
   self-harm, etc. — the standard Llama Guard taxonomy). Catches *unsafe*
   content regardless of topic.
2. Scope   — a cheap keyword heuristic first; only if that's inconclusive do
   we ask the main LLM "is this about stocks/investing?". Catches *off-topic*
   content that is perfectly safe (e.g. "write me a poem") but out of scope
   for this app.

Both layers run before the agent ever sees the request.
"""
import re

from app.errors import GuardrailViolation
from app.guardrails.policies import (
    ALLOWED_SAFETY_CATEGORIES,
    MAX_QUERY_CHARS,
    OFF_TOPIC_MESSAGE,
    STOCK_SCOPE_KEYWORDS,
    UNSAFE_INPUT_MESSAGE,
)
from app.llm import get_guard_llm, get_llm
from app.logging_config import get_logger, log_event

logger = get_logger("guardrails.input")

_TICKER_LIKE = re.compile(r"\b[A-Z]{2,10}(\.NS|\.BO)?\b")


def _check_length(text: str) -> None:
    if not text or not text.strip():
        raise GuardrailViolation("length", "Please enter a question.")
    if len(text) > MAX_QUERY_CHARS:
        raise GuardrailViolation(
            "length", f"Question is too long (max {MAX_QUERY_CHARS} characters). Please shorten it."
        )


def _check_safety(text: str, trace_id: str) -> None:
    guard = get_guard_llm()
    verdict = guard.invoke(text).content.strip().lower()
    is_unsafe = verdict.startswith("unsafe") or "\nunsafe" in verdict
    if not is_unsafe:
        log_event(logger, "info", "Input passed safety guardrail", trace_id)
        return

    match = re.search(r"s\d+", verdict)
    category = match.group(0).upper() if match else "unknown"

    if category in ALLOWED_SAFETY_CATEGORIES:
        log_event(
            logger, "info", "Safety flag allowlisted for this domain", trace_id,
            category=category, note="financial-advice category expected for a stock analyzer",
        )
        return

    log_event(logger, "warning", "Input blocked by safety guardrail", trace_id, category=category)
    raise GuardrailViolation("safety", UNSAFE_INPUT_MESSAGE, category=category)


def _heuristic_in_scope(text: str) -> bool:
    lowered = text.lower()
    if any(keyword in lowered for keyword in STOCK_SCOPE_KEYWORDS):
        return True
    # A short, mostly-uppercase token often signals a ticker (e.g. "TCS", "INFY").
    return bool(_TICKER_LIKE.search(text))


def _llm_in_scope(text: str) -> bool:
    llm = get_llm(temperature=0)
    prompt = (
        "Answer with exactly one word, 'yes' or 'no'. Is the following user "
        f"message related to stock markets, investing, or a specific company's "
        f"shares?\n\nMessage: {text!r}"
    )
    verdict = llm.invoke(prompt).content.strip().lower()
    return verdict.startswith("yes")


def _check_scope(text: str, trace_id: str) -> None:
    if _heuristic_in_scope(text):
        log_event(logger, "info", "Input passed scope guardrail (heuristic)", trace_id)
        return
    if _llm_in_scope(text):
        log_event(logger, "info", "Input passed scope guardrail (LLM fallback)", trace_id)
        return
    log_event(logger, "warning", "Input blocked by scope guardrail", trace_id)
    raise GuardrailViolation("scope", OFF_TOPIC_MESSAGE)


def run_input_guardrails(text: str, trace_id: str) -> None:
    """Raises GuardrailViolation if the request should be blocked."""
    _check_length(text)
    _check_safety(text, trace_id)
    _check_scope(text, trace_id)
