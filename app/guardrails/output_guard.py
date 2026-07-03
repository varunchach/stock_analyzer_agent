"""Output guardrails — applied to the agent's final answer before it reaches
the client. Rules live in policies.py; this module only enforces them.
"""
from app.guardrails.policies import (
    DISCLAIMER_TEXT,
    FORBIDDEN_OUTPUT_PHRASES,
    MAX_RESPONSE_CHARS,
)
from app.logging_config import get_logger, log_event

logger = get_logger("guardrails.output")


def apply_output_guardrails(text: str, trace_id: str) -> tuple[str, list[dict]]:
    """Returns (sanitized_text, guardrail_events) — never raises; always
    returns something safe to show the user."""
    events: list[dict] = []
    sanitized = text or ""

    lowered = sanitized.lower()
    hit_phrases = [p for p in FORBIDDEN_OUTPUT_PHRASES if p in lowered]
    if hit_phrases:
        log_event(logger, "warning", "Output guardrail stripped forbidden phrasing", trace_id, phrases=hit_phrases)
        events.append({"stage": "output_phrase", "result": "flagged", "detail": hit_phrases})
        for phrase in sorted(hit_phrases, key=len, reverse=True):
            sanitized = _redact(sanitized, phrase)
    else:
        events.append({"stage": "output_phrase", "result": "clean"})

    if len(sanitized) > MAX_RESPONSE_CHARS:
        sanitized = sanitized[:MAX_RESPONSE_CHARS].rsplit("\n", 1)[0] + "\n\n*(truncated for length)*"
        events.append({"stage": "output_length", "result": "truncated"})
    else:
        events.append({"stage": "output_length", "result": "ok"})

    if DISCLAIMER_TEXT.lower() not in sanitized.lower():
        sanitized = sanitized.rstrip() + f"\n\n---\n_{DISCLAIMER_TEXT}_"
        events.append({"stage": "disclaimer", "result": "appended"})
    else:
        events.append({"stage": "disclaimer", "result": "present"})

    return sanitized, events


def _redact(text: str, phrase: str) -> str:
    import re

    return re.sub(re.escape(phrase), "[claim removed by guardrail]", text, flags=re.IGNORECASE)
