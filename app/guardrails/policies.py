"""Guardrail policy constants — the rules, kept separate from enforcement logic
so students can see/tweak them without touching the guard code itself.
"""

DISCLAIMER_TEXT = "This is not SEBI-registered investment advice. For educational purposes only."

# Llama Guard's standard taxonomy flags "S6: Specialized Advice" for any
# financial/medical/legal advice-seeking text — which is this app's entire
# purpose. We deliberately allow S6 through the safety guardrail because our
# OUTPUT guardrail already enforces the actual safeguards S6 exists for:
# a mandatory disclaimer, hedged "leaning" phrasing, and a ban on
# "guaranteed return" style claims (see output_guard.py). Every other
# category (S1-S5, S7-S14 — violence, weapons, self-harm, hate, etc.) is
# still a hard block.
ALLOWED_SAFETY_CATEGORIES = {"S6"}

MAX_QUERY_CHARS = 400
MAX_RESPONSE_CHARS = 6000

# Cheap first-pass scope check. If any of these appear, we skip the LLM scope
# classifier entirely (fast path). If none match, we fall back to an LLM check
# before blocking — some valid questions ("what should I do with my savings?")
# won't contain an obvious keyword.
STOCK_SCOPE_KEYWORDS = [
    "stock", "share", "shares", "equity", "nse", "bse", "ticker", "market",
    "invest", "investment", "portfolio", "fundamental", "technical", "rsi",
    "sma", "moving average", "dividend", "earnings", "eps", "p/e", "pe ratio",
    "buy", "sell", "hold", "ipo", "mutual fund", "sensex", "nifty", "trading",
    "trade", "valuation", "market cap", "analyst", "target price", "sentiment",
    "recommendation", "potential",
]

# Forbidden in *model output* — regardless of what the user asked, the agent
# must never phrase a recommendation this way.
FORBIDDEN_OUTPUT_PHRASES = [
    "guaranteed return", "guaranteed returns", "guaranteed profit",
    "sure shot", "100% return", "risk-free", "no risk", "can't lose",
    "definitely will go up", "definitely will go down",
]

OFF_TOPIC_MESSAGE = (
    "I can only help with stock market and investment analysis questions "
    "(e.g. \"Tell me about Infosys stock\" or \"Should I look at Reliance?\"). "
    "Please ask about a specific company, sector, or ticker."
)

UNSAFE_INPUT_MESSAGE = (
    "This request was flagged by our safety filter and can't be processed. "
    "Please rephrase your question and keep it focused on stock market analysis."
)

# Some models expose internal chat-template markup (e.g. gpt-oss's "harmony"
# format uses <|channel|>/<|message|>/<|start|>/<|end|>) when the serving
# backend fails to fully parse a tool call. If any of these leak into the
# final answer, the response is a hallucinated/malformed completion — not a
# real, tool-grounded answer — and must never be shown to the user as-is.
MALFORMED_OUTPUT_MARKERS = ["<|channel|>", "<|message|>", "<|start|>", "<|end|>", "<|constrain|>"]

# Based on diagnostic testing (see docs/test_maas_tool_parsing.py): the
# upstream model/proxy leaks malformed output on roughly 7% of turns, and a
# typical analysis needs ~5 turns (~30% chance of hitting it at least once
# per conversation). 2 retries left a ~9% residual failure rate; 3 brings
# that down to ~2.7% — worth the extra latency on the rare retry path.
MAX_AGENT_RETRIES = 3

MALFORMED_OUTPUT_MESSAGE = (
    "The analysis engine returned an unexpected response. Please try asking again — "
    "this is usually a transient issue with the model, not your question."
)
