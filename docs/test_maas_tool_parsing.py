"""One-off diagnostic script — NOT part of the app.

Tests the raw MaaS chat-completions endpoint directly (bypassing LangChain/
LangGraph entirely), simulating a full multi-turn tool-calling conversation
exactly like our agent does: user -> tool_call -> tool_result -> next turn
-> ... This is the realistic reproduction, since the bug reported was seen
mid-conversation (after at least one tool result was already in context),
not on the very first turn.

Run: python docs/test_maas_tool_parsing.py
"""
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.agent.tools_market import get_fundamental_snapshot, get_live_quote, get_technical_snapshot  # noqa: E402
from app.agent.tools_research import search_market_sentiment, search_stock_news  # noqa: E402

# Note: historical diagnostic — the app's main LLM has since moved off MaaS
# (see README "Why we switched off MaaS" section). To re-run this against
# MaaS specifically, set these three env vars explicitly; they are no longer
# read from the app's main .env (which now points LLM_* at the new provider).
BASE_URL = os.environ.get("MAAS_DIAGNOSTIC_BASE_URL", "https://maas-rhdp.apps.maas.redhatworkshops.io/v1")
if "MAAS_DIAGNOSTIC_API_KEY" not in os.environ:
    sys.exit(
        "This diagnostic targets MaaS specifically and needs its own key "
        "(the app's main LLM key now points elsewhere). Set MAAS_DIAGNOSTIC_API_KEY "
        "and re-run."
    )
API_KEY = os.environ["MAAS_DIAGNOSTIC_API_KEY"]
MODEL = os.environ.get("MAAS_DIAGNOSTIC_MODEL", "gpt-oss-120b")

REAL_TOOLS = {
    "get_live_quote": get_live_quote,
    "get_fundamental_snapshot": get_fundamental_snapshot,
    "get_technical_snapshot": get_technical_snapshot,
    "search_stock_news": search_stock_news,
    "search_market_sentiment": search_market_sentiment,
}

TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "get_live_quote", "description": "Get live stock price for a Yahoo Finance symbol",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
    }},
    {"type": "function", "function": {
        "name": "get_fundamental_snapshot", "description": "Get fundamental ratios (P/E, market cap, etc.) for a symbol",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
    }},
    {"type": "function", "function": {
        "name": "get_technical_snapshot", "description": "Get RSI-14, SMA-20, SMA-50 technical indicators for a symbol",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
    }},
    {"type": "function", "function": {
        "name": "search_stock_news", "description": "Search recent India-focused news for a stock symbol",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
    }},
    {"type": "function", "function": {
        "name": "search_market_sentiment", "description": "Search analyst ratings and market sentiment for a stock",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
    }},
]

MALFORMED_MARKERS = ["<|channel|>", "<|message|>", "<|start|>", "<|end|>", "<|constrain|>", "functions."]


def call_maas(messages: list[dict]) -> dict:
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": messages, "tools": TOOLS_SCHEMA, "tool_choice": "auto"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def check_malformed(content: str) -> list[str]:
    return [m for m in MALFORMED_MARKERS if m in (content or "")]


def run_conversation(run_id: int, user_query: str, max_turns: int = 6):
    print(f"\n{'=' * 70}\nCONVERSATION {run_id}: {user_query!r}\n{'=' * 70}")
    messages = [{"role": "user", "content": user_query}]
    turn_reports = []

    for turn in range(1, max_turns + 1):
        raw = call_maas(messages)
        choice = raw["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason")
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        leaked = check_malformed(content)

        report = {
            "turn": turn, "finish_reason": finish_reason,
            "num_tool_calls": len(tool_calls),
            "tool_call_names": [tc["function"]["name"] for tc in tool_calls],
            "content_preview": content[:150],
            "leaked_markers": leaked,
        }
        turn_reports.append(report)
        tag = "MALFORMED" if leaked else ("TOOL_CALL" if tool_calls else "FINAL_TEXT")
        print(f"  turn {turn}: [{tag}] finish_reason={finish_reason} tool_calls={report['tool_call_names']}")
        if leaked:
            print(f"    >>> LEAKED CONTENT: {content[:400]}")

        if leaked:
            return turn_reports, True

        if not tool_calls:
            # Model returned a final answer — conversation naturally ends.
            return turn_reports, False

        # Append the assistant's tool-call message, then execute each real
        # tool and append its result, exactly like our LangGraph agent does.
        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tool_calls})
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"] or "{}")
            tool_fn = REAL_TOOLS.get(name)
            result = tool_fn.invoke(args) if tool_fn else json.dumps({"error": f"unknown tool {name}"})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    return turn_reports, False


CONVERSATIONS = [
    "Tell me about HDFC Bank's stock analysis. I would like to understand this in detail",
    "Tell me about HDFC Bank's stock analysis. I would like to understand this in detail",
    "Give me a full analysis of TCS including fundamentals, technicals and news",
    "Should I invest in Reliance? Check live price, fundamentals and sentiment",
    "Analyze Infosys stock in depth — price, fundamentals, technicals, and recent news",
]


def main():
    any_malformed = False
    for i, q in enumerate(CONVERSATIONS, 1):
        _, malformed = run_conversation(i, q)
        any_malformed = any_malformed or malformed

    print(f"\n\n{'#' * 70}")
    print("At least one malformed/leaked response observed:", any_malformed)
    print("#" * 70)


if __name__ == "__main__":
    main()
