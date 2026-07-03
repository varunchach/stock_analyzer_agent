"""The stock-analysis agent — a single LangGraph ReAct agent wired to all
tools (ticker resolution, market data, research). Same create_agent pattern
used in the teaching notebooks, just pointed at the MaaS LLM and wrapped for
the FastAPI app (streaming + memory).
"""
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.tools_market import MARKET_TOOLS
from app.agent.tools_research import RESEARCH_TOOLS
from app.agent.tools_ticker import TICKER_TOOLS
from app.llm import get_llm

SYSTEM_PROMPT = """You are TradeSetu's stock research assistant. You help users \
understand a stock's potential using live data — never invent numbers.

Rules:
1. If the user gives a company name instead of an exact symbol, call resolve_ticker first.
2. Always ground every price, ratio, or indicator in a tool result — never guess.
3. Cover, where relevant to the question: live price, fundamentals, technicals \
(RSI/SMA), recent news, market sentiment, and global/macro context.
4. End every analysis with a recommendation leaning of Buy, Hold, or Sell, phrased \
as a *leaning* (e.g. "leaning Hold") — never an unconditional directive.
5. Always include this exact disclaimer at the end: "This is not SEBI-registered \
investment advice. For educational purposes only."
6. Never claim guaranteed returns or "sure shot" outcomes. Be balanced — mention \
both supportive and concerning signals when they exist.
7. Keep the final answer well-structured with short sections, not a wall of text.
"""

ALL_TOOLS = TICKER_TOOLS + MARKET_TOOLS + RESEARCH_TOOLS

_checkpointer = InMemorySaver()
_agent = create_agent(get_llm(), ALL_TOOLS, system_prompt=SYSTEM_PROMPT, checkpointer=_checkpointer)


def get_agent():
    return _agent


def extract_tool_calls(messages) -> list[str]:
    names = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            names.extend(tc["name"] for tc in m.tool_calls)
    return names


def extract_tool_trace(messages) -> list[dict]:
    """Compact, UI-friendly trace: one entry per tool call + its result."""
    trace = []
    pending_calls = {}
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                pending_calls[tc["id"]] = {"tool": tc["name"], "args": tc["args"]}
        if isinstance(m, ToolMessage):
            entry = pending_calls.get(m.tool_call_id, {"tool": m.name, "args": {}})
            entry["result_preview"] = (m.content or "")[:400]
            trace.append(entry)
    return trace
