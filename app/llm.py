"""LLM factory. Swapping providers later only means changing .env — no code
changes elsewhere in the app (agent, guardrails, routes all call get_llm()).
"""
from langchain_openai import ChatOpenAI

from app.config import settings


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Main reasoning/agent LLM — any OpenAI-compatible endpoint (see .env)."""
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=temperature,
        timeout=60,
    )


def get_guard_llm() -> ChatOpenAI:
    """Safety classifier (Llama Guard) — intentionally its own base_url/key/model,
    since the guardrail provider does not have to match the main LLM provider."""
    return ChatOpenAI(
        base_url=settings.guard_base_url,
        api_key=settings.guard_api_key,
        model=settings.guard_model,
        temperature=0,
        timeout=30,
    )
