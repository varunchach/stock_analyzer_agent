"""Central app configuration. Reads secrets from .env — never hardcode keys here."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _append_no_proxy(*hosts: str) -> None:
    """Ensure selected hosts bypass any machine-level proxy settings."""
    existing = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""
    items = [item.strip() for item in existing.split(",") if item.strip()]
    changed = False
    for host in hosts:
        if host not in items:
            items.append(host)
            changed = True
    if changed:
        value = ",".join(items)
        os.environ["NO_PROXY"] = value
        os.environ["no_proxy"] = value


def _mirror_langsmith_env() -> None:
    """Backfill legacy LangChain env names expected by some tracing paths."""
    if os.getenv("LANGSMITH_TRACING") and not os.getenv("LANGCHAIN_TRACING_V2"):
        os.environ["LANGCHAIN_TRACING_V2"] = os.environ["LANGSMITH_TRACING"]
    if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
    if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]
    if os.getenv("LANGSMITH_ENDPOINT") and not os.getenv("LANGCHAIN_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"]


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    # Main reasoning/agent model — provider-agnostic, any OpenAI-compatible endpoint
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    # Safety-classifier model — deliberately a SEPARATE base_url/key/model, since
    # it may live on a different provider than the main LLM (e.g. main LLM on
    # OpenAI, guardrail classifier on a MaaS-hosted Llama Guard)
    guard_base_url: str
    guard_api_key: str
    guard_model: str
    tavily_api_key: str
    langsmith_enabled: bool
    app_env: str
    log_level: str


def get_settings() -> Settings:
    if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
        _append_no_proxy("api.smith.langchain.com", "smith.langchain.com")
        _mirror_langsmith_env()
    return Settings(
        llm_base_url=_require("LLM_BASE_URL"),
        llm_api_key=_require("LLM_API_KEY"),
        llm_model=_require("LLM_MODEL"),
        guard_base_url=_require("GUARD_BASE_URL"),
        guard_api_key=_require("GUARD_API_KEY"),
        guard_model=_require("GUARD_MODEL"),
        tavily_api_key=_require("TAVILY_API_KEY"),
        langsmith_enabled=os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


settings = get_settings()
