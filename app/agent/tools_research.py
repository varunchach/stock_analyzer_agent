"""Research tools — Tavily web search wrapped as focused, stock-specific tools."""
import json

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from app.config import settings
from app.logging_config import get_logger

logger = get_logger("tools.research")

_news = TavilySearch(max_results=4, topic="news", tavily_api_key=settings.tavily_api_key)
_general = TavilySearch(max_results=3, topic="general", tavily_api_key=settings.tavily_api_key)


def _clean_name(symbol: str) -> str:
    return symbol.replace(".NS", "").replace(".BO", "")


@tool
def search_stock_news(symbol: str) -> str:
    """Search recent India-focused news for a stock symbol."""
    name = _clean_name(symbol)
    try:
        result = _news.invoke({"query": f"{name} stock news latest India NSE"})
        return result if isinstance(result, str) else json.dumps(result, default=str)
    except Exception as exc:
        logger.warning("search_stock_news failed", extra={"extra_fields": {"symbol": symbol, "error": str(exc)}})
        return json.dumps({"error": "News search temporarily unavailable"})


@tool
def search_market_sentiment(symbol: str) -> str:
    """Search analyst ratings, broker views, and market sentiment for a stock."""
    name = _clean_name(symbol)
    try:
        result = _general.invoke({"query": f"{name} stock analyst rating sentiment buy sell hold target price India"})
        return result if isinstance(result, str) else json.dumps(result, default=str)
    except Exception as exc:
        logger.warning("search_market_sentiment failed", extra={"extra_fields": {"symbol": symbol, "error": str(exc)}})
        return json.dumps({"error": "Sentiment search temporarily unavailable"})


@tool
def search_global_macro_impact(symbol: str) -> str:
    """Search international/macro news that may affect this stock (Fed, oil, global IT spend, etc.)."""
    name = _clean_name(symbol)
    try:
        result = _general.invoke({"query": f"global macro news impact {name} India stock US Fed sector outlook"})
        return result if isinstance(result, str) else json.dumps(result, default=str)
    except Exception as exc:
        logger.warning("search_global_macro_impact failed", extra={"extra_fields": {"symbol": symbol, "error": str(exc)}})
        return json.dumps({"error": "Macro search temporarily unavailable"})


RESEARCH_TOOLS = [search_stock_news, search_market_sentiment, search_global_macro_impact]
