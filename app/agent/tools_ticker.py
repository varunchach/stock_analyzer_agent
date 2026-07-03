"""Ticker resolution — turns free text ("TCS", "Tata Consultancy", "Reliance
shares") into a valid Yahoo Finance symbol. Prefers NSE, then BSE, then the
best global match. No hardcoded symbol map — resolved live via yfinance search.
"""
import json

import yfinance as yf
from langchain_core.tools import tool

from app.logging_config import get_logger

logger = get_logger("tools.ticker")

_EXCHANGE_PRIORITY = {"NSE": 0, "Bombay": 1}


@tool
def resolve_ticker(company_or_symbol: str) -> str:
    """Resolve a company name or partial symbol (e.g. 'TCS', 'Reliance', 'Infosys')
    to a valid Yahoo Finance ticker symbol. Always call this first if the user
    gave a company name rather than an exact symbol like 'TCS.NS'."""
    query = company_or_symbol.strip()
    try:
        search = yf.Search(query, max_results=8)
        candidates = [c for c in search.quotes if c.get("quoteType") == "EQUITY"]
        if not candidates:
            return json.dumps({"error": f"No matching stock found for '{query}'. Ask the user to clarify."})

        candidates.sort(key=lambda c: _EXCHANGE_PRIORITY.get(c.get("exchDisp"), 99))
        best = candidates[0]
        return json.dumps({
            "resolved_symbol": best["symbol"],
            "name": best.get("longname") or best.get("shortname"),
            "exchange": best.get("exchDisp"),
            "sector": best.get("sector"),
            "alternatives": [c["symbol"] for c in candidates[1:4]],
        })
    except Exception as exc:
        logger.warning("resolve_ticker failed", extra={"extra_fields": {"query": query, "error": str(exc)}})
        return json.dumps({"error": f"Could not resolve ticker for '{query}'"})


TICKER_TOOLS = [resolve_ticker]
