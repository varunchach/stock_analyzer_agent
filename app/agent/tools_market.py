"""Live market data tools — Yahoo Finance via yfinance. No hardcoded prices;
every number is fetched at call time.
"""
import json

import yfinance as yf
from langchain_core.tools import tool

from app.logging_config import get_logger

logger = get_logger("tools.market")


def _norm_symbol(symbol: str) -> str:
    return symbol.strip().upper()


@tool
def get_live_quote(symbol: str) -> str:
    """Fetch live price, day change %, and volume for a Yahoo Finance symbol (e.g. TCS.NS, RELIANCE.NS, AAPL)."""
    sym = _norm_symbol(symbol)
    try:
        ticker = yf.Ticker(sym)
        info = ticker.info
        hist = ticker.history(period="5d")
        if hist.empty and not info:
            return json.dumps({"error": f"No quote data for '{sym}'"})
        last = float(hist["Close"].iloc[-1]) if not hist.empty else info.get("currentPrice")
        prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else info.get("previousClose")
        change_pct = round((last - prev) / prev * 100, 2) if prev else None
        payload = {
            "symbol": sym,
            "price": round(last, 2) if last else None,
            "currency": info.get("currency", "INR"),
            "change_pct_1d": change_pct,
            "volume": int(hist["Volume"].iloc[-1]) if not hist.empty else info.get("volume"),
            "market_state": info.get("marketState"),
            "as_of": str(hist.index[-1].date()) if not hist.empty else None,
        }
        return json.dumps(payload, default=str)
    except Exception as exc:
        logger.warning("get_live_quote failed", extra={"extra_fields": {"symbol": sym, "error": str(exc)}})
        return json.dumps({"error": f"Could not fetch live quote for '{sym}'"})


@tool
def get_fundamental_snapshot(symbol: str) -> str:
    """Return key fundamental ratios and financial snapshot from live market data."""
    sym = _norm_symbol(symbol)
    try:
        info = yf.Ticker(sym).info
        keys = [
            "symbol", "shortName", "currency", "sector", "industry",
            "marketCap", "trailingPE", "forwardPE", "priceToBook",
            "dividendYield", "profitMargins", "revenueGrowth",
            "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "epsTrailingTwelveMonths",
        ]
        data = {k: info.get(k) for k in keys if info.get(k) is not None}
        if not data:
            return json.dumps({"error": f"No fundamentals for '{sym}'"})
        data["symbol"] = sym
        return json.dumps(data, default=str)
    except Exception as exc:
        logger.warning("get_fundamental_snapshot failed", extra={"extra_fields": {"symbol": sym, "error": str(exc)}})
        return json.dumps({"error": f"Could not fetch fundamentals for '{sym}'"})


@tool
def get_technical_snapshot(symbol: str) -> str:
    """Compute RSI-14, SMA-20, SMA-50 and price-vs-MA trend from recent history."""
    sym = _norm_symbol(symbol)
    try:
        hist = yf.Ticker(sym).history(period="6mo")
        if hist.empty or len(hist) < 20:
            return json.dumps({"error": f"Not enough history for '{sym}'"})
        close = hist["Close"]
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi14 = float((100 - (100 / (1 + rs))).iloc[-1])
        price = float(close.iloc[-1])
        trend = "above_SMA20" if price > sma20 else "below_SMA20"
        payload = {
            "symbol": sym,
            "price": round(price, 2),
            "rsi_14": round(rsi14, 2),
            "sma_20": round(sma20, 2),
            "sma_50": round(sma50, 2) if sma50 else None,
            "trend_vs_sma20": trend,
            "history_days": len(hist),
        }
        return json.dumps(payload, default=str)
    except Exception as exc:
        logger.warning("get_technical_snapshot failed", extra={"extra_fields": {"symbol": sym, "error": str(exc)}})
        return json.dumps({"error": f"Could not compute technicals for '{sym}'"})


@tool
def get_price_history(symbol: str, period: str = "3mo") -> str:
    """Return a compact daily closing-price series for charting (date + close only)."""
    sym = _norm_symbol(symbol)
    try:
        hist = yf.Ticker(sym).history(period=period)
        if hist.empty:
            return json.dumps({"error": f"No history for '{sym}'"})
        series = [
            {"date": str(idx.date()), "close": round(float(row["Close"]), 2)}
            for idx, row in hist.iterrows()
        ]
        return json.dumps({"symbol": sym, "series": series})
    except Exception as exc:
        logger.warning("get_price_history failed", extra={"extra_fields": {"symbol": sym, "error": str(exc)}})
        return json.dumps({"error": f"Could not fetch history for '{sym}'"})


MARKET_TOOLS = [get_live_quote, get_fundamental_snapshot, get_technical_snapshot, get_price_history]
