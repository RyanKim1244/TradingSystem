"""주가/펀더멘털/매크로 시계열 수집기."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from fredapi import Fred

from ..config import MACRO_SERIES, settings


def fetch_price_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """yfinance로 OHLCV + 일간 수익률 시계열을 반환."""
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price history returned for {ticker}")
    df = df.rename(columns=str.lower)
    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df.index = df.index.tz_localize(None)
    return df


def fetch_fundamentals(ticker: str) -> dict:
    """주요 재무비율 스냅샷."""
    info = yf.Ticker(ticker).info
    keys = (
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "debtToEquity",
        "returnOnEquity",
        "profitMargins",
        "earningsGrowth",
        "revenueGrowth",
        "marketCap",
        "beta",
    )
    return {k: info.get(k) for k in keys}


def fetch_macro(start: str | None = None) -> pd.DataFrame:
    """FRED 매크로 시계열을 단일 DataFrame으로 머지."""
    if not settings.fred_api_key:
        raise RuntimeError("FRED_API_KEY not set in environment")
    fred = Fred(api_key=settings.fred_api_key)
    start = start or (datetime.utcnow() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    series = {name: fred.get_series(code, observation_start=start) for name, code in MACRO_SERIES.items()}
    macro = pd.DataFrame(series).sort_index().ffill()
    macro["yield_spread"] = macro["yield_10y"] - macro["yield_2y"]
    macro.index = pd.to_datetime(macro.index)
    return macro


def build_feature_frame(ticker: str, period: str = "5y") -> pd.DataFrame:
    """가격 + 매크로를 일자 기준으로 병합한 단일 피처 테이블."""
    prices = fetch_price_history(ticker, period=period)
    macro = fetch_macro(start=prices.index.min().strftime("%Y-%m-%d"))
    macro = macro.reindex(prices.index.union(macro.index)).ffill().reindex(prices.index)
    return prices.join(macro, how="left")
