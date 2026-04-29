"""Phase 1 진입점: 단일 티커에 대한 데이터 + 센티먼트 스냅샷."""
from __future__ import annotations

import argparse
import json

from .brain.sentiment import score_news
from .sensors.market_data import build_feature_frame, fetch_fundamentals
from .sensors.news import fetch_recent_news


def snapshot(ticker: str, history_period: str = "1y", news_limit: int = 10) -> dict:
    features = build_feature_frame(ticker, period=history_period)
    fundamentals = fetch_fundamentals(ticker)
    news = fetch_recent_news(ticker, limit=news_limit)
    sentiment = score_news(ticker, news)

    last = features.tail(1).iloc[0]
    return {
        "ticker": ticker,
        "as_of": str(features.index[-1].date()),
        "price": {
            "close": float(last["close"]),
            "return_5d": float(last.get("return_5d") or 0.0),
            "volatility_20d": float(last.get("volatility_20d") or 0.0),
        },
        "macro": {
            "vix": float(last.get("vix") or 0.0),
            "yield_spread": float(last.get("yield_spread") or 0.0),
            "fed_funds_rate": float(last.get("fed_funds_rate") or 0.0),
        },
        "fundamentals": fundamentals,
        "sentiment": sentiment.as_dict(),
        "news_sample": [item.title for item in news[:5]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha-Engine Phase 1 snapshot")
    parser.add_argument("ticker", help="e.g. AAPL")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--news", type=int, default=10)
    args = parser.parse_args()

    result = snapshot(args.ticker.upper(), history_period=args.period, news_limit=args.news)
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
