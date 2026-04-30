"""학습 CLI: `python -m src.brain.train AAPL`."""
from __future__ import annotations

import argparse
import json

from .features import build_training_frame
from .model import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost return predictor")
    parser.add_argument("ticker")
    parser.add_argument("--period", default="5y", help="yfinance history window")
    parser.add_argument("--horizon", type=int, default=5, help="forward return horizon (trading days)")
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    X, y = build_training_frame(args.ticker, period=args.period, horizon_days=args.horizon)
    report = train(args.ticker, X, y, horizon_days=args.horizon, test_frac=args.test_frac)
    print(json.dumps(report.as_dict(), indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
