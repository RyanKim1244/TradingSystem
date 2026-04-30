"""예측 CLI: `python -m src.brain.predict AAPL`."""
from __future__ import annotations

import argparse
import json

from .features import build_inference_row
from .model import predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict forward return with a trained model")
    parser.add_argument("ticker")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--period", default="1y")
    args = parser.parse_args()

    row = build_inference_row(args.ticker, period=args.period)
    result = predict(args.ticker, row, horizon_days=args.horizon)
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
