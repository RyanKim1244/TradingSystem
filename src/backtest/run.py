"""백테스트 CLI: `python -m src.backtest.run AAPL`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .walkforward import walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest of Alpha-Engine model")
    parser.add_argument("ticker")
    parser.add_argument("--period", default="10y", help="yfinance history window")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--train-days", type=int, default=756, help="~3y training window")
    parser.add_argument("--test-days", type=int, default=126, help="~6mo out-of-sample slice")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--save-curve", default=None, help="optional CSV path for equity curve")
    args = parser.parse_args()

    result = walk_forward(
        args.ticker,
        period=args.period,
        horizon_days=args.horizon,
        train_days=args.train_days,
        test_days=args.test_days,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
    )
    print(json.dumps(result.summary(), indent=2, default=str, ensure_ascii=False))

    if args.save_curve:
        path = Path(args.save_curve)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(path, header=["equity"])
        print(f"\nequity curve saved to {path}")


if __name__ == "__main__":
    main()
