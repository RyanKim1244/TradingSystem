"""전체 파이프라인 진입점: `python -m src.decide AAPL`."""
from __future__ import annotations

import argparse
import json

from .shield.decision import decide


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Alpha-Engine decision pipeline")
    parser.add_argument("ticker")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--period", default="1y")
    parser.add_argument("--news", type=int, default=10)
    args = parser.parse_args()

    result = decide(
        args.ticker.upper(),
        horizon_days=args.horizon,
        history_period=args.period,
        news_limit=args.news,
    )
    print(json.dumps(result.as_dict(), indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
