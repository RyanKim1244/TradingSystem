"""ML 모델용 피처 엔지니어링: 기술적 지표 + 매크로."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..sensors.market_data import build_feature_frame

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "volatility_20d",
    "rsi_14",
    "sma_ratio_20",
    "sma_ratio_50",
    "fed_funds_rate",
    "cpi",
    "unemployment",
    "yield_10y",
    "yield_2y",
    "yield_spread",
    "vix",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi_14"] = _rsi(df["close"], 14)
    sma_20 = df["close"].rolling(20).mean()
    sma_50 = df["close"].rolling(50).mean()
    df["sma_ratio_20"] = df["close"] / sma_20 - 1.0
    df["sma_ratio_50"] = df["close"] / sma_50 - 1.0
    return df


def build_training_frame(
    ticker: str,
    period: str = "5y",
    horizon_days: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """피처 테이블(X)과 미래 horizon_days 수익률 타겟(y)을 반환.

    누설 방지: y는 close.shift(-horizon)으로 계산하고, 마지막 horizon 행은 학습 시
    drop된다 (예측 시점얱 X만 있고 y가 없으므로 자동으로 빠짐).
    """
    raw = build_feature_frame(ticker, period=period)
    enriched = _add_technicals(raw)
    target = enriched["close"].pct_change(horizon_days).shift(-horizon_days)

    X = enriched[FEATURE_COLUMNS]
    y = target.rename(f"return_{horizon_days}d_fwd")

    combined = X.join(y).dropna()
    return combined[FEATURE_COLUMNS], combined[y.name]


def build_inference_row(ticker: str, period: str = "1y") -> pd.DataFrame:
    """가장 최근 한 행만 반환 (예측용). 타겟은 없음."""
    raw = build_feature_frame(ticker, period=period)
    enriched = _add_technicals(raw)
    latest = enriched[FEATURE_COLUMNS].dropna().tail(1)
    if latest.empty:
        raise ValueError(f"No complete feature row available for {ticker}")
    return latest
