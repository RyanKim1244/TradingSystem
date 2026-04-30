"""워크포워드 백테스트: 슬라이딩 윈도우로 재학습 → 다음 구간 예측 → 시뮬레이션."""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..brain.features import FEATURE_COLUMNS, _add_technicals
from ..config import ShieldConfig, shield_config
from ..sensors.market_data import build_feature_frame
from .engine import BacktestResult, compute_metrics, simulate


def _backtest_signal(row: pd.Series, predicted_return: float, cfg: ShieldConfig) -> str:
    """백테스트용 시그널 (센티먼트 미포함). 라이브 decide()와 의도 일치."""
    vix = row.get("vix")
    spread = row.get("yield_spread")
    rsi = row.get("rsi_14")
    if vix is not None and pd.notna(vix) and vix > cfg.vix_block:
        return "BLOCK"
    if spread is not None and pd.notna(spread) and spread < cfg.yield_spread_block:
        return "BLOCK"
    if predicted_return >= cfg.model_buy_threshold:
        return "BUY"
    if predicted_return <= cfg.model_sell_threshold:
        return "REDUCE"
    if rsi is not None and pd.notna(rsi) and rsi <= cfg.rsi_oversold:
        return "BUY"
    return "HOLD"


def _new_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )


def _build_full_frame(ticker: str, period: str, horizon_days: int) -> pd.DataFrame:
    raw = build_feature_frame(ticker, period=period)
    enriched = _add_technicals(raw)
    target = enriched["close"].pct_change(horizon_days).shift(-horizon_days)
    full = enriched.join(target.rename("y"))
    keep_cols = list(set(["close", "y", *FEATURE_COLUMNS]))
    return full[keep_cols].dropna()


def walk_forward(
    ticker: str,
    period: str = "10y",
    horizon_days: int = 5,
    train_days: int = 756,
    test_days: int = 126,
    cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    cfg: ShieldConfig | None = None,
) -> BacktestResult:
    cfg = cfg or shield_config
    full = _build_full_frame(ticker, period, horizon_days)
    if len(full) < train_days + test_days:
        raise ValueError(
            f"Need at least {train_days + test_days} rows; got {len(full)}. "
            "Try a longer --period or shorter --train-days."
        )

    signals: list[dict] = []
    cursor = train_days
    while cursor + test_days <= len(full):
        train = full.iloc[cursor - train_days : cursor]
        test = full.iloc[cursor : cursor + test_days]

        model = _new_model()
        model.fit(train[FEATURE_COLUMNS], train["y"], verbose=False)
        preds = model.predict(test[FEATURE_COLUMNS])

        for (date, row), pred in zip(test.iterrows(), preds):
            signals.append(
                {
                    "date": date,
                    "close": row["close"],
                    "predicted_return": float(pred),
                    "signal": _backtest_signal(row, float(pred), cfg),
                }
            )
        cursor += test_days

    sig_df = pd.DataFrame(signals).set_index("date").sort_index()
    equity, daily_ret, n_trades = simulate(
        sig_df["close"], sig_df["signal"], cost_bps=cost_bps, slippage_bps=slippage_bps
    )
    metrics = compute_metrics(equity, daily_ret)

    return BacktestResult(
        ticker=ticker.upper(),
        horizon_days=horizon_days,
        n_days=len(equity),
        n_trades=n_trades,
        equity_curve=equity,
        signals=sig_df,
        **metrics,
    )
