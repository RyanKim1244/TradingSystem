"""백테스트 시뮬레이터 + 성과 지표.

벡터화 시뮬레이션. 일별로 'BUY'/'HOLD'/'REDUCE'/'BLOCK' 시그널을 받아
다음 날 종가 기준 포지션 변경, 수수료/슬리피지 반영.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    ticker: str
    horizon_days: int
    n_days: int
    n_trades: int
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    equity_curve: pd.Series = field(repr=False)
    signals: pd.DataFrame = field(repr=False)

    def summary(self) -> dict:
        return {
            "ticker": self.ticker,
            "horizon_days": self.horizon_days,
            "n_days": self.n_days,
            "n_trades": self.n_trades,
            "total_return_pct": self.total_return * 100,
            "annualized_return_pct": self.annualized_return * 100,
            "annualized_vol_pct": self.annualized_vol * 100,
            "sharpe": self.sharpe,
            "max_drawdown_pct": self.max_drawdown * 100,
            "win_rate_pct": self.win_rate * 100,
        }


def _signal_to_position(signals: pd.Series) -> pd.Series:
    """시그널 → 의도 포지션 (0 또는 1). HOLD는 직전 포지션 유지."""
    pos = 0
    out = []
    for s in signals:
        if s == "BUY":
            pos = 1
        elif s in ("REDUCE", "BLOCK"):
            pos = 0
        out.append(pos)
    return pd.Series(out, index=signals.index, dtype=float)


def simulate(
    prices: pd.Series,
    signals: pd.Series,
    cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> tuple[pd.Series, pd.Series, int]:
    """일별 close 가격과 시그널을 받아 (equity_curve, daily_returns, n_trades) 반환."""
    aligned = pd.concat([prices.rename("close"), signals.rename("signal")], axis=1).dropna()
    intended = _signal_to_position(aligned["signal"])
    # 체결 지연: 오늘 시그널은 내일 보유 → shift(1)
    held = intended.shift(1).fillna(0.0)

    daily_ret = aligned["close"].pct_change().fillna(0.0)
    strat_ret = held * daily_ret

    turn = held.diff().abs().fillna(held.iloc[0])  # 첫 진입도 1회 거래
    cost_per_turn = (cost_bps + slippage_bps) / 10_000.0
    strat_ret = strat_ret - turn * cost_per_turn

    equity = (1.0 + strat_ret).cumprod()
    n_trades = int((turn > 0).sum())
    return equity, strat_ret, n_trades


def compute_metrics(equity: pd.Series, daily_ret: pd.Series) -> dict:
    if equity.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }
    total_return = float(equity.iloc[-1] - 1.0)
    n_days = len(daily_ret)
    years = n_days / TRADING_DAYS_PER_YEAR
    ann_ret = float((equity.iloc[-1]) ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else 0.0
    ann_vol = float(daily_ret.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    mdd = float(drawdown.min())
    nonzero = daily_ret[daily_ret != 0]
    win_rate = float((nonzero > 0).mean()) if len(nonzero) else 0.0
    return {
        "total_return": total_return,
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate": win_rate,
    }
