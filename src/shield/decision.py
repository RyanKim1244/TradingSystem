"""센서/모델/센티먼트 → 최종 매매 결정."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..brain.features import build_inference_row
from ..brain.model import load as load_model
from ..brain.sentiment import score_news
from ..config import ShieldConfig, shield_config
from ..sensors.news import fetch_recent_news
from .rules import cap_weight, macro_block, model_signal, scale_in_ok

Action = Literal["BUY", "HOLD", "REDUCE", "BLOCK"]


@dataclass
class Decision:
    ticker: str
    as_of: str
    action: Action
    target_weight: float
    reasons: list[str]
    macro_block: bool
    signals: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of,
            "action": self.action,
            "target_weight": self.target_weight,
            "reasons": self.reasons,
            "macro_block": self.macro_block,
            "signals": self.signals,
        }


def _maybe_predict(ticker: str, latest_row, horizon_days: int) -> float | None:
    try:
        bundle = load_model(ticker, horizon_days)
    except FileNotFoundError:
        return None
    model = bundle["model"]
    cols = bundle["features"]
    return float(model.predict(latest_row[cols])[0])


def decide(
    ticker: str,
    horizon_days: int = 5,
    history_period: str = "1y",
    news_limit: int = 10,
    cfg: ShieldConfig | None = None,
) -> Decision:
    cfg = cfg or shield_config
    row = build_inference_row(ticker, period=history_period)
    last = row.iloc[-1]
    as_of = str(row.index[-1].date())

    vix = float(last.get("vix")) if "vix" in row.columns else None
    yield_spread = float(last.get("yield_spread")) if "yield_spread" in row.columns else None
    rsi = float(last.get("rsi_14")) if "rsi_14" in row.columns else None

    blocked, macro_reasons = macro_block(vix, yield_spread, cfg)

    news = fetch_recent_news(ticker, limit=news_limit)
    sentiment = score_news(ticker, news)

    predicted_return = _maybe_predict(ticker, row, horizon_days)
    base_action, model_reasons = model_signal(predicted_return, sentiment.score, cfg)
    scale_ok, scale_reasons = scale_in_ok(rsi, sentiment.score, cfg)

    reasons: list[str] = []
    reasons.extend(macro_reasons)
    reasons.extend(model_reasons)

    if blocked:
        action: Action = "BLOCK"
        target_weight = 0.0
        reasons.append("macro hard limit triggered → block all new buys")
    elif base_action == "BUY":
        action = "BUY"
        target_weight = cap_weight(cfg.max_position_weight, cfg)
    elif base_action == "REDUCE":
        action = "REDUCE"
        target_weight = 0.0
    else:
        if scale_ok:
            action = "BUY"
            target_weight = cap_weight(cfg.max_position_weight / 2, cfg)
            reasons.append("scale-in: oversold RSI + non-negative sentiment")
        else:
            action = "HOLD"
            target_weight = 0.0
            reasons.extend(scale_reasons)

    return Decision(
        ticker=ticker.upper(),
        as_of=as_of,
        action=action,
        target_weight=target_weight,
        reasons=reasons,
        macro_block=blocked,
        signals={
            "vix": vix,
            "yield_spread": yield_spread,
            "rsi_14": rsi,
            "sentiment": sentiment.as_dict(),
            "predicted_return": predicted_return,
            "horizon_days": horizon_days,
        },
    )
