"""순수 규칙 함수: 입력 신호 → (allow_buy, reasons)."""
from __future__ import annotations

from ..config import ShieldConfig


def macro_block(vix: float, yield_spread: float, cfg: ShieldConfig) -> tuple[bool, list[str]]:
    """거시경제 하드 리미트 — True면 전 종목 신규 매수 차단."""
    reasons: list[str] = []
    if vix is not None and vix > cfg.vix_block:
        reasons.append(f"VIX {vix:.1f} > {cfg.vix_block} (panic regime)")
    if yield_spread is not None and yield_spread < cfg.yield_spread_block:
        reasons.append(f"yield_spread {yield_spread:.2f} < {cfg.yield_spread_block} (deep inversion)")
    return bool(reasons), reasons


def scale_in_ok(rsi: float, sentiment_score: float, cfg: ShieldConfig) -> tuple[bool, list[str]]:
    """안전장치 결합형 1차 매수 조건."""
    reasons: list[str] = []
    sentiment_ok = sentiment_score is not None and sentiment_score >= cfg.sentiment_floor
    rsi_oversold = rsi is not None and rsi <= cfg.rsi_oversold

    if not sentiment_ok:
        reasons.append(f"sentiment {sentiment_score} < floor {cfg.sentiment_floor}")
    if not rsi_oversold:
        reasons.append(f"RSI {rsi} above oversold {cfg.rsi_oversold}")
    return sentiment_ok and rsi_oversold, reasons


def model_signal(
    predicted_return: float | None,
    sentiment_score: float,
    cfg: ShieldConfig,
) -> tuple[str, list[str]]:
    """모델 예측 + 센티먼트 결합 → 'BUY' / 'HOLD' / 'REDUCE'."""
    reasons: list[str] = []
    if predicted_return is None:
        reasons.append("no model prediction available")
        return "HOLD", reasons

    if predicted_return >= cfg.model_buy_threshold and sentiment_score >= cfg.sentiment_floor:
        reasons.append(
            f"model E[r]={predicted_return:.4f} >= {cfg.model_buy_threshold} & sentiment {sentiment_score:.2f} ok"
        )
        return "BUY", reasons
    if predicted_return <= cfg.model_sell_threshold:
        reasons.append(f"model E[r]={predicted_return:.4f} <= {cfg.model_sell_threshold}")
        return "REDUCE", reasons
    reasons.append(f"model E[r]={predicted_return:.4f} in dead zone")
    return "HOLD", reasons


def cap_weight(target_weight: float, cfg: ShieldConfig) -> float:
    return max(0.0, min(target_weight, cfg.max_position_weight))
