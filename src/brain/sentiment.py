"""Gemini로 뉴스 헤드라인을 -1.0 ~ +1.0 센티먼트 점수로 변환."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from google import genai

from ..config import settings
from ..sensors.news import NewsItem

_PROMPT = """You are a quantitative equity analyst.
For ticker {ticker}, read the news items below and judge the *forward* market impact.
Return JSON with fields:
  score: float in [-1.0, +1.0]   (-1 = strongly bearish, +1 = strongly bullish)
  confidence: float in [0.0, 1.0]
  rationale: one short sentence

Only respond with the JSON object. Do not wrap it in markdown.

NEWS:
{news_block}
"""


@dataclass
class SentimentResult:
    score: float
    confidence: float
    rationale: str
    n_items: int

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "n_items": self.n_items,
        }


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response: {text!r}")
    return json.loads(match.group(0))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def score_news(
    ticker: str,
    items: list[NewsItem],
    model_name: str = "gemini-2.5-flash",
) -> SentimentResult:
    """뉴스 헤드라인 묶음을 단일 센티먼트 점수로 환산."""
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    if not items:
        return SentimentResult(score=0.0, confidence=0.0, rationale="no news", n_items=0)

    client = genai.Client(api_key=settings.gemini_api_key)
    news_block = "\n\n".join(item.to_prompt_block() for item in items)
    prompt = _PROMPT.format(ticker=ticker, news_block=news_block)

    response = client.models.generate_content(model=model_name, contents=prompt)
    payload = _extract_json(response.text)
    return SentimentResult(
        score=_clamp(payload.get("score", 0.0), -1.0, 1.0),
        confidence=_clamp(payload.get("confidence", 0.0), 0.0, 1.0),
        rationale=str(payload.get("rationale", "")).strip(),
        n_items=len(items),
    )
