"""티커별 최신 뉴스 헤드라인 수집."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import yfinance as yf


@dataclass
class NewsItem:
    title: str
    publisher: str
    link: str
    published_at: datetime | None
    summary: str | None = None

    def to_prompt_block(self) -> str:
        when = self.published_at.strftime("%Y-%m-%d") if self.published_at else "?"
        body = self.summary or ""
        return f"[{when}] ({self.publisher}) {self.title}\n{body}".strip()


def _parse_timestamp(raw: int | float | None) -> datetime | None:
    if raw is None:
        return None
    try:
        return datetime.utcfromtimestamp(int(raw))
    except (ValueError, OSError):
        return None


def fetch_recent_news(ticker: str, limit: int = 10) -> list[NewsItem]:
    """yfinance가 노출하는 Yahoo Finance 뉴스 피드를 정규화된 형태로 반환."""
    raw_items = yf.Ticker(ticker).news or []
    items: list[NewsItem] = []
    for raw in raw_items[:limit]:
        content = raw.get("content", raw)
        title = content.get("title") or raw.get("title")
        if not title:
            continue
        publisher = (
            content.get("provider", {}).get("displayName")
            if isinstance(content.get("provider"), dict)
            else raw.get("publisher", "unknown")
        )
        link = (
            content.get("canonicalUrl", {}).get("url")
            if isinstance(content.get("canonicalUrl"), dict)
            else raw.get("link", "")
        )
        published_at = _parse_timestamp(raw.get("providerPublishTime"))
        if published_at is None and content.get("pubDate"):
            try:
                published_at = datetime.fromisoformat(content["pubDate"].replace("Z", "+00:00"))
            except ValueError:
                published_at = None
        items.append(
            NewsItem(
                title=title,
                publisher=publisher or "unknown",
                link=link or "",
                published_at=published_at,
                summary=content.get("summary"),
            )
        )
    return items
