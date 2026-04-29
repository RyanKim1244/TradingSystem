"""환경 변수 및 글로벌 상수 로더."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    fred_api_key: str | None

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            fred_api_key=os.getenv("FRED_API_KEY"),
        )


settings = Settings.load()

MACRO_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "yield_10y": "DGS10",
    "yield_2y": "DGS2",
    "vix": "VIXCLS",
}
