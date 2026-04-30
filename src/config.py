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
        def _clean(value: str | None) -> str | None:
            if value is None:
                return None
            cleaned = value.strip().strip("﻿").strip("\"'")
            return cleaned or None

        return cls(
            gemini_api_key=_clean(os.getenv("GEMINI_API_KEY")),
            fred_api_key=_clean(os.getenv("FRED_API_KEY")),
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
