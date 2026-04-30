# Alpha-Engine

거시경제(Macro), 기업 가치(Fundamental), 시장 심리(Sentiment) 데이터를 융합하여
초과 수익(Alpha)을 추구하는 자동화 매매 시스템.

## 4-Tier 아키텍처

1. **Sensors** (`src/sensors/`) — 시장 데이터 수집 (yfinance, FRED, 뉴스)
2. **Brain** (`src/brain/`) — Gemini 감성 분석 + XGBoost 예측 (Phase 2)
3. **Shield** (`src/shield/`) — VIX/펀더멘털 기반 리스크 관리 (Phase 3)
4. **Execution** (`src/execution/`) — 시그널 산출 및 Telegram 알림 (Phase 5)

## 개발 단계

| Phase | 내용 | 상태 |
|-------|------|------|
| 1 | 데이터 센서 + Gemini 센티먼트 엔진 | 완료 |
| 2 | XGBoost 예측 모델 | 완료 |
| 3 | 리스크 방패 로직 | 예정 |
| 4 | Backtrader 백테스팅 | 예정 |
| 5 | AWS 배포 + Telegram 봇 | 예정 |

## 시작하기

```bash
pip install -r requirements.txt
cp .env.example .env   # 발급받은 API 키 입력

# Phase 1: 단일 종목 데이터 + 센티먼트 스냅샷
python -m src.main AAPL

# Phase 2: 학습 → 예측
python -m src.brain.train AAPL --period 5y --horizon 5
python -m src.brain.predict AAPL --horizon 5
```

### API 키 발급
- **Gemini**: https://aistudio.google.com/app/apikey
- **FRED**: https://fred.stlouisfed.org/docs/api/api_key.html
