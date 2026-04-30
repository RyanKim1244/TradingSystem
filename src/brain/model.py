"""XGBoost 회귀 모델: 미래 수익률 예측."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

MODELS_DIR = Path("models")


@dataclass
class TrainReport:
    ticker: str
    horizon_days: int
    n_train: int
    n_test: int
    train_mae: float
    test_mae: float
    test_r2: float
    feature_importance: dict[str, float]
    model_path: Path

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "horizon_days": self.horizon_days,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_mae": self.train_mae,
            "test_mae": self.test_mae,
            "test_r2": self.test_r2,
            "feature_importance": self.feature_importance,
            "model_path": str(self.model_path),
        }


def _time_split(X: pd.DataFrame, y: pd.Series, test_frac: float = 0.2):
    cut = int(len(X) * (1 - test_frac))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def train(
    ticker: str,
    X: pd.DataFrame,
    y: pd.Series,
    horizon_days: int,
    test_frac: float = 0.2,
) -> TrainReport:
    X_tr, X_te, y_tr, y_te = _time_split(X, y, test_frac=test_frac)

    model = XGBRegressor(
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
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    train_mae = float(mean_absolute_error(y_tr, model.predict(X_tr)))
    pred_te = model.predict(X_te)
    test_mae = float(mean_absolute_error(y_te, pred_te))
    test_r2 = float(r2_score(y_te, pred_te)) if len(y_te) > 1 else float("nan")

    importance = dict(
        sorted(
            zip(X.columns, map(float, model.feature_importances_)),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{ticker.upper()}_h{horizon_days}.pkl"
    joblib.dump({"model": model, "features": list(X.columns), "horizon_days": horizon_days}, model_path)

    return TrainReport(
        ticker=ticker.upper(),
        horizon_days=horizon_days,
        n_train=len(X_tr),
        n_test=len(X_te),
        train_mae=train_mae,
        test_mae=test_mae,
        test_r2=test_r2,
        feature_importance=importance,
        model_path=model_path,
    )


def load(ticker: str, horizon_days: int) -> dict:
    path = MODELS_DIR / f"{ticker.upper()}_h{horizon_days}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run training first.")
    return joblib.load(path)


def predict(ticker: str, latest_row: pd.DataFrame, horizon_days: int) -> dict:
    bundle = load(ticker, horizon_days)
    model = bundle["model"]
    cols = bundle["features"]
    X = latest_row[cols]
    yhat = float(model.predict(X)[0])
    return {
        "ticker": ticker.upper(),
        "as_of": str(X.index[-1].date()),
        "horizon_days": horizon_days,
        "predicted_return": yhat,
        "predicted_return_pct": yhat * 100.0,
        "features": {col: float(X.iloc[-1][col]) if pd.notna(X.iloc[-1][col]) else None for col in cols},
    }
