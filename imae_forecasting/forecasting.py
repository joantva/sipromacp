from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from .data_utils import load_imae_dataset


def _forecast_feature(series: pd.Series, horizon: int = 12) -> np.ndarray:
    """Proyección simple por estacionalidad mensual + leve tendencia lineal."""
    s = series.dropna()
    if len(s) < 24:
        return np.repeat(float(s.iloc[-1]), horizon)

    monthly = s.groupby(s.index.month).mean()
    if len(monthly) < 12:
        base = float(s.iloc[-1])
        return np.repeat(base, horizon)

    # tendencia lineal para ajustar nivel
    x = np.arange(len(s))
    coeffs = np.polyfit(x, s.values, deg=1)
    trend_slope = coeffs[0]

    forecasts = []
    last_date = s.index[-1]
    for h in range(1, horizon + 1):
        next_month = (last_date.month + h - 1) % 12 + 1
        seasonal_base = monthly.loc[next_month]
        trend_adj = trend_slope * h
        forecasts.append(float(seasonal_base + trend_adj))
    return np.array(forecasts)


def _build_future_features(df: pd.DataFrame, feature_cols: list[str], horizon: int = 12) -> pd.DataFrame:
    start = df.index.max() + pd.offsets.MonthBegin(1)
    future_index = pd.date_range(start, periods=horizon, freq="MS")

    future = pd.DataFrame(index=future_index)
    for col in feature_cols:
        future[col] = _forecast_feature(df[col], horizon=horizon)
    return future


def generate_12m_forecast(
    data_path: str = "data.xlsx",
    artifacts_dir: str = "artifacts",
    output_path: str = "artifacts/imae_forecast_12m.csv",
    horizon: int = 12,
) -> pd.DataFrame:
    """Genera proyección de IMAE a 12 meses usando el mejor modelo entrenado."""
    artifacts = Path(artifacts_dir)
    model = joblib.load(artifacts / "best_imae_model.joblib")
    metadata: Dict[str, object] = json.loads((artifacts / "model_metadata.json").read_text(encoding="utf-8"))

    bundle = load_imae_dataset(data_path)
    feature_cols = metadata["features"]

    future_X = _build_future_features(bundle.data, feature_cols, horizon=horizon)
    preds = model.predict(future_X)

    out_df = future_X.copy()
    out_df["imae_proyectado"] = preds
    out_df.index.name = "fecha"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path)
    return out_df
