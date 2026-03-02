from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_utils import load_imae_dataset, split_features_target


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _time_series_cv_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes: List[float] = []
    rmses: List[float] = []
    r2s: List[float] = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        maes.append(mean_absolute_error(y_test, preds))
        rmses.append(_rmse(y_test, preds))
        r2s.append(r2_score(y_test, preds))

    return {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "r2": float(np.mean(r2s)),
    }


def train_and_evaluate_models(
    data_path: str = "data.xlsx",
    output_dir: str = "artifacts",
    n_splits: int = 5,
) -> Dict[str, object]:
    """Construye y evalúa modelos candidatos para explicar/proyectar el IMAE."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_imae_dataset(data_path)
    X, y = split_features_target(bundle.data, bundle.target_col)

    models: Dict[str, Pipeline] = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline(
            [("model", RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=2))]
        ),
    }

    results = []
    for name, model in models.items():
        metrics = _time_series_cv_metrics(model, X, y, n_splits=n_splits)
        results.append({"model": name, **metrics})

    results_df = pd.DataFrame(results).sort_values("rmse")
    best_name = str(results_df.iloc[0]["model"])
    best_model = models[best_name]
    best_model.fit(X, y)

    model_path = out / "best_imae_model.joblib"
    eval_path = out / "model_evaluation.csv"
    metadata_path = out / "model_metadata.json"

    joblib.dump(best_model, model_path)
    results_df.to_csv(eval_path, index=False)

    importances = {}
    if best_name == "random_forest":
        rf = best_model.named_steps["model"]
        importances = dict(sorted(zip(X.columns, rf.feature_importances_), key=lambda kv: kv[1], reverse=True))
    else:
        est = best_model.named_steps["model"]
        if hasattr(est, "coef_"):
            importances = dict(
                sorted(zip(X.columns, np.abs(est.coef_)), key=lambda kv: kv[1], reverse=True)
            )

    metadata = {
        "target": bundle.target_col,
        "n_obs": int(len(X)),
        "selected_model": best_name,
        "features": list(X.columns),
        "top_feature_importance": dict(list(importances.items())[:10]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "evaluation": results_df,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "features": list(X.columns),
        "target": bundle.target_col,
    }
