from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


@dataclass
class DatasetBundle:
    data: pd.DataFrame
    target_col: str


def load_imae_dataset(path: str | Path) -> DatasetBundle:
    """Carga data.xlsx y detecta la columna objetivo IMAE."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_excel(path)
    if df.empty:
        raise ValueError("El archivo Excel está vacío.")

    original_cols = list(df.columns)
    df.columns = [str(c).strip() for c in original_cols]

    date_candidates = [c for c in df.columns if "fecha" in c.lower() or "date" in c.lower()]
    if date_candidates:
        date_col = date_candidates[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)
    elif isinstance(df.index, pd.RangeIndex):
        df.index = pd.date_range("2000-01-01", periods=len(df), freq="MS")

    target_candidates = [c for c in df.columns if "imae" in c.lower()]
    if not target_candidates:
        raise ValueError("No se encontró una columna objetivo que contenga 'IMAE' en su nombre.")

    target_col = target_candidates[0]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="all")
    if df[target_col].isna().all():
        raise ValueError("La columna objetivo IMAE no tiene datos numéricos válidos.")

    return DatasetBundle(data=df, target_col=target_col)


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    model_df = df.dropna().copy()
    X = model_df.drop(columns=[target_col])
    y = model_df[target_col]
    if X.empty:
        raise ValueError("No hay indicadores explicativos disponibles tras limpiar datos.")
    return X, y
