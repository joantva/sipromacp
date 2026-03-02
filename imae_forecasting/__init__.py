"""Herramientas para construir, evaluar y usar modelos de proyección del IMAE."""

from .modeling import train_and_evaluate_models
from .forecasting import generate_12m_forecast

__all__ = ["train_and_evaluate_models", "generate_12m_forecast"]
