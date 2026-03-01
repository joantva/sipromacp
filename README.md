# Proyección del IMAE con Machine Learning

Este repositorio contiene dos módulos:

1. **Construcción y evaluación de modelos** (`imae_forecasting/modeling.py`):
   - Carga `data.xlsx`.
   - Detecta automáticamente la columna objetivo `IMAE`.
   - Evalúa varios modelos (Regresión Lineal, Ridge y Random Forest) con `TimeSeriesSplit`.
   - Selecciona y guarda el mejor modelo.
   - Exporta métricas e importancia de indicadores.

2. **Uso del modelo y proyección a 12 meses** (`imae_forecasting/forecasting.py`):
   - Toma el mejor modelo entrenado.
   - Proyecta indicadores explicativos para los próximos meses.
   - Genera la proyección de IMAE para horizonte configurable (por defecto 12 meses).

## Estructura

- `run_modeling.py`: script CLI para entrenamiento/evaluación.
- `run_forecast.py`: script CLI para uso y proyecciones.
- `artifacts/` (se crea automáticamente):
  - `best_imae_model.joblib`
  - `model_evaluation.csv`
  - `model_metadata.json`
  - `imae_forecast_12m.csv`

## Uso

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Entrenar y evaluar modelos:

```bash
python run_modeling.py --data data.xlsx --out artifacts --splits 5
```

Proyectar próximos 12 meses:

```bash
python run_forecast.py --data data.xlsx --artifacts artifacts --output artifacts/imae_forecast_12m.csv --horizon 12
```

## Supuestos del archivo `data.xlsx`

- Debe contener una columna de fecha (`fecha` o `date`) o, en su defecto, se asumirá frecuencia mensual consecutiva.
- Debe existir al menos una columna cuyo nombre contenga `IMAE` (objetivo).
- El resto de columnas numéricas se usan como indicadores explicativos.
