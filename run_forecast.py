from __future__ import annotations

import argparse

from imae_forecasting.forecasting import generate_12m_forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="Uso del modelo para proyección IMAE")
    parser.add_argument("--data", default="data.xlsx", help="Ruta al archivo Excel de entrada")
    parser.add_argument("--artifacts", default="artifacts", help="Directorio con modelo entrenado")
    parser.add_argument("--output", default="artifacts/imae_forecast_12m.csv", help="CSV de proyección")
    parser.add_argument("--horizon", default=12, type=int, help="Horizonte de proyección")
    args = parser.parse_args()

    forecast = generate_12m_forecast(
        data_path=args.data,
        artifacts_dir=args.artifacts,
        output_path=args.output,
        horizon=args.horizon,
    )
    print(forecast[["imae_proyectado"]].tail(args.horizon))


if __name__ == "__main__":
    main()
