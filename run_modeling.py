from __future__ import annotations

import argparse

from imae_forecasting.modeling import train_and_evaluate_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Construcción y evaluación de modelos para IMAE")
    parser.add_argument("--data", default="data.xlsx", help="Ruta al archivo Excel de entrada")
    parser.add_argument("--out", default="artifacts", help="Directorio de salida")
    parser.add_argument("--splits", default=5, type=int, help="Número de cortes TimeSeriesSplit")
    args = parser.parse_args()

    result = train_and_evaluate_models(data_path=args.data, output_dir=args.out, n_splits=args.splits)
    print("Modelo seleccionado:", result["evaluation"].iloc[0]["model"])
    print("Métricas guardadas en:", f"{args.out}/model_evaluation.csv")


if __name__ == "__main__":
    main()
