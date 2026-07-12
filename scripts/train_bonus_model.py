from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.churn_model import DEFAULT_CLEAN_DATA_PATH, DEFAULT_MODEL_PATH, train_model_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina o modelo usado pela interface web da fase bonus."
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_CLEAN_DATA_PATH),
        help="Caminho para data/cleaned_dataset.csv.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_MODEL_PATH),
        help="Caminho do artefato .pkl gerado.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        metadata = train_model_artifact(args.data, args.output)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Erro ao treinar modelo: {exc}", file=sys.stderr)
        return 1

    print(f"Modelo salvo em: {args.output}")
    print(f"Modelo: {metadata['model_name']}")
    print("Metricas no teste:")
    for metric, value in metadata["metrics"].items():
        print(f"- {metric}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
