from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "Customertravel.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "cleaned_dataset.csv"

INCOME_MAP = {
    "Low Income": 0,
    "Middle Income": 1,
    "High Income": 2,
}

BINARY_MAP = {
    "No": 0,
    "Yes": 1,
}

REQUIRED_COLUMNS = [
    "Age",
    "FrequentFlyer",
    "AnnualIncomeClass",
    "ServicesOpted",
    "AccountSyncedToSocialMedia",
    "BookedHotelOrNot",
    "Target",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Limpa e codifica o dataset Customertravel.csv.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="CSV bruto de entrada.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="CSV limpo de saida.")
    return parser


def validate_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes no CSV: {', '.join(missing_columns)}")


def clean_data(input_path: str | Path = DEFAULT_INPUT, output_path: str | Path = DEFAULT_OUTPUT) -> dict[str, int]:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"CSV bruto nao encontrado em {input_path}. "
            "Baixe o Customertravel.csv e coloque em data/Customertravel.csv."
        )

    df = pd.read_csv(input_path)
    validate_columns(df)

    initial_rows = len(df)
    duplicated_rows = int(df.duplicated().sum())

    df = df.copy()
    df.replace("No Record", pd.NA, inplace=True)
    null_rows = int(df.isna().any(axis=1).sum())
    df.dropna(inplace=True)

    df["AnnualIncomeClass"] = df["AnnualIncomeClass"].map(INCOME_MAP)
    df["FrequentFlyer"] = df["FrequentFlyer"].map(BINARY_MAP)
    df["AccountSyncedToSocialMedia"] = df["AccountSyncedToSocialMedia"].map(BINARY_MAP)
    df["BookedHotelOrNot"] = df["BookedHotelOrNot"].map(BINARY_MAP)

    encoded_null_rows = int(df.isna().any(axis=1).sum())
    if encoded_null_rows:
        raise ValueError(
            "Foram encontrados valores categoricos nao mapeados. "
            "Confira se as categorias seguem o formato esperado."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return {
        "initial_rows": initial_rows,
        "duplicated_rows": duplicated_rows,
        "removed_null_rows": null_rows,
        "final_rows": len(df),
    }


def main() -> int:
    args = build_parser().parse_args()

    try:
        summary = clean_data(args.input, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erro ao limpar dados: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset limpo salvo em: {args.output}")
    print(f"Linhas iniciais: {summary['initial_rows']}")
    print(f"Linhas duplicadas encontradas: {summary['duplicated_rows']}")
    print(f"Linhas removidas por nulos: {summary['removed_null_rows']}")
    print(f"Linhas finais: {summary['final_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
