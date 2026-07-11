from pathlib import Path
import re

import pandas as pd


RAW_DATA_PATH = Path("data/ai4i2020.csv")
OUTPUT_PATH = Path("data/processed/telemetry_clean.csv")


def clean_column_name(column: str) -> str:
    column = column.strip().lower()
    column = re.sub(r"[\[\]\(\)]", "", column)
    column = re.sub(r"[^a-z0-9]+", "_", column)
    return column.strip("_")


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_PATH}."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print("Original shape:", df.shape)

    df.columns = [
        clean_column_name(column)
        for column in df.columns
    ]

    print("\nCleaned columns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    duplicate_count = int(df.duplicated().sum())
    print("\nDuplicate rows:", duplicate_count)

    df = df.drop_duplicates().copy()

    if "machine_failure" not in df.columns:
        raise ValueError(
            "The machine_failure target column was not found."
        )

    df["timestamp"] = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=len(df),
        freq="min"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nClean shape:", df.shape)
    print("\nTarget distribution:")
    print(df["machine_failure"].value_counts())
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()