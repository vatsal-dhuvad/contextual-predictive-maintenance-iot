# Dataset loading, validation, and cleaning
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/ai4i2020.csv")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Download AI4I 2020 and place it inside data/."
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst five rows:")
    print(df.head())


if __name__ == "__main__":
    main()