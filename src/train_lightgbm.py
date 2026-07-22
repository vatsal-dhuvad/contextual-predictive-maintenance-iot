# LightGBM training with SMOTE and cross-validation
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/modeling_dataset.csv"
)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} does not exist."
        )

    df = pd.read_csv(INPUT_PATH)

    y = df["machine_failure"]

    print("Class counts:")
    print(y.value_counts())

    print("\nClass percentages:")
    print(
        y.value_counts(normalize=True)
        * 100
    )


if __name__ == "__main__":
    main()