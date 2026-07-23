from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)


INPUT_PATH = Path(
    "data/processed/modeling_dataset.csv"
)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} does not exist."
        )

    df = pd.read_csv(INPUT_PATH)

    X = df.drop(
        columns=["machine_failure"]
    )

    y = df["machine_failure"]

    pipeline = Pipeline(
        steps=[
            (
                "smote",
                SMOTE(
                    random_state=42,
                    k_neighbors=3
                )
            ),
            (
                "model",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    random_state=42,
                    verbosity=-1
                )
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    print("Macro F1 scores:", scores)
    print("Mean Macro F1:", scores.mean())


if __name__ == "__main__":
    main()