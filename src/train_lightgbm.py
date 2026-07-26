from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)


INPUT_PATH = Path(
    "data/processed/modeling_dataset.csv"
)

MODEL_PATH = Path(
    "models/lightgbm_smote_pipeline.joblib"
)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    X = df.drop(
        columns=["machine_failure"]
    )

    y = df["machine_failure"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=42
        )
    )

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
                    n_estimators=400,
                    learning_rate=0.04,
                    num_leaves=31,
                    random_state=42,
                    n_jobs=-1,
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
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    print("CV Macro F1 scores:", scores)
    print("Mean CV Macro F1:", scores.mean())

    pipeline.fit(
        X_train,
        y_train
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print(f"Model saved locally to {MODEL_PATH}")


if __name__ == "__main__":
    main()