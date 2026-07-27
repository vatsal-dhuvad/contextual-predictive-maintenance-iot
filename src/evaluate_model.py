from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path(
    "data/processed/modeling_dataset.csv"
)

MODEL_PATH = Path(
    "models/lightgbm_smote_pipeline.joblib"
)

METRICS_PATH = Path(
    "reports/lightgbm_metrics.json"
)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run train_lightgbm.py first."
        )

    df = pd.read_csv(DATA_PATH)

    X = df.drop(
        columns=["machine_failure"]
    )

    y = df["machine_failure"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)

    metrics = {
        "test_macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro"
            )
        ),
        "failure_precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),
        "failure_recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions
        ).tolist(),
    }

    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print(json.dumps(
        metrics,
        indent=2
    ))

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2
        )


if __name__ == "__main__":
    main()