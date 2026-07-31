from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path(
    "data/processed/modeling_dataset.csv"
)

MODEL_PATH = Path(
    "models/lightgbm_smote_pipeline.joblib"
)

RESULT_PATH = Path(
    "reports/threshold_metrics.json"
)

PLOT_PATH = Path(
    "reports/precision_recall_curve.png"
)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

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

    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    precision, recall, thresholds = (
        precision_recall_curve(
            y_test,
            probabilities
        )
    )

    threshold_precision = precision[:-1]
    threshold_recall = recall[:-1]

    failure_f1_values = (
        2
        * threshold_precision
        * threshold_recall
        / (
            threshold_precision
            + threshold_recall
            + 1e-12
        )
    )

    best_index = int(
        np.argmax(failure_f1_values)
    )

    best_threshold = float(
        thresholds[best_index]
    )

    default_predictions = (
        probabilities >= 0.50
    ).astype(int)

    tuned_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    results = {
        "average_precision": float(
            average_precision_score(
                y_test,
                probabilities
            )
        ),
        "best_threshold": best_threshold,
        "best_failure_f1": float(
            failure_f1_values[best_index]
        ),
        "default_macro_f1": float(
            f1_score(
                y_test,
                default_predictions,
                average="macro"
            )
        ),
        "tuned_macro_f1": float(
            f1_score(
                y_test,
                tuned_predictions,
                average="macro"
            )
        ),
    }

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=2
        )

    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        PLOT_PATH,
        dpi=200
    )
    plt.close()

    print(json.dumps(
        results,
        indent=2
    ))


if __name__ == "__main__":
    main()