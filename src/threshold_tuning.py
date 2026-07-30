# Precision-recall analysis and threshold tuning
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path(
    "data/processed/modeling_dataset.csv"
)

MODEL_PATH = Path(
    "models/lightgbm_smote_pipeline.joblib"
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

    precision, recall, _ = (
        precision_recall_curve(
            y_test,
            probabilities
        )
    )

    average_precision = (
        average_precision_score(
            y_test,
            probabilities
        )
    )

    print(
        "Average Precision:",
        round(average_precision, 4)
    )

    PLOT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
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


if __name__ == "__main__":
    main()