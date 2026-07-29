from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path(
    "data/processed/modeling_dataset.csv"
)

MODEL_PATH = Path(
    "models/lightgbm_smote_pipeline.joblib"
)

RESULT_PATH = Path(
    "reports/noise_robustness.csv"
)

PLOT_PATH = Path(
    "reports/noise_robustness.png"
)

NOISE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.20,
]


def inject_noise(
    X: pd.DataFrame,
    noise_level: float,
    rng: np.random.Generator
) -> pd.DataFrame:
    noisy = X.copy()

    continuous_columns = [
        column
        for column in noisy.columns
        if noisy[column].nunique() > 2
    ]

    for column in continuous_columns:
        column_std = noisy[column].std()

        if column_std == 0 or pd.isna(column_std):
            continue

        noise = rng.normal(
            loc=0,
            scale=column_std * noise_level,
            size=len(noisy)
        )

        noisy[column] += noise

    return noisy


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

    results = []

    for noise_level in NOISE_LEVELS:
        rng = np.random.default_rng(42)

        X_noisy = inject_noise(
            X_test,
            noise_level,
            rng
        )

        predictions = model.predict(
            X_noisy
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        results.append({
            "noise_level": noise_level,
            "macro_f1": macro_f1
        })

        print(
            f"Noise {noise_level:.2f} "
            f"→ Macro F1 {macro_f1:.4f}"
        )

    result_df = pd.DataFrame(results)

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        RESULT_PATH,
        index=False
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        result_df["noise_level"],
        result_df["macro_f1"],
        marker="o"
    )

    plt.xlabel("Noise Level")
    plt.ylabel("Macro F1")
    plt.title(
        "Model Robustness under Sensor Noise"
    )
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=200
    )

    plt.close()


if __name__ == "__main__":
    main()