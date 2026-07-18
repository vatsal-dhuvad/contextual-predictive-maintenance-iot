# Internal features versus contextual features comparison
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)


INPUT_PATH = Path(
    "data/processed/context_features.csv"
)

CONTEXT_COLUMNS = [
    "ambient_temperature_c",
    "humidity_pct",
    "factory_load_index",
    "load_temperature_interaction",
]


def evaluate_features(
    X: pd.DataFrame,
    y: pd.Series,
    label: str
) -> float:
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    mean_score = float(scores.mean())

    print(f"\n{label}")
    print("Fold scores:", scores)
    print("Mean Macro F1:", round(mean_score, 4))

    return mean_score


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    y = df["machine_failure"]

    drop_columns = [
        column
        for column in [
            "machine_failure",
            "timestamp",
            "product_id",
            "udi",
            "type",
            "twf",
            "hdf",
            "pwf",
            "osf",
            "rnf",
        ]
        if column in df.columns
    ]

    X_all = df.drop(
        columns=drop_columns
    )

    X_all = X_all.select_dtypes(
        include="number"
    )

    available_context_columns = [
        column
        for column in CONTEXT_COLUMNS
        if column in X_all.columns
    ]

    X_internal = X_all.drop(
        columns=available_context_columns
    )

    internal_score = evaluate_features(
        X_internal,
        y,
        "Internal telemetry only"
    )

    contextual_score = evaluate_features(
        X_all,
        y,
        "Internal plus contextual features"
    )

    improvement = (
        contextual_score - internal_score
    )

    print(
        "\nMacro F1 improvement:",
        round(improvement, 4)
    )


if __name__ == "__main__":
    main()