from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/telemetry_signal_features.csv"
)

CONTEXT_PATH = Path(
    "data/processed/external_context.csv"
)

OUTPUT_PATH = Path(
    "data/processed/context_fused.csv"
)


def create_context(
    df: pd.DataFrame
) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    hour = df["timestamp"].dt.hour.to_numpy()

    return pd.DataFrame({
        "timestamp": df["timestamp"],
        "ambient_temperature_c": (
            df["air_temperature_k"].to_numpy()
            - 273.15
            + rng.normal(0, 1.5, len(df))
        ),
        "humidity_pct": np.clip(
            55
            + 10 * np.sin(2 * np.pi * hour / 24)
            + rng.normal(0, 4, len(df)),
            20,
            95
        ),
        "factory_load_index": np.clip(
            0.65
            + 0.15 * np.sin(2 * np.pi * hour / 24)
            + rng.normal(0, 0.05, len(df)),
            0,
            1
        ),
    })


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} does not exist."
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["timestamp"]
    )

    context = create_context(df)

    CONTEXT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    context.to_csv(
        CONTEXT_PATH,
        index=False
    )

    fused = df.merge(
        context,
        on="timestamp",
        how="left",
        validate="one_to_one"
    )

    if len(fused) != len(df):
        raise ValueError(
            "Context fusion changed the number of rows."
        )

    context_columns = [
        "ambient_temperature_c",
        "humidity_pct",
        "factory_load_index",
    ]

    if fused[
        context_columns
    ].isnull().any().any():
        raise ValueError(
            "Missing contextual values after fusion."
        )

    fused.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(fused[[
        "timestamp",
        *context_columns
    ]].head())

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()