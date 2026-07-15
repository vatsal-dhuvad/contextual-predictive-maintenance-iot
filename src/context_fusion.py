# External contextual data simulation and fusion
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/telemetry_signal_features.csv"
)

CONTEXT_PATH = Path(
    "data/processed/external_context.csv"
)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} does not exist."
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["timestamp"]
    )

    rng = np.random.default_rng(42)

    hour = df["timestamp"].dt.hour.to_numpy()

    context = pd.DataFrame({
        "timestamp": df["timestamp"],
        "ambient_temperature_c": (
            df["air_temperature_k"].to_numpy()
            - 273.15
            + rng.normal(
                0,
                1.5,
                len(df)
            )
        ),
        "humidity_pct": np.clip(
            55
            + 10 * np.sin(
                2 * np.pi * hour / 24
            )
            + rng.normal(
                0,
                4,
                len(df)
            ),
            20,
            95
        ),
        "factory_load_index": np.clip(
            0.65
            + 0.15 * np.sin(
                2 * np.pi * hour / 24
            )
            + rng.normal(
                0,
                0.05,
                len(df)
            ),
            0,
            1
        ),
    })

    CONTEXT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    context.to_csv(
        CONTEXT_PATH,
        index=False
    )

    print(context.head())
    print(f"Saved to: {CONTEXT_PATH}")


if __name__ == "__main__":
    main()