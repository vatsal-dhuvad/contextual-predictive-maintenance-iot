from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/telemetry_clean.csv")
OUTPUT_PATH = Path(
    "data/processed/telemetry_signal_features.csv"
)

SENSOR_COLUMNS = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

ROLLING_WINDOW = 20


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} does not exist. "
            "Run data_ingestion.py first."
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    missing_columns = [
        column
        for column in SENSOR_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing sensor columns: {missing_columns}"
        )

    for column in SENSOR_COLUMNS:
        rolling = df[column].rolling(
            window=ROLLING_WINDOW,
            min_periods=1
        )

        df[f"{column}_rolling_mean"] = (
            rolling.mean()
        )

        df[f"{column}_rolling_std"] = (
            rolling.std().fillna(0)
        )

        df[f"{column}_rolling_var"] = (
            rolling.var().fillna(0)
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    generated_columns = [
        column
        for column in df.columns
        if "rolling_" in column
    ]

    print(
        "Generated rolling features:",
        len(generated_columns)
    )

    print(df[generated_columns].head())
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()