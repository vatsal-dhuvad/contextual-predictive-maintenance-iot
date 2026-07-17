# Predictive maintenance feature engineering
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/context_fused.csv"
)

OUTPUT_PATH = Path(
    "data/processed/context_features.csv"
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

    df["temperature_difference_k"] = (
        df["process_temperature_k"]
        - df["air_temperature_k"]
    )

    df["mechanical_power_proxy"] = (
        df["torque_nm"]
        * df["rotational_speed_rpm"]
    )

    df["wear_torque_interaction"] = (
        df["tool_wear_min"]
        * df["torque_nm"]
    )

    df["load_temperature_interaction"] = (
        df["factory_load_index"]
        * df["ambient_temperature_c"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    created_features = [
        "temperature_difference_k",
        "mechanical_power_proxy",
        "wear_torque_interaction",
        "load_temperature_interaction",
    ]

    print(df[created_features].head())
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()