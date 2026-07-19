from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/context_fused.csv"
)

OUTPUT_PATH = Path(
    "data/processed/modeling_dataset.csv"
)

LEAKAGE_COLUMNS = [
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
]


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

    existing_leakage_columns = [
        column
        for column in LEAKAGE_COLUMNS
        if column in df.columns
    ]

    df = df.drop(
        columns=existing_leakage_columns
    )

    identifier_columns = [
        column
        for column in [
            "udi",
            "product_id",
            "timestamp",
        ]
        if column in df.columns
    ]

    df = df.drop(
        columns=identifier_columns
    )

    if "type" in df.columns:
        df = pd.get_dummies(
            df,
            columns=["type"],
            drop_first=False,
            dtype=int
        )

    non_numeric_columns = (
        df.select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    if non_numeric_columns:
        raise ValueError(
            "Non-numeric columns remain: "
            f"{non_numeric_columns}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("Final shape:", df.shape)
    print("\nTarget distribution:")
    print(df["machine_failure"].value_counts())
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()