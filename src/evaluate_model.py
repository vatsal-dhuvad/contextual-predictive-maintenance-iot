# Classification model evaluation
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path(
    "data/processed/telemetry_signal_features.csv"
)

REPORT_DIR = Path("reports")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} does not exist."
        )

    df = pd.read_csv(INPUT_PATH)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Shape:", df.shape)

    class_counts = (
        df["machine_failure"]
        .value_counts()
        .sort_index()
    )

    class_percentages = (
        df["machine_failure"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print("\nClass counts:")
    print(class_counts)

    print("\nClass percentages:")
    print(class_percentages)

    plt.figure(figsize=(7, 5))
    class_counts.plot(kind="bar")
    plt.title("Machine Failure Class Distribution")
    plt.xlabel("Machine Failure")
    plt.ylabel("Record Count")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "class_distribution.png",
        dpi=200
    )
    plt.close()

    sensor_columns = [
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ]

    df[sensor_columns].hist(
        figsize=(12, 8),
        bins=30
    )

    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "sensor_distributions.png",
        dpi=200
    )
    plt.close()

    print("EDA reports generated successfully.")


if __name__ == "__main__":
    main()