# Sensor noise injection and robustness testing
import numpy as np
import pandas as pd


def inject_noise(
    X: pd.DataFrame,
    noise_level: float,
    random_state: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(
        random_state
    )

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

        noisy[column] = (
            noisy[column] + noise
        )

    return noisy


if __name__ == "__main__":
    sample = pd.DataFrame({
        "sensor_a": [1.0, 2.0, 3.0],
        "sensor_b": [10.0, 20.0, 30.0],
    })

    print(inject_noise(
        sample,
        noise_level=0.10
    ))