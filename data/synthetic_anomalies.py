"""
Synthetic anomaly datasets: tabular transactions and time series with injected anomalies.
"""
import numpy as np
import pandas as pd


def generate_transactions(n_normal: int = 5000, n_anomalies: int = 100,
                           seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Normal transactions
    normal = pd.DataFrame({
        "amount": rng.lognormal(4.0, 0.8, n_normal),
        "hour": rng.integers(8, 22, n_normal),
        "merchant_category": rng.integers(0, 10, n_normal),
        "distance_from_home_km": rng.exponential(15, n_normal),
        "n_transactions_24h": rng.poisson(3, n_normal),
        "is_anomaly": 0,
    })
    # Anomalous transactions (unusual patterns)
    anomalies = pd.DataFrame({
        "amount": rng.lognormal(8.0, 1.5, n_anomalies),      # very high amounts
        "hour": rng.integers(0, 6, n_anomalies),               # unusual hours
        "merchant_category": rng.integers(8, 10, n_anomalies),
        "distance_from_home_km": rng.exponential(500, n_anomalies),  # far away
        "n_transactions_24h": rng.poisson(15, n_anomalies),   # burst activity
        "is_anomaly": 1,
    })
    df = pd.concat([normal, anomalies], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Generated {len(df)} transactions | anomaly rate = {df['is_anomaly'].mean():.1%}")
    return df


def generate_time_series_with_anomalies(n: int = 500, anomaly_rate: float = 0.05,
                                         seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 100 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 50) + rng.normal(0, 2, n)
    labels = np.zeros(n, dtype=int)
    n_anomalies = int(n * anomaly_rate)
    anomaly_idx = rng.choice(n, n_anomalies, replace=False)
    values[anomaly_idx] += rng.choice([-1, 1], n_anomalies) * rng.uniform(20, 50, n_anomalies)
    labels[anomaly_idx] = 1
    df = pd.DataFrame({"t": t, "value": values.round(3), "is_anomaly": labels})
    print(f"Generated time series n={n} | anomaly rate = {labels.mean():.1%}")
    return df
