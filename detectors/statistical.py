"""
Statistical anomaly detectors: Z-score, IQR, and CUSUM for time series.
"""
from __future__ import annotations
import numpy as np


def zscore_anomalies(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Flag observations more than `threshold` standard deviations from mean."""
    z = np.abs((values - np.mean(values)) / (np.std(values) + 1e-9))
    return (z > threshold).astype(int)


def iqr_anomalies(values: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Tukey IQR method: flag values outside [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return ((values < lower) | (values > upper)).astype(int)


def cusum(values: np.ndarray, target: float = None, k: float = 0.5,
          h: float = 5.0) -> np.ndarray:
    """
    Cumulative Sum control chart.
    Detects sustained shifts from target mean.
    k: allowance (half the tolerable shift), h: decision interval.
    """
    mu = target if target is not None else np.mean(values)
    std = np.std(values)
    if std == 0:
        return np.zeros(len(values), dtype=int)

    s_pos = np.zeros(len(values))
    s_neg = np.zeros(len(values))
    alarms = np.zeros(len(values), dtype=int)

    for i in range(1, len(values)):
        xi = (values[i] - mu) / std
        s_pos[i] = max(0, s_pos[i-1] + xi - k)
        s_neg[i] = max(0, s_neg[i-1] - xi - k)
        if s_pos[i] > h or s_neg[i] > h:
            alarms[i] = 1

    return alarms
