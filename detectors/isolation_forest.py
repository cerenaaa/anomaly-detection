"""
Isolation Forest anomaly detector with calibrated scores.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


@dataclass
class AnomalyResult:
    scores: np.ndarray
    labels: np.ndarray
    threshold: float
    contamination: float


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 random_state: int = 42):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=n_estimators,
                                      contamination=contamination,
                                      random_state=random_state)

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        return self

    def predict(self, X: np.ndarray) -> AnomalyResult:
        Xs = self.scaler.transform(X)
        raw_scores = -self.model.score_samples(Xs)
        score_range = raw_scores.max() - raw_scores.min()
        scores = (raw_scores - raw_scores.min()) / (score_range + 1e-9)
        threshold = np.percentile(scores, (1 - self.contamination) * 100)
        labels = (scores >= threshold).astype(int)
        return AnomalyResult(scores=scores, labels=labels,
                              threshold=threshold, contamination=self.contamination)
