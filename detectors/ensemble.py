"""Ensemble anomaly detector: fuses scores from multiple detectors."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class EnsembleResult:
    scores: np.ndarray
    labels: np.ndarray
    detector_weights: dict[str, float]


class AnomalyEnsemble:
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.detectors: dict = {}

    def add(self, name: str, detector):
        self.detectors[name] = detector
        return self

    def predict(self, X: np.ndarray) -> EnsembleResult:
        all_scores = {}
        for name, det in self.detectors.items():
            result = det.predict(X)
            s = result.scores if hasattr(result, "scores") else result
            s = np.array(s, dtype=float)
            # Normalize to [0,1]
            s = (s - s.min()) / (s.ptp() + 1e-9)
            all_scores[name] = s

        # Equal weighting
        weights = {k: 1.0 / len(all_scores) for k in all_scores}
        combined = sum(w * all_scores[k] for k, w in weights.items())
        threshold = np.percentile(combined, (1 - self.contamination) * 100)
        labels = (combined >= threshold).astype(int)
        return EnsembleResult(scores=combined, labels=labels, detector_weights=weights)
