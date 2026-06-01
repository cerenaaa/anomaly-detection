"""Anomaly detection demo."""
import argparse
import numpy as np
from data.synthetic_anomalies import generate_transactions
from detectors.isolation_forest import IsolationForestDetector
from detectors.statistical import zscore_anomalies, iqr_anomalies
from evaluation.metrics import evaluate

FEATURE_COLS = ["amount", "hour", "merchant_category", "distance_from_home_km", "n_transactions_24h"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="isolation_forest")
    args = parser.parse_args()

    print("Generating transaction data...")
    df = generate_transactions()
    X = df[FEATURE_COLS].values
    y = df["is_anomaly"].values

    if args.method == "isolation_forest":
        print("
Running Isolation Forest...")
        det = IsolationForestDetector(contamination=0.02)
        det.fit(X)
        result = det.predict(X)
        evaluate(y, result.scores, result.labels)

    elif args.method == "statistical":
        print("
Running Z-score on amount...")
        labels = zscore_anomalies(df["amount"].values, threshold=3.0)
        scores = np.abs((df["amount"].values - df["amount"].mean()) / df["amount"].std())
        evaluate(y, scores, labels)

if __name__ == "__main__":
    main()
