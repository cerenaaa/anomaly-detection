# Anomaly Detection

[![CI](https://github.com/cerenaaa/anomaly-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/anomaly-detection/actions)

Unsupervised anomaly detection for tabular and time series data: Isolation Forest, Autoencoder, ECOD, and statistical control charts. Built for fraud detection, system monitoring, and data quality.

## Methods

| Method | Type | Strengths |
|---|---|---|
| **Isolation Forest** | Tree-based | Fast, handles high-dim, no distribution assumption |
| **Autoencoder** | Deep learning | Captures complex feature interactions |
| **ECOD** | Statistical | No hyperparams, theoretically grounded |
| **Z-score / IQR** | Statistical | Univariate, interpretable |
| **CUSUM** | Control chart | Detects drift in time series |

## Structure
```
anomaly-detection/
├── detectors/
│   ├── isolation_forest.py    # IF with calibrated anomaly scores
│   ├── autoencoder.py         # Reconstruction-error autoencoder
│   ├── statistical.py         # Z-score, IQR, CUSUM
│   └── ensemble.py            # Score fusion across detectors
├── data/
│   └── synthetic_anomalies.py # Transactional + time series anomaly datasets
├── evaluation/
│   └── metrics.py             # AUROC, precision@K, average precision
└── detect.py
```

## Quickstart
```bash
pip install -r requirements.txt
python detect.py --method ensemble
```
