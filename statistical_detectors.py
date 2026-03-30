"""
statistical_detectors.py
Z-score, IQR, and multi-rule statistical anomaly detectors.
These are Layer 1 in Amazon's screening pipeline — fast, no ML required.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os


def engineer_features(df):
    """Add derived features needed by all models."""
    df = df.copy()
    df['log_amount'] = np.log1p(df['amount'])
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] <= 6) | (df['hour'] >= 22)).astype(int)

    mu = df['log_amount'].mean()
    sigma = df['log_amount'].std()
    df['amount_zscore'] = (df['log_amount'] - mu) / sigma
    return df, mu, sigma


def zscore_detector(df, threshold=2.5):
    """Flag transactions where |z-score of log(amount)| > threshold."""
    return (df['amount_zscore'].abs() > threshold).astype(int)


def iqr_detector(df, multiplier=2.0):
    """Flag transactions outside multiplier*IQR fence on log(amount)."""
    Q1 = df['log_amount'].quantile(0.25)
    Q3 = df['log_amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return ((df['log_amount'] < lower) | (df['log_amount'] > upper)).astype(int)


def multirule_detector(df):
    """Combine amount, country risk, time-of-day, and session signals."""
    return (
        (df['amount_zscore'].abs() > 3.0) |
        ((df['country_risk'] == 1) & (df['is_night'] == 1)) |
        (df['sessions_today'] > 10)
    ).astype(int)


def evaluate_statistical(df, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    df, _, _ = engineer_features(df)

    df['pred_zscore'] = zscore_detector(df)
    df['pred_iqr'] = iqr_detector(df)
    df['pred_rules'] = multirule_detector(df)

    for name, col in [('Z-Score', 'pred_zscore'),
                      ('IQR', 'pred_iqr'),
                      ('Multi-Rule', 'pred_rules')]:
        print(f"\n=== {name} DETECTOR ===")
        print(classification_report(df['is_anomaly'], df[col],
                                    target_names=['Normal', 'Anomaly']))

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Statistical Detectors — Confusion Matrices')
    for ax, (name, col) in zip(axes, [('Z-Score', 'pred_zscore'),
                                       ('IQR', 'pred_iqr'),
                                       ('Multi-Rule', 'pred_rules')]):
        cm = confusion_matrix(df['is_anomaly'], df[col])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        ax.set_title(f'{name}')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    path = os.path.join(output_dir, 'statistical_confusion_matrices.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[Statistical] Saved: {path}")
    return df


if __name__ == "__main__":
    from data_generator import generate_transactions
    df = generate_transactions()
    evaluate_statistical(df)
