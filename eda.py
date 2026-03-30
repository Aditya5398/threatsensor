"""
eda.py
Exploratory Data Analysis: distributions, correlations, IQR outlier analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def run_eda(df, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(df.describe().round(2))

    print("\n=== STATISTICS BY CLASS ===")
    print(df.groupby('is_anomaly')[['amount', 'hour', 'sessions_today']].agg(
        ['mean', 'median', 'std']).round(2))

    # ── IQR OUTLIER ANALYSIS ───────────────────────────────────────────────
    Q1 = df['amount'].quantile(0.25)
    Q3 = df['amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    iqr_outliers = df[(df.amount < lower) | (df.amount > upper)]

    print(f"\n=== IQR ANALYSIS ===")
    print(f"Q1=${Q1:.2f}  Q3=${Q3:.2f}  IQR=${IQR:.2f}")
    print(f"Fences: [{lower:.2f}, {upper:.2f}]")
    print(f"IQR outliers: {len(iqr_outliers):,}  "
          f"(of which {iqr_outliers.is_anomaly.sum()} are real anomalies = "
          f"{iqr_outliers.is_anomaly.mean() * 100:.1f}% precision)")

    # ── PLOTS ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('ThreatSensor — Exploratory Data Analysis', fontsize=16)

    norm = df[df.is_anomaly == 0]
    anom = df[df.is_anomaly == 1]

    # 1. Raw amount distribution
    axes[0, 0].hist(norm.amount, bins=100, alpha=0.6, color='steelblue', label='Normal', density=True)
    axes[0, 0].hist(anom.amount, bins=50, alpha=0.7, color='tomato', label='Anomaly', density=True)
    axes[0, 0].set_xlim(0, 5000)
    axes[0, 0].set_title('Amount (Raw) — right-skewed log-normal')
    axes[0, 0].set_xlabel('Transaction Amount ($)')
    axes[0, 0].legend()

    # 2. Log-transformed amount — both look Gaussian now
    axes[0, 1].hist(np.log1p(norm.amount), bins=60, alpha=0.6, color='steelblue', label='Normal', density=True)
    axes[0, 1].hist(np.log1p(anom.amount), bins=40, alpha=0.7, color='tomato', label='Anomaly', density=True)
    axes[0, 1].set_title('log(Amount+1) — classes separate clearly')
    axes[0, 1].set_xlabel('log(Amount + 1)')
    axes[0, 1].legend()

    # 3. Hour of day
    axes[0, 2].hist(norm.hour, bins=24, alpha=0.6, color='steelblue', label='Normal', density=True)
    axes[0, 2].hist(anom.hour, bins=24, alpha=0.7, color='tomato', label='Anomaly', density=True)
    axes[0, 2].set_title('Hour of Day — anomalies spike at night')
    axes[0, 2].set_xlabel('Hour')
    axes[0, 2].legend()

    # 4. Boxplots
    axes[1, 0].boxplot(
        [np.log1p(norm.amount), np.log1p(anom.amount)],
        labels=['Normal', 'Anomaly'],
        patch_artist=True,
        boxprops=dict(facecolor='lightsteelblue')
    )
    axes[1, 0].set_title('Boxplot: log(Amount) by Class')
    axes[1, 0].set_ylabel('log(Amount + 1)')

    # 5. Correlation heatmap
    corr = df[['amount', 'hour', 'num_items', 'country_risk',
               'sessions_today', 'is_anomaly']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                ax=axes[1, 1], center=0, square=True)
    axes[1, 1].set_title('Feature Correlation Heatmap')

    # 6. Sessions today
    axes[1, 2].hist(norm.sessions_today, bins=20, alpha=0.6, color='steelblue', label='Normal', density=True)
    axes[1, 2].hist(anom.sessions_today, bins=20, alpha=0.7, color='tomato', label='Anomaly', density=True)
    axes[1, 2].set_title('Sessions Today — high activity = suspicious')
    axes[1, 2].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, 'eda_plots.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[EDA] Saved: {path}")


if __name__ == "__main__":
    from data_generator import generate_transactions
    df = generate_transactions()
    run_eda(df)
