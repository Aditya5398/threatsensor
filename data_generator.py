"""
data_generator.py
Generates 50,000 synthetic financial transactions with injected anomalies.
Mimics real transaction data distributions used in financial screening.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_transactions(n_normal=49500, n_anomaly=500, random_seed=42):
    np.random.seed(random_seed)

    # ── NORMAL TRANSACTIONS ────────────────────────────────────────────────
    normal = {
        # Log-normal: most txns $10-$2000, long right tail
        'amount': np.random.lognormal(mean=5, sigma=1.2, size=n_normal),
        # Business hours peak: mean=1pm, std=4hrs
        'hour': np.clip(np.random.normal(13, 4, n_normal), 0, 23).astype(int),
        # Poisson: avg 2.5 items per transaction
        'num_items': np.random.poisson(lam=2.5, size=n_normal) + 1,
        # Weekdays busier than weekends
        'day_of_week': np.random.choice(range(7), size=n_normal,
                                         p=[0.18, 0.18, 0.17, 0.17, 0.15, 0.08, 0.07]),
        # 95% of txns from low-risk countries
        'country_risk': np.random.choice([0, 1], size=n_normal, p=[0.95, 0.05]),
        # Sessions/day: Poisson avg 1.5
        'sessions_today': np.random.poisson(lam=1.5, size=n_normal) + 1,
        'is_anomaly': np.zeros(n_normal, dtype=int)
    }

    # ── ANOMALOUS TRANSACTIONS ─────────────────────────────────────────────
    # Two types: very large amounts OR near-zero (structuring)
    anomaly_amounts = np.where(
        np.random.random(n_anomaly) > 0.5,
        np.random.lognormal(8, 0.8, n_anomaly),    # Large: $3k-$50k
        np.random.uniform(0.01, 5, n_anomaly)       # Tiny: structuring
    )
    anomaly = {
        'amount': anomaly_amounts,
        'hour': np.random.choice([0, 1, 2, 3, 4, 22, 23], size=n_anomaly),
        'num_items': np.random.choice([1, 99, 100], size=n_anomaly),
        'day_of_week': np.random.choice(range(7), size=n_anomaly,
                                         p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.25]),
        'country_risk': np.random.choice([0, 1], size=n_anomaly, p=[0.4, 0.6]),
        'sessions_today': np.random.choice([1, 15, 20, 25], size=n_anomaly),
        'is_anomaly': np.ones(n_anomaly, dtype=int)
    }

    df = pd.concat([pd.DataFrame(normal), pd.DataFrame(anomaly)], ignore_index=True)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Add timestamps over 90 days
    start = datetime(2024, 1, 1)
    df['timestamp'] = [start + timedelta(hours=int(np.random.randint(0, 90 * 24)))
                       for _ in range(len(df))]
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"[DataGenerator] {len(df):,} total | {df.is_anomaly.sum():,} anomalies "
          f"({df.is_anomaly.mean() * 100:.2f}%)")
    return df


if __name__ == "__main__":
    df = generate_transactions()
    print(df.head())
    print(df.describe().round(2))
