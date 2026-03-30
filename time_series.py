"""
time_series.py
ARIMA-based anomaly detection on transaction volume time series.
Detects abnormal spikes/drops in hourly transaction counts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os


def build_hourly_series(df):
    """Aggregate transactions to hourly counts."""
    df['hour_bucket'] = df['timestamp'].dt.floor('h')
    hourly = df.groupby('hour_bucket').agg(
        total_transactions=('amount', 'count'),
        total_amount=('amount', 'sum'),
        anomaly_count=('is_anomaly', 'sum')
    )
    # Fill gaps with 0 (no transactions in that hour)
    full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
    hourly = hourly.reindex(full_range, fill_value=0)
    return hourly['total_transactions']


def run_adf_test(ts):
    """
    Augmented Dickey-Fuller test for stationarity.
    H0: non-stationary (has unit root)
    p < 0.05 → reject H0 → stationary → d=0
    """
    result = adfuller(ts, autolag='AIC')
    print("\n=== ADF STATIONARITY TEST ===")
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"p-value       : {result[1]:.6f}")
    if result[1] < 0.05:
        print("RESULT: STATIONARY (p < 0.05) → use d=0 in ARIMA")
    else:
        print("RESULT: NON-STATIONARY → apply differencing (d=1)")
    return result[1] < 0.05


def fit_arima(train_ts, order=(2, 0, 2)):
    """Fit ARIMA model on training series."""
    print(f"\n[ARIMA] Fitting ARIMA{order}...")
    model = ARIMA(train_ts, order=order)
    result = model.fit()
    print(f"[ARIMA] AIC={result.aic:.2f}  BIC={result.bic:.2f}")
    return result


def detect_volume_anomalies(ts, model_result, test_steps=48, alpha=0.05):
    """
    Forecast test_steps ahead and flag actual values outside confidence interval.
    Anything outside the 95% CI = statistically anomalous volume.
    """
    forecast = model_result.get_forecast(steps=test_steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=alpha)

    test_ts = ts[-test_steps:]
    lower = conf_int.iloc[:, 0].values
    upper = conf_int.iloc[:, 1].values

    anomalous = test_ts[(test_ts.values < lower) | (test_ts.values > upper)]
    print(f"\n[ARIMA] Detected {len(anomalous)} anomalous hours "
          f"out of {test_steps} in test window")
    return forecast_mean, conf_int, test_ts, anomalous


def plot_arima_results(ts, forecast_mean, conf_int, test_ts, anomalous, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle('ThreatSensor — ARIMA Time Series Analysis', fontsize=14)

    # 1. Full series
    axes[0, 0].plot(ts.values[:300], color='steelblue', linewidth=0.8)
    axes[0, 0].set_title('Hourly Transaction Volume (first 300 hrs)')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Count')

    # 2. Distribution of hourly volume
    axes[0, 1].hist(ts.values, bins=30, color='steelblue', edgecolor='white')
    axes[0, 1].set_title('Distribution of Hourly Volume')
    axes[0, 1].set_xlabel('Transactions per Hour')

    # 3. ACF
    plot_acf(ts, lags=48, ax=axes[1, 0])
    axes[1, 0].set_title('ACF — use to identify MA order (q)')

    # 4. PACF
    plot_pacf(ts, lags=48, ax=axes[1, 1])
    axes[1, 1].set_title('PACF — use to identify AR order (p)')

    plt.tight_layout()
    path = os.path.join(output_dir, 'arima_diagnostics.png')
    plt.savefig(path, dpi=150)
    plt.close()

    # Forecast plot
    fig2, ax = plt.subplots(figsize=(14, 5))
    train_tail = ts[-148:-48]
    ax.plot(range(len(train_tail)), train_tail.values, 'b-', label='Historical', lw=1)
    ax.plot(range(len(train_tail), len(train_tail) + len(test_ts)),
            test_ts.values, 'g-', label='Actual', lw=1.5)
    ax.plot(range(len(train_tail), len(train_tail) + len(forecast_mean)),
            forecast_mean.values, 'r--', label='ARIMA Forecast', lw=2)
    ax.fill_between(
        range(len(train_tail), len(train_tail) + len(forecast_mean)),
        conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values,
        alpha=0.2, color='red', label='95% CI'
    )
    ax.set_title('ARIMA Forecast — Anomalous Volume Detection')
    ax.set_xlabel('Hour Index')
    ax.set_ylabel('Transaction Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(output_dir, 'arima_forecast.png')
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"[ARIMA] Saved: {path} and {path2}")


def run_timeseries_pipeline(df, output_dir="outputs"):
    ts = build_hourly_series(df)
    print(f"[TimeSeries] {len(ts)} hourly observations | "
          f"mean={ts.mean():.1f} txns/hr")

    is_stationary = run_adf_test(ts)
    d = 0 if is_stationary else 1

    train_ts = ts[:-48]
    arima_result = fit_arima(train_ts, order=(2, d, 2))
    forecast_mean, conf_int, test_ts, anomalous = detect_volume_anomalies(
        ts, arima_result, test_steps=48
    )
    plot_arima_results(ts, forecast_mean, conf_int, test_ts, anomalous, output_dir)
    return arima_result


if __name__ == "__main__":
    from data_generator import generate_transactions
    df = generate_transactions()
    run_timeseries_pipeline(df)
