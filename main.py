"""
main.py — ThreatSensor
Run the complete anomaly detection pipeline end to end.
"""

from data_generator import generate_transactions
from eda import run_eda
from statistical_detectors import engineer_features, evaluate_statistical
from ml_models import run_ml_pipeline
from time_series import run_timeseries_pipeline

OUTPUT_DIR = "outputs"


def main():
    print("=" * 60)
    print("  THREATSENSOR — Transaction Anomaly Detection")
    print("  Amazon Denied Party Screening — ML Pipeline")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1/5] Generating transaction data...")
    df = generate_transactions(n_normal=49500, n_anomaly=500)

    # Step 2: EDA
    print("\n[2/5] Running exploratory data analysis...")
    run_eda(df, output_dir=OUTPUT_DIR)

    # Step 3: Feature engineering + statistical detectors
    print("\n[3/5] Running statistical detectors...")
    df, mu, sigma = engineer_features(df)
    evaluate_statistical(df, output_dir=OUTPUT_DIR)

    # Step 4: ML models (Isolation Forest, RF, XGBoost, Ensemble)
    print("\n[4/5] Training ML models...")
    results = run_ml_pipeline(df, output_dir=OUTPUT_DIR)

    # Step 5: Time series (ARIMA)
    print("\n[5/5] Running ARIMA time-series analysis...")
    run_timeseries_pipeline(df, output_dir=OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  All charts saved to: {OUTPUT_DIR}/")
    print("=" * 60)

    print("\nGenerated files:")
    import os
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  outputs/{f}")


if __name__ == "__main__":
    main()
