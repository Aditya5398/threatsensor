"""
ml_models.py
Isolation Forest, Random Forest, and XGBoost anomaly detectors.
Includes SMOTE for class imbalance, threshold tuning, and full evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, precision_recall_curve,
                             roc_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import os

FEATURES = ['log_amount', 'hour', 'num_items', 'country_risk',
            'sessions_today', 'is_weekend', 'is_night']


def train_isolation_forest(X_scaled, contamination=0.01, n_estimators=200):
    """
    Train Isolation Forest — unsupervised, no labels needed.
    Short path length through random trees = anomaly.
    """
    print("\n[IsolationForest] Training...")
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=256,          # Optimal per original paper
        contamination=contamination,
        max_features=1.0,
        random_state=42
    )
    model.fit(X_scaled)
    # score_samples returns negative anomaly score; negate so higher = more anomalous
    scores = -model.score_samples(X_scaled)
    preds = (model.predict(X_scaled) == -1).astype(int)  # -1 = anomaly in sklearn
    print(f"[IsolationForest] Flagged {preds.sum():,} anomalies")
    return model, scores, preds


def train_random_forest(X_train, y_train, X_test, threshold=0.30):
    """
    Train Random Forest with SMOTE oversampling + class weighting.
    Handles the 99:1 class imbalance problem.
    """
    print("\n[RandomForest] Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[RandomForest] After SMOTE: {len(X_res):,} samples, "
          f"{y_res.sum():,} anomalies ({y_res.mean():.1%})")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        class_weight='balanced',  # Belt AND suspenders alongside SMOTE
        n_jobs=-1,
        random_state=42
    )
    print("[RandomForest] Training...")
    model.fit(X_res, y_res)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    print(f"[RandomForest] Flagged {preds.sum():,} anomalies (threshold={threshold})")
    return model, probs, preds


def train_xgboost(X_train, y_train, X_test, threshold=0.30):
    """
    Train XGBoost with scale_pos_weight for imbalance.
    Gradient boosting: trees built sequentially, each correcting prior errors.
    """
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\n[XGBoost] scale_pos_weight={scale_weight:.1f} (anomaly counts as {scale_weight:.0f}x more)")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric='aucpr',      # Optimize directly for PR-AUC
        use_label_encoder=False,
        random_state=42,
        verbosity=0
    )
    print("[XGBoost] Training...")
    model.fit(X_train, y_train,
              eval_set=[(X_test, None)] if False else [(X_train, y_train)],
              verbose=False)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    print(f"[XGBoost] Flagged {preds.sum():,} anomalies (threshold={threshold})")
    return model, probs, preds


def plot_feature_importance(rf_model, output_dir):
    fi = pd.DataFrame({
        'feature': FEATURES,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(fi['feature'], fi['importance'], color='steelblue')
    plt.xlabel('Importance (Mean Decrease in Impurity)')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved: {path}")


def plot_curves(y_true, model_scores_dict, output_dir):
    """Plot ROC and Precision-Recall curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['gray', 'green', 'steelblue', 'darkorange']

    for (name, scores), color in zip(model_scores_dict.items(), colors):
        mm = MinMaxScaler()
        scores_norm = mm.fit_transform(np.array(scores).reshape(-1, 1)).flatten()

        # ROC
        fpr, tpr, _ = roc_curve(y_true, scores_norm)
        auroc = roc_auc_score(y_true, scores_norm)
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auroc:.3f})')

        # PR
        prec, rec, _ = precision_recall_curve(y_true, scores_norm)
        pr_auc = average_precision_score(y_true, scores_norm)
        axes[1].plot(rec, prec, color=color, lw=2, label=f'{name} (AUC={pr_auc:.3f})')

    axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate (Recall)')
    axes[0].set_title('ROC Curves')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(y=0.01, color='k', linestyle='--', label='Random Classifier')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curves (Primary Metric)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'all_model_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved: {path}")


def run_ml_pipeline(df, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    X = df[FEATURES].copy()
    y = df['is_anomaly'].copy()

    # Train/test split — stratified to preserve anomaly rate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")

    # Scale features — mean=0, std=1
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)    # CRITICAL: use same scaler, fit on train only

    # ── TRAIN MODELS ──────────────────────────────────────────────────────
    iso_model, iso_scores_all, _ = train_isolation_forest(
        scaler.transform(X), contamination=0.01
    )
    iso_test_scores = -iso_model.score_samples(X_test_sc)

    rf_model, rf_probs, rf_preds = train_random_forest(X_train_sc, y_train, X_test_sc)
    xgb_model, xgb_probs, xgb_preds = train_xgboost(X_train_sc, y_train, X_test_sc)

    # ── ENSEMBLE ──────────────────────────────────────────────────────────
    mm = MinMaxScaler()
    iso_norm = mm.fit_transform(iso_test_scores.reshape(-1, 1)).flatten()
    ensemble = 0.20 * iso_norm + 0.35 * rf_probs + 0.45 * xgb_probs
    ensemble_preds = (ensemble >= 0.35).astype(int)

    # ── PRINT RESULTS ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Model':<22} {'PR-AUC':>8} {'AUROC':>8} {'Recall':>8} {'Precision':>10}")
    print("-" * 60)
    for name, probs, preds in [
        ('Isolation Forest', iso_norm, (iso_norm >= 0.5).astype(int)),
        ('Random Forest',    rf_probs, rf_preds),
        ('XGBoost',          xgb_probs, xgb_preds),
        ('Ensemble',         ensemble, ensemble_preds),
    ]:
        from sklearn.metrics import precision_score, recall_score
        pr = average_precision_score(y_test, probs)
        au = roc_auc_score(y_test, probs)
        re = recall_score(y_test, preds)
        pr_s = precision_score(y_test, preds, zero_division=0)
        print(f"{name:<22} {pr:>8.4f} {au:>8.4f} {re:>8.4f} {pr_s:>10.4f}")

    print("\n=== ENSEMBLE FULL REPORT ===")
    print(classification_report(y_test, ensemble_preds, target_names=['Normal', 'Anomaly']))

    # ── PLOTS ─────────────────────────────────────────────────────────────
    plot_feature_importance(rf_model, output_dir)
    plot_curves(y_test,
                {'IsoForest': iso_norm, 'RandomForest': rf_probs,
                 'XGBoost': xgb_probs, 'Ensemble': ensemble},
                output_dir)

    return {
        'scaler': scaler,
        'iso_model': iso_model,
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'X_test': X_test,
        'y_test': y_test,
        'ensemble_preds': ensemble_preds,
        'ensemble_scores': ensemble
    }


if __name__ == "__main__":
    from data_generator import generate_transactions
    from statistical_detectors import engineer_features
    df = generate_transactions()
    df, _, _ = engineer_features(df)
    run_ml_pipeline(df)
