# 03_baseline_model.py
"""
BASELINE MODEL
--------------
Quick logistic regression to validate pipeline and get baseline performance.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import joblib

print("="*80)
print("BASELINE MODEL - LOGISTIC REGRESSION")
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)

# Paths
BASE = Path(r"I:\enron_modeling")
OUTPUT_DIR = BASE / "outputs"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load data
print("\n[STEP 1/6] Loading train/test data...")
train = pd.read_parquet(OUTPUT_DIR / "modeling_data_train.parquet")
test = pd.read_parquet(OUTPUT_DIR / "modeling_data_test.parquet")
print(f" Train: {len(train):,} rows")
print(f" Test: {len(test):,} rows")

# Get feature names
with open(OUTPUT_DIR / "feature_names.txt") as f:
    feature_names = [line.strip() for line in f if line.strip()]
print(f" Features: {len(feature_names)}")

# Prepare X and y
print("\n[STEP 2/6] Preparing features and target...")
X_train = train[feature_names].values
y_train = train['target_binary'].values
X_test = test[feature_names].values
y_test = test['target_binary'].values

print(f" X_train shape: {X_train.shape}")
print(f" X_test shape: {X_test.shape}")
print(f" Positive class rate (train): {y_train.mean()*100:.2f}%")
print(f" Positive class rate (test): {y_test.mean()*100:.2f}%")

# Scale features
print("\n[STEP 3/6] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(" Features standardized")

# Train model
print("\n[STEP 4/6] Training Logistic Regression...")
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)
print(" Model trained")

# Predictions
print("\n[STEP 5/6] Making predictions...")
y_train_pred = model.predict(X_train_scaled)
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
print(" Predictions generated")

# Evaluation
print("\n[STEP 6/6] Evaluating model...")
print("\n--- TRAINING SET ---")
print(classification_report(y_train, y_train_pred,
                          target_names=['Low/Med', 'High/Crit'],
                          digits=3))

train_roc_auc = roc_auc_score(y_train, y_train_proba)
train_pr_auc = average_precision_score(y_train, y_train_proba)
print(f"ROC-AUC: {train_roc_auc:.3f}")
print(f"PR-AUC: {train_pr_auc:.3f}")

print("\n--- TEST SET ---")
print(classification_report(y_test, y_test_pred,
                          target_names=['Low/Med', 'High/Crit'],
                          digits=3))

test_roc_auc = roc_auc_score(y_test, y_test_proba)
test_pr_auc = average_precision_score(y_test, y_test_proba)
print(f"ROC-AUC: {test_roc_auc:.3f}")
print(f"PR-AUC: {test_pr_auc:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix (Test):")
print("           Predicted")
print("           Low/Med High/Crit")
print(f"Actual Low/Med    {cm[0,0]:6d}   {cm[0,1]:6d}")
print(f"Actual High/Crit  {cm[1,0]:6d}   {cm[1,1]:6d}")

# Feature importance
print("\n--- TOP 10 IMPORTANT FEATURES ---")
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_[0]
})
feature_importance['abs_coef'] = feature_importance['coefficient'].abs()
feature_importance = feature_importance.sort_values('abs_coef', ascending=False)
print(feature_importance.head(10)[['feature', 'coefficient']].to_string(index=False))

# Save feature importance
feature_importance.to_csv(OUTPUT_DIR / "baseline_feature_importance.csv", index=False)
print("\nFeature importance saved")

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, linewidth=2, label=f'PR-AUC = {test_pr_auc:.3f}')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve (Baseline Logistic Regression)',
         fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "baseline_pr_curve.png", dpi=150)
print("PR curve saved")
plt.close()

# Save model
model_bundle = {
    'model': model,
    'scaler': scaler,
    'features': feature_names,
    'metrics': {
        'train_roc_auc': train_roc_auc,
        'train_pr_auc': train_pr_auc,
        'test_roc_auc': test_roc_auc,
        'test_pr_auc': test_pr_auc,
    }
}
joblib.dump(model_bundle, MODEL_DIR / "model_baseline_logreg.pkl")
print(f"Model saved: models/model_baseline_logreg.pkl")

# Summary
print("\n" + "="*80)
print("BASELINE MODEL COMPLETE")
print("="*80)
print(f"\nPerformance Summary:")
print(f" Train PR-AUC: {train_pr_auc:.3f}")
print(f" Test PR-AUC: {test_pr_auc:.3f}")
print(f" Test ROC-AUC: {test_roc_auc:.3f}")

if test_pr_auc > 0.1:
    print(f"\nModel beats random baseline")
    print(f" Pipeline is working correctly")
else:
    print(f"\nModel not beating baseline - check features")

print(f"\nTop 3 Features:")
for i, row in feature_importance.head(3).iterrows():
    print(f" {i+1}. {row['feature']}: {row['coefficient']:.4f}")

print(f"\nReady for advanced models")
print(f"Completed: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)