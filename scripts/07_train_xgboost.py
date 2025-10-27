# 07_train_xgboost.py
"""
XGBOOST MODEL
-------------
Train XGBoost - typically the best performer.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import joblib

print("="*80)
print("XGBOOST MODEL TRAINING")
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)

# Paths
BASE = Path(r"I:\enron_modeling")
OUTPUT_DIR = BASE / "outputs"
MODEL_DIR = BASE / "models"

# Load data
print("\n[STEP 1/7] Loading train/test data...")
train = pd.read_parquet(OUTPUT_DIR / "modeling_data_train.parquet")
test = pd.read_parquet(OUTPUT_DIR / "modeling_data_test.parquet")
print(f" Train: {len(train):,} rows")
print(f" Test: {len(test):,} rows")

# Get feature names
with open(OUTPUT_DIR / "feature_names.txt") as f:
    feature_names = [line.strip() for line in f if line.strip()]
print(f" Features: {len(feature_names)}")

# Prepare X and y
print("\n[STEP 2/7] Preparing features and target...")
X_train = train[feature_names].values
y_train = train['target_binary'].values
X_test = test[feature_names].values
y_test = test['target_binary'].values

print(f" X_train shape: {X_train.shape}")
print(f" X_test shape: {X_test.shape}")

# Scale features
print("\n[STEP 3/7] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(" Features standardized")

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f" Scale pos weight: {scale_pos_weight:.2f}")

# Train XGBoost
print("\n[STEP 4/7] Training XGBoost...")
print(" Hyperparameters:")
print(f"  - n_estimators: 400")
print(f"  - max_depth: 6")
print(f"  - learning_rate: 0.1")
print(f"  - subsample: 0.8")
print(f"  - colsample_bytree: 0.8")
print(f"  - scale_pos_weight: {scale_pos_weight:.2f}")

model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='aucpr'
)

print("\n Training... (this will take 3-5 minutes)")
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)
print("\n Model trained")

# Predictions
print("\n[STEP 5/7] Making predictions...")
y_train_pred = model.predict(X_train_scaled)
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
print(" Predictions generated")

# Evaluation
print("\n[STEP 6/7] Evaluating model...")
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
print("\n--- FEATURE IMPORTANCE (TOP 10) ---")
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print(feature_importance.head(10).to_string(index=False))

# Save feature importance
feature_importance.to_csv(OUTPUT_DIR / "xgboost_feature_importance.csv", index=False)
print("\n Feature importance saved")

# Visualizations
print("\n[STEP 7/7] Creating visualizations...")

# 1. Feature Importance Chart
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
top_20 = feature_importance.head(20)
bars = ax.barh(range(len(top_20)), top_20['importance'], color='#d32f2f', alpha=0.7)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'])
ax.set_xlabel('Importance (Gain)', fontsize=12)
ax.set_title('XGBoost - Top 20 Feature Importance', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for i, (bar, val) in enumerate(zip(bars, top_20['importance'])):
    ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "xgboost_feature_importance.png", dpi=150)
print(" Feature importance chart saved")
plt.close()

# 2. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(recall, precision, linewidth=2, color='#d32f2f',
       label=f'XGBoost (PR-AUC = {test_pr_auc:.3f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve - XGBoost', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "xgboost_pr_curve.png", dpi=150)
print(" PR curve saved")
plt.close()

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(fpr, tpr, linewidth=2, color='#d32f2f',
       label=f'XGBoost (ROC-AUC = {test_roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Baseline')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve - XGBoost', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "xgboost_roc_curve.png", dpi=150)
print(" ROC curve saved")
plt.close()

# Save model
print("\n[FINAL] Saving model...")
model_bundle = {
    'model': model,
    'scaler': scaler,
    'features': feature_names,
    'metrics': {
        'train_roc_auc': float(train_roc_auc),
        'train_pr_auc': float(train_pr_auc),
        'test_roc_auc': float(test_roc_auc),
        'test_pr_auc': float(test_pr_auc),
    },
    'feature_importance': feature_importance.to_dict('records')
}
joblib.dump(model_bundle, MODEL_DIR / "model_xgboost.pkl")
print(f" Model saved: models/model_xgboost.pkl")

# Summary
print("\n" + "="*80)
print("XGBOOST TRAINING COMPLETE")
print("="*80)
print(f"\nPerformance Summary:")
print(f" Train PR-AUC: {train_pr_auc:.3f}")
print(f" Test PR-AUC: {test_pr_auc:.3f}")
print(f" Test ROC-AUC: {test_roc_auc:.3f}")

# Compare to other models
baseline = joblib.load(MODEL_DIR / "model_baseline_logreg.pkl")
rf = joblib.load(MODEL_DIR / "model_rf.pkl")

print(f"\nModel Comparison (Test PR-AUC):")
print(f" Logistic Regression: {baseline['metrics']['test_pr_auc']:.3f}")
print(f" Random Forest: {rf['metrics']['test_pr_auc']:.3f}")
print(f" XGBoost: {test_pr_auc:.3f}")

best_pr_auc = max(
    baseline['metrics']['test_pr_auc'],
    rf['metrics']['test_pr_auc'],
    test_pr_auc
)

if test_pr_auc == best_pr_auc:
    print(f"\n XGBoost is the BEST model (PR-AUC: {test_pr_auc:.3f})")
else:
    print(f"\n XGBoost not the best (best: {best_pr_auc:.3f})")

print(f"\nTop 5 Features:")
for i, row in feature_importance.head(5).iterrows():
    print(f" {i+1}. {row['feature']}: {row['importance']:.4f}")

print(f"\nAll 3 models complete")
print(f"Completed: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)