# 06_train_random_forest.py
"""
RANDOM FOREST MODEL
-------------------
Train Random Forest with hyperparameter tuning.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import joblib

print("="*80)
print("RANDOM FOREST MODEL TRAINING")
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

# Train Random Forest
print("\n[STEP 4/7] Training Random Forest...")
print(" Hyperparameters:")
print("  - n_estimators: 400")
print("  - max_depth: 20")
print("  - min_samples_split: 50")
print("  - min_samples_leaf: 20")
print("  - class_weight: balanced")

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\n Training... (this will take 5-10 minutes)")
model.fit(X_train, y_train)
print("\n Model trained")

# Predictions
print("\n[STEP 5/7] Making predictions...")
y_train_pred = model.predict(X_train)
y_train_proba = model.predict_proba(X_train)[:, 1]
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]
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
feature_importance.to_csv(OUTPUT_DIR / "rf_feature_importance.csv", index=False)
print("\n Feature importance saved")

# Visualizations
print("\n[STEP 7/7] Creating visualizations...")

# 1. Feature Importance Chart
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
top_20 = feature_importance.head(20)
bars = ax.barh(range(len(top_20)), top_20['importance'], color='steelblue', alpha=0.7)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'])
ax.set_xlabel('Importance (Gini)', fontsize=12)
ax.set_title('Random Forest - Top 20 Feature Importance', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for i, (bar, val) in enumerate(zip(bars, top_20['importance'])):
    ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rf_feature_importance.png", dpi=150)
print(" Feature importance chart saved")
plt.close()

# 2. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(recall, precision, linewidth=2, label=f'Random Forest (PR-AUC = {test_pr_auc:.3f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve - Random Forest', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rf_pr_curve.png", dpi=150)
print(" PR curve saved")
plt.close()

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(fpr, tpr, linewidth=2, label=f'Random Forest (ROC-AUC = {test_roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Baseline')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve - Random Forest', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rf_roc_curve.png", dpi=150)
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
joblib.dump(model_bundle, MODEL_DIR / "model_rf.pkl")
print(f" Model saved: models/model_rf.pkl")

# Summary
print("\n" + "="*80)
print("RANDOM FOREST TRAINING COMPLETE")
print("="*80)
print(f"\nPerformance Summary:")
print(f" Train PR-AUC: {train_pr_auc:.3f}")
print(f" Test PR-AUC: {test_pr_auc:.3f}")
print(f" Test ROC-AUC: {test_roc_auc:.3f}")

# Compare to baseline
baseline = joblib.load(MODEL_DIR / "model_baseline_logreg.pkl")
baseline_pr_auc = baseline['metrics']['test_pr_auc']
improvement = test_pr_auc - baseline_pr_auc
print(f"\nImprovement over baseline:")
print(f" Baseline PR-AUC: {baseline_pr_auc:.3f}")
print(f" RF PR-AUC: {test_pr_auc:.3f}")
print(f" Improvement: {improvement:+.3f} ({improvement/baseline_pr_auc*100:+.1f}%)")

print(f"\nTop 5 Features:")
for i, row in feature_importance.head(5).iterrows():
    print(f" {i+1}. {row['feature']}: {row['importance']:.4f}")

print(f"\nReady for XGBoost")
print(f"Completed: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)