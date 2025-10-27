# 10_model_comparison_report.py
"""
MODEL COMPARISON REPORT
-----------------------
Create comprehensive comparison of all 3 models for presentation.
Addresses Client Feedback #4: "How did we calculate the risks?"

Author: Nimur
Date: Saturday 10:00 PM
"""
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

print("="*80)
print("MODEL COMPARISON REPORT")
print("="*80)

# Paths
BASE = Path(r"I:\enron_modeling")
MODEL_DIR = BASE / "models"
OUTPUT_DIR = BASE / "outputs"

# Load all models
print("\n[1/3] Loading models...")
logreg = joblib.load(MODEL_DIR / "model_baseline_logreg.pkl")
rf = joblib.load(MODEL_DIR / "model_rf.pkl")
xgb = joblib.load(MODEL_DIR / "model_xgboost.pkl")

print(" ✓ Logistic Regression")
print(" ✓ Random Forest")
print(" ✓ XGBoost")

# Extract metrics
print("\n[2/3] Comparing performance...")

models_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Train PR-AUC': [
        logreg['metrics']['train_pr_auc'],
        rf['metrics']['train_pr_auc'],
        xgb['metrics']['train_pr_auc']
    ],
    'Test PR-AUC': [
        logreg['metrics']['test_pr_auc'],
        rf['metrics']['test_pr_auc'],
        xgb['metrics']['test_pr_auc']
    ],
    'Train ROC-AUC': [
        logreg['metrics']['train_roc_auc'],
        rf['metrics']['train_roc_auc'],
        xgb['metrics']['train_roc_auc']
    ],
    'Test ROC-AUC': [
        logreg['metrics']['test_roc_auc'],
        rf['metrics']['test_roc_auc'],
        xgb['metrics']['test_roc_auc']
    ]
})

print("\nModel Performance Comparison:")
print(models_comparison.to_string(index=False))

# Identify best model
best_model_idx = models_comparison['Test PR-AUC'].idxmax()
best_model_name = models_comparison.loc[best_model_idx, 'Model']
best_pr_auc = models_comparison.loc[best_model_idx, 'Test PR-AUC']

print(f"\n🏆 BEST MODEL: {best_model_name} (Test PR-AUC: {best_pr_auc:.3f})")

# Save comparison table
models_comparison.to_csv(OUTPUT_DIR / "model_comparison_table.csv", index=False)
print(f"\n ✓ Comparison table saved")

# Visualizations
print("\n[3/3] Creating comparison visualizations...")

# 1. Bar chart comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PR-AUC comparison
ax1 = axes[0]
x_pos = range(len(models_comparison))
bars1 = ax1.bar(x_pos, models_comparison['Test PR-AUC'],
                color=['#1976d2', '#388e3c', '#d32f2f'], alpha=0.7)

# Highlight best model
bars1[best_model_idx].set_color('#ffd700')
bars1[best_model_idx].set_edgecolor('black')
bars1[best_model_idx].set_linewidth(2)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(models_comparison['Model'], rotation=15, ha='right')
ax1.set_ylabel('PR-AUC Score', fontsize=12)
ax1.set_title('Model Comparison - PR-AUC (Test Set)', fontsize=13, fontweight='bold')
ax1.set_ylim([0, 1])
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add baseline line
ax1.axhline(y=0.04, color='red', linestyle='--', linewidth=1,
            label='Random baseline (~0.04)', alpha=0.7)
ax1.legend()

# ROC-AUC comparison
ax2 = axes[1]
bars2 = ax2.bar(x_pos, models_comparison['Test ROC-AUC'],
                color=['#1976d2', '#388e3c', '#d32f2f'], alpha=0.7)

# Highlight best model
bars2[best_model_idx].set_color('#ffd700')
bars2[best_model_idx].set_edgecolor('black')
bars2[best_model_idx].set_linewidth(2)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(models_comparison['Model'], rotation=15, ha='right')
ax2.set_ylabel('ROC-AUC Score', fontsize=12)
ax2.set_title('Model Comparison - ROC-AUC (Test Set)', fontsize=13, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add baseline line
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1,
            label='Random baseline (0.5)', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison_chart.png", dpi=150)
print(" ✓ Comparison chart saved")
plt.close()

# 2. Overfitting analysis (Train vs Test)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

x_pos = range(len(models_comparison))
width = 0.35

train_bars = ax.bar([x - width/2 for x in x_pos],
                     models_comparison['Test PR-AUC'],
                     width, label='Test', color='steelblue', alpha=0.7)
test_bars = ax.bar([x + width/2 for x in x_pos],
                    models_comparison['Train PR-AUC'],
                    width, label='Train', color='orange', alpha=0.7)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('PR-AUC Score', fontsize=12)
ax.set_title('Train vs Test Performance (Overfitting Check)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models_comparison['Model'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_overfitting_check.png", dpi=150)
print(" ✓ Overfitting check saved")
plt.close()

# Summary report
print("\n" + "="*80)
print("MODEL COMPARISON COMPLETE")
print("="*80)

print(f"\nSUMMARY:")
print(f" Best Model: {best_model_name}")
print(f" Best PR-AUC: {best_pr_auc:.3f}")

print(f"\nImprovement from Baseline:")
baseline_prauc = models_comparison.loc[0, 'Test PR-AUC']
improvement = best_pr_auc - baseline_prauc
print(f" Baseline: {baseline_prauc:.3f}")
print(f" Best: {best_pr_auc:.3f}")
print(f" Improvement: {improvement:+.3f} ({improvement/baseline_prauc*100:+.1f}%)")

print(f"\nOverfitting Check:")
for idx, row in models_comparison.iterrows():
    train_test_gap = row['Train PR-AUC'] - row['Test PR-AUC']
    status = "✓ Good" if train_test_gap < 0.15 else "⚠ Overfitting"
    print(f" {row['Model']:20s}: Gap = {train_test_gap:.3f} {status}")

print(f"\nFiles Created:")
print(f" • outputs/model_comparison_table.csv")
print(f" • outputs/model_comparison_chart.png")
print(f" • outputs/model_overfitting_check.png")

print(f"\n💡 Use comparison chart in presentation to show:")
print(f"   'We tested 3 algorithms and selected the best'")
print("="*80)