# 08_shap_analysis_fixed.py
"""
SHAP ANALYSIS - FIXED VERSION
------------------------------
Workaround for XGBoost/SHAP compatibility issue.
Author: Nimur
Date: Saturday 5:00 PM
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SHAP ANALYSIS - EXPLAINING MODEL PREDICTIONS (FIXED)")
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)

# Paths
BASE = Path(r"I:\enron_modeling")
OUTPUT_DIR = BASE / "outputs"
MODEL_DIR = BASE / "models"
SHAP_DIR = BASE / "shap_analysis"
SHAP_DIR.mkdir(exist_ok=True)

# Load best model (XGBoost)
print("\n[STEP 1/5] Loading XGBoost model...")
model_bundle = joblib.load(MODEL_DIR / "model_xgboost.pkl")
model = model_bundle['model']
scaler = model_bundle['scaler']
feature_names = model_bundle['features']
print(f" ✓ Model loaded")
print(f" Features: {len(feature_names)}")

# Load test data
print("\n[STEP 2/5] Loading test data...")
test = pd.read_parquet(OUTPUT_DIR / "modeling_data_test.parquet")
X_test = test[feature_names].values
X_test_scaled = scaler.transform(X_test)
y_test = test['target_binary'].values

# Sample for analysis
np.random.seed(42)
sample_size = min(500, len(X_test))
sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
X_sample = X_test_scaled[sample_idx]
y_sample = y_test[sample_idx]
print(f" Test data: {len(X_test):,} rows")
print(f" Sample: {sample_size} rows")

# Alternative: Use model's feature_importances_ (built-in XGBoost)
print("\n[STEP 3/5] Computing feature importance...")
print(" Using XGBoost native feature importance (gain-based)")

# Get feature importance from XGBoost directly
feature_importance = model.feature_importances_
shap_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
})
shap_df = shap_df.sort_values('importance', ascending=False)

print("\nTop 10 Features (by XGBoost Importance):")
print(shap_df.head(10).to_string(index=False))

# Save importance
shap_df.to_csv(SHAP_DIR / "feature_importance_xgboost.csv", index=False)
print(f"\n ✓ Feature importance saved")

# Compute permutation importance (alternative to SHAP)
print("\n[STEP 4/5] Computing permutation importance...")
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_sample, y_sample, 
    n_repeats=10, 
    random_state=42, 
    n_jobs=-1
)

perm_df = pd.DataFrame({
    'feature': feature_names,
    'perm_importance': result.importances_mean,
    'perm_std': result.importances_std
})
perm_df = perm_df.sort_values('perm_importance', ascending=False)
perm_df.to_csv(SHAP_DIR / "permutation_importance_xgboost.csv", index=False)
print(" ✓ Permutation importance computed")

print("\nTop 10 Features (by Permutation):")
print(perm_df.head(10).to_string(index=False))

# Visualizations
print("\n[STEP 5/5] Creating visualizations...")

# 1. Feature Importance Bar Chart (XGBoost native)
print(" Creating feature importance chart...")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
top_20 = shap_df.head(20)
bars = ax.barh(range(len(top_20)), top_20['importance'],
               color='#1976d2', alpha=0.7)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'])
ax.set_xlabel('XGBoost Feature Importance (Gain)', fontsize=12)
ax.set_title('Feature Importance - XGBoost', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_20['importance'])):
    ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(SHAP_DIR / "feature_importance_xgboost.png", dpi=150)
print(" ✓ Feature importance chart saved")
plt.close()

# 2. Permutation Importance Chart
print(" Creating permutation importance chart...")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
top_20_perm = perm_df.head(20)
bars = ax.barh(range(len(top_20_perm)), top_20_perm['perm_importance'],
               xerr=top_20_perm['perm_std'],
               color='#d32f2f', alpha=0.7)
ax.set_yticks(range(len(top_20_perm)))
ax.set_yticklabels(top_20_perm['feature'])
ax.set_xlabel('Permutation Importance (Mean Decrease in Accuracy)', fontsize=12)
ax.set_title('Permutation Feature Importance - XGBoost', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(SHAP_DIR / "permutation_importance_xgboost.png", dpi=150)
print(" ✓ Permutation importance chart saved")
plt.close()

# 3. Feature vs Target Correlation
print(" Creating feature correlation analysis...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

top_6_features = shap_df.head(6)['feature'].tolist()

for idx, feat in enumerate(top_6_features):
    ax = axes[idx]
    feat_idx = feature_names.index(feat)
    
    # Separate by class
    X_sample_feat = X_sample[:, feat_idx]
    
    # Box plot by class
    low_risk = X_sample_feat[y_sample == 0]
    high_risk = X_sample_feat[y_sample == 1]
    
    bp = ax.boxplot([low_risk, high_risk], 
                     labels=['Low Risk', 'High Risk'],
                     patch_artist=True)
    
    bp['boxes'][0].set_facecolor('#66bb6a')
    bp['boxes'][1].set_facecolor('#ef5350')
    
    ax.set_ylabel('Feature Value', fontsize=10)
    ax.set_title(f'{feat}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Feature Distributions by Risk Level', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(SHAP_DIR / "feature_distributions_by_risk.png", dpi=150)
print(" ✓ Feature distributions saved")
plt.close()

# 4. Top 10 Feature Comparison
print(" Creating feature comparison chart...")
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Compare both importance measures
comparison_df = shap_df.head(10).merge(
    perm_df[['feature', 'perm_importance']], 
    on='feature'
)

x = np.arange(len(comparison_df))
width = 0.35

bars1 = ax.bar(x - width/2, comparison_df['importance'], width, 
               label='XGBoost Gain', color='#1976d2', alpha=0.7)
bars2 = ax.bar(x + width/2, comparison_df['perm_importance'], width,
               label='Permutation', color='#d32f2f', alpha=0.7)

ax.set_xlabel('Features', fontsize=12)
ax.set_ylabel('Importance', fontsize=12)
ax.set_title('Feature Importance Comparison (Top 10)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['feature'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(SHAP_DIR / "feature_importance_comparison.png", dpi=150)
print(" ✓ Comparison chart saved")
plt.close()

# Summary
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("="*80)

print(f"\nTop 10 Features (XGBoost Gain):")
for i, row in shap_df.head(10).iterrows():
    print(f" {i+1:2d}. {row['feature']:25s} {row['importance']:.4f}")

print(f"\nTop 10 Features (Permutation):")
for i, row in perm_df.head(10).iterrows():
    print(f" {i+1:2d}. {row['feature']:25s} {row['perm_importance']:.4f}")

print(f"\nFiles Created:")
print(f" • shap_analysis/feature_importance_xgboost.csv")
print(f" • shap_analysis/permutation_importance_xgboost.csv")
print(f" • shap_analysis/feature_importance_xgboost.png")
print(f" • shap_analysis/permutation_importance_xgboost.png")
print(f" • shap_analysis/feature_distributions_by_risk.png")
print(f" • shap_analysis/feature_importance_comparison.png")

print(f"\n✅ Ready for handoff package creation!")
print(f"Completed: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)