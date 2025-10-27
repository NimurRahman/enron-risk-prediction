# 08_shap_analysis.py
"""
SHAP ANALYSIS
-------------
Explain model predictions using SHAP values.
Author: Nimur
Date: Saturday 4:20 PM
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import shap
import matplotlib.pyplot as plt

print("="*80)
print("SHAP ANALYSIS - EXPLAINING MODEL PREDICTIONS")
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

# Sample for SHAP (use subset for speed)
np.random.seed(42)
sample_size = min(500, len(X_test))
sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
X_sample = X_test_scaled[sample_idx]
print(f" Test data: {len(X_test):,} rows")
print(f" SHAP sample: {sample_size} rows")

# Compute SHAP values
print("\n[STEP 3/5] Computing SHAP values...")
print(" (This takes 5-10 minutes for XGBoost)")

# Create explainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values
print(" Computing... (be patient)")
shap_values = explainer.shap_values(X_sample)
print(" ✓ SHAP values computed")

# SHAP feature importance
print("\n[STEP 4/5] Calculating SHAP-based feature importance...")
shap_importance = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    'feature': feature_names,
    'shap_importance': shap_importance
})
shap_df = shap_df.sort_values('shap_importance', ascending=False)

print("\nTop 10 Features (by SHAP):")
print(shap_df.head(10).to_string(index=False))

# Save SHAP importance
shap_df.to_csv(SHAP_DIR / "shap_importance_xgboost.csv", index=False)
print(f"\n ✓ SHAP importance saved")

# Visualizations
print("\n[STEP 5/5] Creating SHAP visualizations...")

# 1. SHAP Summary Plot
print(" Creating summary plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_names,
    max_display=20,
    show=False
)
plt.title('SHAP Summary Plot - Feature Impact on Risk Prediction',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(SHAP_DIR / "shap_summary_xgboost.png", dpi=150, bbox_inches='tight')
print(" ✓ Summary plot saved")
plt.close()

# 2. SHAP Feature Importance Bar Chart
print(" Creating feature importance chart...")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
top_20 = shap_df.head(20)
bars = ax.barh(range(len(top_20)), top_20['shap_importance'],
               color='#1976d2', alpha=0.7)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'])
ax.set_xlabel('Mean |SHAP Value| (Average Impact on Model Output)', fontsize=12)
ax.set_title('SHAP Feature Importance - XGBoost', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_20['shap_importance'])):
    ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(SHAP_DIR / "shap_importance_bar_xgboost.png", dpi=150)
print(" ✓ Importance bar chart saved")
plt.close()

# 3. SHAP Waterfall Plot (example prediction)
print(" Creating waterfall plot (example prediction)...")
y_test = test['target_binary'].values
high_risk_idx = np.where(y_test[sample_idx] == 1)[0]

if len(high_risk_idx) > 0:
    example_idx = high_risk_idx[0]
    
    plt.figure(figsize=(10, 8))
    shap_exp = shap.Explanation(
        values=shap_values[example_idx],
        base_values=explainer.expected_value,
        data=X_sample[example_idx],
        feature_names=feature_names
    )
    shap.waterfall_plot(shap_exp, max_display=15, show=False)
    plt.title('SHAP Waterfall Plot - Example High-Risk Prediction',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_waterfall_example.png", dpi=150, bbox_inches='tight')
    print(" ✓ Waterfall plot saved")
    plt.close()

# 4. Dependence plots for top 3 features
print(" Creating dependence plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, feat in enumerate(shap_df.head(3)['feature']):
    ax = axes[idx]
    feat_idx = feature_names.index(feat)
    
    # Scatter plot: feature value vs SHAP value
    scatter = ax.scatter(
        X_sample[:, feat_idx],
        shap_values[:, feat_idx],
        c=shap_values[:, feat_idx],
        cmap='coolwarm',
        alpha=0.6,
        s=20
    )
    
    ax.set_xlabel(f'{feat} Value', fontsize=11)
    ax.set_ylabel(f'SHAP Value for {feat}', fontsize=11)
    ax.set_title(f'{feat}', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='SHAP Value')

plt.suptitle('SHAP Dependence Plots - Top 3 Features',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(SHAP_DIR / "shap_dependence_top3.png", dpi=150, bbox_inches='tight')
print(" ✓ Dependence plots saved")
plt.close()

# Summary
print("\n" + "="*80)
print("SHAP ANALYSIS COMPLETE")
print("="*80)

print(f"\nSHAP Feature Importance (Top 10):")
for i, row in shap_df.head(10).iterrows():
    print(f" {i+1:2d}. {row['feature']:25s} {row['shap_importance']:.4f}")

print(f"\nFiles Created:")
print(f" • shap_analysis/shap_importance_xgboost.csv")
print(f" • shap_analysis/shap_summary_xgboost.png")
print(f" • shap_analysis/shap_importance_bar_xgboost.png")
print(f" • shap_analysis/shap_waterfall_example.png")
print(f" • shap_analysis/shap_dependence_top3.png")

print(f"\n✅ Ready to create model API for Sumit!")
print(f"Completed: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)