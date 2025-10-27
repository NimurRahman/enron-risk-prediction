# 04_visualize_features.py
"""
FEATURE VISUALIZATION
---------------------
Create visual explanations of features for client presentation.
Addresses Client Feedback #4: "How did we calculate the risks?"
Author: Nimur
Date: Saturday 3:30 PM
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*80)
print("FEATURE VISUALIZATION FOR CLIENT")
print("="*80)

# Paths
BASE = Path(r"I:\enron_modeling")
OUTPUT_DIR = BASE / "outputs"

# Load data
print("\n[1/4] Loading data...")
df = pd.read_parquet(OUTPUT_DIR / "modeling_data_full.parquet")
feature_importance = pd.read_csv(OUTPUT_DIR / "baseline_feature_importance.csv")
print(f" Data: {len(df):,} rows")
print(f" Features: {len(feature_importance)}")

# Get top features
top_features = feature_importance.nlargest(8, 'abs_coef')['feature'].tolist()

print("\n[2/4] Creating feature distributions by risk...")
# Feature distributions by risk band
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, feat in enumerate(top_features):
    ax = axes[i]
    
    # Box plot by risk band
    risk_order = ['Low', 'Medium', 'High', 'Critical']
    data_to_plot = [df[df['risk_band'] == band][feat].dropna() for band in risk_order]
    
    bp = ax.boxplot(data_to_plot, labels=risk_order, patch_artist=True)
    
    # Color boxes
    colors = ['#66bb6a', '#ffb74d', '#ef5350', '#8d6e63']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_title(f'{feat}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Value', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_distributions_by_risk.png", dpi=150)
print(" ✓ Feature distributions saved")
plt.close()

print("\n[3/4] Creating feature importance chart...")
# Bar chart of feature importance
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
top_10 = feature_importance.head(10)
colors_imp = ['#d32f2f' if x > 0 else '#1976d2' for x in top_10['coefficient']]

bars = ax.barh(range(len(top_10)), top_10['abs_coef'], color=colors_imp, alpha=0.7)
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10['feature'])
ax.set_xlabel('Absolute Coefficient (Importance)', fontsize=12)
ax.set_title('Top 10 Features for Risk Prediction\n(Baseline Logistic Regression)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_10['abs_coef'])):
    ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance_chart.png", dpi=150)
print(" ✓ Feature importance chart saved")
plt.close()

print("\n[4/4] Creating feature correlation with risk...")
# Scatter plots of top 4 features vs risk score
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Sample data for plotting (too many points otherwise)
df_sample = df.sample(n=min(5000, len(df)), random_state=42)

for i, feat in enumerate(top_features[:4]):
    ax = axes[i]
    
    # Scatter with color by risk band
    colors_map = {'Low': '#66bb6a', 'Medium': '#ffb74d', 'High': '#ef5350', 'Critical': '#8d6e63'}
    for band in ['Low', 'Medium', 'High', 'Critical']:
        data = df_sample[df_sample['risk_band'] == band]
        ax.scatter(data[feat], data['risk_score'],
                  c=colors_map[band], label=band, alpha=0.5, s=20)
    
    ax.set_xlabel(feat, fontsize=11)
    ax.set_ylabel('Risk Score', fontsize=11)
    ax.set_title(f'{feat} vs Risk Score', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_vs_risk_scatter.png", dpi=150)
print(" ✓ Feature vs risk scatter saved")
plt.close()

# Summary table for presentation
print("\n[BONUS] Creating feature summary table...")
feature_summary = []
for feat in top_features:
    low_mean = df[df['risk_band'] == 'Low'][feat].mean()
    crit_mean = df[df['risk_band'] == 'Critical'][feat].mean()
    ratio = crit_mean / (low_mean + 1e-9)
    
    feature_summary.append({
        'Feature': feat,
        'Low Risk Mean': f'{low_mean:.3f}',
        'Critical Risk Mean': f'{crit_mean:.3f}',
        'Critical/Low Ratio': f'{ratio:.2f}x',
        'Importance': f"{feature_importance[feature_importance['feature']==feat]['abs_coef'].values[0]:.3f}"
    })

summary_df = pd.DataFrame(feature_summary)
summary_df.to_csv(OUTPUT_DIR / "feature_summary_for_presentation.csv", index=False)
print(" ✓ Feature summary table saved")

print("\n" + "="*80)
print("FEATURE VISUALIZATION COMPLETE")
print("="*80)
print(f"\nFiles Created (for presentation):")
print(f" • feature_distributions_by_risk.png")
print(f"   → Shows how each feature differs by risk level")
print(f" • feature_importance_chart.png")
print(f"   → Top 10 features ranked by importance")
print(f" • feature_vs_risk_scatter.png")
print(f"   → Relationship between features and risk score")
print(f" • feature_summary_for_presentation.csv")
print(f"   → Table comparing Low vs Critical risk")
print(f"\n💡 Use these in your presentation to explain:")
print(f"   'How did we calculate the risks?'")
print("="*80)