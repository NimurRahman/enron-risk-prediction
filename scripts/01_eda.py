# 01_eda.py
"""
EXPLORATORY DATA ANALYSIS
-------------------------
Understand the data before modeling.
Author: Nimur
Date: Saturday 10:45 AM
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*80)
print("EXPLORATORY DATA ANALYSIS - ENRON RISK PREDICTION")
print("="*80)

# Paths
BASE = Path(r"I:\enron_modeling")
DATA_DIR = BASE / "data"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
print("\n[1/6] Loading data...")
try:
    features = pd.read_parquet(DATA_DIR / "node_week_features_enhanced.parquet")
    print(f" ✓ Features: {len(features):,} rows, {len(features.columns)} columns")
except Exception as e:
    print(f" ✗ Features: {e}")
    features = None

try:
    risk = pd.read_parquet(DATA_DIR / "node_week_risk_enhanced.parquet")
    print(f" ✓ Risk: {len(risk):,} rows, {len(risk.columns)} columns")
except Exception as e:
    print(f" ✗ Risk: {e}")
    risk = None

try:
    nodes = pd.read_csv(DATA_DIR / "nodes.csv")
    print(f" ✓ Nodes: {len(nodes):,} rows")
except Exception as e:
    print(f" ✗ Nodes: {e}")
    nodes = None

# Check if we have data
if features is None or risk is None:
    print("\n❌ ERROR: Cannot proceed without features and risk data")
    print("Please copy data files to I:\\enron_modeling\\data\\")
    exit(1)

# Basic info
print("\n[2/6] Data overview...")
print(f"\nFeatures columns:")
print(features.columns.tolist())
print(f"\nRisk columns:")
print(risk.columns.tolist())

print(f"\nDate range:")
if 'week_start' in features.columns:
    print(f" Features: {features['week_start'].min()} to {features['week_start'].max()}")
if 'week_start' in risk.columns:
    print(f" Risk: {risk['week_start'].min()} to {risk['week_start'].max()}")

# Missing values
print("\n[3/6] Missing values check...")
print("\nFeatures missing values:")
missing = features.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print(missing)
else:
    print(" ✓ No missing values")

print("\nRisk missing values:")
missing_risk = risk.isnull().sum()
missing_risk = missing_risk[missing_risk > 0].sort_values(ascending=False)
if len(missing_risk) > 0:
    print(missing_risk)
else:
    print(" ✓ No missing values")

# Risk distribution
print("\n[4/6] Risk distribution analysis...")
if 'risk_band' in risk.columns:
    print("\nRisk band counts:")
    band_counts = risk['risk_band'].value_counts()
    print(band_counts)
    
    print("\nRisk band percentages:")
    band_pct = risk['risk_band'].value_counts(normalize=True) * 100
    print(band_pct)

if 'risk_score' in risk.columns:
    print("\nRisk score statistics:")
    print(risk['risk_score'].describe())

# Feature statistics
print("\n[5/6] Feature statistics...")
numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['node_id', 'week_start']]

if len(numeric_cols) > 0:
    print(f"\nNumeric features ({len(numeric_cols)} total):")
    print("\nSample statistics (first 10 features):")
    print(features[numeric_cols[:10]].describe())
    
    full_stats = features[numeric_cols].describe()
    full_stats.to_csv(OUTPUT_DIR / "feature_statistics.csv")
    print(f"\n ✓ Full statistics saved to: outputs/feature_statistics.csv")

# Visualizations
print("\n[6/6] Creating visualizations...")
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Risk band distribution
if 'risk_band' in risk.columns:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    band_counts = risk['risk_band'].value_counts()
    colors = {'Low': '#66bb6a', 'Medium': '#ffb74d', 'High': '#ef5350', 'Critical': '#8d6e63'}
    band_order = ['Low', 'Medium', 'High', 'Critical']
    band_counts = band_counts.reindex(band_order, fill_value=0)
    
    bars = ax.bar(band_counts.index, band_counts.values,
                  color=[colors.get(b, 'gray') for b in band_counts.index])
    ax.set_xlabel('Risk Band', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Risk Band Distribution (All Data)', fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "risk_band_distribution.png", dpi=150)
    print(" ✓ Risk band distribution saved")
    plt.close()

# 2. Risk score distribution
if 'risk_score' in risk.columns:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(risk['risk_score'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Risk Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
    ax.axvline(risk['risk_score'].mean(), color='red', linestyle='--',
               label=f"Mean: {risk['risk_score'].mean():.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "risk_score_distribution.png", dpi=150)
    print(" ✓ Risk score distribution saved")
    plt.close()

# Summary report
print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)
print(f"\nSummary:")
print(f" Total records: {len(features):,}")
print(f" Unique individuals: {features['node_id'].nunique():,}")
print(f" Time period: {features['week_start'].nunique()} weeks")
print(f" Features: {len(numeric_cols)}")

if 'risk_band' in risk.columns:
    critical_pct = (risk['risk_band'] == 'Critical').sum() / len(risk) * 100
    high_pct = (risk['risk_band'] == 'High').sum() / len(risk) * 100
    print(f" High risk: {high_pct:.1f}%")
    print(f" Critical risk: {critical_pct:.1f}%")

print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f" • feature_statistics.csv")
print(f" • risk_band_distribution.png")
print(f" • risk_score_distribution.png")

print("\n✅ Ready for feature engineering!")
print("="*80)