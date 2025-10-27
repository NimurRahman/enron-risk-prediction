# 02_prepare_modeling_data.py
"""
PREPARE MODELING DATASET
------------------------
Merge features + risk, create train/test split, handle missing values.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("="*80)
print("PREPARING MODELING DATASET")
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)

# Paths
BASE = Path(r"I:\enron_modeling")
DATA_DIR = BASE / "data"
OUTPUT_DIR = BASE / "outputs"

# Load data
print("\n[STEP 1/7] Loading data...")
features = pd.read_parquet(DATA_DIR / "node_week_features_enhanced.parquet")
risk = pd.read_parquet(DATA_DIR / "node_week_risk_enhanced.parquet")
print(f" Features: {len(features):,} rows")
print(f" Risk: {len(risk):,} rows")

# Merge features + risk
print("\n[STEP 2/7] Merging features and risk...")
df = features.merge(
    risk[['node_id', 'week_start', 'risk_score', 'risk_band']],
    on=['node_id', 'week_start'],
    how='inner'
)
print(f" Merged: {len(df):,} rows")

# Fix date issue - filter to valid Enron timeframe (2000-2002)
print("\n[STEP 3/7] Filtering to valid date range...")
df['week_start'] = pd.to_datetime(df['week_start'])
df = df[(df['week_start'] >= '2000-01-01') & (df['week_start'] <= '2002-12-31')]
print(f" After date filter: {len(df):,} rows")

# Identify feature columns
print("\n[STEP 4/7] Identifying feature columns...")
drop_cols = ['node_id', 'week_start', 'risk_score', 'risk_band', 'risk_pct']
drop_cols = [c for c in drop_cols if c in df.columns]
feature_cols = [c for c in df.columns if c not in drop_cols]
feature_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64]]
print(f" Feature columns: {len(feature_cols)}")
print(f" First 10 features: {feature_cols[:10]}")

# Save feature names
with open(OUTPUT_DIR / "feature_names.txt", "w") as f:
    f.write("\n".join(feature_cols))
print(f" ✓ Feature names saved")

# Handle missing values
print("\n[STEP 5/7] Handling missing values...")
missing_before = df[feature_cols].isnull().sum().sum()
print(f" Missing values before: {missing_before:,}")
df[feature_cols] = df[feature_cols].fillna(0)
missing_after = df[feature_cols].isnull().sum().sum()
print(f" Missing values after: {missing_after:,}")

# Create target variable
print("\n[STEP 6/7] Creating target variable...")
# Binary: High/Critical (1) vs Low/Elevated (0)
df['target_binary'] = df['risk_band'].isin(['High', 'Critical']).astype(int)

# Map risk bands
risk_band_map = {'Low': 0, 'Elevated': 1, 'High': 2, 'Critical': 3}
df['target_multiclass'] = df['risk_band'].map(risk_band_map)

print(" Target variables created:")
print(f" - target_binary: {df['target_binary'].value_counts().to_dict()}")
print(f" - target_multiclass: {df['target_multiclass'].value_counts().to_dict()}")

# Time-based train/test split
print("\n[STEP 7/7] Creating train/test split (time-based)...")
df = df.sort_values('week_start').reset_index(drop=True)
weeks = sorted(df['week_start'].unique())
split_idx = int(len(weeks) * 0.8)
train_weeks = weeks[:split_idx]
test_weeks = weeks[split_idx:]

print(f" Total weeks: {len(weeks)}")
print(f" Train weeks: {len(train_weeks)} (first 80%)")
print(f" Test weeks: {len(test_weeks)} (last 20%)")
print(f" Train period: {train_weeks[0].date()} to {train_weeks[-1].date()}")
print(f" Test period: {test_weeks[0].date()} to {test_weeks[-1].date()}")

train_mask = df['week_start'].isin(train_weeks)
test_mask = df['week_start'].isin(test_weeks)
train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

print(f"\n Train set: {len(train_df):,} rows")
print(f" Test set: {len(test_df):,} rows")

# Save datasets
print("\n[STEP 8/8] Saving processed datasets...")
df.to_parquet(OUTPUT_DIR / "modeling_data_full.parquet", index=False)
train_df.to_parquet(OUTPUT_DIR / "modeling_data_train.parquet", index=False)
test_df.to_parquet(OUTPUT_DIR / "modeling_data_test.parquet", index=False)
print(f" ✓ Full dataset saved")
print(f" ✓ Train saved")
print(f" ✓ Test saved")

# Summary
print("\n" + "="*80)
print("DATA PREPARATION COMPLETE")
print("="*80)
print(f"\nDataset Summary:")
print(f" Total records: {len(df):,}")
print(f" Features: {len(feature_cols)}")
print(f" Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
print(f" Test: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
print(f" Positive rate (High/Critical): {df['target_binary'].mean()*100:.1f}%")
print(f"\n✅ Ready for model training!")
print(f"Completed: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)