# 11_generate_predictions_enhanced.py
"""
GENERATE ENHANCED PREDICTIONS FOR DASHBOARD
--------------------------------------------
Creates predictions_test.csv with 7 enhancements:
1. Top contributing features per person (SHAP)
2. Peer comparison (percentiles)
3. Actionable recommendations
4. Historical context (rolling stats)
5. Confidence intervals
6. Multi-model ensemble
7. Risk change indicators

Author: Nimur
Date: October 18, 2025
"""
import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path
from datetime import datetime

print("="*80)
print("GENERATING ENHANCED PREDICTIONS FOR DASHBOARD")
print("="*80)

# Paths
BASE = Path(r"I:\enron_modeling")
MODEL_DIR = BASE / "models"
OUTPUT_DIR = BASE / "outputs"
HANDOFF_DIR = BASE / "handoff_to_sumit"

# Load test data
print("\n[1/10] Loading test data...")
test_df = pd.read_parquet(OUTPUT_DIR / "modeling_data_full.parquet")
print(f" Test data: {len(test_df):,} rows")

# Load feature names
with open(OUTPUT_DIR / "feature_names.txt") as f:
    feature_names = [line.strip() for line in f if line.strip()]

print(f" Features: {len(feature_names)}")

# Load all 3 models
print("\n[2/10] Loading all 3 models...")
logreg_bundle = joblib.load(MODEL_DIR / "model_baseline_logreg.pkl")
rf_bundle = joblib.load(MODEL_DIR / "model_rf.pkl")
xgb_bundle = joblib.load(MODEL_DIR / "model_xgboost.pkl")

print(" ✓ Logistic Regression")
print(" ✓ Random Forest")
print(" ✓ XGBoost")

# Prepare features
print("\n[3/10] Preparing features...")
X_test = test_df[feature_names].values
X_test_scaled = xgb_bundle['scaler'].transform(X_test)

# Enhancement #6: Multi-Model Ensemble
print("\n[4/10] Enhancement #6: Multi-model ensemble predictions...")
logreg_proba = logreg_bundle['model'].predict_proba(X_test_scaled)[:, 1]
rf_proba = rf_bundle['model'].predict_proba(X_test_scaled)[:, 1]
xgb_proba = xgb_bundle['model'].predict_proba(X_test_scaled)[:, 1]

# Ensemble: Average of all 3
ensemble_proba = (logreg_proba + rf_proba + xgb_proba) / 3

# Model agreement: Standard deviation (low = high agreement)
model_agreement = np.std([logreg_proba, rf_proba, xgb_proba], axis=0)
agreement_label = np.where(model_agreement < 0.1, "high",
                          np.where(model_agreement < 0.2, "medium", "low"))

print(f" ✓ Ensemble predictions generated")
print(f" ✓ Model agreement calculated")

# Enhancement #5: Confidence Intervals (using model agreement as proxy)
print("\n[5/10] Enhancement #5: Confidence intervals...")
# Use XGBoost as primary, confidence based on model agreement
confidence_margin = model_agreement * 1.96  # 95% confidence
confidence_low = np.clip(xgb_proba - confidence_margin, 0, 1)
confidence_high = np.clip(xgb_proba + confidence_margin, 0, 1)

print(f" ✓ Confidence intervals calculated")

# Get 4-class probabilities from XGBoost
print("\n[6/10] Getting risk band probabilities...")
# Map binary to 4-class (approximate)
proba_critical = np.where(xgb_proba > 0.75, xgb_proba, 0)
proba_high = np.where((xgb_proba > 0.50) & (xgb_proba <= 0.75), xgb_proba, 0)
proba_medium = np.where((xgb_proba > 0.20) & (xgb_proba <= 0.50), xgb_proba, 0)
proba_low = np.where(xgb_proba <= 0.20, 1 - xgb_proba, 0)

# Normalize so they sum to 1
total = proba_low + proba_medium + proba_high + proba_critical + 1e-9
proba_low = proba_low / total
proba_medium = proba_medium / total
proba_high = proba_high / total
proba_critical = proba_critical / total

print(" ✓ 4-class probabilities calculated")

# Enhancement #1: Top Contributing Features (SHAP)
print("\n[7/10] Enhancement #1: Computing SHAP values for top features...")
print(" (This will take 2-3 minutes...)")

# Use a sample for SHAP (too slow for all 20K)
SHAP_SAMPLE_SIZE = len(X_test_scaled)
sample_indices = np.random.choice(len(X_test_scaled), 
                                  min(SHAP_SAMPLE_SIZE, len(X_test_scaled)), 
                                  replace=False)
X_shap_sample = X_test_scaled[sample_indices]

# Compute SHAP
explainer = shap.TreeExplainer(xgb_bundle['model'])
shap_values = explainer.shap_values(X_shap_sample)

# Get top 3 features for each person in sample
top_features_list = []
for i in range(len(X_shap_sample)):
    # Get absolute SHAP values for this person
    shap_abs = np.abs(shap_values[i])
    
    # Get top 3 indices
    top_3_idx = np.argsort(shap_abs)[-3:][::-1]
    
    # Format as "feature=value"
    top_3_str = []
    for idx in top_3_idx:
        feat_name = feature_names[idx]
        feat_value = X_test_scaled[sample_indices[i], idx]
        top_3_str.append(f"{feat_name}={feat_value:.2f}")
    
    top_features_list.append("; ".join(top_3_str))

# For non-sampled rows, use average top features
avg_top_features = "degree_ma4; total_emails_ma4; betweenness_ma4"

# Create full list (sampled get real SHAP, others get average)
all_top_features = [avg_top_features] * len(X_test)
for i, sample_idx in enumerate(sample_indices):
    all_top_features[sample_idx] = top_features_list[i]

print(f" ✓ SHAP top features computed for {SHAP_SAMPLE_SIZE:,} samples")

# Enhancement #3: Actionable Recommendations
print("\n[8/10] Enhancement #3: Generating recommendations...")
recommendations = []

for i in range(len(test_df)):
    risk = xgb_proba[i]
    
    if risk < 0.5:
        rec = "Maintain current patterns"
    elif risk < 0.75:
        # Extract top feature from top_features_list
        top_feat = all_top_features[i].split(";")[0].split("=")[0]
        rec = f"Monitor closely; focus on reducing {top_feat}"
    else:
        rec = "Urgent: Reduce after-hours emails, decrease connection count"
    
    recommendations.append(rec)

print(" ✓ Recommendations generated")

# Enhancement #2: Peer Comparison
print("\n[9/10] Enhancement #2: Calculating peer comparisons...")
percentiles = pd.Series(xgb_proba).rank(pct=True) * 100

# Format as percentile labels
percentile_labels = []
for p in percentiles:
    if p >= 99:
        percentile_labels.append("Top 1% (Critical)")
    elif p >= 95:
        percentile_labels.append("Top 5% (High)")
    elif p >= 80:
        percentile_labels.append("Top 20% (Elevated)")
    else:
        percentile_labels.append("Normal")

print(" ✓ Peer comparisons calculated")

# Enhancement #4 & #7: Historical Context & Risk Change
print("\n[10/10] Enhancement #4 & #7: Computing historical context...")

# Create a copy with risk scores
results_df = test_df.copy()
results_df['risk_score'] = xgb_proba

# Sort by node_id and week_start
results_df = results_df.sort_values(['node_id', 'week_start'])

# Calculate rolling statistics per person
def calc_historical_stats(group):
    group = group.copy()
    group['avg_risk_last_4_weeks'] = group['risk_score'].rolling(4, min_periods=1).mean()
    group['max_risk_ever'] = group['risk_score'].expanding().max()
    group['risk_volatility'] = group['risk_score'].rolling(4, min_periods=1).std().fillna(0)
    
    # Risk change
    group['risk_score_last_week'] = group['risk_score'].shift(1)
    group['risk_delta'] = group['risk_score'] - group['risk_score_last_week']
    
    # Trend
    group['trend'] = group['risk_delta'].apply(
        lambda x: "⬆ RISING" if pd.notna(x) and x > 0.05 
        else "⬇ FALLING" if pd.notna(x) and x < -0.05 
        else "➡ STABLE"
    )
    return group

historical_df = results_df.groupby('node_id', group_keys=False).apply(calc_historical_stats)

# Reset to match original test_df order
historical_df = historical_df.sort_values(['node_id', 'week_start']).reset_index(drop=True)
test_df_reset = test_df.sort_values(['node_id', 'week_start']).reset_index(drop=True)

print(" ✓ Historical context calculated")

# Create final predictions DataFrame
print("\n[FINAL] Creating enhanced predictions DataFrame...")

predictions = pd.DataFrame({
    # Basic info
    'node_id': test_df_reset['node_id'].values,
    'week_start': test_df_reset['week_start'].values,
    
    # True labels
    'y_true': test_df_reset['risk_band'].values,
    
    # Predictions (reorder to match sorted data)
    'risk_score': historical_df['risk_score'].values,
    'y_pred': np.where(historical_df['risk_score'].values > 0.5, 'High/Critical', 'Low/Medium'),
    
    # 4-class probabilities (need to reorder these too)
})

# Reorder other arrays to match sorted data
sort_indices = test_df.sort_values(['node_id', 'week_start']).index
proba_low_sorted = proba_low[sort_indices]
proba_medium_sorted = proba_medium[sort_indices]
proba_high_sorted = proba_high[sort_indices]
proba_critical_sorted = proba_critical[sort_indices]
all_top_features_sorted = [all_top_features[i] for i in sort_indices]
recommendations_sorted = [recommendations[i] for i in sort_indices]
percentiles_sorted = percentiles.iloc[sort_indices].values
percentile_labels_sorted = [percentile_labels[i] for i in sort_indices]
confidence_low_sorted = confidence_low[sort_indices]
confidence_high_sorted = confidence_high[sort_indices]
logreg_proba_sorted = logreg_proba[sort_indices]
rf_proba_sorted = rf_proba[sort_indices]
xgb_proba_sorted = xgb_proba[sort_indices]
ensemble_proba_sorted = ensemble_proba[sort_indices]
agreement_label_sorted = [agreement_label[i] for i in sort_indices]

# Add all columns
predictions['proba_Low'] = proba_low_sorted
predictions['proba_Medium'] = proba_medium_sorted
predictions['proba_High'] = proba_high_sorted
predictions['proba_Critical'] = proba_critical_sorted
predictions['top_features'] = all_top_features_sorted
predictions['percentile'] = percentiles_sorted
predictions['risk_category'] = percentile_labels_sorted
predictions['recommendation'] = recommendations_sorted
predictions['avg_risk_last_4_weeks'] = historical_df['avg_risk_last_4_weeks'].values
predictions['max_risk_ever'] = historical_df['max_risk_ever'].values
predictions['risk_volatility'] = historical_df['risk_volatility'].values
predictions['confidence_low'] = confidence_low_sorted
predictions['confidence_high'] = confidence_high_sorted
predictions['risk_score_logreg'] = logreg_proba_sorted
predictions['risk_score_rf'] = rf_proba_sorted
predictions['risk_score_xgb'] = xgb_proba_sorted
predictions['risk_score_ensemble'] = ensemble_proba_sorted
predictions['model_agreement'] = agreement_label_sorted
predictions['risk_score_last_week'] = historical_df['risk_score_last_week'].values
predictions['risk_delta'] = historical_df['risk_delta'].values
predictions['trend'] = historical_df['trend'].values

# Save to outputs
output_path = OUTPUT_DIR / "predictions_test_enhanced.csv"
predictions.to_csv(output_path, index=False)
print(f"\n ✓ Saved to: {output_path}")

# Also save to handoff directory
handoff_path = HANDOFF_DIR / "predictions_test.csv"
predictions.to_csv(handoff_path, index=False)
print(f" ✓ Saved to: {handoff_path}")

# Summary
print("\n" + "="*80)
print("ENHANCED PREDICTIONS COMPLETE")
print("="*80)

print(f"\nDataset Summary:")
print(f" Total predictions: {len(predictions):,}")
print(f" Unique individuals: {predictions['node_id'].nunique():,}")
print(f" Weeks covered: {predictions['week_start'].nunique()}")

print(f"\nRisk Distribution:")
print(predictions['y_pred'].value_counts())

print(f"\nModel Agreement:")
print(predictions['model_agreement'].value_counts())

print(f"\nTrend Distribution:")
print(predictions['trend'].value_counts())

print(f"\n✅ 7 ENHANCEMENTS INCLUDED:")
print(" 1. ✅ Top contributing features (SHAP)")
print(" 2. ✅ Peer comparison (percentiles)")
print(" 3. ✅ Actionable recommendations")
print(" 4. ✅ Historical context (rolling stats)")
print(" 5. ✅ Confidence intervals")
print(" 6. ✅ Multi-model ensemble")
print(" 7. ✅ Risk change indicators")

print(f"\nColumns in CSV ({len(predictions.columns)}):")
for col in predictions.columns:
    print(f" - {col}")

print(f"\n💡 Dashboard can now show:")
print(" • Individual risk scores with confidence bands")
print(" • WHY each person is high-risk (top features)")
print(" • Peer comparison (percentile rankings)")
print(" • Actionable next steps (recommendations)")
print(" • Historical trends (4-week averages)")
print(" • Model reliability (ensemble agreement)")
print(" • Risk trajectory (rising/falling/stable)")

print("\n🎉 Ready for Sumit's dashboard integration!")
print("="*80)