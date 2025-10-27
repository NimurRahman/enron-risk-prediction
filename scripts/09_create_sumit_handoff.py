# 09_create_sumit_handoff.py
"""
CREATE HANDOFF PACKAGE FOR SUMIT
---------------------------------
Package all models and files Sumit needs for AI Agent.

What Sumit needs:
- Trained models (.pkl files)
- SHAP files
- Feature names
- Performance metrics

Author: Nimur
Date: Saturday 10 PM
"""
import shutil
from pathlib import Path
import joblib
import json
from datetime import datetime

print("="*80)
print("CREATING HANDOFF PACKAGE FOR SUMIT")
print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)

# Paths
BASE = Path(r"I:\enron_modeling")
MODEL_DIR = BASE / "models"
SHAP_DIR = BASE / "shap_analysis"
OUTPUT_DIR = BASE / "outputs"

# Create handoff directory
HANDOFF_DIR = BASE / "handoff_to_sumit"
HANDOFF_DIR.mkdir(exist_ok=True)

# Subdirectories
(HANDOFF_DIR / "models").mkdir(exist_ok=True)
(HANDOFF_DIR / "shap_analysis").mkdir(exist_ok=True)
(HANDOFF_DIR / "documentation").mkdir(exist_ok=True)

print("\n[1/6] Copying model files...")
# Copy all trained models
models_to_copy = [
    "model_baseline_logreg.pkl",
    "model_rf.pkl",
    "model_xgboost.pkl"
]

for model_file in models_to_copy:
    src = MODEL_DIR / model_file
    dst = HANDOFF_DIR / "models" / model_file
    if src.exists():
        shutil.copy(src, dst)
        print(f" ✓ {model_file}")
    else:
        print(f" ✗ {model_file} not found")

print("\n[2/6] Copying SHAP files...")
# Copy SHAP analysis files
shap_files = list(SHAP_DIR.glob("*"))
for shap_file in shap_files:
    dst = HANDOFF_DIR / "shap_analysis" / shap_file.name
    shutil.copy(shap_file, dst)
    print(f" ✓ {shap_file.name}")

print("\n[3/6] Copying feature names...")
# Copy feature names
shutil.copy(
    OUTPUT_DIR / "feature_names.txt",
    HANDOFF_DIR / "models" / "feature_names.txt"
)
print(" ✓ feature_names.txt")

print("\n[4/6] Creating model performance summary...")
# Load all models and extract metrics
logreg = joblib.load(MODEL_DIR / "model_baseline_logreg.pkl")
rf = joblib.load(MODEL_DIR / "model_rf.pkl")
xgb = joblib.load(MODEL_DIR / "model_xgboost.pkl")

performance_summary = {
    "model_comparison": {
        "logistic_regression": {
            "test_pr_auc": logreg['metrics']['test_pr_auc'],
            "test_roc_auc": logreg['metrics']['test_roc_auc'],
            "file": "model_baseline_logreg.pkl"
        },
        "random_forest": {
            "test_pr_auc": rf['metrics']['test_pr_auc'],
            "test_roc_auc": rf['metrics']['test_roc_auc'],
            "file": "model_rf.pkl"
        },
        "xgboost": {
            "test_pr_auc": xgb['metrics']['test_pr_auc'],
            "test_roc_auc": xgb['metrics']['test_roc_auc'],
            "file": "model_xgboost.pkl",
            "best_model": True
        }
    },
    "recommendation": "Use XGBoost (best PR-AUC)",
    "features": logreg['features'],
    "created": datetime.now().isoformat()
}

with open(HANDOFF_DIR / "models" / "model_performance.json", "w") as f:
    json.dump(performance_summary, f, indent=2)

print(" ✓ model_performance.json")

print("\n[5/6] Creating README for Sumit...")
readme_content = f"""# Model Handoff Package for Sumit

**From:** Nimur (Predictive Modeling)  
**To:** Sumit (AI Agent)  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Status:** Ready for integration

---

## What's In This Package

### models/
- **model_xgboost.pkl** - ⭐ BEST MODEL (PR-AUC: {xgb['metrics']['test_pr_auc']:.3f})
- **model_rf.pkl** - Random Forest (PR-AUC: {rf['metrics']['test_pr_auc']:.3f})
- **model_baseline_logreg.pkl** - Baseline (PR-AUC: {logreg['metrics']['test_pr_auc']:.3f})
- **feature_names.txt** - List of {len(logreg['features'])} features (ORDER MATTERS!)
- **model_performance.json** - Performance metrics

### shap_analysis/
- **shap_importance_xgboost.csv** - Feature importance from SHAP
- **shap_summary_xgboost.png** - Visual explanation chart
- **shap_waterfall_example.png** - Example prediction explanation
- **shap_dependence_top3.png** - Feature impact plots

---

## How to Use Models

### Loading a Model
```python
import joblib

# Load model bundle
bundle = joblib.load("models/model_xgboost.pkl")
model = bundle['model']       # Trained XGBoost
scaler = bundle['scaler']     # StandardScaler
features = bundle['features'] # List of {len(logreg['features'])} feature names (ORDER MATTERS!)
```

### Making Predictions
```python
import pandas as pd
import numpy as np

# Load your data
data = pd.read_csv("your_data.csv")

# Extract features in CORRECT ORDER
X = data[features].values

# Scale features
X_scaled = scaler.transform(X)

# Predict
risk_probabilities = model.predict_proba(X_scaled)[:, 1]  # Probability of high risk
risk_predictions = model.predict(X_scaled)                 # Binary: 0=Low, 1=High
```

### Using SHAP for Explanations
```python
import shap
import pandas as pd

# Load SHAP importance
shap_importance = pd.read_csv("shap_analysis/shap_importance_xgboost.csv")

# Get top 5 features
top_features = shap_importance.head(5)['feature'].tolist()

# For AI Agent: "John Doe is high-risk due to: {{top_features}}"
```

---

## Integration Checklist

- [ ] Test model loading
- [ ] Verify predictions work
- [ ] Check feature order matches
- [ ] Test SHAP integration
- [ ] Verify AI Agent can explain predictions
- [ ] Test "what-if" simulator

---

## Performance Summary

**Best Model:** XGBoost  
**Test PR-AUC:** {xgb['metrics']['test_pr_auc']:.3f}  
**Test ROC-AUC:** {xgb['metrics']['test_roc_auc']:.3f}  

**Model Ranking:**
1. XGBoost: {xgb['metrics']['test_pr_auc']:.3f} ⭐
2. Random Forest: {rf['metrics']['test_pr_auc']:.3f}
3. Logistic: {logreg['metrics']['test_pr_auc']:.3f}

---

## Questions?

Contact Nimur if:
- Models won't load
- Feature order issues
- Integration problems
- Need clarification on SHAP usage

---

**Status:** ✅ Ready for AI Agent integration
"""

with open(HANDOFF_DIR / "documentation" / "README.md", "w", encoding='utf-8') as f:
    f.write(readme_content)

print(" ✓ README.md")

print("\n[6/6] Creating file manifest...")
# Create manifest of all files
manifest = {
    "handoff_created": datetime.now().isoformat(),
    "from": "Nimur",
    "to": "Sumit",
    "files": {
        "models": [f.name for f in (HANDOFF_DIR / "models").glob("*")],
        "shap_analysis": [f.name for f in (HANDOFF_DIR / "shap_analysis").glob("*")],
        "documentation": [f.name for f in (HANDOFF_DIR / "documentation").glob("*")]
    },
    "total_files": sum([
        len(list((HANDOFF_DIR / "models").glob("*"))),
        len(list((HANDOFF_DIR / "shap_analysis").glob("*"))),
        len(list((HANDOFF_DIR / "documentation").glob("*")))
    ])
}

with open(HANDOFF_DIR / "MANIFEST.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(" ✓ MANIFEST.json")

# Summary
print("\n" + "="*80)
print("HANDOFF PACKAGE COMPLETE")
print("="*80)

print(f"\nPackage Location: {HANDOFF_DIR}")
print(f"\nContents:")
print(f" models/ ({len(manifest['files']['models'])} files)")
for f in manifest['files']['models']:
    print(f"   - {f}")

print(f"\n shap_analysis/ ({len(manifest['files']['shap_analysis'])} files)")
for f in manifest['files']['shap_analysis']:
    print(f"   - {f}")

print(f"\n documentation/ ({len(manifest['files']['documentation'])} files)")
for f in manifest['files']['documentation']:
    print(f"   - {f}")

print(f"\nTotal files: {manifest['total_files']}")

print(f"\n📦 HANDOFF READY FOR SUMIT")

print(f"\n📧 Message to send Sumit:")
print("─" * 70)
print(f"""
Hey Sumit,

Models are ready for you!

Location: I:\\enron_modeling\\handoff_to_sumit\\

What's included:
- 3 trained models (XGBoost is best: PR-AUC {xgb['metrics']['test_pr_auc']:.3f})
- REAL SHAP analysis files (individual prediction explanations!)
- Feature names ({len(logreg['features'])} features)
- Complete documentation (README.md)

Read the README first - it has:
- How to load models
- How to make predictions  
- How to use SHAP for explanations
- Integration checklist

Best model: model_xgboost.pkl (XGBoost 2.0.3 - kept for SHAP compatibility)
Performance: PR-AUC {xgb['metrics']['test_pr_auc']:.3f}, ROC-AUC {xgb['metrics']['test_roc_auc']:.3f}

Your AI Agent can now explain individual predictions using SHAP!

Ready when you are!
- Nimur
""")
print("─" * 70)

print("\n✅ Handoff package complete!")
print(f"Completed: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)