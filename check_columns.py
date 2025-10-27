import pandas as pd
from pathlib import Path

data_dir = Path("I:/enron_modeling/data")
outputs_dir = Path("I:/enron_modeling/outputs")

print("=== CHECKING DATA COLUMNS ===\n")

# Check edges
edges_path = data_dir / "edges_weekly_weighted.parquet"
if edges_path.exists():
    edges = pd.read_parquet(edges_path)
    print(f"EDGES columns: {list(edges.columns)}")
    print(f"EDGES shape: {edges.shape}\n")

# Check risk
risk_path = data_dir / "node_week_risk.parquet"
if risk_path.exists():
    risk = pd.read_parquet(risk_path)
    print(f"RISK columns: {list(risk.columns)}")
    print(f"RISK shape: {risk.shape}\n")

# Check predictions
preds_path = outputs_dir / "predictions_test_enhanced.csv"
if preds_path.exists():
    preds = pd.read_csv(preds_path, nrows=5)
    print(f"PREDICTIONS columns: {list(preds.columns)}")
    print(f"PREDICTIONS sample:")
    print(preds.head())
else:
    print("predictions_test_enhanced.csv NOT FOUND in outputs\n")
    # Try data folder
    preds_path2 = data_dir / "predictions_test_enhanced.csv"
    if preds_path2.exists():
        preds = pd.read_csv(preds_path2, nrows=5)
        print(f"Found in data folder! PREDICTIONS columns: {list(preds.columns)}")