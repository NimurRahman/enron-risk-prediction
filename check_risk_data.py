import pandas as pd

risk = pd.read_parquet("I:/enron_modeling/data/node_week_risk.parquet")

print("Risk bands in node_week_risk.parquet:")
print(risk['risk_band'].value_counts())
print("\nColumns:", list(risk.columns))
print("\nSample data:")
print(risk.head())