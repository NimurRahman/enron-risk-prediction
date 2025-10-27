import pickle, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

path = Path(r"I:\enron_dashboard\data\model_rf.pkl")
with open(path, "rb") as f:
    model = pickle.load(f)

print("Model type:", type(model))

# Try sklearn's recorded feature names
names = getattr(model, "feature_names_in_", None)
if names is not None:
    print("\nfeature_names_in_:\n", list(names))

# If it's a Pipeline, try ColumnTransformer
if isinstance(model, Pipeline):
    print("\nPipeline steps:", [n for n,_ in model.steps])
    for step_name, step in model.steps:
        if isinstance(step, ColumnTransformer):
            try:
                cols = step.get_feature_names_out().tolist()
                print("\nColumnTransformer features:\n", cols)
            except Exception as e:
                print("\nColumnTransformer names not available:", e)

# Fallback: look at training matrix headers
try:
    mi = pd.read_csv(r"I:\enron_dashboard\data\model_input.csv", nrows=5)
    print("\nmodel_input.csv columns:\n", mi.columns.tolist())
except Exception as e:
    print("\nCouldn't read model_input.csv:", e)
