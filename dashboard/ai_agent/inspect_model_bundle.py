from pathlib import Path
import json, sys
import pandas as pd

# Robust loader: joblib first, then pickle
def load_any(path):
    obj = None
    try:
        import joblib
        obj = joblib.load(path)
        return obj, "joblib.load"
    except Exception as e:
        try:
            import pickle
            with open(path, "rb") as f:
                obj = pickle.load(f)
            return obj, "pickle.load"
        except Exception as e2:
            print("Failed to load:", e, "|", e2)
            sys.exit(1)

path = Path(r"I:\enron_dashboard\data\model_rf.pkl")
obj, loader = load_any(path)
print("Loader:", loader)
print("Top-level type:", type(obj))

# If it's a dict, show keys and try to pull an estimator + feature names
est = obj
feat_names = None
if isinstance(obj, dict):
    print("Dict keys:", list(obj.keys()))
    # common keys people use
    for k in ["model", "estimator", "pipeline", "clf", "rf", "best_estimator_", "estimator_"]:
        if k in obj:
            est = obj[k]
            print("Found estimator under key:", k)
            break
    # look for embedded feature names
    for k in ["feature_names", "features", "feature_list", "columns", "input_features"]:
        if k in obj and isinstance(obj[k], (list, tuple)):
            feat_names = list(obj[k])
            print("Found feature names under key:", k)
            break

# Try to introspect sklearn-style names on the estimator
try:
    names = getattr(est, "feature_names_in_", None)
    if names is not None:
        feat_names = list(names)
        print("feature_names_in_ found on estimator.")
except Exception:
    pass

# If it's a Pipeline with a ColumnTransformer, ask it
try:
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    if isinstance(est, Pipeline):
        print("Pipeline steps:", [n for n,_ in est.steps])
        for step_name, step in est.steps:
            if isinstance(step, ColumnTransformer):
                try:
                    feat_names = step.get_feature_names_out().tolist()
                    print("Feature names from ColumnTransformer.")
                except Exception as e:
                    print("ColumnTransformer feature names not available:", e)
except Exception as e:
    print("Pipeline inspection error:", e)

# Fallback: read from your training matrix headers
if feat_names is None:
    try:
        mi = pd.read_csv(r"I:\enron_dashboard\data\model_input.csv", nrows=5)
        feat_names = [c for c in mi.columns if c.lower() not in {"y","target","label"}]
        print("Using model_input.csv columns as features.")
    except Exception as e:
        print("Couldn't read model_input.csv:", e)

# Print + save feature list
if feat_names:
    print("\nExpected feature count:", len(feat_names))
    print("First 25 features:", feat_names[:25], "…")
    out = Path(r"I:\enron_dashboard\data\model_expected_features.txt")
    out.write_text("\n".join(map(str, feat_names)), encoding="utf-8")
    print(f"Saved feature list to {out}")
else:
    print("No feature names discovered.")

# Smoke test: try to predict with first row of model_input.csv if possible
try:
    mi = pd.read_csv(r"I:\enron_dashboard\data\model_input.csv")
    X = mi[feat_names] if feat_names else mi
    # strip target if present
    for t in ["y","target","label"]:
        if t in X.columns: X = X.drop(columns=[t])
    row = X.iloc[[0]]
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(row)[0]
        print("predict_proba OK. Example:", p)
    elif hasattr(est, "predict"):
        p = est.predict(row)[0]
        print("predict OK. Example:", p)
    else:
        print("Loaded estimator has no predict/predict_proba.")
except Exception as e:
    print("Prediction smoke test failed:", e)
