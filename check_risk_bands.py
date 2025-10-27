import pandas as pd

preds = pd.read_csv("I:/enron_modeling/outputs/predictions_test_enhanced.csv")

print("Risk bands in predictions (y_pred column):")
print(preds['y_pred'].value_counts())
print("\nUnique values:", preds['y_pred'].unique())