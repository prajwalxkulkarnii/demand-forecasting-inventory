"""
Demand Forecasting Training & Evaluation script.
Run:  python notebooks/03_forecasting.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_pipeline import DataPipeline
from forecasting import DemandForecaster
from visualizer import Visualizer

DATA    = os.path.join(os.path.dirname(__file__), "..", "data", "sales_data.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  Demand Forecasting — Train & Evaluate  (Notebook 03)")
print("=" * 60)

# Pipeline
pipeline = DataPipeline(DATA)
pipeline.load_data().clean_data().engineer_features()
X_train, X_test, y_train, y_test = pipeline.split_data(test_months=3)

print(f"\nTrain: {len(X_train):,} samples | Test: {len(X_test):,} samples")

# Train
forecaster = DemandForecaster(model_type="gradient_boosting")
print("\nTraining Gradient Boosting model...")
forecaster.train(X_train, y_train)

# Evaluate
print("\n--- Test Set Evaluation ---")
preds = forecaster.predict(X_test)
metrics = forecaster.evaluate(y_test, preds)
for k, v in metrics.items():
    print(f"  {k:<6}: {v}")

# Feature importance plot
fi = forecaster.feature_importances(pipeline.feature_columns)
print(f"\nTop 5 features:\n{fi.head(5).to_string(index=False)}")
fig, ax = plt.subplots(figsize=(10, 6))
top_fi = fi.tail(10)
ax.barh(top_fi["feature"], top_fi["importance"], color="steelblue")
ax.set_title("Top 10 Feature Importances", fontsize=13)
ax.set_xlabel("Importance")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "07_feature_importance.png"), dpi=140)
plt.close(fig)

# Forecast vs actual for top store/item
df = pipeline.df
store1 = df["store"].cat.categories[0]
item1  = df["item"].cat.categories[0]
sub = df[(df["store"]==store1) & (df["item"]==item1)]
sub_X = sub[pipeline.feature_columns]
sub_y = sub["sales"]
sub_pred = forecaster.predict(sub_X)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(sub["date"].values, sub_y.values, label="Actual", color="steelblue", lw=1.5)
ax.plot(sub["date"].values, sub_pred, label="Predicted", color="tomato", lw=1.5, linestyle="--")
ax.set_title(f"Forecast vs Actual — {store1} / {item1}", fontsize=13)
ax.set_ylabel("Sales (units)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "08_forecast_vs_actual.png"), dpi=140)
plt.close(fig)

print(f"\n✅ Plots saved to: {OUT_DIR}")
print("Done.")
