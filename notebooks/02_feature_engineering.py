"""
Feature Engineering notebook-style script.
Run:  python notebooks/02_feature_engineering.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_pipeline import DataPipeline

DATA    = os.path.join(os.path.dirname(__file__), "..", "data", "sales_data.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  Feature Engineering  (Notebook 02)")
print("=" * 60)

pipeline = DataPipeline(DATA)
pipeline.load_data().clean_data()
print(f"\nBefore feature engineering: {pipeline.df.shape}")
print(f"Columns: {list(pipeline.df.columns)}")

pipeline.engineer_features()
print(f"\nAfter feature engineering:  {pipeline.df.shape}")
print(f"Feature columns ({len(pipeline.feature_columns)}):")
for f in pipeline.feature_columns:
    print(f"  • {f}")

# Correlation heatmap of features
fig, ax = plt.subplots(figsize=(14, 10))
corr_cols = pipeline.feature_columns + ["sales"]
corr = pipeline.df[corr_cols].corr()
mask = (corr.abs() < 0.0)   # show all
sns.heatmap(
    corr, ax=ax, annot=False, cmap="coolwarm",
    center=0, linewidths=0.3,
    cbar_kws={"label": "Pearson r"}
)
ax.set_title("Feature Correlation Matrix", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "05_feature_correlation.png"), dpi=130)
plt.close(fig)
print(f"\n✅ Correlation heatmap saved → {OUT_DIR}/05_feature_correlation.png")

# Lag feature distribution
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, lag in zip(axes, [1, 7, 14, 28]):
    ax.hist(pipeline.df[f"lag_{lag}"].dropna(), bins=40, color="teal", edgecolor="white", alpha=0.8)
    ax.set_title(f"lag_{lag}")
    ax.set_xlabel("Sales")
fig.suptitle("Distribution of Lag Features", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "06_lag_distributions.png"), dpi=130)
plt.close(fig)
print(f"✅ Lag distributions saved → {OUT_DIR}/06_lag_distributions.png")

print("\nSummary:", pipeline.get_summary())
