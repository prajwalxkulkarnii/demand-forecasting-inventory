"""
Notebook-style EDA for Demand Forecasting.
Run:  python notebooks/01_eda.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)
DATA    = os.path.join(os.path.dirname(__file__), "..", "data", "sales_data.csv")

print("=" * 60)
print("  Demand Forecasting — EDA  (Notebook 01)")
print("=" * 60)

df = pd.read_csv(DATA, parse_dates=["date"])
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nDate range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Stores: {df['store'].nunique()}, Items: {df['item'].nunique()}")
print(f"\nDescribe:\n{df.describe().to_string()}")
print(f"\nMissing values:\n{df.isnull().sum().to_string()}")

sns.set_theme(style="whitegrid")

# 1. Sales distribution
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df["sales"], bins=60, color="steelblue", edgecolor="white", alpha=0.85)
ax.set_title("Sales Distribution", fontsize=13)
ax.set_xlabel("Daily Sales (units)")
ax.set_ylabel("Frequency")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_sales_distribution.png"), dpi=140)
plt.close(fig)

# 2. Total sales per store
fig, ax = plt.subplots(figsize=(10, 4))
store_sales = df.groupby("store")["sales"].sum().sort_values(ascending=True)
ax.barh(store_sales.index, store_sales.values, color=sns.color_palette("viridis", len(store_sales)))
ax.set_title("Total Sales by Store", fontsize=13)
ax.set_xlabel("Total Units Sold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_store_sales.png"), dpi=140)
plt.close(fig)

# 3. Monthly trend (all stores)
df["month"] = df["date"].dt.to_period("M")
monthly = df.groupby("month")["sales"].sum()
fig, ax = plt.subplots(figsize=(14, 4))
monthly.plot(ax=ax, color="darkorange", lw=2)
ax.fill_between(range(len(monthly)), monthly.values, alpha=0.2, color="darkorange")
ax.set_title("Monthly Total Sales Across All Stores", fontsize=13)
ax.set_ylabel("Total Sales")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_monthly_trend.png"), dpi=140)
plt.close(fig)

# 4. Day-of-week pattern
df["dow"] = df["date"].dt.day_name()
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_avg = df.groupby("dow")["sales"].mean().reindex(dow_order)
fig, ax = plt.subplots(figsize=(9, 4))
dow_avg.plot(kind="bar", ax=ax, color=sns.color_palette("cool", 7), edgecolor="white")
ax.set_title("Average Sales by Day of Week", fontsize=13)
ax.set_ylabel("Avg Units Sold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "04_dow_pattern.png"), dpi=140)
plt.close(fig)

print(f"\n✅ All EDA plots saved to: {OUT_DIR}")
