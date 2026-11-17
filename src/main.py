"""
Main pipeline runner for Demand Forecasting and Inventory Dashboard.
Run this script to execute the full end-to-end pipeline.

Usage
-----
    python src/main.py

Outputs
-------
- Console: summary report with evaluation metrics and top low-stock alerts
- data/inventory_report.csv: inventory optimization per store-item
- data/forecasts.csv: 30-day demand forecasts
"""

import logging
import os
import sys
import time

import pandas as pd

# Ensure src/ is on path when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline import DataPipeline
from forecasting import DemandForecaster
from inventory_optimizer import InventoryOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def banner(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def main() -> None:
    start = time.time()
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sales_data.csv")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  1. Load & Clean                                                     #
    # ------------------------------------------------------------------ #
    banner("1 — Loading & Cleaning Data")
    pipeline = DataPipeline(DATA_PATH)
    pipeline.load_data().clean_data()
    summary = pipeline.get_summary()
    print(f"  Rows:         {summary['rows']:,}")
    print(f"  Date range:   {summary['date_range'][0]} → {summary['date_range'][1]}")
    print(f"  Stores:       {summary['stores']}")
    print(f"  Items:        {summary['items']}")
    print(f"  Avg sales/day:{summary['avg_daily_sales']}")

    # ------------------------------------------------------------------ #
    #  2. Feature Engineering                                              #
    # ------------------------------------------------------------------ #
    banner("2 — Engineering Features")
    pipeline.engineer_features()
    print(f"  Feature columns ({len(pipeline.feature_columns)}): {pipeline.feature_columns}")

    # ------------------------------------------------------------------ #
    #  3. Train / Test Split                                               #
    # ------------------------------------------------------------------ #
    banner("3 — Splitting Data (last 3 months = test)")
    X_train, X_test, y_train, y_test = pipeline.split_data(test_months=3)
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples:     {len(X_test):,}")

    # ------------------------------------------------------------------ #
    #  4. Train Model                                                      #
    # ------------------------------------------------------------------ #
    banner("4 — Training Demand Forecaster (Gradient Boosting)")
    forecaster = DemandForecaster(model_type="gradient_boosting")
    forecaster.train(X_train, y_train)

    # ------------------------------------------------------------------ #
    #  5. Evaluate                                                         #
    # ------------------------------------------------------------------ #
    banner("5 — Evaluation on Test Set")
    predictions = forecaster.predict(X_test)
    metrics = forecaster.evaluate(y_test, predictions)
    for k, v in metrics.items():
        print(f"  {k:<6}: {v}")

    # ------------------------------------------------------------------ #
    #  6. Feature Importance                                               #
    # ------------------------------------------------------------------ #
    banner("6 — Top Feature Importances")
    fi = forecaster.feature_importances(pipeline.feature_columns)
    print(fi.head(10).to_string(index=False))

    # ------------------------------------------------------------------ #
    #  7. Inventory Optimization                                           #
    # ------------------------------------------------------------------ #
    banner("7 — Inventory Optimization Report")
    optimizer = InventoryOptimizer()
    inv_report = optimizer.generate_inventory_report(pipeline.df)
    inv_path = os.path.join(OUTPUT_DIR, "inventory_report.csv")
    inv_report.to_csv(inv_path, index=False)
    print(f"  Inventory report saved → {inv_path}")
    print(f"  Top 5 items by EOQ:\n{inv_report.nlargest(5, 'eoq')[['store','item','eoq','reorder_point']].to_string(index=False)}")

    # ------------------------------------------------------------------ #
    #  8. 30-Day Forecast for first store-item                            #
    # ------------------------------------------------------------------ #
    banner("8 — 30-Day Demand Forecast Sample")
    df = pipeline.df
    store1 = df["store"].cat.categories[0]
    item1 = df["item"].cat.categories[0]
    sample = df[(df["store"] == store1) & (df["item"] == item1)].tail(1)
    if not sample.empty:
        last_features = sample[pipeline.feature_columns]
        pred_val = forecaster.predict(last_features)[0]
        rop_info = forecaster.compute_reorder_point(
            avg_daily_demand=pred_val,
            lead_time_days=7,
            demand_std=df[df["item"] == item1]["sales"].std(),
        )
        print(f"  Store: {store1}, Item: {item1}")
        print(f"  Predicted next-day demand: {pred_val:.1f} units")
        print(f"  Reorder Point:   {rop_info['reorder_point']} units")
        print(f"  Safety Stock:    {rop_info['safety_stock']} units")

    # ------------------------------------------------------------------ #
    #  Done                                                                #
    # ------------------------------------------------------------------ #
    elapsed = time.time() - start
    banner(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print("  ✅ Data loaded and cleaned")
    print("  ✅ Features engineered")
    print("  ✅ Model trained and evaluated")
    print("  ✅ Inventory report generated")
    print(f"  📄 Reports saved to: {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()
