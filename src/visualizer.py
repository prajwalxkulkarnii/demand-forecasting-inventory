"""
Demand Forecasting Visualizer
Generates plots: sales trends, forecast vs actual, feature importance,
inventory levels, demand heatmaps. Saves to outputs/ directory.
"""

import logging
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = "viridis"
FIG_SIZE = (12, 6)


def _save(fig: plt.Figure, name: str) -> str:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", path)
    return path


class Visualizer:
    """Collection of visualisation helpers for demand forecasting."""

    # ------------------------------------------------------------------
    def plot_sales_trend(
        self,
        df: pd.DataFrame,
        store_id: Optional[str] = None,
        item_id: Optional[str] = None,
        rolling_window: int = 30,
    ) -> str:
        """
        Time-series plot of daily sales with rolling-average overlay.

        Parameters
        ----------
        df           : Full sales DataFrame (needs 'date' and 'sales').
        store_id     : Filter to this store (None = all stores).
        item_id      : Filter to this item  (None = all items).
        rolling_window: Days for rolling mean overlay.

        Returns
        -------
        str : file path of saved PNG.
        """
        sub = df.copy()
        if store_id:
            sub = sub[sub["store"] == store_id]
        if item_id:
            sub = sub[sub["item"] == item_id]

        daily = sub.groupby("date")["sales"].sum().reset_index()
        daily["rolling"] = daily["sales"].rolling(rolling_window, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.fill_between(daily["date"], daily["sales"], alpha=0.3, color="steelblue", label="Daily sales")
        ax.plot(daily["date"], daily["rolling"], color="darkorange", lw=2, label=f"{rolling_window}-day rolling avg")
        ax.set_title(f"Sales Trend — store={store_id or 'ALL'}, item={item_id or 'ALL'}", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Units Sold")
        ax.legend()
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        return _save(fig, "sales_trend.png")

    # ------------------------------------------------------------------
    def plot_forecast_vs_actual(
        self,
        actual: pd.Series,
        predicted: np.ndarray,
        dates: pd.Series,
        title: str = "Forecast vs Actual",
    ) -> str:
        """Line chart comparing model predictions to true sales."""
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.plot(dates.values, actual.values, label="Actual", color="steelblue", lw=1.5)
        ax.plot(dates.values, predicted, label="Predicted", color="tomato", lw=1.5, linestyle="--")
        ax.fill_between(
            dates.values,
            actual.values,
            predicted,
            alpha=0.15,
            color="gray",
            label="Error band",
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales (units)")
        ax.legend()
        fig.tight_layout()
        return _save(fig, "forecast_vs_actual.png")

    # ------------------------------------------------------------------
    def plot_feature_importance(
        self, model, feature_names: List[str], top_n: int = 15
    ) -> str:
        """Horizontal bar chart of feature importances (tree models)."""
        if not hasattr(model, "feature_importances_"):
            logger.warning("Model has no feature_importances_; skipping.")
            return ""
        fi = (
            pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
            .sort_values("importance", ascending=True)
            .tail(top_n)
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(fi["feature"], fi["importance"], color=sns.color_palette(PALETTE, len(fi)))
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_title(f"Top {top_n} Feature Importances", fontsize=14)
        ax.set_xlabel("Importance")
        fig.tight_layout()
        return _save(fig, "feature_importance.png")

    # ------------------------------------------------------------------
    def plot_inventory_levels(self, inventory_df: pd.DataFrame) -> str:
        """
        Grouped bar chart: average daily demand vs reorder point per item
        (aggregated across stores).
        """
        agg = (
            inventory_df.groupby("item")[["avg_daily_demand", "reorder_point", "safety_stock"]]
            .mean()
            .reset_index()
            .sort_values("reorder_point", ascending=False)
            .head(20)
        )
        x = np.arange(len(agg))
        width = 0.3

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, agg["avg_daily_demand"], width, label="Avg Daily Demand", color="steelblue")
        ax.bar(x, agg["reorder_point"], width, label="Reorder Point", color="tomato")
        ax.bar(x + width, agg["safety_stock"], width, label="Safety Stock", color="mediumseagreen")
        ax.set_xticks(x)
        ax.set_xticklabels(agg["item"], rotation=45, ha="right", fontsize=9)
        ax.set_title("Inventory Levels: Demand vs Reorder Points (Top 20 Items)", fontsize=14)
        ax.set_ylabel("Units")
        ax.legend()
        fig.tight_layout()
        return _save(fig, "inventory_levels.png")

    # ------------------------------------------------------------------
    def plot_demand_heatmap(self, df: pd.DataFrame) -> str:
        """Heatmap of average daily sales: stores (rows) × items (cols)."""
        pivot = df.pivot_table(
            index="store", columns="item", values="sales", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(18, 6))
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            linewidths=0.3,
            fmt=".1f",
            annot=pivot.shape[0] <= 10 and pivot.shape[1] <= 20,
            cbar_kws={"label": "Avg Daily Sales"},
        )
        ax.set_title("Average Daily Demand: Store × Item", fontsize=14)
        ax.set_xlabel("Item")
        ax.set_ylabel("Store")
        fig.tight_layout()
        return _save(fig, "demand_heatmap.png")

    # ------------------------------------------------------------------
    def plot_monthly_sales(self, df: pd.DataFrame) -> str:
        """Multi-year monthly sales trend lines."""
        df = df.copy()
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        monthly = df.groupby(["year", "month"])["sales"].sum().reset_index()

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        for yr, grp in monthly.groupby("year"):
            ax.plot(grp["month"], grp["sales"], marker="o", label=str(yr), lw=2)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
        ax.set_title("Monthly Sales Trend by Year", fontsize=14)
        ax.set_ylabel("Total Sales")
        ax.legend(title="Year")
        fig.tight_layout()
        return _save(fig, "monthly_sales_trend.png")
