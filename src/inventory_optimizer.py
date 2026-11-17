"""
src/inventory_optimizer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
InventoryOptimizer: compute EOQ, safety stock, reorder points,
and generate full inventory status reports.

Formulae
--------
EOQ         = sqrt(2 × D × S / H)
Safety Stock = z × σ_d × sqrt(L)
ROP         = d̄ × L + Safety Stock

Usage
-----
    optimizer = InventoryOptimizer()
    eoq = optimizer.compute_eoq(5000, 50, 2.0)
    ss  = optimizer.compute_safety_stock(std_demand=10, lead_time=7)
    rop = optimizer.compute_reorder_point(avg_demand=20, lead_time=7, safety_stock=ss)
    report = optimizer.generate_inventory_report(sales_df)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


class InventoryOptimizer:
    """Collection of classical inventory-science formulae and report generators.

    Parameters
    ----------
    lead_time_days : int
        Default supplier lead time used when not explicitly provided.
    service_level : float
        Default service level (probability of not stocking out).
    ordering_cost : float
        Fixed cost per purchase order (\$).
    holding_cost_rate : float
        Annual holding cost as a fraction of unit cost (e.g. 0.25 → 25 %).
    """

    def __init__(
        self,
        lead_time_days: int = 7,
        service_level: float = 0.95,
        ordering_cost: float = 50.0,
        holding_cost_rate: float = 0.25,
    ) -> None:
        self.lead_time_days = lead_time_days
        self.service_level = service_level
        self.ordering_cost = ordering_cost
        self.holding_cost_rate = holding_cost_rate

    # ── Core formulae ─────────────────────────────────────────────────────────

    def compute_eoq(
        self,
        annual_demand: float,
        ordering_cost: Optional[float] = None,
        holding_cost_per_unit: Optional[float] = None,
    ) -> float:
        """Economic Order Quantity (Wilson formula).

        .. math::
            EOQ = \\sqrt{\\frac{2 D S}{H}}

        Parameters
        ----------
        annual_demand : float
            Annual units demanded (D).
        ordering_cost : float, optional
            Cost per order (S). Defaults to ``self.ordering_cost``.
        holding_cost_per_unit : float, optional
            Annual holding cost per unit (H).  If not provided, assumed 1.0.

        Returns
        -------
        float
            Optimal order quantity (units).
        """
        S = ordering_cost if ordering_cost is not None else self.ordering_cost
        H = holding_cost_per_unit if holding_cost_per_unit is not None else 1.0
        if H <= 0:
            raise ValueError("holding_cost_per_unit must be > 0.")
        eoq = math.sqrt(2 * annual_demand * S / H)
        return round(eoq, 2)

    def compute_safety_stock(
        self,
        std_demand: float,
        lead_time: Optional[int] = None,
        z_score: Optional[float] = None,
    ) -> float:
        """Safety stock to cover demand variability during lead time.

        .. math::
            SS = z \\cdot \\sigma_d \\cdot \\sqrt{L}

        Parameters
        ----------
        std_demand : float
            Standard deviation of daily demand (σ_d).
        lead_time : int, optional
            Lead time in days (L). Defaults to ``self.lead_time_days``.
        z_score : float, optional
            Service-level z-score. If not provided, computed from
            ``self.service_level``.

        Returns
        -------
        float
            Safety stock (units).
        """
        L = lead_time if lead_time is not None else self.lead_time_days
        z = z_score if z_score is not None else float(norm.ppf(self.service_level))
        ss = z * std_demand * math.sqrt(L)
        return round(max(0.0, ss), 2)

    def compute_reorder_point(
        self,
        avg_demand: float,
        lead_time: Optional[int] = None,
        safety_stock: Optional[float] = None,
    ) -> float:
        """Reorder Point: average demand during lead time + safety stock.

        .. math::
            ROP = \\bar{d} \\cdot L + SS

        Parameters
        ----------
        avg_demand : float
            Average daily demand (d̄).
        lead_time : int, optional
            Lead time in days. Defaults to ``self.lead_time_days``.
        safety_stock : float, optional
            Safety stock. If None, calculated with ``std_demand = 0.2 × avg_demand``.

        Returns
        -------
        float
            Reorder point (units).
        """
        L = lead_time if lead_time is not None else self.lead_time_days
        if safety_stock is None:
            std_demand = avg_demand * 0.20
            safety_stock = self.compute_safety_stock(std_demand, L)
        rop = avg_demand * L + safety_stock
        return round(rop, 2)

    # ── Report generation ─────────────────────────────────────────────────────

    def generate_inventory_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate a full inventory status report per (store, item).

        For each (store, item) pair the method computes:
        - ``avg_daily_sales``    — mean daily quantity
        - ``std_daily_sales``    — std of daily quantity
        - ``annual_demand``      — avg × 365
        - ``safety_stock``       — z × σ × √L
        - ``reorder_point``      — avg × L + SS
        - ``eoq``                — Economic Order Quantity
        - ``simulated_stock``    — random current stock (for demo purposes)
        - ``below_reorder``      — boolean flag

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: ``store``, ``item``, ``sales`` (or ``quantity``).

        Returns
        -------
        pd.DataFrame
            One row per (store, item) with computed inventory metrics.
        """
        sales_col = "sales" if "sales" in df.columns else "quantity"
        logger.info(
            "Generating inventory report for %d (store, item) pairs …",
            df.groupby(["store", "item"], observed=True).ngroups,
        )

        agg = (
            df.groupby(["store", "item"], observed=True)[sales_col]
            .agg(avg_daily_sales="mean", std_daily_sales="std")
            .reset_index()
        )
        agg["std_daily_sales"] = agg["std_daily_sales"].fillna(0.0)
        agg["annual_demand"] = agg["avg_daily_sales"] * 365

        # Inventory science metrics
        agg["safety_stock"] = agg.apply(
            lambda r: self.compute_safety_stock(r["std_daily_sales"]), axis=1
        )
        agg["reorder_point"] = agg.apply(
            lambda r: self.compute_reorder_point(
                r["avg_daily_sales"], safety_stock=r["safety_stock"]
            ),
            axis=1,
        )
        agg["eoq"] = agg.apply(
            lambda r: self.compute_eoq(
                r["annual_demand"],
                holding_cost_per_unit=max(0.1, r["avg_daily_sales"] * self.holding_cost_rate),
            ),
            axis=1,
        )

        # Simulated current stock level (normally distributed around 2× ROP)
        rng = np.random.default_rng(seed=0)
        agg["simulated_stock"] = np.maximum(
            0,
            rng.normal(loc=agg["reorder_point"] * 2, scale=agg["reorder_point"] * 0.4),
        ).round(1)

        agg["below_reorder"] = agg["simulated_stock"] < agg["reorder_point"]

        logger.info(
            "Report complete. %d / %d items below reorder point.",
            agg["below_reorder"].sum(),
            len(agg),
        )
        return agg

    def get_low_stock_alerts(self, inventory_df: pd.DataFrame) -> pd.DataFrame:
        """Filter inventory report to items below their reorder point.

        Parameters
        ----------
        inventory_df : pd.DataFrame
            Output of :meth:`generate_inventory_report`.

        Returns
        -------
        pd.DataFrame
            Subset where ``simulated_stock < reorder_point``, sorted by
            urgency (largest shortfall first).
        """
        alerts = inventory_df[inventory_df["below_reorder"]].copy()
        alerts["shortfall"] = (
            alerts["reorder_point"] - alerts["simulated_stock"]
        ).round(2)
        alerts.sort_values("shortfall", ascending=False, inplace=True)
        alerts.reset_index(drop=True, inplace=True)
        logger.info("Low-stock alerts: %d items.", len(alerts))
        return alerts
