"""
src/database.py
~~~~~~~~~~~~~~~
DatabaseManager: SQLAlchemy-based persistence layer for the inventory DB.

Usage
-----
    with DatabaseManager("data/inventory.db") as db:
        db.create_tables()
        db.insert_sales_data(df)
        result = db.query("SELECT COUNT(*) FROM fact_sales")
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

# ── DDL (mirrors sql/schema.sql) ─────────────────────────────────────────────
_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS dim_stores (
    store_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    store_name TEXT    NOT NULL UNIQUE,
    region     TEXT    NOT NULL DEFAULT 'Unknown',
    city       TEXT    NOT NULL DEFAULT 'Unknown',
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dim_items (
    item_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    item_name  TEXT    NOT NULL UNIQUE,
    category   TEXT    NOT NULL DEFAULT 'General',
    unit_cost  REAL    NOT NULL DEFAULT 1.0,
    unit_price REAL    NOT NULL DEFAULT 1.0,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS fact_sales (
    sale_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    date       TEXT    NOT NULL,
    store_id   INTEGER NOT NULL REFERENCES dim_stores(store_id),
    item_id    INTEGER NOT NULL REFERENCES dim_items(item_id),
    quantity   INTEGER NOT NULL DEFAULT 0,
    revenue    REAL    NOT NULL DEFAULT 0.0,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fs_date       ON fact_sales (date);
CREATE INDEX IF NOT EXISTS idx_fs_store_item ON fact_sales (store_id, item_id, date);

CREATE TABLE IF NOT EXISTS fact_inventory (
    inventory_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    date               TEXT NOT NULL,
    store_id           INTEGER NOT NULL REFERENCES dim_stores(store_id),
    item_id            INTEGER NOT NULL REFERENCES dim_items(item_id),
    stock_level        REAL NOT NULL DEFAULT 0.0,
    reorder_point      REAL NOT NULL DEFAULT 0.0,
    safety_stock       REAL NOT NULL DEFAULT 0.0,
    economic_order_qty REAL NOT NULL DEFAULT 0.0,
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fi_date       ON fact_inventory (date);
CREATE INDEX IF NOT EXISTS idx_fi_store_item ON fact_inventory (store_id, item_id, date);

CREATE TABLE IF NOT EXISTS fact_forecasts (
    forecast_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    store_id         INTEGER NOT NULL REFERENCES dim_stores(store_id),
    item_id          INTEGER NOT NULL REFERENCES dim_items(item_id),
    forecast_date    TEXT NOT NULL,
    predicted_demand REAL NOT NULL,
    lower_bound      REAL,
    upper_bound      REAL,
    model_name       TEXT NOT NULL DEFAULT 'gradient_boosting',
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ff_date       ON fact_forecasts (forecast_date);
CREATE INDEX IF NOT EXISTS idx_ff_store_item ON fact_forecasts (store_id, item_id, forecast_date);
"""


class DatabaseManager:
    """SQLAlchemy-backed manager for the inventory star-schema SQLite database.

    Parameters
    ----------
    db_path : str
        File path for the SQLite database (created if it does not exist).
    """

    def __init__(self, db_path: str = "data/inventory.db") -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.engine: Engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        logger.info("DatabaseManager connected to %s", db_path)

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Dispose the connection pool."""
        self.engine.dispose()
        logger.info("Database connection closed.")

    # ── Schema ────────────────────────────────────────────────────────────────

    def create_tables(self) -> None:
        """Execute DDL to create all tables and indexes."""
        with self.engine.begin() as conn:
            for stmt in _DDL.split(";"):
                stmt = stmt.strip()
                if stmt:
                    try:
                        conn.execute(text(stmt))
                    except SQLAlchemyError as exc:
                        logger.warning("DDL warning (non-fatal): %s", exc)
        logger.info("All tables created / verified.")

    # ── Dimension helpers ─────────────────────────────────────────────────────

    def _upsert_dim_stores(self, stores: pd.Series) -> dict[str, int]:
        """Ensure all stores exist in dim_stores; return name → id map."""
        with self.engine.begin() as conn:
            for name in stores.unique():
                conn.execute(
                    text(
                        "INSERT OR IGNORE INTO dim_stores (store_name, region, city) "
                        "VALUES (:name, 'Region-A', 'City-A')"
                    ),
                    {"name": name},
                )
            rows = conn.execute(
                text("SELECT store_name, store_id FROM dim_stores")
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def _upsert_dim_items(self, items: pd.Series) -> dict[str, int]:
        """Ensure all items exist in dim_items; return name → id map."""
        with self.engine.begin() as conn:
            for name in items.unique():
                conn.execute(
                    text(
                        "INSERT OR IGNORE INTO dim_items (item_name, category, unit_cost, unit_price) "
                        "VALUES (:name, 'General', 1.0, 1.5)"
                    ),
                    {"name": name},
                )
            rows = conn.execute(
                text("SELECT item_name, item_id FROM dim_items")
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ── Insert methods ────────────────────────────────────────────────────────

    def insert_sales_data(self, df: pd.DataFrame) -> int:
        """Bulk-insert sales records into ``fact_sales``.

        The DataFrame must contain columns:
        ``date``, ``store``, ``item``, ``sales`` (or ``quantity``), ``price``.

        Parameters
        ----------
        df : pd.DataFrame
            Raw or processed sales data.

        Returns
        -------
        int
            Number of rows inserted.
        """
        store_map = self._upsert_dim_stores(df["store"].astype(str))
        item_map = self._upsert_dim_items(df["item"].astype(str))

        sales_col = "sales" if "sales" in df.columns else "quantity"
        price_col = "price" if "price" in df.columns else None

        rows = []
        for _, row in df.iterrows():
            store_name = str(row["store"])
            item_name = str(row["item"])
            qty = int(row[sales_col])
            price = float(row[price_col]) if price_col else 1.0
            rows.append(
                {
                    "date": str(row["date"])[:10],
                    "store_id": store_map[store_name],
                    "item_id": item_map[item_name],
                    "quantity": qty,
                    "revenue": round(qty * price, 4),
                }
            )

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO fact_sales (date, store_id, item_id, quantity, revenue) "
                    "VALUES (:date, :store_id, :item_id, :quantity, :revenue)"
                ),
                rows,
            )
        logger.info("Inserted %d rows into fact_sales.", len(rows))
        return len(rows)

    def insert_forecast(self, forecast_df: pd.DataFrame, model_name: str = "gradient_boosting") -> int:
        """Insert forecast results into ``fact_forecasts``.

        Parameters
        ----------
        forecast_df : pd.DataFrame
            Must contain: ``date`` (or ``forecast_date``), ``store``, ``item``,
            ``predicted_demand``, optionally ``lower_bound``, ``upper_bound``.
        model_name : str
            Name of the model that produced the forecast.

        Returns
        -------
        int
            Number of rows inserted.
        """
        store_map = self._upsert_dim_stores(forecast_df["store"].astype(str))
        item_map = self._upsert_dim_items(forecast_df["item"].astype(str))

        date_col = "date" if "date" in forecast_df.columns else "forecast_date"
        rows = []
        for _, row in forecast_df.iterrows():
            rows.append(
                {
                    "store_id": store_map[str(row["store"])],
                    "item_id": item_map[str(row["item"])],
                    "forecast_date": str(row[date_col])[:10],
                    "predicted_demand": float(row["predicted_demand"]),
                    "lower_bound": float(row.get("lower_bound", 0)),
                    "upper_bound": float(row.get("upper_bound", 0)),
                    "model_name": model_name,
                }
            )

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO fact_forecasts "
                    "(store_id, item_id, forecast_date, predicted_demand, "
                    " lower_bound, upper_bound, model_name) "
                    "VALUES (:store_id, :item_id, :forecast_date, :predicted_demand, "
                    "        :lower_bound, :upper_bound, :model_name)"
                ),
                rows,
            )
        logger.info("Inserted %d rows into fact_forecasts.", len(rows))
        return len(rows)

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a raw SQL query and return results as a DataFrame.

        Parameters
        ----------
        sql : str
            SQL SELECT statement.

        Returns
        -------
        pd.DataFrame
        """
        with self.engine.connect() as conn:
            result = pd.read_sql(text(sql), conn)
        logger.debug("Query returned %d rows.", len(result))
        return result
