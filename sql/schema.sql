-- =============================================================================
-- sql/schema.sql
-- Star-schema DDL for the Demand Forecasting & Inventory Dashboard
-- Compatible with SQLite 3.x
-- =============================================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── Dimension: Stores ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_stores (
    store_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    store_name  TEXT    NOT NULL UNIQUE,
    region      TEXT    NOT NULL,          -- e.g. "North", "South", "East", "West"
    city        TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── Dimension: Items ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_items (
    item_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    item_name       TEXT    NOT NULL UNIQUE,
    category        TEXT    NOT NULL,      -- e.g. "Beverages", "Apparel"
    unit_cost       REAL    NOT NULL,      -- cost to procure one unit
    unit_price      REAL    NOT NULL,      -- retail selling price
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── Fact: Sales ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fact_sales (
    sale_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,          -- ISO-8601 'YYYY-MM-DD'
    store_id    INTEGER NOT NULL REFERENCES dim_stores(store_id),
    item_id     INTEGER NOT NULL REFERENCES dim_items(item_id),
    quantity    INTEGER NOT NULL CHECK (quantity >= 0),
    revenue     REAL    NOT NULL CHECK (revenue >= 0),
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fact_sales_date     ON fact_sales (date);
CREATE INDEX IF NOT EXISTS idx_fact_sales_store    ON fact_sales (store_id);
CREATE INDEX IF NOT EXISTS idx_fact_sales_item     ON fact_sales (item_id);
CREATE INDEX IF NOT EXISTS idx_fact_sales_store_item ON fact_sales (store_id, item_id, date);

-- ── Fact: Inventory ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fact_inventory (
    inventory_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    date                TEXT    NOT NULL,
    store_id            INTEGER NOT NULL REFERENCES dim_stores(store_id),
    item_id             INTEGER NOT NULL REFERENCES dim_items(item_id),
    stock_level         REAL    NOT NULL CHECK (stock_level >= 0),
    reorder_point       REAL    NOT NULL,
    safety_stock        REAL    NOT NULL,
    economic_order_qty  REAL    NOT NULL,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fact_inv_date       ON fact_inventory (date);
CREATE INDEX IF NOT EXISTS idx_fact_inv_store_item ON fact_inventory (store_id, item_id, date);

-- ── Fact: Forecasts ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fact_forecasts (
    forecast_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    store_id         INTEGER NOT NULL REFERENCES dim_stores(store_id),
    item_id          INTEGER NOT NULL REFERENCES dim_items(item_id),
    forecast_date    TEXT    NOT NULL,
    predicted_demand REAL    NOT NULL,
    lower_bound      REAL,
    upper_bound      REAL,
    model_name       TEXT    NOT NULL DEFAULT 'gradient_boosting',
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fact_fcst_date      ON fact_forecasts (forecast_date);
CREATE INDEX IF NOT EXISTS idx_fact_fcst_store_item ON fact_forecasts (store_id, item_id, forecast_date);
