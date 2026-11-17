-- =============================================================================
-- sql/queries.sql
-- Useful analytics queries for the Demand Forecasting & Inventory Dashboard
-- =============================================================================

-- ── 1. Top 10 items by total sales quantity per store ─────────────────────────
-- Returns the best-selling items for each store, ranked by total units sold.
SELECT
    s.store_name,
    i.item_name,
    i.category,
    SUM(fs.quantity)                        AS total_units_sold,
    ROUND(SUM(fs.revenue), 2)               AS total_revenue,
    RANK() OVER (
        PARTITION BY fs.store_id
        ORDER BY SUM(fs.quantity) DESC
    )                                       AS rank_in_store
FROM fact_sales      fs
JOIN dim_stores      s  ON s.store_id = fs.store_id
JOIN dim_items       i  ON i.item_id  = fs.item_id
GROUP BY fs.store_id, fs.item_id
QUALIFY rank_in_store <= 10
ORDER BY s.store_name, rank_in_store;


-- ── 2. Items currently below reorder point (inventory alert) ─────────────────
-- Scan the LATEST inventory snapshot for each store/item and flag low stock.
WITH latest_inventory AS (
    SELECT
        store_id,
        item_id,
        date,
        stock_level,
        reorder_point,
        safety_stock,
        ROW_NUMBER() OVER (
            PARTITION BY store_id, item_id
            ORDER BY date DESC
        ) AS rn
    FROM fact_inventory
)
SELECT
    s.store_name,
    i.item_name,
    i.category,
    li.date               AS snapshot_date,
    li.stock_level,
    li.reorder_point,
    li.safety_stock,
    ROUND(li.reorder_point - li.stock_level, 1) AS units_short
FROM latest_inventory li
JOIN dim_stores s ON s.store_id = li.store_id
JOIN dim_items  i ON i.item_id  = li.item_id
WHERE li.rn = 1
  AND li.stock_level < li.reorder_point
ORDER BY units_short DESC;


-- ── 3. Monthly sales trend per store ─────────────────────────────────────────
-- Aggregates quantity and revenue by year-month for each store.
SELECT
    s.store_name,
    strftime('%Y-%m', fs.date)              AS year_month,
    SUM(fs.quantity)                        AS total_units,
    ROUND(SUM(fs.revenue), 2)               AS total_revenue,
    ROUND(AVG(fs.quantity), 2)              AS avg_daily_units
FROM fact_sales  fs
JOIN dim_stores  s ON s.store_id = fs.store_id
GROUP BY fs.store_id, year_month
ORDER BY s.store_name, year_month;


-- ── 4. Forecast accuracy: predicted vs actual (last 90 days) ─────────────────
-- Joins forecasts to actuals on the same date/store/item, computes APE.
SELECT
    s.store_name,
    i.item_name,
    ff.forecast_date,
    ff.model_name,
    ROUND(fs.quantity,        2)                     AS actual_demand,
    ROUND(ff.predicted_demand, 2)                    AS predicted_demand,
    ROUND(
        ABS(fs.quantity - ff.predicted_demand)
        / NULLIF(fs.quantity, 0) * 100, 2
    )                                                AS ape_pct,
    ff.lower_bound,
    ff.upper_bound
FROM fact_forecasts ff
JOIN fact_sales  fs ON  fs.store_id = ff.store_id
                    AND fs.item_id  = ff.item_id
                    AND fs.date     = ff.forecast_date
JOIN dim_stores  s  ON s.store_id  = ff.store_id
JOIN dim_items   i  ON i.item_id   = ff.item_id
WHERE ff.forecast_date >= date('now', '-90 days')
ORDER BY ff.forecast_date, s.store_name, i.item_name;


-- ── 5. Demand volatility by item (std-dev of daily sales) ────────────────────
-- High-volatility items need larger safety stocks.
SELECT
    i.item_name,
    i.category,
    ROUND(AVG(fs.quantity),  2)   AS avg_daily_sales,
    ROUND(
        SQRT(
            AVG(fs.quantity * fs.quantity) - AVG(fs.quantity) * AVG(fs.quantity)
        ), 2
    )                             AS std_daily_sales,
    COUNT(*)                      AS observation_count
FROM fact_sales  fs
JOIN dim_items   i ON i.item_id = fs.item_id
GROUP BY fs.item_id
ORDER BY std_daily_sales DESC;


-- ── 6. Rolling 30-day average sales per store/item ───────────────────────────
-- Useful for smoothed trend lines and reorder-point calculations.
SELECT
    s.store_name,
    i.item_name,
    fs.date,
    fs.quantity,
    ROUND(
        AVG(fs.quantity) OVER (
            PARTITION BY fs.store_id, fs.item_id
            ORDER BY fs.date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ), 2
    ) AS rolling_30d_avg
FROM fact_sales  fs
JOIN dim_stores  s ON s.store_id = fs.store_id
JOIN dim_items   i ON i.item_id  = fs.item_id
ORDER BY s.store_name, i.item_name, fs.date;
