# Power BI Dashboard Setup Guide
## Demand Forecasting & Inventory Dashboard

> [!IMPORTANT]
> Connect Power BI Desktop to the project's SQLite database or CSV exports. Follow the steps below in order.

---

## 1. Data Sources

| File | Description |
|------|-------------|
| `data/sales_data.csv` | Raw and engineered sales records (913k+ rows) |
| `data/inventory_report.csv` | EOQ, safety stock, reorder points per store-item |
| `data/forecasts.csv` | 30-day forward demand forecasts |

### Connecting CSV Files
1. Open **Power BI Desktop** → **Get Data** → **Text/CSV**
2. Browse to each CSV and import
3. Use **Power Query Editor** to set correct data types:
   - `date` → Date
   - `sales`, `price` → Decimal Number
   - `store`, `item` → Text

---

## 2. Data Model Relationships

```
dim_stores (store_id) ──< fact_sales (store_id)
dim_items  (item_id)  ──< fact_sales (item_id)
dim_stores (store_id) ──< fact_inventory (store_id)
dim_items  (item_id)  ──< fact_inventory (item_id)
fact_forecasts references both dims via store_id, item_id
```

---

## 3. DAX Measures

Paste each measure into Power BI's **Modeling → New Measure**:

```dax
-- Total Sales
Total Sales = SUM(sales_data[sales])

-- Average Daily Sales
Avg Daily Sales = AVERAGE(sales_data[sales])

-- Revenue (if price column exists)
Total Revenue = SUMX(sales_data, sales_data[sales] * sales_data[price])

-- Items Below Reorder Point
Items Below Reorder = 
COUNTROWS(
    FILTER(
        inventory_report,
        inventory_report[avg_daily_demand] * 7 > 
            inventory_report[reorder_point] - inventory_report[safety_stock]
    )
)

-- Forecast Accuracy % (MAPE-based)
Forecast Accuracy % = 
VAR mape = 
    AVERAGEX(
        forecasts,
        ABS(forecasts[predicted_demand] - forecasts[actual]) / 
            MAX(forecasts[actual], 0.001)
    )
RETURN FORMAT(1 - mape, "0.0%")

-- Safety Stock Coverage Days
Safety Stock Days = 
DIVIDE(
    AVERAGE(inventory_report[safety_stock]),
    AVERAGE(inventory_report[avg_daily_demand])
)

-- Top Store by Sales
Top Store = 
TOPN(1, VALUES(sales_data[store]), [Total Sales])

-- Month-over-Month Growth
MoM Growth % = 
VAR current = CALCULATE([Total Sales], DATESMTD(sales_data[date]))
VAR previous = CALCULATE([Total Sales], DATEADD(DATESMTD(sales_data[date]), -1, MONTH))
RETURN DIVIDE(current - previous, previous)
```

---

## 4. Dashboard Pages

### Page 1 — Executive Summary
| Visual | Fields |
|--------|--------|
| KPI Card | Total Sales |
| KPI Card | Total Revenue |
| KPI Card | Avg Daily Sales |
| KPI Card | Items Below Reorder |
| Line Chart | Date → Total Sales (with forecast overlay) |
| Bar Chart | Store → Total Sales |
| Table | Top 10 Items by Revenue |

### Page 2 — Demand Forecasting
| Visual | Fields |
|--------|--------|
| Line Chart | Date → Actual vs Predicted Sales |
| Ribbon Chart | Month → Sales by Store |
| Scatter Plot | Actual vs Predicted (diagonal = perfect fit) |
| Matrix | Store × Item → Avg Daily Sales |
| Slicer | Store, Item, Date Range |

### Page 3 — Inventory Alerts 🚨
| Visual | Fields |
|--------|--------|
| Table (sorted by deficit) | Item, Store, Current Stock, Reorder Point, Deficit |
| Gauge | Safety Stock Coverage Days |
| Bar Chart | Item → EOQ quantity |
| Conditional Format | Red = below reorder, Green = adequate |

### Page 4 — Store Performance
| Visual | Fields |
|--------|--------|
| Map / Bar | Store → Revenue (by region if region column added) |
| Line | Store → MoM Growth % |
| Heatmap Matrix | Store × Month → Sales |
| Bar | Store → Items Below Reorder Count |

---

## 5. Useful Power Query Transformations

```m
// Add Year-Month column
= Table.AddColumn(Source, "YearMonth", 
    each Date.Year([date]) * 100 + Date.Month([date]))

// Flag low stock
= Table.AddColumn(Source, "LowStockFlag",
    each if [avg_daily_demand] * 7 > [current_stock] then "⚠️ LOW" else "OK")
```

---

## 6. Performance Tips

> [!TIP]
> - Import mode is faster than DirectQuery for static CSVs
> - Create a Date dimension table for proper time intelligence
> - Use Incremental Refresh if connecting to a live database
> - Disable Auto Date/Time in Options for better performance with large datasets
