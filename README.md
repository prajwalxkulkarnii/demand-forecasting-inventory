# 📦 Demand Forecasting & Inventory Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.15%2B-blueviolet?logo=plotly)

> **A production-grade end-to-end pipeline** that ingests 913 k+ store-item sales records, engineers lag / rolling-average features, trains a gradient-boosting forecaster, computes inventory reorder points, and surfaces results in an interactive dashboard.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
│  data/generate_data.py  →  data/sales_data.csv  (913 k rows)       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PIPELINE LAYER  (src/)                         │
│                                                                     │
│  data_pipeline.py          │  Clean, type-cast, engineer features   │
│  ├─ lag_1 … lag_28         │  Lag features (1, 7, 14, 28 days)      │
│  ├─ rolling_7 / rolling_30 │  Rolling mean / std windows             │
│  └─ date parts + encoding  │  dow, month, quarter, is_weekend …     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MODELLING LAYER  (src/)                         │
│                                                                     │
│  forecasting.py            │  XGBoost / GBM / Linear Regression     │
│  ├─ train()                │  Fit model, log CV metrics              │
│  ├─ predict()              │  Point forecasts                        │
│  ├─ evaluate()             │  MAE, RMSE, MAPE, R²                    │
│  └─ forecast_future()      │  Next-N-day rolling forecast            │
│                            │                                        │
│  inventory_optimizer.py    │  EOQ, Safety Stock, Reorder Point      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STORAGE LAYER  (src/)                           │
│                                                                     │
│  database.py               │  SQLAlchemy → SQLite                   │
│  ├─ dim_stores             │  Store dimension                        │
│  ├─ dim_items              │  Item dimension                         │
│  ├─ fact_sales             │  Transactional sales                    │
│  ├─ fact_inventory         │  Stock levels & reorder signals         │
│  └─ fact_forecasts         │  Model predictions + confidence bands   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 VISUALISATION & DASHBOARD LAYER                     │
│                                                                     │
│  src/visualizer.py         │  Matplotlib / Seaborn / Plotly PNGs   │
│  dashboard/README.md       │  Power BI connection + DAX measures    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| 🔢 **Large-scale synthetic data** | 10 stores × 50 items × 5 years = 913 k daily rows |
| 📐 **Rich feature engineering** | Lag-1/7/14/28, rolling mean/std 7 & 30, date cyclicals |
| 🤖 **Multiple ML backends** | XGBoost (default), Gradient Boosting, Linear Regression |
| 📉 **Comprehensive metrics** | MAE, RMSE, MAPE, R² reported per model |
| 🏭 **Inventory science** | EOQ, Safety Stock (z-score configurable), Reorder Point |
| 🗄️ **SQLite persistence** | Full star-schema via SQLAlchemy; plug-in Power BI |
| 📊 **Rich visualisations** | 5 chart types (trend, forecast, heatmap, feature imp., levels) |
| 🧪 **Tested** | pytest suite — no real data file required |

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Data wrangling:** Pandas 2.x, NumPy
- **Machine learning:** scikit-learn, XGBoost (optional), statsmodels
- **Visualisation:** Matplotlib, Seaborn, Plotly
- **Storage:** SQLite via SQLAlchemy 2.x
- **Notebooks:** Jupyter / plain Python scripts
- **Testing:** pytest

---

## 🚀 Installation

```bash
# 1. Clone the repo
git clone https://github.com/prajwalxkulakrni/demand-forecasting-inventory.git
cd demand-forecasting-inventory

# 2. Create & activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic data  (~30 s)
python data/generate_data.py

# 5. Run end-to-end pipeline
python src/main.py
```

---

## 📖 Usage Guide

### Generate data
```bash
python data/generate_data.py
# → data/sales_data.csv  (913 k rows)
```

### Run the full pipeline
```bash
python src/main.py
# Loads data → engineers features → trains model → evaluates
# → saves to SQLite → generates PNG charts in outputs/
```

### Individual modules
```python
from src.data_pipeline import DataPipeline
from src.forecasting import DemandForecaster
from src.inventory_optimizer import InventoryOptimizer

pipeline = DataPipeline("data/sales_data.csv")
pipeline.clean_data()
pipeline.engineer_features()
X_train, X_test, y_train, y_test = pipeline.split_data()

forecaster = DemandForecaster(model_type="gradient_boosting")
forecaster.train(X_train, y_train)
preds = forecaster.predict(X_test)
metrics = forecaster.evaluate(y_test, preds)
print(metrics)

optimizer = InventoryOptimizer()
eoq = optimizer.compute_eoq(annual_demand=5000, ordering_cost=50, holding_cost_per_unit=2)
print(f"EOQ: {eoq:.1f} units")
```

### Exploratory notebooks
```bash
python notebooks/01_eda.py              # EDA + plots → notebooks/plots/
python notebooks/02_feature_engineering.py
python notebooks/03_forecasting.py
```

### Run tests
```bash
pytest tests/ -v
```

---

## 📸 Dashboard Screenshots

The **Power BI** dashboard (see `dashboard/README.md`) contains four pages:

1. **Executive Summary** — Total Sales KPI, YoY growth, top-performing stores card, monthly trend line.
2. **Demand Forecasting** — Actual vs. predicted time series, 95 % confidence band, MAPE gauge.
3. **Inventory Alerts** — Traffic-light table of items below reorder point, safety-stock days coverage.
4. **Store Performance** — Clustered bar by region, decomposed seasonality, heatmap of store × item.

Static PNG equivalents are written to `outputs/` by `src/visualizer.py`.

---

## 📁 Directory Structure

```
demand-forecasting-inventory/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   ├── generate_data.py        # Synthetic data generator
│   └── sales_data.csv          # Generated (gitignored if large)
│
├── sql/
│   ├── schema.sql              # Star-schema DDL
│   └── queries.sql             # Analytics queries
│
├── src/
│   ├── data_pipeline.py        # Cleaning + feature engineering
│   ├── forecasting.py          # ML models + reorder point
│   ├── inventory_optimizer.py  # EOQ / safety stock
│   ├── database.py             # SQLAlchemy persistence
│   ├── visualizer.py           # Charts (Matplotlib / Plotly)
│   └── main.py                 # End-to-end runner
│
├── notebooks/
│   ├── 01_eda.py
│   ├── 02_feature_engineering.py
│   └── 03_forecasting.py
│
├── dashboard/
│   └── README.md               # Power BI integration guide
│
├── tests/
│   └── test_forecasting.py
│
└── outputs/                    # Auto-created; PNG charts saved here
```

---

## 📄 License

[MIT](LICENSE) © 2024 prajwalxkulakrni
