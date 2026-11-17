"""
data/generate_data.py
~~~~~~~~~~~~~~~~~~~~~
Generate a synthetic sales dataset with ~913 k rows:
  10 stores × 50 items × 1826 days (2019-01-01 → 2023-12-31)

Each row represents the daily sales of one item in one store.
Realistic patterns include:
  - Long-term upward trend
  - Annual seasonality (summer peaks, winter peaks depending on item category)
  - Day-of-week effect (weekends higher for most items)
  - Random Gaussian noise

Output: data/sales_data.csv
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ── Parameters ─────────────────────────────────────────────────────────────────
N_STORES = 10
N_ITEMS = 50
START_DATE = "2019-01-01"
END_DATE = "2023-12-31"
OUT_PATH = os.path.join(os.path.dirname(__file__), "sales_data.csv")

STORE_NAMES = [f"store_{i}" for i in range(1, N_STORES + 1)]
ITEM_NAMES = [f"item_{i}" for i in range(1, N_ITEMS + 1)]

# Assign items to two broad seasonal categories
#   "summer"  items peak in Jul-Aug  (e.g. beverages, sunscreen)
#   "winter"  items peak in Nov-Dec  (e.g. hot drinks, coats)
ITEM_CATEGORIES = {}
for idx, item in enumerate(ITEM_NAMES):
    ITEM_CATEGORIES[item] = "summer" if idx % 2 == 0 else "winter"

# Base price ranges per item (slight price variation across stores)
ITEM_BASE_PRICES = {item: round(rng.uniform(2.0, 50.0), 2) for item in ITEM_NAMES}

# Store-level sales multiplier (some stores are busier)
STORE_MULTIPLIERS = {store: rng.uniform(0.6, 1.8) for store in STORE_NAMES}


def seasonal_index(day_of_year: np.ndarray, category: str) -> np.ndarray:
    """Return a seasonal multiplier in the range [0.6, 1.4]."""
    # Normalise to [0, 2π]
    angle = 2 * np.pi * day_of_year / 365.0
    if category == "summer":
        # Peaks around day ~200 (mid-July)
        raw = np.cos(angle - 2 * np.pi * 200 / 365)
    else:
        # Peaks around day ~335 (early December)
        raw = np.cos(angle - 2 * np.pi * 335 / 365)
    # Scale to [0.6, 1.4]
    return 0.6 + 0.4 * (raw + 1) / 2 * 2


def dow_multiplier(day_of_week: np.ndarray) -> np.ndarray:
    """Weekend uplift: Sat/Sun ~20 % higher."""
    mult = np.where(day_of_week >= 5, 1.20, 1.00)
    return mult


def generate_sales_for_store_item(
    dates: pd.DatetimeIndex,
    store: str,
    item: str,
    base_sales: float,
) -> pd.DataFrame:
    """Return a DataFrame of daily sales rows for one (store, item) pair."""
    n = len(dates)

    # Day-of-year for seasonality
    doy = dates.day_of_year.to_numpy()
    dow = dates.day_of_week.to_numpy()

    # Trend: 0 → ~15 % growth over 5 years
    trend = 1.0 + 0.15 * np.linspace(0, 1, n)

    category = ITEM_CATEGORIES[item]
    season = seasonal_index(doy, category)
    dow_mult = dow_multiplier(dow)
    store_mult = STORE_MULTIPLIERS[store]

    # Base sales signal
    signal = base_sales * trend * season * dow_mult * store_mult

    # Gaussian noise (std ≈ 10 % of local signal)
    noise = rng.normal(0, signal * 0.10)
    sales = np.maximum(0, np.round(signal + noise).astype(int))

    # Price: base ± 5 % random variation per row
    base_price = ITEM_BASE_PRICES[item]
    prices = np.round(base_price * rng.uniform(0.95, 1.05, size=n), 2)

    return pd.DataFrame(
        {
            "date": dates,
            "store": store,
            "item": item,
            "sales": sales,
            "price": prices,
            "day_of_week": dow,
            "month": dates.month.to_numpy(),
            "year": dates.year.to_numpy(),
        }
    )


def main() -> None:
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    print(f"Date range: {dates[0].date()} → {dates[-1].date()}  ({len(dates)} days)")
    print(f"Stores: {N_STORES}   Items: {N_ITEMS}")
    print(f"Expected rows: {len(dates) * N_STORES * N_ITEMS:,}")
    print("Generating … this may take ~30 seconds")

    chunks: list[pd.DataFrame] = []
    combos = [(s, i) for s in STORE_NAMES for i in ITEM_NAMES]

    for store, item in tqdm(combos, desc="store × item pairs"):
        # Random base daily sales between 5 and 60 units
        base = rng.uniform(5, 60)
        df_chunk = generate_sales_for_store_item(dates, store, item, base)
        chunks.append(df_chunk)

    df = pd.concat(chunks, ignore_index=True)
    df.sort_values(["date", "store", "item"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n✅ Saved  →  {OUT_PATH}")
    print(f"   Total rows : {len(df):,}")
    print(f"   Columns    : {list(df.columns)}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
