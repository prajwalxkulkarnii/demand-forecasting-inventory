"""
src/data_pipeline.py
~~~~~~~~~~~~~~~~~~~~
DataPipeline: load, clean, and feature-engineer the sales CSV.

Usage
-----
    pipeline = DataPipeline("data/sales_data.csv")
    pipeline.clean_data()
    pipeline.engineer_features()
    X_train, X_test, y_train, y_test = pipeline.split_data(test_months=3)
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

FEATURE_COLUMNS = [
    "store",
    "item",
    "day_of_week",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "is_month_end",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_7",
    "rolling_30",
    "rolling_std_7",
]
TARGET_COLUMN = "sales"


class DataPipeline:
    """End-to-end data preparation pipeline for the demand forecasting project.

    Parameters
    ----------
    filepath : str
        Path to the raw ``sales_data.csv`` file.

    Attributes
    ----------
    df : pd.DataFrame
        The working copy of the dataset (mutated in-place by each step).
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None
        self._load()

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the CSV from disk and perform minimal type casting."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Sales data not found at '{self.filepath}'. "
                "Run  python data/generate_data.py  first."
            )
        logger.info("Loading data from %s …", self.filepath)
        self.df = pd.read_csv(
            self.filepath,
            parse_dates=["date"],
            dtype={
                "store": "category",
                "item": "category",
                "sales": "int32",
                "day_of_week": "int8",
                "month": "int8",
                "year": "int16",
            },
        )
        logger.info("Loaded %d rows × %d columns.", *self.df.shape)

    # ── Public ────────────────────────────────────────────────────────────────

    def clean_data(self) -> pd.DataFrame:
        """Remove duplicates, handle nulls, clip outliers.

        Returns
        -------
        pd.DataFrame
            The cleaned DataFrame (also updates ``self.df``).
        """
        df = self.df.copy()
        before = len(df)

        # Drop full duplicates
        df.drop_duplicates(inplace=True)
        logger.info("Dropped %d duplicate rows.", before - len(df))

        # Drop rows where the target is missing
        null_sales = df["sales"].isna().sum()
        df.dropna(subset=["sales"], inplace=True)
        logger.info("Dropped %d rows with null sales.", null_sales)

        # Clip negative sales to 0
        negative_count = (df["sales"] < 0).sum()
        df["sales"] = df["sales"].clip(lower=0)
        logger.info("Clipped %d negative sales values to 0.", negative_count)

        # Clip extreme outliers: cap at 99.9th percentile per item
        p999 = df.groupby("item", observed=True)["sales"].transform(
            lambda x: x.quantile(0.999)
        )
        outlier_count = (df["sales"] > p999).sum()
        df["sales"] = df["sales"].clip(upper=p999)
        logger.info("Capped %d outlier sales values at item-level p99.9.", outlier_count)

        # Ensure date column is datetime
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(["store", "item", "date"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.df = df
        logger.info("Clean data: %d rows × %d columns.", *df.shape)
        return self.df

    def engineer_features(self) -> pd.DataFrame:
        """Create lag, rolling-window, and calendar features.

        Lag features
        ~~~~~~~~~~~~
        ``lag_1``, ``lag_7``, ``lag_14``, ``lag_28`` — sales N days prior for
        the same (store, item) pair.

        Rolling window features
        ~~~~~~~~~~~~~~~~~~~~~~~
        ``rolling_7``, ``rolling_30``  — trailing mean over 7 / 30 days.
        ``rolling_std_7``              — trailing std over 7 days.

        Calendar features
        ~~~~~~~~~~~~~~~~~
        ``quarter``, ``is_weekend``, ``is_month_end``.

        Returns
        -------
        pd.DataFrame
            Feature-enriched DataFrame (also updates ``self.df``).
        """
        if self.df is None:
            raise RuntimeError("Call clean_data() before engineer_features().")

        df = self.df.copy()
        logger.info("Engineering features …")

        # ── Calendar additions ───────────────────────────────────────────────
        df["quarter"] = df["date"].dt.quarter.astype("int8")
        df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
        df["is_month_end"] = df["date"].dt.is_month_end.astype("int8")

        # ── Lag / rolling features (computed within each store-item group) ───
        df.sort_values(["store", "item", "date"], inplace=True)

        grouped = df.groupby(["store", "item"], observed=True)["sales"]

        for lag in (1, 7, 14, 28):
            df[f"lag_{lag}"] = grouped.shift(lag)

        df["rolling_7"] = (
            grouped.transform(lambda x: x.shift(1).rolling(7).mean())
        )
        df["rolling_30"] = (
            grouped.transform(lambda x: x.shift(1).rolling(30).mean())
        )
        df["rolling_std_7"] = (
            grouped.transform(lambda x: x.shift(1).rolling(7).std())
        )

        # ── Drop rows that have NaN in lag columns (burn-in period) ──────────
        before = len(df)
        df.dropna(subset=["lag_28"], inplace=True)
        logger.info(
            "Dropped %d burn-in rows (insufficient lag history).", before - len(df)
        )

        # Encode categoricals as integer codes for ML models
        df["store"] = df["store"].cat.codes.astype("int16")
        df["item"] = df["item"].cat.codes.astype("int16")

        df.reset_index(drop=True, inplace=True)
        self.df = df
        logger.info("Feature engineering done: %d rows × %d columns.", *df.shape)
        return self.df

    def split_data(
        self,
        test_months: int = 3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Chronological train / test split.

        Parameters
        ----------
        test_months : int
            Number of trailing months to hold out as the test set.

        Returns
        -------
        X_train, X_test : pd.DataFrame
        y_train, y_test : pd.Series
        """
        if self.df is None:
            raise RuntimeError("Call engineer_features() before split_data().")

        cutoff = self.df["date"].max() - pd.DateOffset(months=test_months)
        logger.info(
            "Train/test split at %s  (test_months=%d).", cutoff.date(), test_months
        )

        train = self.df[self.df["date"] <= cutoff]
        test = self.df[self.df["date"] > cutoff]

        X_train = train[FEATURE_COLUMNS]
        y_train = train[TARGET_COLUMN]
        X_test = test[FEATURE_COLUMNS]
        y_test = test[TARGET_COLUMN]

        logger.info(
            "Train: %d rows | Test: %d rows (%.1f%% of total).",
            len(train),
            len(test),
            100 * len(test) / len(self.df),
        )
        return X_train, X_test, y_train, y_test

    @property
    def feature_names(self) -> list[str]:
        """Return the list of model feature column names."""
        return FEATURE_COLUMNS
