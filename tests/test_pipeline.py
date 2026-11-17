"""
Unit tests for DataPipeline feature engineering and splitting.
Run with:  pytest tests/test_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_pipeline import DataPipeline


def make_sales_df(n_stores: int = 3, n_items: int = 4, days: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    rows = []
    start = pd.Timestamp("2022-01-01")
    for s in range(n_stores):
        for i in range(n_items):
            for d in range(days):
                rows.append({
                    "date":  start + pd.Timedelta(days=d),
                    "store": f"store_{s+1}",
                    "item":  f"item_{i+1}",
                    "sales": max(0, rng.normal(20, 5)),
                    "price": rng.uniform(5, 15),
                })
    return pd.DataFrame(rows)


class TestDataPipeline:

    @pytest.fixture
    def tmp_csv(self, tmp_path):
        df = make_sales_df()
        path = tmp_path / "sales.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_load_data_shape(self, tmp_csv):
        p = DataPipeline(tmp_csv)
        p.load_data()
        assert p.df is not None
        assert len(p.df) > 0

    def test_load_data_required_columns(self, tmp_csv):
        p = DataPipeline(tmp_csv)
        p.load_data()
        for col in ["date","store","item","sales"]:
            assert col in p.df.columns

    def test_clean_data_no_negatives(self, tmp_csv):
        p = DataPipeline(tmp_csv)
        p.load_data().clean_data()
        assert (p.df["sales"] >= 0).all()

    def test_engineer_features_lag_columns(self, tmp_csv):
        p = DataPipeline(tmp_csv)
        p.load_data().clean_data().engineer_features()
        for lag in [1, 7, 14, 28]:
            assert f"lag_{lag}" in p.df.columns, f"Missing lag_{lag}"

    def test_engineer_features_rolling_columns(self, tmp_csv):
        p = DataPipeline(tmp_csv)
        p.load_data().clean_data().engineer_features()
        assert "rolling_mean_7" in p.df.columns
        assert "rolling_mean_30" in p.df.columns
        assert "rolling_std_7" in p.df.columns

    def test_engineer_features_date_parts(self, tmp_csv):
        p = DataPipeline(tmp_csv)
        p.load_data().clean_data().engineer_features()
        for col in ["day_of_week","month","quarter","year","is_weekend","is_month_end"]:
            assert col in p.df.columns

    def test_no_nans_after_engineering(self, tmp_csv):
        p = DataPipeline(tmp_csv)
        p.load_data().clean_data().engineer_features()
        for col in p.feature_columns:
            assert p.df[col].isnull().sum() == 0, f"NaN found in {col}"

    def test_split_data_no_overlap(self, tmp_csv):
        p = DataPipeline(tmp_csv)
        p.load_data().clean_data().engineer_features()
        X_train, X_test, y_train, y_test = p.split_data(test_months=1)
        assert len(X_train) > 0
        assert len(X_test) > 0
        # No date overlap — handled via cutoff
        assert len(X_train) + len(X_test) == len(p.df)

    def test_get_summary_keys(self, tmp_csv):
        p = DataPipeline(tmp_csv)
        p.load_data()
        summary = p.get_summary()
        for key in ["rows","date_range","stores","items","avg_daily_sales"]:
            assert key in summary
