"""
Unit tests for DemandForecaster and InventoryOptimizer.
Run with:  pytest tests/test_forecasting.py -v
"""

import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from forecasting import DemandForecaster
from inventory_optimizer import InventoryOptimizer


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def sample_X_y():
    """Small synthetic feature matrix and target."""
    rng = np.random.default_rng(42)
    n = 500
    X = pd.DataFrame({
        "store_code":        rng.integers(0, 5, n),
        "item_code":         rng.integers(0, 10, n),
        "lag_1":             rng.uniform(0, 50, n),
        "lag_7":             rng.uniform(0, 50, n),
        "lag_14":            rng.uniform(0, 50, n),
        "lag_28":            rng.uniform(0, 50, n),
        "rolling_mean_7":    rng.uniform(10, 40, n),
        "rolling_mean_30":   rng.uniform(10, 40, n),
        "rolling_std_7":     rng.uniform(0, 10, n),
        "day_of_week":       rng.integers(0, 7, n),
        "month":             rng.integers(1, 13, n),
        "quarter":           rng.integers(1, 5, n),
        "year":              rng.integers(2019, 2024, n),
        "day_of_year":       rng.integers(1, 366, n),
        "is_weekend":        rng.integers(0, 2, n),
        "is_month_end":      rng.integers(0, 2, n),
        "is_month_start":    rng.integers(0, 2, n),
    })
    y = pd.Series(rng.uniform(0, 60, n), name="sales")
    return X, y


@pytest.fixture
def trained_forecaster(sample_X_y):
    X, y = sample_X_y
    f = DemandForecaster(model_type="gradient_boosting")
    f.train(X, y)
    return f, X, y


# ------------------------------------------------------------------ #
#  DemandForecaster                                                    #
# ------------------------------------------------------------------ #

class TestDemandForecaster:

    def test_init_gradient_boosting(self):
        f = DemandForecaster(model_type="gradient_boosting")
        assert f.model_type == "gradient_boosting"

    def test_init_linear(self):
        f = DemandForecaster(model_type="linear")
        assert f.model_type == "linear"

    def test_predict_before_train_raises(self, sample_X_y):
        X, _ = sample_X_y
        f = DemandForecaster(model_type="linear")
        with pytest.raises(RuntimeError, match="not been trained"):
            f.predict(X)

    def test_train_returns_self(self, sample_X_y):
        X, y = sample_X_y
        f = DemandForecaster(model_type="linear")
        result = f.train(X, y)
        assert result is f

    def test_predict_non_negative(self, trained_forecaster):
        f, X, _ = trained_forecaster
        preds = f.predict(X)
        assert (preds >= 0).all(), "All predictions must be ≥ 0"

    def test_predict_shape(self, trained_forecaster):
        f, X, _ = trained_forecaster
        preds = f.predict(X)
        assert preds.shape == (len(X),)

    def test_evaluate_keys(self, trained_forecaster):
        f, X, y = trained_forecaster
        preds = f.predict(X)
        metrics = f.evaluate(y, preds)
        assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE", "R2"}

    def test_evaluate_mae_positive(self, trained_forecaster):
        f, X, y = trained_forecaster
        preds = f.predict(X)
        metrics = f.evaluate(y, preds)
        assert metrics["MAE"] >= 0
        assert metrics["RMSE"] >= 0

    def test_reorder_point_basic(self, trained_forecaster):
        f, _, _ = trained_forecaster
        result = f.compute_reorder_point(avg_daily_demand=10, lead_time_days=7, demand_std=2)
        assert "reorder_point" in result
        assert result["reorder_point"] > 0

    def test_reorder_point_zero_demand(self, trained_forecaster):
        f, _, _ = trained_forecaster
        result = f.compute_reorder_point(avg_daily_demand=0, lead_time_days=7, demand_std=0)
        assert result["reorder_point"] == 0.0

    def test_feature_importances(self, trained_forecaster):
        f, X, _ = trained_forecaster
        fi = f.feature_importances(list(X.columns))
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) == len(X.columns)
        assert abs(fi["importance"].sum() - 1.0) < 1e-6


# ------------------------------------------------------------------ #
#  InventoryOptimizer                                                  #
# ------------------------------------------------------------------ #

class TestInventoryOptimizer:

    def test_eoq_basic(self):
        opt = InventoryOptimizer()
        eoq = opt.compute_eoq(annual_demand=5000, ordering_cost=50, holding_cost_per_unit=2.5)
        expected = round((2 * 5000 * 50 / 2.5) ** 0.5, 2)
        assert abs(eoq - expected) < 0.01

    def test_eoq_zero_holding_raises(self):
        opt = InventoryOptimizer()
        with pytest.raises(ValueError):
            opt.compute_eoq(5000, 50, holding_cost_per_unit=0)

    def test_safety_stock(self):
        opt = InventoryOptimizer()
        ss = opt.compute_safety_stock(std_daily_demand=5, lead_time_days=4, z_score=1.645)
        expected = round(1.645 * 5 * (4 ** 0.5), 2)
        assert abs(ss - expected) < 0.01

    def test_reorder_point_formula(self):
        opt = InventoryOptimizer()
        rop = opt.compute_reorder_point(avg_daily_demand=10, lead_time_days=5, safety_stock=15)
        assert rop == round(10 * 5 + 15, 2)

    def test_generate_report_columns(self):
        opt = InventoryOptimizer()
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "store": pd.Categorical(["s1"] * 100 + ["s2"] * 100),
            "item":  pd.Categorical(["i1"] * 200),
            "sales": rng.uniform(5, 30, 200),
            "date":  pd.date_range("2022-01-01", periods=200),
        })
        report = opt.generate_inventory_report(df)
        for col in ["store","item","avg_daily_demand","eoq","safety_stock","reorder_point"]:
            assert col in report.columns, f"Missing column: {col}"

    def test_low_stock_alerts(self):
        opt = InventoryOptimizer()
        inv = pd.DataFrame({
            "store": ["s1","s1","s2"],
            "item":  ["i1","i2","i1"],
            "reorder_point": [50, 30, 20],
            "current_stock": [40, 35, 10],  # s1/i1 and s2/i1 below reorder
        })
        alerts = opt.get_low_stock_alerts(inv, current_stock_col="current_stock")
        assert len(alerts) == 2
        assert "stock_deficit" in alerts.columns
