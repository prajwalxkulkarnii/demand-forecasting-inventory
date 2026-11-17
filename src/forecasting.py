"""
src/forecasting.py
~~~~~~~~~~~~~~~~~~
DemandForecaster: train, predict, evaluate, and generate future forecasts.

Supported model backends
------------------------
- "gradient_boosting"  — sklearn GradientBoostingRegressor (default)
- "xgboost"            — XGBRegressor (requires ``pip install xgboost``)
- "linear"             — sklearn Ridge regression

Usage
-----
    forecaster = DemandForecaster(model_type="gradient_boosting")
    forecaster.train(X_train, y_train)
    preds = forecaster.predict(X_test)
    metrics = forecaster.evaluate(y_test, preds)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, ignoring zero-actual rows."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ── Main class ────────────────────────────────────────────────────────────────

class DemandForecaster:
    """Train and serve a demand forecasting model.

    Parameters
    ----------
    model_type : str
        One of ``"gradient_boosting"``, ``"xgboost"``, ``"linear"``.
    random_state : int
        Random seed for reproducibility.
    """

    SUPPORTED_MODELS = ("gradient_boosting", "xgboost", "linear")

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        random_state: int = 42,
    ) -> None:
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"model_type must be one of {self.SUPPORTED_MODELS}, got '{model_type}'."
            )
        self.model_type = model_type
        self.random_state = random_state
        self.model: Any = None
        self._build_model()

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_model(self) -> None:
        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                min_samples_leaf=10,
                random_state=self.random_state,
            )
        elif self.model_type == "xgboost":
            try:
                from xgboost import XGBRegressor  # noqa: PLC0415

                self.model = XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.07,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    verbosity=0,
                    n_jobs=-1,
                )
            except ImportError as exc:
                raise ImportError(
                    "xgboost is not installed. Run  pip install xgboost  or "
                    "choose model_type='gradient_boosting'."
                ) from exc
        elif self.model_type == "linear":
            self.model = Ridge(alpha=1.0)

        logger.info("Forecaster initialised: %s", self.model_type)

    # ── Public ────────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> "DemandForecaster":
        """Fit the model on training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix.
        y_train : pd.Series
            Target sales values.

        Returns
        -------
        self : DemandForecaster
            Fitted instance (chainable).
        """
        logger.info("Training %s on %d rows …", self.model_type, len(X_train))
        self.model.fit(X_train, y_train)
        train_preds = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_preds)
        logger.info("Training MAE (in-sample): %.4f", train_mae)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate point predictions.

        Parameters
        ----------
        X_test : pd.DataFrame
            Feature matrix for the prediction set.

        Returns
        -------
        np.ndarray
            Non-negative predicted demand values.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        preds = self.model.predict(X_test)
        return np.maximum(0, preds)

    def evaluate(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute regression evaluation metrics.

        Parameters
        ----------
        y_true : array-like
            Actual target values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        dict
            Keys: ``mae``, ``rmse``, ``mape``, ``r2``.
        """
        y_true_np = np.asarray(y_true, dtype=float)
        y_pred_np = np.asarray(y_pred, dtype=float)

        mae = mean_absolute_error(y_true_np, y_pred_np)
        rmse = math.sqrt(mean_squared_error(y_true_np, y_pred_np))
        mape = _mape(y_true_np, y_pred_np)
        r2 = r2_score(y_true_np, y_pred_np)

        metrics = {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}
        logger.info(
            "Evaluation — MAE: %.4f | RMSE: %.4f | MAPE: %.2f%% | R²: %.4f",
            mae, rmse, mape, r2,
        )
        return metrics

    def forecast_future(
        self,
        pipeline,  # DataPipeline
        days: int = 30,
    ) -> pd.DataFrame:
        """Generate a rolling multi-step forecast for the next ``days`` days.

        Strategy: seed the forecast with the last ``lag_28`` rows of real data,
        then auto-regressively fill lags from previously predicted values.

        Parameters
        ----------
        pipeline : DataPipeline
            A fitted pipeline whose ``df`` contains the full history.
        days : int
            Number of future calendar days to forecast.

        Returns
        -------
        pd.DataFrame
            Columns: ``date``, ``store``, ``item``, ``predicted_demand``,
            ``lower_bound``, ``upper_bound``.
        """
        from src.data_pipeline import FEATURE_COLUMNS  # noqa: PLC0415

        df = pipeline.df.copy()
        last_date = df["date"].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=days, freq="D"
        )

        store_item_groups = df.groupby(["store", "item"], observed=True)
        records = []

        for (store, item), grp in store_item_groups:
            grp = grp.sort_values("date").tail(60).copy()

            for fdate in future_dates:
                # Build a feature row using existing history
                last_row = grp.iloc[-1]
                new_row: Dict[str, Any] = {
                    "date": fdate,
                    "store": store,
                    "item": item,
                    "day_of_week": int(fdate.dayofweek),
                    "month": int(fdate.month),
                    "quarter": int(fdate.quarter),
                    "year": int(fdate.year),
                    "is_weekend": int(fdate.dayofweek >= 5),
                    "is_month_end": int(fdate.is_month_end),
                    "lag_1": float(grp["sales"].iloc[-1]),
                    "lag_7": float(grp["sales"].iloc[-7]) if len(grp) >= 7 else float(grp["sales"].mean()),
                    "lag_14": float(grp["sales"].iloc[-14]) if len(grp) >= 14 else float(grp["sales"].mean()),
                    "lag_28": float(grp["sales"].iloc[-28]) if len(grp) >= 28 else float(grp["sales"].mean()),
                    "rolling_7": float(grp["sales"].tail(7).mean()),
                    "rolling_30": float(grp["sales"].tail(30).mean()),
                    "rolling_std_7": float(grp["sales"].tail(7).std()),
                    "sales": np.nan,  # unknown
                }

                X_row = pd.DataFrame([new_row])[FEATURE_COLUMNS]
                pred = float(self.predict(X_row)[0])

                # Simple ±1 std confidence band using 7-day rolling std
                std7 = new_row["rolling_std_7"] if not np.isnan(new_row["rolling_std_7"]) else 0.0
                z = 1.96  # 95 % confidence
                lower = max(0.0, pred - z * std7)
                upper = pred + z * std7

                records.append(
                    {
                        "date": fdate,
                        "store": store,
                        "item": item,
                        "predicted_demand": round(pred, 2),
                        "lower_bound": round(lower, 2),
                        "upper_bound": round(upper, 2),
                    }
                )

                # Append predicted value as "sales" so next iteration uses it as lag
                new_row_with_sales = {**new_row, "sales": pred}
                grp = pd.concat(
                    [grp, pd.DataFrame([new_row_with_sales])], ignore_index=True
                )

        forecast_df = pd.DataFrame(records)
        logger.info(
            "Generated %d forecast rows for %d future days.", len(forecast_df), days
        )
        return forecast_df

    def compute_reorder_point(
        self,
        avg_demand: float,
        lead_time_days: int = 7,
        service_level: float = 0.95,
        std_demand: Optional[float] = None,
    ) -> float:
        """Calculate the reorder point using the safety-stock formula.

        .. math::
            ROP = \\bar{d} \\cdot L + z_{\\alpha} \\cdot \\sigma_d \\cdot \\sqrt{L}

        Parameters
        ----------
        avg_demand : float
            Average daily demand (units/day).
        lead_time_days : int
            Supplier lead time in days.
        service_level : float
            Desired service level (e.g. 0.95 → 95 %).
        std_demand : float, optional
            Standard deviation of daily demand. Defaults to 20 % of avg.

        Returns
        -------
        float
            Reorder point in units.
        """
        if std_demand is None:
            std_demand = avg_demand * 0.20

        z = float(norm.ppf(service_level))
        safety_stock = z * std_demand * math.sqrt(lead_time_days)
        rop = avg_demand * lead_time_days + safety_stock
        return round(rop, 2)

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importances if the backend supports it."""
        return getattr(self.model, "feature_importances_", None)
