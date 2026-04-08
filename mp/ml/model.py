"""LightGBM-based stock ranking model.

Trains a regression model on cross-sectional factor data to predict
forward returns.  Uses time-series cross-validation (expanding window)
to prevent look-ahead bias.

Usage:
    from mp.ml.model import StockRanker
    ranker = StockRanker()
    metrics = ranker.train(dataset_df)
    scores  = ranker.predict(features_df)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger

from mp.ml.dataset import FACTOR_COLUMNS

FEATURE_COLS = FACTOR_COLUMNS  # re-export for convenience
DEFAULT_MODEL_PATH = str(Path("data") / "model.lgb")


class StockRanker:
    """LightGBM wrapper for stock scoring."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model: Optional[lgb.Booster] = None
        self.model_path = model_path
        self.feature_importance: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
        """Train with time-series cross-validation (expanding window).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``date``, FEATURE_COLS, and ``fwd_ret``.
        n_splits : int
            Number of temporal CV folds.

        Returns
        -------
        dict with ``cv_mae_mean``, ``cv_mae_std``, ``cv_ic_mean``,
        ``best_rounds``, ``feature_importance`` (DataFrame).
        """
        required = set(FEATURE_COLS) | {"date", "fwd_ret"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = df.sort_values("date").reset_index(drop=True)
        X = df[FEATURE_COLS].values
        y = df["fwd_ret"].values
        dates = df["date"].values

        # --- temporal folds (expanding window) ---
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        fold_size = n_dates // (n_splits + 1)

        if fold_size < 20:
            logger.warning("Very few dates per fold ({}). Results may be noisy.", fold_size)

        cv_maes: List[float] = []
        cv_ics: List[float] = []
        best_rounds_list: List[int] = []

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
        }

        for fold in range(n_splits):
            # Train: dates[0 : (fold+1)*fold_size]
            # Val:   dates[(fold+1)*fold_size : (fold+2)*fold_size]
            train_end = unique_dates[(fold + 1) * fold_size - 1]
            val_start = unique_dates[(fold + 1) * fold_size]
            val_end_idx = min((fold + 2) * fold_size - 1, n_dates - 1)
            val_end = unique_dates[val_end_idx]

            train_mask = dates <= train_end
            val_mask = (dates >= val_start) & (dates <= val_end)

            if train_mask.sum() < 100 or val_mask.sum() < 20:
                logger.debug("Fold {} skipped: train={}, val={}", fold,
                             train_mask.sum(), val_mask.sum())
                continue

            dtrain = lgb.Dataset(X[train_mask], label=y[train_mask])
            dval = lgb.Dataset(X[val_mask], label=y[val_mask], reference=dtrain)

            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ]
            booster = lgb.train(
                params, dtrain,
                num_boost_round=1000,
                valid_sets=[dval],
                callbacks=callbacks,
            )

            best_iter = booster.best_iteration
            best_rounds_list.append(best_iter)

            preds = booster.predict(X[val_mask])
            mae = float(np.mean(np.abs(preds - y[val_mask])))
            cv_maes.append(mae)

            # Rank IC (Spearman) on validation fold
            from scipy.stats import spearmanr
            ic, _ = spearmanr(preds, y[val_mask])
            if not np.isnan(ic):
                cv_ics.append(float(ic))

            logger.info("Fold {}: MAE={:.4f}, IC={:.3f}, rounds={}",
                        fold, mae, ic if not np.isnan(ic) else 0.0, best_iter)

        # --- Final model: retrain on all data ---
        best_rounds = int(np.median(best_rounds_list)) if best_rounds_list else 200
        logger.info("Retraining on full data with {} rounds", best_rounds)

        dtrain_all = lgb.Dataset(X, label=y)
        self.model = lgb.train(params, dtrain_all, num_boost_round=best_rounds)

        # Feature importance
        importance = self.model.feature_importance(importance_type="gain")
        self.feature_importance = dict(zip(FEATURE_COLS, importance.tolist()))

        # Save model
        self.save()

        metrics = {
            "cv_mae_mean": float(np.mean(cv_maes)) if cv_maes else float("nan"),
            "cv_mae_std": float(np.std(cv_maes)) if cv_maes else float("nan"),
            "cv_ic_mean": float(np.mean(cv_ics)) if cv_ics else float("nan"),
            "cv_ic_std": float(np.std(cv_ics)) if cv_ics else float("nan"),
            "best_rounds": best_rounds,
            "n_folds": len(cv_maes),
            "n_train_rows": len(y),
            "feature_importance": self.feature_importance_report(),
        }
        logger.info("Training complete: MAE={:.4f}+/-{:.4f}, IC={:.3f}+/-{:.3f}",
                     metrics["cv_mae_mean"], metrics["cv_mae_std"],
                     metrics["cv_ic_mean"], metrics["cv_ic_std"])
        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict scores. Higher = more bullish (higher predicted return).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain FEATURE_COLS.

        Returns
        -------
        np.ndarray of predicted scores, same length as df.
        """
        if self.model is None:
            if not self.load():
                raise RuntimeError("No model loaded. Train or load a model first.")

        missing = set(FEATURE_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = df[FEATURE_COLS].values
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None):
        """Save model to disk."""
        path = path or self.model_path
        if self.model is None:
            logger.warning("No model to save")
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save_model(path)
        logger.info("Model saved to {}", path)

    def load(self, path: Optional[str] = None) -> bool:
        """Load model from disk. Returns True if successful."""
        path = path or self.model_path
        if not os.path.exists(path):
            logger.warning("Model file not found: {}", path)
            return False

        try:
            self.model = lgb.Booster(model_file=path)
            # Restore feature importance
            importance = self.model.feature_importance(importance_type="gain")
            self.feature_importance = dict(zip(FEATURE_COLS, importance.tolist()))
            logger.info("Model loaded from {}", path)
            return True
        except Exception as e:
            logger.error("Failed to load model: {}", e)
            return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def feature_importance_report(self) -> pd.DataFrame:
        """Feature importance sorted descending."""
        if not self.feature_importance:
            return pd.DataFrame(columns=["feature", "importance"])

        rows = sorted(self.feature_importance.items(), key=lambda x: -x[1])
        df = pd.DataFrame(rows, columns=["feature", "importance"])
        total = df["importance"].sum()
        if total > 0:
            df["pct"] = (df["importance"] / total * 100).round(1)
        return df
