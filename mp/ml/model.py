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

from mp.ml.dataset import CURATED_COLUMNS, FACTOR_COLUMNS, FUNDAMENTAL_COLUMNS, EXCESS_LABEL

FEATURE_COLS = FACTOR_COLUMNS  # re-export for convenience
DEFAULT_MODEL_PATH = str(Path("data") / "model.lgb")


def _topk_metrics(preds: np.ndarray, actuals: np.ndarray,
                  dates: np.ndarray, k: int = 10) -> Dict[str, float]:
    """Compute Precision@K and HitRate@K per date, then average."""
    df = pd.DataFrame({"date": dates, "pred": preds, "actual": actuals})
    hit_rates, precisions = [], []
    for _, g in df.groupby("date"):
        if len(g) < k:
            continue
        top = g.nlargest(k, "pred")
        hit_rates.append(float((top["actual"] > 0).mean()))
        actual_topk = set(g.nlargest(k, "actual").index)
        precisions.append(len(set(top.index) & actual_topk) / k)
    return {
        "hit_rate_at_k": float(np.mean(hit_rates)) if hit_rates else float("nan"),
        "precision_at_k": float(np.mean(precisions)) if precisions else float("nan"),
    }


class StockRanker:
    """LightGBM wrapper for stock scoring."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 feature_cols: Optional[List[str]] = None,
                 objective: str = "regression",
                 label_col: str = "fwd_ret",
                 lgb_params: Optional[Dict[str, object]] = None):
        self.model: Optional[lgb.Booster] = None
        self.model_path = model_path
        self.feature_cols = feature_cols or FACTOR_COLUMNS
        self.feature_importance: Dict[str, float] = {}
        self.objective = objective
        self.label_col = label_col
        self._lgb_params_override = lgb_params or {}
        self.score_type = "predicted_return"  # raw LightGBM regression output

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
        label = self.label_col
        required = set(self.feature_cols) | {"date", label}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = df.sort_values("date").reset_index(drop=True)
        X = df[self.feature_cols].values
        y = df[label].values
        dates = df["date"].values

        # --- temporal folds (expanding window) ---
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        fold_size = n_dates // (n_splits + 1)

        if fold_size < 20:
            logger.warning("Very few dates per fold ({}). Results may be noisy.", fold_size)

        cv_maes: List[float] = []
        cv_ics: List[float] = []
        cv_hit_rates: List[float] = []
        cv_precisions: List[float] = []
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

            dtrain = lgb.Dataset(X[train_mask], label=y[train_mask],
                                 feature_name=list(self.feature_cols))
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

            topk = _topk_metrics(preds, y[val_mask], dates[val_mask], k=10)
            if not np.isnan(topk["hit_rate_at_k"]):
                cv_hit_rates.append(topk["hit_rate_at_k"])
            if not np.isnan(topk["precision_at_k"]):
                cv_precisions.append(topk["precision_at_k"])
            logger.info("Fold {}: MAE={:.4f}, IC={:.3f}, HitRate@10={:.2f}, Precision@10={:.2f}, rounds={}",
                        fold, mae, ic if not np.isnan(ic) else 0.0,
                        topk["hit_rate_at_k"], topk["precision_at_k"], best_iter)

        # --- Final model: retrain on all data ---
        best_rounds = int(np.median(best_rounds_list)) if best_rounds_list else 200
        logger.info("Retraining on full data with {} rounds", best_rounds)

        dtrain_all = lgb.Dataset(X, label=y, feature_name=list(self.feature_cols))
        self.model = lgb.train(params, dtrain_all, num_boost_round=best_rounds)

        # Feature importance
        importance = self.model.feature_importance(importance_type="gain")
        self.feature_importance = dict(zip(self.feature_cols, importance.tolist()))

        # Save model
        self.save()

        metrics = {
            "cv_mae_mean": float(np.mean(cv_maes)) if cv_maes else float("nan"),
            "cv_mae_std": float(np.std(cv_maes)) if cv_maes else float("nan"),
            "cv_ic_mean": float(np.mean(cv_ics)) if cv_ics else float("nan"),
            "cv_ic_std": float(np.std(cv_ics)) if cv_ics else float("nan"),
            "cv_hit_rate_mean": float(np.mean(cv_hit_rates)) if cv_hit_rates else float("nan"),
            "cv_precision_mean": float(np.mean(cv_precisions)) if cv_precisions else float("nan"),
            "best_rounds": best_rounds,
            "n_folds": len(cv_maes),
            "n_train_rows": len(y),
            "feature_importance": self.feature_importance_report(),
        }
        logger.info(
            "Training complete: MAE={:.4f}±{:.4f}, IC={:.3f}±{:.3f}, HitRate@10={:.2f}, Precision@10={:.2f}",
            metrics["cv_mae_mean"], metrics["cv_mae_std"],
            metrics["cv_ic_mean"], metrics["cv_ic_std"],
            metrics["cv_hit_rate_mean"], metrics["cv_precision_mean"],
        )
        return metrics

    def train_fast(self, df: pd.DataFrame, val_frac: float = 0.15) -> dict:
        """Single-pass training for walk-forward loops (no CV, no auto-save).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``date``, FEATURE_COLS, and the label column
            (``fwd_ret`` by default, or ``excess_ret`` / others).
        val_frac : float
            Fraction of the most recent dates used as validation set.

        Returns
        -------
        dict with ``mae``, ``ic``, ``best_rounds``, and optionally
        ``hit_rate_at_k``, ``precision_at_k``.
        """
        label = self.label_col
        required = set(self.feature_cols) | {"date", label}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = df.sort_values("date").reset_index(drop=True)
        X = df[self.feature_cols].values
        y = df[label].values
        dates = df["date"].values

        is_rank = self.objective in ("lambdarank", "rank_xndcg")

        # LambdaRank requires integer relevance labels (0..N).
        # Quantize continuous returns into 5 grades per date group.
        if is_rank:
            y_raw = y.copy()  # keep original for IC calculation
            grade = np.zeros(len(y), dtype=np.int32)
            for dt in np.unique(dates):
                mask = dates == dt
                vals = y[mask]
                # 5 quantile bins: 0(worst) .. 4(best)
                try:
                    bins = pd.qcut(vals, 5, labels=False, duplicates="drop")
                except ValueError:
                    bins = np.zeros(len(vals), dtype=int)
                grade[mask] = bins.astype(np.int32)
            y = grade

        # Split by time: last val_frac as validation
        n = len(df)
        split_idx = int(n * (1 - val_frac))

        # --- Build LGB params ---
        if is_rank:
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5, 10, 20],
                "learning_rate": 0.05, "num_leaves": 31,
                "min_child_samples": 50, "subsample": 0.8,
                "colsample_bytree": 0.8, "reg_alpha": 0.1,
                "reg_lambda": 0.1, "verbose": -1,
            }
        else:
            params = {
                "objective": "regression", "metric": "mae",
                "learning_rate": 0.05, "num_leaves": 31,
                "min_child_samples": 50, "subsample": 0.8,
                "colsample_bytree": 0.8, "reg_alpha": 0.1,
                "reg_lambda": 0.1, "verbose": -1,
            }

        # Apply user overrides
        params.update(self._lgb_params_override)

        if split_idx < 100 or n - split_idx < 20:
            # Fallback: train on everything, no early stopping
            if is_rank:
                groups = pd.Series(dates).groupby(dates).size().values
                dtrain = lgb.Dataset(X, label=y, group=groups)
            else:
                dtrain = lgb.Dataset(X, label=y)
            self.model = lgb.train(params, dtrain, num_boost_round=200)
            return {"mae": float("nan"), "ic": float("nan"), "best_rounds": 200}

        if is_rank:
            train_groups = pd.Series(dates[:split_idx]).groupby(dates[:split_idx]).size().values
            val_groups = pd.Series(dates[split_idx:]).groupby(dates[split_idx:]).size().values
            dtrain = lgb.Dataset(X[:split_idx], label=y[:split_idx],
                                 group=train_groups,
                                 feature_name=list(self.feature_cols))
            dval = lgb.Dataset(X[split_idx:], label=y[split_idx:],
                               group=val_groups, reference=dtrain)
        else:
            dtrain = lgb.Dataset(X[:split_idx], label=y[:split_idx],
                                 feature_name=list(self.feature_cols))
            dval = lgb.Dataset(X[split_idx:], label=y[split_idx:], reference=dtrain)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        self.model = lgb.train(
            params, dtrain, num_boost_round=1000,
            valid_sets=[dval], callbacks=callbacks,
        )

        preds = self.model.predict(X[split_idx:])

        # Use raw continuous labels for metrics when lambdarank quantized them
        y_eval = y_raw[split_idx:] if is_rank else y[split_idx:]
        mae = float(np.mean(np.abs(preds - y_eval)))

        from scipy.stats import spearmanr
        ic, _ = spearmanr(preds, y_eval)
        ic = float(ic) if not np.isnan(ic) else 0.0

        # Top-K metrics on validation set
        topk = _topk_metrics(preds, y_eval, dates[split_idx:], k=10)

        return {
            "mae": mae,
            "ic": ic,
            "best_rounds": self.model.best_iteration or 200,
            **topk,
        }

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

        missing = set(self.feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        # Check for NaN in fundamental columns before prediction
        fund_cols = [c for c in FUNDAMENTAL_COLUMNS if c in df.columns]
        if fund_cols:
            nan_pct = df[fund_cols].isna().mean()
            bad_cols = nan_pct[nan_pct > 0.5]
            if not bad_cols.empty:
                detail = ", ".join(f"{c}({v:.0%})" for c, v in bad_cols.items())
                logger.warning("⚠ 基本面数据大面积缺失: {} — 预测结果可能不准", detail)

        X = df[self.feature_cols].values
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
            feat_names = self.model.feature_name()
            # Only adopt model's feature names if they are real (not generic Column_N)
            if (feat_names and len(feat_names) == len(importance)
                    and not feat_names[0].startswith("Column_")):
                self.feature_cols = feat_names
                self.feature_importance = dict(zip(feat_names, importance.tolist()))
            else:
                self.feature_importance = dict(zip(self.feature_cols, importance.tolist()))
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


# ======================================================================
# Two-Stage Ranker
# ======================================================================

class TwoStageRanker:
    """Stage-1 coarse ranking → top percentile → Stage-2 fine ranking.

    Stage-1: existing StockRanker (regression on fwd_ret) filters the
    universe to the top ``top_pct`` stocks per date.

    Stage-2: a second StockRanker (optionally lambdarank on excess_ret)
    with additional cross-sectional features re-ranks the survivors.

    Usage:
        ranker = TwoStageRanker()
        metrics = ranker.train_fast(panel_df)
        scores = ranker.predict(today_df)
    """

    def __init__(
        self,
        top_pct: float = 0.2,
        stage1_feature_cols: Optional[List[str]] = None,
        stage2_objective: str = "regression",
        stage2_label: str = "excess_ret",
    ):
        from mp.ml.stage2_features import STAGE2_COLUMNS

        self.top_pct = top_pct
        s1_cols = stage1_feature_cols or FACTOR_COLUMNS

        self.stage1 = StockRanker(
            feature_cols=s1_cols,
            objective="regression",
            label_col="fwd_ret",
        )
        self.stage2 = StockRanker(
            feature_cols=list(s1_cols) + STAGE2_COLUMNS,
            objective=stage2_objective,
            label_col=stage2_label,
        )

    # ----- training -----

    def train_fast(self, df: pd.DataFrame, val_frac: float = 0.15) -> dict:
        """Train both stages on *df*.

        1. Train Stage-1 (regression on fwd_ret) on full data.
        2. Score all rows; keep top ``top_pct`` per date.
        3. Compute Stage-2 features for the filtered subset.
        4. Train Stage-2 on the filtered subset.
        """
        from mp.ml.stage2_features import compute_stage2_features

        # Stage 1
        m1 = self.stage1.train_fast(df, val_frac)
        logger.info("Stage-1 trained: IC={:.3f}, HitRate@K={:.3f}",
                     m1.get("ic", 0), m1.get("hit_rate_at_k", 0))

        # Score full panel and filter
        s1_scores = self.stage1.predict(df)
        df = df.copy()
        df["stage1_score"] = s1_scores

        # Keep top_pct per date by rank threshold
        rank_thr = df.groupby("date")["stage1_score"].transform(
            lambda x: x.quantile(1 - self.top_pct)
        )
        filtered = df[df["stage1_score"] >= rank_thr].reset_index(drop=True)
        logger.info("Stage-2 training on {} / {} rows (top {:.0%} per date)",
                     len(filtered), len(df), self.top_pct)

        # Stage 2 features
        filtered = compute_stage2_features(filtered)

        # Stage 2 train
        m2 = self.stage2.train_fast(filtered, val_frac)
        logger.info("Stage-2 trained: IC={:.3f}, HitRate@K={:.3f}",
                     m2.get("ic", 0), m2.get("hit_rate_at_k", 0))

        return {"stage1": m1, "stage2": m2}

    # ----- prediction -----

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Score stocks: Stage-1 filter → Stage-2 rerank.

        Returns an array of length ``len(df)``. Stocks not passing Stage-1
        receive ``-np.inf``.
        """
        from mp.ml.stage2_features import compute_stage2_features

        s1_scores = self.stage1.predict(df)
        k = max(int(len(df) * self.top_pct), 1)
        top_idx = np.argsort(s1_scores)[-k:]

        # Prepare Stage-2 input
        top_df = df.iloc[top_idx].copy()
        top_df["stage1_score"] = s1_scores[top_idx]
        top_df = compute_stage2_features(top_df)

        s2_scores = self.stage2.predict(top_df)

        # Merge back
        final = np.full(len(df), -np.inf)
        final[top_idx] = s2_scores
        return final


# ======================================================================
# Blend Ranker  (post-hoc rank-blend of complementary models)
# ======================================================================

class BlendRanker:
    """Rank-normalized blend of two complementary models.

    The winning approach from experimentation: train one model on excess
    returns (ZZ500-relative), another on extreme samples only, then blend
    their rank-normalized scores.

    Default blend: 0.80 * excess_ret + 0.20 * extreme30.

    Usage:
        ranker = BlendRanker()
        metrics = ranker.train_fast(panel_df)
        scores = ranker.predict(today_df)
    """

    def __init__(
        self,
        weight_primary: float = 0.80,
        extreme_pctile: float = 0.30,
        primary_label: str = "excess_ret",
        primary_params: Optional[Dict] = None,
        extreme_params: Optional[Dict] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        self.weight_primary = weight_primary
        self.weight_secondary = 1.0 - weight_primary
        self.extreme_pctile = extreme_pctile
        self.feature_cols = feature_cols or FACTOR_COLUMNS
        self.score_type = "rank_percentile"  # rank-normalized blend output ∈ [0,1]

        self.primary = StockRanker(
            feature_cols=self.feature_cols,
            objective="regression",
            label_col=primary_label,
            lgb_params=primary_params,
        )
        self.extreme = StockRanker(
            feature_cols=self.feature_cols,
            objective="regression",
            label_col="excess_ret",   # market-neutral: stable spread across bull/bear
            lgb_params=extreme_params,
        )

    def _filter_extremes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only top/bottom ``extreme_pctile`` per date by excess_ret.

        Uses excess_ret (vs ZZ500 benchmark) instead of raw fwd_ret so the
        extremes represent true relative winners/losers regardless of market
        regime.  Falls back to fwd_ret if excess_ret is unavailable.
        """
        pct = self.extreme_pctile
        rank_col = "excess_ret" if "excess_ret" in df.columns else "fwd_ret"

        def _keep(g: pd.DataFrame) -> pd.DataFrame:
            lo = g[rank_col].quantile(pct)
            hi = g[rank_col].quantile(1 - pct)
            return g[(g[rank_col] <= lo) | (g[rank_col] >= hi)]

        return df.groupby("date", group_keys=False).apply(_keep)

    def train_fast(self, df: pd.DataFrame, val_frac: float = 0.15) -> dict:
        """Train both component models.

        1. Primary model: trained on all data with ``primary_label``.
        2. Extreme model: trained on top/bottom extremes with ``excess_ret``.
        """
        # Primary (excess_ret)
        m1 = self.primary.train_fast(df, val_frac)
        logger.info("BlendRanker primary: IC={:.3f}", m1.get("ic", 0))

        # Extreme — filter requires excess_ret; fall back to fwd_ret rows only
        filter_col = "excess_ret" if "excess_ret" in df.columns else "fwd_ret"
        extreme_df = self._filter_extremes(
            df[df[filter_col].notna()].copy()
        )
        m2 = self.extreme.train_fast(extreme_df, val_frac)
        logger.info("BlendRanker extreme: IC={:.3f}, train rows={}",
                     m2.get("ic", 0), len(extreme_df))

        return {"primary": m1, "extreme": m2}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Score by rank-normalized blend of both models."""
        s1 = self.primary.predict(df)
        s2 = self.extreme.predict(df)

        # Rank-normalize to [0, 1]
        r1 = pd.Series(s1).rank(pct=True).values
        r2 = pd.Series(s2).rank(pct=True).values

        return self.weight_primary * r1 + self.weight_secondary * r2

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Return raw scores from primary model (actual predicted returns)."""
        return self.primary.predict(df)

    def feature_importance_report(self) -> pd.DataFrame:
        """Delegate to primary model's feature importance."""
        return self.primary.feature_importance_report()

    def save(self, path_prefix: str = "data/blend"):
        """Save both component models."""
        self.primary.save(f"{path_prefix}_primary.lgb")
        self.extreme.save(f"{path_prefix}_extreme.lgb")

    def load(self, path_prefix: str = "data/blend") -> bool:
        """Load both component models."""
        ok1 = self.primary.load(f"{path_prefix}_primary.lgb")
        ok2 = self.extreme.load(f"{path_prefix}_extreme.lgb")
        return ok1 and ok2
