"""Lock-in test: 60d StockRanker model artifact must always be 64-feature.

Regression guard — if a future cron path or refactor accidentally writes a
narrower-feature 60d model (e.g., a stale CURATED 32-feature retrain), this
test fails immediately in CI rather than going silently into production.

Context: P3-1c StockRanker fallback clobber bug already happened once; this
test makes sure 60d-specific regressions are also caught. See docs/dialog/
round 46.
"""
import pytest
import lightgbm as lgb
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model_60d.lgb"


def test_model_60d_feature_count():
    if not MODEL_PATH.exists():
        pytest.skip(f"{MODEL_PATH} not present; bootstrap needed before test runs")
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    assert booster.num_feature() == 64, (
        f"data/model_60d.lgb has {booster.num_feature()} features, "
        "expected 64 (FACTOR_COLUMNS post-Q15 widening). "
        "Did a cron path retrain on stale CURATED? Check docs/dialog/ round 46."
    )
