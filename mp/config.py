"""Configuration loader."""

from pathlib import Path

import yaml
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"


class DataConfig(BaseModel):
    source: str = "akshare"
    cache_dir: str = "data/cache"
    db_url: str = "sqlite:///data/market.db"


class StrategyConfig(BaseModel):
    mode: str = "rotation"
    universe: str = "hs300"
    rebalance_freq: str = "monthly"
    top_n: int = 5
    start_date: str = "2024-01-01"
    end_date: str | None = None


class RiskConfig(BaseModel):
    max_position_pct: float = 0.20
    stop_loss_pct: float = 0.08
    trailing_stop_pct: float = 0.12
    max_drawdown_pct: float = 0.15
    max_sectors: int = 5
    vol_target: float = 0.15


class ExecutionConfig(BaseModel):
    mode: str = "simulation"
    broker: str | None = None
    slippage_bps: int = 5
    commission_bps: int = 3


class Settings(BaseModel):
    data: DataConfig = DataConfig()
    strategy: StrategyConfig = StrategyConfig()
    risk: RiskConfig = RiskConfig()
    execution: ExecutionConfig = ExecutionConfig()


def load_settings(path: Path = _DEFAULT_CONFIG) -> Settings:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Settings(**raw)
