"""Factor registry - register and lookup factor calculation functions."""

from typing import Callable

import pandas as pd

# Factor function signature: (bars, valuation, financial) -> Series indexed by code
FactorFunc = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.Series]

_REGISTRY: dict[str, FactorFunc] = {}


def register(name: str):
    """Decorator to register a factor function."""
    def decorator(func: FactorFunc):
        _REGISTRY[name] = func
        return func
    return decorator


def get_factor_func(name: str) -> FactorFunc:
    if name not in _REGISTRY:
        raise KeyError(f"Factor '{name}' not registered. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_factors() -> list[str]:
    return list(_REGISTRY.keys())
