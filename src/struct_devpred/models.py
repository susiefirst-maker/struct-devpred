"""Classical-model factories for the benchmark study.

Three families — all cheap enough to run across 7 descriptor × 6 endpoint
combinations in minutes. No neural nets in Phase 4.
"""

from __future__ import annotations

from typing import Callable

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


ModelFactory = Callable[[], object]


def ridge_factory(alpha: float = 1.0) -> ModelFactory:
    return lambda: Ridge(alpha=alpha, random_state=0)


def random_forest_factory(n_estimators: int = 300, max_depth: int | None = None) -> ModelFactory:
    return lambda: RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=0,
        n_jobs=-1,
    )


def gradient_boosting_factory(
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
) -> ModelFactory:
    return lambda: GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=0,
    )


MODEL_FACTORIES: dict[str, ModelFactory] = {
    "ridge": ridge_factory(),
    "random_forest": random_forest_factory(),
    "gradient_boosting": gradient_boosting_factory(),
}

# Whether each model family requires standardization of input features.
STANDARDIZE_INPUTS = {
    "ridge": True,
    "random_forest": False,
    "gradient_boosting": False,
}
