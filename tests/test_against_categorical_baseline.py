"""Regression guard: declaring a feature circular shouldn't hurt on non-cyclic data.

If a user accidentally marks a numeric feature as circular, ArcBoost should
do at least as well as LightGBM's binned-categorical baseline on the same
data — i.e., the arc-search shouldn't be catastrophically worse than
subset-search on bins of an arbitrary feature. This guards against subtle
mis-tunings (e.g., wrong default for max_bin_circular, off-by-one in the
prefix-sum loop).
"""
from __future__ import annotations

import numpy as np

from .conftest import circular_finder_required


@circular_finder_required
def test_circular_does_not_regress_on_uniform_real_feature(arcboost):
    rng = np.random.default_rng(13)
    n = 4000
    x = rng.uniform(-3.0, 3.0, size=n)
    y = np.sin(2.0 * x) + rng.normal(0.0, 0.1, size=n)
    X = x.reshape(-1, 1).astype(np.float32)

    common = {
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "deterministic": True,
        "seed": 0,
    }

    # Treat as numeric (best-case baseline).
    ds_num = arcboost.Dataset(X, label=y, free_raw_data=False)
    booster_num = arcboost.train(common, ds_num, num_boost_round=100)
    pred_num = booster_num.predict(X)
    rmse_num = float(np.sqrt(np.mean((pred_num - y) ** 2)))

    # Treat as circular with period 6.0 (the data range), shifted so that x in [-3, 3]
    # maps to phi in [0, 6).
    ds_circ = arcboost.Dataset(X + 3.0, label=y, free_raw_data=False)
    booster_circ = arcboost.train(
        {**common, "circular_feature": "0", "circular_period": 6.0, "max_bin_circular": 64},
        ds_circ,
        num_boost_round=100,
    )
    pred_circ = booster_circ.predict(X + 3.0)
    rmse_circ = float(np.sqrt(np.mean((pred_circ - y) ** 2)))

    # Loose bound — circular should be within 1.5x of numeric on data where the
    # cyclic structure is artificial. Tighten when the real arc-search lands.
    assert rmse_circ <= 1.5 * rmse_num, (
        f"circular RMSE ({rmse_circ:.4f}) regressed >1.5x vs numeric ({rmse_num:.4f}) "
        "on a non-cyclic feature — investigate arc-search loop or default tuning."
    )
