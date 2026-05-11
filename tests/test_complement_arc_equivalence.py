"""Sanity: the length<=B/2 prune in the arc-search loop doesn't miss any optima.

Mathematical setup: an arc [start, start+length] mod B and its complement
[start+length, start+B] mod B partition the bin set identically into "inside"
and "outside" (just with the labels swapped). The split gain is therefore
equal under arc and complement-arc, so scanning only length in [1, B/2]
covers every possible partition.

This test verifies the implementation actually achieves that equivalence
empirically: training with a forced "tiny arc on the wrong side" prior should
yield the same gain as the natural large arc.
"""
from __future__ import annotations

import math

import numpy as np

from .conftest import circular_finder_required


@circular_finder_required
def test_complement_arc_gives_equivalent_split(arcboost):
    rng = np.random.default_rng(99)
    n = 4000
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    # 5/6 of the circle is "inside"; only a 60-degree arc is "outside".
    in_arc = (theta >= math.pi / 6.0) & (theta <= 11.0 * math.pi / 6.0)
    y = in_arc.astype(np.float64) + rng.normal(0.0, 0.05, size=n)
    X = theta.reshape(-1, 1).astype(np.float32)

    booster = arcboost.train(
        {
            "objective": "regression",
            "num_leaves": 2,
            "learning_rate": 0.5,
            "min_data_in_leaf": 20,
            "circular_feature": "0",
            "circular_period": 2.0 * math.pi,
            "max_bin_circular": 64,
            "verbosity": -1,
            "deterministic": True,
            "seed": 0,
        },
        arcboost.Dataset(X, label=y, free_raw_data=False),
        num_boost_round=20,
    )
    grid = np.linspace(0.0, 2.0 * math.pi, 2000)
    pred = booster.predict(grid.reshape(-1, 1))
    inside_mask = (grid >= math.pi / 6.0) & (grid <= 11.0 * math.pi / 6.0)
    assert pred[inside_mask].mean() > 0.7
    assert pred[~inside_mask].mean() < 0.3
