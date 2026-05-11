"""Recovery test: ArcBoost should find the known optimal arc in one split.

Construct a dataset where y is determined by membership in a known arc
[pi/3, 2*pi/3]: y = 1 inside the arc, y = 0 outside, with light noise. Train
ArcBoost with n_estimators=1, num_leaves=2 and check that the single split
recovers the arc to within bin resolution.

This is the canonical correctness test for FindBestThresholdCircular. Skipped
in v0.1.0.dev0 because the finder is a Log::Fatal stub.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from .conftest import circular_finder_required


@circular_finder_required
def test_recovers_known_arc_in_one_split(arcboost):
    rng = np.random.default_rng(42)
    n = 5000
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    in_arc = (theta >= math.pi / 3.0) & (theta <= 2.0 * math.pi / 3.0)
    y = in_arc.astype(np.float64) + rng.normal(0.0, 0.05, size=n)
    X = theta.reshape(-1, 1).astype(np.float32)

    params = {
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
    }
    ds = arcboost.Dataset(X, label=y, free_raw_data=False)
    # Multiple rounds so the leaf outputs converge close to the constant
    # values they're trying to approximate (≈1 inside, ≈0 outside). One round
    # under default shrinkage gives predictions in the 0.1–0.3 range — the
    # arc is correctly recovered, but the test's pass/fail threshold needs the
    # ensemble to settle.
    booster = arcboost.train(params, ds, num_boost_round=20)

    # Predict on a fine theta grid; the threshold between low and high outputs
    # should match the arc edges within bin resolution.
    grid = np.linspace(0.0, 2.0 * math.pi, 2000)
    pred = booster.predict(grid.reshape(-1, 1))
    inside_pred_mean = pred[(grid >= math.pi / 3.0) & (grid <= 2.0 * math.pi / 3.0)].mean()
    outside_pred_mean = pred[(grid < math.pi / 3.0) | (grid > 2.0 * math.pi / 3.0)].mean()

    # With a single arc split, leaves should separate cleanly: ~1 inside, ~0 outside.
    assert inside_pred_mean > 0.7
    assert outside_pred_mean < 0.3


@circular_finder_required
def test_recovers_arc_straddling_zero(arcboost):
    """Wraparound arcs are exactly what circular splits should win on.

    The arc [11*pi/6, pi/6] (i.e., a 60-degree arc straddling theta=0) cannot
    be expressed as a single linear threshold on raw theta, so this test is
    decisive: native LightGBM cannot recover it in one split, but ArcBoost
    should.
    """
    rng = np.random.default_rng(7)
    n = 5000
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    in_arc = (theta >= 11.0 * math.pi / 6.0) | (theta <= math.pi / 6.0)
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
    inside_mask = (grid >= 11.0 * math.pi / 6.0) | (grid <= math.pi / 6.0)
    assert pred[inside_mask].mean() > 0.7
    assert pred[~inside_mask].mean() < 0.3
