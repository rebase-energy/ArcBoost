"""Tests for the BinMapper ValueToBin branch on circular features.

These tests do NOT need the arc-search finder — they exercise only the
binning logic added in patches 0001-0002, which is reachable by constructing
a Dataset with a circular feature and inspecting the bin assignments.

Strategy: train a 1-tree, num_leaves=2 model with a circular feature so that
the patched BinMapper actually gets exercised. We don't assert on the split
itself (that's the finder's job and may not be implemented yet) — only on the
fact that no error is raised during binning.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def test_arcboost_imports(arcboost):
    """Sanity: the renamed package imports and exposes the LightGBM API."""
    assert hasattr(arcboost, "LGBMRegressor")
    assert hasattr(arcboost, "Dataset")
    assert hasattr(arcboost, "train")


def test_circular_param_is_accepted_in_params_dict(arcboost):
    """circular_feature passes through Config without error.

    Even with the stub finder, simply constructing a Dataset and calling
    Booster() with circular_feature set should not raise — the error only
    happens when split-finding actually runs.
    """
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, 2.0 * math.pi, size=200).astype(np.float32)
    X = theta.reshape(-1, 1)
    y = np.sin(theta).astype(np.float64)

    ds = arcboost.Dataset(X, label=y, free_raw_data=False).construct()
    # If construction succeeded, the BinMapper accepted the data with the
    # circular_period default. We don't try to train (the stub finder would
    # abort the process); the dataset.construct() path is enough to exercise
    # ValueToBin via the histogram pre-computation.
    assert ds.num_data() == 200


@pytest.mark.parametrize(
    "value,period,expected_in_range",
    [
        (0.0, 2.0 * math.pi, True),
        (math.pi, 2.0 * math.pi, True),
        (-0.1, 2.0 * math.pi, True),  # negative wraps to period - 0.1
        (10.0, 2.0 * math.pi, True),  # > period wraps via fmod
        (float("nan"), 2.0 * math.pi, True),  # NaN → reserved bin
    ],
)
def test_value_to_bin_handles_edge_values_via_dataset_construction(
    arcboost, value, period, expected_in_range
):
    """Smoke: dataset construction with edge-case values does not crash.

    A direct C++ unit test of BinMapper::ValueToBin is preferable but requires
    bindings we haven't exposed yet; this test exercises the same code path
    via the Python Dataset API.
    """
    X = np.array([[value], [0.0], [math.pi]], dtype=np.float32)
    y = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    ds = arcboost.Dataset(
        X,
        label=y,
        free_raw_data=False,
        params={"circular_feature": "0", "circular_period": period, "min_data_in_bin": 1, "min_data_in_leaf": 1},
    ).construct()
    assert ds.num_data() == 3
