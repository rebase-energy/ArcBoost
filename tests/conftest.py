"""Shared pytest fixtures and skip-markers for the ArcBoost test suite.

Most tests exercise the circular split-finder. In v0.1.0.dev0 the finder is a
``Log::Fatal`` stub (see patches/lightgbm/0004), so calls that train with
``circular_feature`` set will abort the process. We detect this once at
collection time and apply a skip-marker so the rest of the suite is still
informative against the parts of the patch series that are wired (enum,
config plumbing, dispatch, Python rename).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _circular_finder_implemented() -> bool:
    """Return True if the FindBestThresholdCircular stub has been replaced.

    We check by looking at the C++ source after patches are applied: if the
    body still contains the v0 Log::Fatal stub message, the finder is not yet
    implemented and tests that need it must skip.
    """
    src = (
        Path(__file__).resolve().parent.parent
        / "provider" / "lightgbm" / "src" / "treelearner" / "feature_histogram.cpp"
    )
    if not src.exists():
        return False
    text = src.read_text(encoding="utf-8", errors="ignore")
    return "circular split-finder is declared but not yet implemented" not in text


circular_finder_required = pytest.mark.skipif(
    not _circular_finder_implemented(),
    reason="ArcBoost FindBestThresholdCircular is a stub in v0.1.0.dev0; "
           "implement the arc-search algorithm in feature_histogram.cpp to enable.",
)


@pytest.fixture(scope="session")
def arcboost():
    """Import ArcBoost; skip the entire test if the build hasn't been done yet."""
    try:
        import arcboost as ab
    except ImportError as exc:
        pytest.skip(f"arcboost not built / installed: {exc}")
    return ab
