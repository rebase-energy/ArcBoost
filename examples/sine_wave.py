"""Predict a sine wave: ArcBoost vs vanilla LightGBM with (sin x, cos x) features.

Target: y = sin(x) + small Gaussian noise, x ~ Uniform[0, 2*pi).

Models:
  - **ArcBoost**: x as a single circular feature (period = 2*pi). No feature
    engineering. The booster has to discover the sine shape from arcs of x.
  - **Vanilla LightGBM** with the two features (sin x, cos x). This is the
    standard "feature-engineered" baseline; sin(x) is literally one of the
    inputs, so the booster has the perfect representation handed to it.

Expected story:
  Vanilla (sin, cos) should win — sin(x) IS in the feature set, so each
  threshold on it is a clean axis-aligned cut against the target. ArcBoost
  has to approximate the smooth sine curve by piecewise-constant leaf
  outputs over bins of x, so its noise floor is bounded below by the
  bin-discretization variance of sin within a bin (roughly pi/(B*sqrt(3))
  for B uniform bins on [0, 2*pi)).

  This demo measures *how close* ArcBoost gets without manual feature
  engineering. Both models are given the same boosting budget and bin
  resolution per feature.

Outputs:
    examples/figures/sine_wave.png
    examples/figures/sine_wave.csv
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "examples" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(FIG_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(FIG_DIR / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

PERIOD = 2.0 * np.pi
N_TRAIN = 5_000
N_TEST = 2_000
N_TREES = 200
NUM_LEAVES = 8
LEARNING_RATE = 0.1
MIN_DATA_IN_LEAF = 20
MAX_BIN = 64  # per feature; both methods get the same per-feature resolution
NOISE_STD = 0.05
SEED = 7


def make_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, PERIOD, size=n).astype(np.float32)
    y = (np.sin(x) + rng.normal(0.0, NOISE_STD, size=n)).astype(np.float64)
    return x, y


def common_params() -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": NUM_LEAVES,
        "learning_rate": LEARNING_RATE,
        "min_data_in_leaf": MIN_DATA_IN_LEAF,
        "min_data_in_bin": 1,
        "max_bin": MAX_BIN,
        "use_missing": False,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "seed": SEED,
    }


def train_arcboost(arcboost, x: np.ndarray, y: np.ndarray):
    X = x.reshape(-1, 1)
    params = common_params()
    params.update(
        {
            "circular_feature": "0",
            "circular_period": PERIOD,
            "max_bin_circular": MAX_BIN,
        }
    )
    dataset_params = {
        "use_missing": False,
        "circular_feature": "0",
        "circular_period": PERIOD,
        "max_bin_circular": MAX_BIN,
    }
    ds = arcboost.Dataset(X, label=y, params=dataset_params, free_raw_data=False)
    return arcboost.train(params, ds, num_boost_round=N_TREES)


def train_vanilla(lgb, x: np.ndarray, y: np.ndarray):
    X = np.column_stack([np.sin(x), np.cos(x)]).astype(np.float32)
    params = common_params()
    ds = lgb.Dataset(X, label=y, params={"use_missing": False}, free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=N_TREES)


def rmse_per_round(booster, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.empty(N_TREES, dtype=np.float64)
    for k in range(1, N_TREES + 1):
        pred = booster.predict(X, num_iteration=k)
        out[k - 1] = float(np.sqrt(np.mean((pred - y) ** 2)))
    return out


def cumulative_splits(booster) -> np.ndarray:
    dump = booster.dump_model()
    per_tree = np.array(
        [int(t["num_leaves"]) - 1 for t in dump["tree_info"]], dtype=np.int64
    )
    return np.cumsum(per_tree)


def main() -> int:
    import arcboost
    import lightgbm as vanilla_lgb
    import matplotlib.pyplot as plt

    x_train, y_train = make_dataset(N_TRAIN, SEED)
    x_test, y_test = make_dataset(N_TEST, SEED + 1)

    grid = np.linspace(0.0, PERIOD, 1000, dtype=np.float32)
    y_grid_true = np.sin(grid)

    ab_booster = train_arcboost(arcboost, x_train, y_train)
    v_booster = train_vanilla(vanilla_lgb, x_train, y_train)

    # Per-round RMSE on test set.
    ab_rmse = rmse_per_round(ab_booster, x_test.reshape(-1, 1), y_test)
    sincos_test = np.column_stack([np.sin(x_test), np.cos(x_test)]).astype(np.float32)
    v_rmse = rmse_per_round(v_booster, sincos_test, y_test)

    # Cumulative split count per tree.
    ab_splits = cumulative_splits(ab_booster)
    v_splits = cumulative_splits(v_booster)

    # Predictions on a fine grid using the full ensemble.
    ab_grid_pred = ab_booster.predict(grid.reshape(-1, 1))
    sincos_grid = np.column_stack([np.sin(grid), np.cos(grid)]).astype(np.float32)
    v_grid_pred = v_booster.predict(sincos_grid)

    csv_path = FIG_DIR / "sine_wave.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["model", "n_trees", "cum_splits", "test_rmse"]
        )
        writer.writeheader()
        for n in range(1, N_TREES + 1):
            writer.writerow(
                {
                    "model": "ArcBoost",
                    "n_trees": n,
                    "cum_splits": int(ab_splits[n - 1]),
                    "test_rmse": float(ab_rmse[n - 1]),
                }
            )
            writer.writerow(
                {
                    "model": "Vanilla (sin x, cos x)",
                    "n_trees": n,
                    "cum_splits": int(v_splits[n - 1]),
                    "test_rmse": float(v_rmse[n - 1]),
                }
            )

    fig, (ax_pred, ax_curve) = plt.subplots(2, 1, figsize=(11, 8.4))

    ax_pred.plot(grid, y_grid_true, color="#202020", linewidth=2.2, label="True sin(x)")
    ax_pred.plot(
        grid,
        ab_grid_pred,
        color="#0072b2",
        linewidth=1.7,
        label=f"ArcBoost (RMSE {ab_rmse[-1]:.4f})",
    )
    ax_pred.plot(
        grid,
        v_grid_pred,
        color="#d55e00",
        linestyle="--",
        linewidth=1.7,
        label=f"Vanilla LightGBM with (sin x, cos x) (RMSE {v_rmse[-1]:.4f})",
    )
    ax_pred.set_title(
        f"Predictions on y = sin(x) after {N_TREES} boosting rounds "
        f"(noise std = {NOISE_STD})"
    )
    ax_pred.set_xlabel("x")
    ax_pred.set_ylabel("y")
    ax_pred.set_xlim(0.0, PERIOD)
    ax_pred.set_ylim(-1.25, 1.25)
    ax_pred.grid(alpha=0.25)
    ax_pred.legend(loc="lower left", frameon=False)

    # Noise floor reference.
    ax_curve.axhline(
        NOISE_STD,
        color="#888888",
        linestyle=":",
        linewidth=1.4,
        label=f"noise floor ({NOISE_STD:.3f})",
    )
    ax_curve.plot(
        ab_splits, ab_rmse, color="#0072b2", linewidth=2.0, label="ArcBoost"
    )
    ax_curve.plot(
        v_splits,
        v_rmse,
        color="#d55e00",
        linestyle="--",
        linewidth=2.0,
        label="Vanilla LightGBM with (sin x, cos x)",
    )
    ax_curve.set_xscale("log")
    ax_curve.set_xlabel("cumulative split count")
    ax_curve.set_ylabel("held-out RMSE")
    ax_curve.set_title("Convergence")
    ax_curve.set_ylim(0.0, max(0.55, float(max(ab_rmse[0], v_rmse[0]) * 1.05)))
    ax_curve.grid(alpha=0.25, which="both")
    ax_curve.legend(loc="upper right", frameon=False)

    fig.suptitle(
        "Sine wave: ArcBoost on circular x vs vanilla LightGBM on (sin x, cos x)"
    )
    fig.tight_layout()
    fig_path = FIG_DIR / "sine_wave.png"
    fig.savefig(fig_path, dpi=150)

    print()
    print(f"  ArcBoost                 final RMSE = {ab_rmse[-1]:.5f}")
    print(f"  Vanilla (sin x, cos x)   final RMSE = {v_rmse[-1]:.5f}")
    print(f"  noise floor              {NOISE_STD:.5f}")
    print(
        f"  ArcBoost residual above noise: {max(0.0, ab_rmse[-1] - NOISE_STD):.5f}"
    )
    print(
        f"  Vanilla  residual above noise: {max(0.0, v_rmse[-1] - NOISE_STD):.5f}"
    )
    print(f"wrote {fig_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
