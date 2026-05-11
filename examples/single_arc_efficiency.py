"""ArcBoost vs vanilla LightGBM on a single arc, single rotation.

The target is one binary 30-degree arc on a 360-degree circle, centered at
70 degrees (deliberately off-axis from both the sin and cos coordinate axes
so neither aligns naturally with a single threshold):

    y = 1.0  if  theta in [55, 85] degrees,
        0.0  otherwise.

No noise, no wraparound, no sub-period folding. circular_period = 360, the
full single rotation.

Why this is the simplest demonstration of ArcBoost's structural advantage:

  - **ArcBoost** finds the arc as a single split. RMSE -> 0 at 1 split.
  - **Vanilla LightGBM on raw theta** has to bracket the arc with two
    threshold splits (one per edge). RMSE -> 0 at 2 splits.
  - **Vanilla LightGBM on (sin theta, cos theta)** also needs two splits:
    the arc center at 70 degrees is off-axis, so a single axis-aligned split
    in (sin, cos) space (sin > c1) captures a 180-degree half-plane, much
    wider than the 30-degree arc. Two intersected bounds (sin > c1 AND
    cos > c2) are needed to isolate it. RMSE -> 0 at 2 splits.

The 2-to-1 ratio is the irreducible cost of bracketing both edges of an arc
with linear thresholds. It compounds linearly with the number of arcs but
it's already visible on the very simplest case: one arc.

Outputs:
    examples/figures/single_arc_efficiency.png
    examples/figures/single_arc_efficiency.csv
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

PERIOD = 360.0
ARC_CENTER = 70.0
ARC_HALF_WIDTH = 15.0
ARC_START = ARC_CENTER - ARC_HALF_WIDTH
ARC_END = ARC_CENTER + ARC_HALF_WIDTH

N_TRAIN = 15_000
N_TEST = 5_000
SEED = 7
MAX_SPLITS = 8


def in_arc(theta_degrees: np.ndarray) -> np.ndarray:
    theta = np.mod(theta_degrees, PERIOD)
    return (theta >= ARC_START) & (theta <= ARC_END)


def make_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, PERIOD, size=n).astype(np.float32)
    y = in_arc(theta).astype(np.float64)
    return theta, y


def to_sincos(theta_degrees: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(theta_degrees)
    return np.column_stack([np.sin(radians), np.cos(radians)]).astype(np.float32)


def common_lgbm_params(num_leaves: int) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": num_leaves,
        "learning_rate": 1.0,
        "min_data_in_leaf": 5,
        "min_data_in_bin": 1,
        "max_bin": 360,
        "use_missing": False,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "seed": SEED,
    }


def train_arcboost(arcboost, theta: np.ndarray, y: np.ndarray, num_leaves: int):
    X = theta.reshape(-1, 1)
    params = common_lgbm_params(num_leaves)
    params.update(
        {
            "circular_feature": "0",
            "circular_period": PERIOD,
            "max_bin_circular": 360,
        }
    )
    dataset_params = {
        "use_missing": False,
        "circular_feature": "0",
        "circular_period": PERIOD,
        "max_bin_circular": 360,
    }
    train_set = arcboost.Dataset(X, label=y, params=dataset_params, free_raw_data=False)
    return arcboost.train(params, train_set, num_boost_round=1)


def train_vanilla(lgb, X: np.ndarray, y: np.ndarray, num_leaves: int):
    params = common_lgbm_params(num_leaves)
    train_set = lgb.Dataset(
        X, label=y, params={"use_missing": False}, free_raw_data=False
    )
    return lgb.train(params, train_set, num_boost_round=1)


def split_count(booster) -> int:
    return sum(int(tree["num_leaves"]) - 1 for tree in booster.dump_model()["tree_info"])


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def main() -> int:
    import arcboost
    import lightgbm as vanilla_lgb
    import matplotlib.pyplot as plt

    theta_train, y_train = make_dataset(N_TRAIN, SEED)
    theta_test, y_test = make_dataset(N_TEST, SEED + 1)

    grid_theta = np.linspace(0.0, PERIOD, 1441, dtype=np.float32)
    y_grid = in_arc(grid_theta).astype(np.float64)

    methods = {
        "ArcBoost circular theta": {
            "color": "#0072b2",
            "linestyle": "-",
            "fit": lambda nl: train_arcboost(arcboost, theta_train, y_train, nl),
            "predict": lambda b, theta: b.predict(theta.reshape(-1, 1)),
            "rows": [],
        },
        "Vanilla raw theta": {
            "color": "#d55e00",
            "linestyle": "--",
            "fit": lambda nl: train_vanilla(
                vanilla_lgb, theta_train.reshape(-1, 1), y_train, nl
            ),
            "predict": lambda b, theta: b.predict(theta.reshape(-1, 1)),
            "rows": [],
        },
        "Vanilla (sin theta, cos theta)": {
            "color": "#009e73",
            "linestyle": ":",
            "fit": lambda nl: train_vanilla(
                vanilla_lgb, to_sincos(theta_train), y_train, nl
            ),
            "predict": lambda b, theta: b.predict(to_sincos(theta)),
            "rows": [],
        },
    }
    prediction_at_one = {}

    for split_budget in range(1, MAX_SPLITS + 1):
        for name, spec in methods.items():
            booster = spec["fit"](split_budget + 1)
            actual_splits = split_count(booster)
            preds = spec["predict"](booster, theta_test)
            test_rmse = rmse(preds, y_test)
            spec["rows"].append(
                {
                    "model": name,
                    "requested_splits": split_budget,
                    "actual_splits": actual_splits,
                    "test_rmse": test_rmse,
                }
            )
            if split_budget == 1:
                prediction_at_one[name] = spec["predict"](booster, grid_theta)

    csv_path = FIG_DIR / "single_arc_efficiency.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "requested_splits", "actual_splits", "test_rmse"],
        )
        writer.writeheader()
        for spec in methods.values():
            writer.writerows(spec["rows"])

    fig, (ax_pred, ax_curve) = plt.subplots(2, 1, figsize=(11, 8.4))

    ax_pred.axvspan(ARC_START, ARC_END, color="#d9ead3", alpha=0.85, linewidth=0)
    ax_pred.plot(grid_theta, y_grid, color="#202020", linewidth=2.2, label="True arc target")
    for name, spec in methods.items():
        if name not in prediction_at_one:
            continue
        row1 = spec["rows"][0]
        ax_pred.plot(
            grid_theta,
            prediction_at_one[name],
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=1.9,
            label=f"{name}, 1 split (RMSE {row1['test_rmse']:.3f})",
        )
    ax_pred.set_title(
        f"Single arc on a single rotation (center={ARC_CENTER:.0f} deg, "
        f"width={2*ARC_HALF_WIDTH:.0f} deg) at the same 1-split budget"
    )
    ax_pred.set_xlim(0.0, PERIOD)
    ax_pred.set_ylim(-0.10, 1.10)
    ax_pred.set_xlabel("theta (degrees)")
    ax_pred.set_ylabel("prediction")
    ax_pred.grid(alpha=0.22)
    ax_pred.legend(loc="upper right", frameon=False)

    for name, spec in methods.items():
        xs = [row["actual_splits"] for row in spec["rows"]]
        ys = [row["test_rmse"] for row in spec["rows"]]
        ax_curve.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.1,
            color=spec["color"],
            linestyle=spec["linestyle"],
            label=name,
        )
    ax_curve.axvline(1, color="#0072b2", alpha=0.30, linewidth=1.0)
    ax_curve.axvline(2, color="#202020", alpha=0.30, linewidth=1.0)
    ax_curve.text(1.05, 0.45, "ArcBoost converges", color="#0072b2", fontsize=10)
    ax_curve.text(2.05, 0.40, "linear baselines converge", color="#202020", fontsize=10)
    ax_curve.set_title("Held-out RMSE versus one-tree split budget")
    ax_curve.set_xlabel("actual split count")
    ax_curve.set_ylabel("RMSE")
    ax_curve.set_xlim(1, MAX_SPLITS)
    ax_curve.set_ylim(-0.02, 0.50)
    ax_curve.grid(alpha=0.22)
    ax_curve.legend(frameon=False)

    fig.suptitle(
        "ArcBoost: 1 arc-split per arc. Vanilla LightGBM (raw theta or sin/cos): 2 threshold-splits per arc."
    )
    fig.tight_layout()
    fig_path = FIG_DIR / "single_arc_efficiency.png"
    fig.savefig(fig_path, dpi=150)

    print()
    for name, spec in methods.items():
        for row in spec["rows"][:3]:
            print(
                f"  {name:32s} splits={row['actual_splits']}  RMSE={row['test_rmse']:.5f}"
            )
        print()
    print(f"wrote {fig_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
