"""Minimal ArcBoost power demo: one wraparound arc, one split.

Target:
    y = 1 when theta is in [350, 360) U [0, 10], else 0

On raw theta in [0, 360), native LightGBM needs two linear thresholds to
represent that wraparound interval. ArcBoost can represent the same set as one
circular arc split: [350, 10] mod 360.

Outputs:
    examples/figures/wraparound_arc_power.png
    examples/figures/wraparound_arc_power.csv
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "examples" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Keep matplotlib from trying to write into ~/.matplotlib in sandboxed envs.
os.environ.setdefault("MPLCONFIGDIR", str(FIG_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(FIG_DIR / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

PERIOD = 360.0
ARC_HALF_WIDTH = 10.0
N_TRAIN = 6000
N_TEST = 2000
SEED = 42


def in_wraparound_arc(theta_degrees: np.ndarray) -> np.ndarray:
    """Return True inside the 20-degree arc centered on 0 degrees."""
    wrapped = np.mod(theta_degrees, PERIOD)
    return (wrapped <= ARC_HALF_WIDTH) | (wrapped >= PERIOD - ARC_HALF_WIDTH)


def make_dataset(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    theta = rng.uniform(0.0, PERIOD, size=n).astype(np.float32)
    y = in_wraparound_arc(theta).astype(np.float64)
    return theta.reshape(-1, 1), y


def train_one_tree(lgb, X: np.ndarray, y: np.ndarray, *, num_leaves: int, circular: bool):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": num_leaves,
        "learning_rate": 1.0,
        "min_data_in_leaf": 10,
        "min_data_in_bin": 1,
        "max_bin": 360,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "seed": SEED,
    }
    dataset_params = {}
    if circular:
        circular_params = {
            "circular_feature": "0",
            "circular_period": PERIOD,
            "max_bin_circular": 360,
        }
        params.update(circular_params)
        dataset_params.update(circular_params)

    train_set = lgb.Dataset(X, label=y, params=dataset_params, free_raw_data=False)
    return lgb.train(params, train_set, num_boost_round=1)


def split_count(booster) -> int:
    return sum(int(tree["num_leaves"]) - 1 for tree in booster.dump_model()["tree_info"])


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def main() -> int:
    import lightgbm as vanilla_lgb
    import arcboost
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)
    X_train, y_train = make_dataset(N_TRAIN, rng)
    X_test, y_test = make_dataset(N_TEST, rng)

    models = [
        {
            "name": "Vanilla raw theta, 1 split",
            "color": "#d55e00",
            "booster": train_one_tree(vanilla_lgb, X_train, y_train, num_leaves=2, circular=False),
        },
        {
            "name": "Vanilla raw theta, 2 splits",
            "color": "#009e73",
            "booster": train_one_tree(vanilla_lgb, X_train, y_train, num_leaves=3, circular=False),
        },
        {
            "name": "ArcBoost circular theta, 1 split",
            "color": "#0072b2",
            "booster": train_one_tree(arcboost, X_train, y_train, num_leaves=2, circular=True),
        },
    ]

    grid = np.linspace(0.0, PERIOD, 1441, endpoint=True, dtype=np.float32).reshape(-1, 1)
    y_grid = in_wraparound_arc(grid[:, 0]).astype(np.float64)

    rows = []
    for model in models:
        pred_test = model["booster"].predict(X_test)
        pred_grid = model["booster"].predict(grid)
        model["pred_grid"] = pred_grid
        model["splits"] = split_count(model["booster"])
        model["test_rmse"] = rmse(pred_test, y_test)
        model["grid_rmse"] = rmse(pred_grid, y_grid)
        rows.append(
            {
                "model": model["name"],
                "splits": model["splits"],
                "test_rmse": model["test_rmse"],
                "grid_rmse": model["grid_rmse"],
            }
        )

    csv_path = FIG_DIR / "wraparound_arc_power.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "splits", "test_rmse", "grid_rmse"])
        writer.writeheader()
        writer.writerows(rows)

    fig, (ax_curve, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(11, 4.6),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )

    x_grid = grid[:, 0]
    ax_curve.axvspan(0, ARC_HALF_WIDTH, color="#d9ead3", alpha=0.8, linewidth=0)
    ax_curve.axvspan(PERIOD - ARC_HALF_WIDTH, PERIOD, color="#d9ead3", alpha=0.8, linewidth=0)
    ax_curve.plot(x_grid, y_grid, color="#222222", linewidth=2.2, label="True wraparound arc")
    for model in models:
        linestyle = "--" if model["splits"] > 1 else "-"
        ax_curve.plot(
            x_grid,
            model["pred_grid"],
            color=model["color"],
            linewidth=2.0,
            linestyle=linestyle,
            label=f"{model['name']} (RMSE {model['test_rmse']:.3f})",
        )
    ax_curve.set_title("Same raw angle feature, tiny split budget")
    ax_curve.set_xlabel("theta (degrees)")
    ax_curve.set_ylabel("prediction")
    ax_curve.set_xlim(0, PERIOD)
    ax_curve.set_ylim(-0.08, 1.08)
    ax_curve.grid(alpha=0.22)
    ax_curve.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False)

    bar_labels = [f"{row['splits']} split" if row["splits"] == 1 else f"{row['splits']} splits" for row in rows]
    bar_colors = [model["color"] for model in models]
    bar_values = [row["test_rmse"] for row in rows]
    bars = ax_bar.bar(range(len(rows)), bar_values, color=bar_colors, width=0.68)
    ax_bar.set_title("Held-out RMSE")
    ax_bar.set_xticks(range(len(rows)), bar_labels, rotation=20, ha="right")
    ax_bar.set_ylabel("lower is better")
    ax_bar.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, bar_values, strict=False):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("ArcBoost captures a wraparound interval in one split")
    fig.tight_layout()
    fig_path = FIG_DIR / "wraparound_arc_power.png"
    fig.savefig(fig_path, dpi=150)

    for row in rows:
        print(f"{row['model']}: splits={row['splits']}, test_rmse={row['test_rmse']:.4f}")
    print(f"wrote {fig_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
