"""ArcBoost split-efficiency demo on five disjoint angular arcs.

The target is 1 inside five small angle windows and 0 elsewhere. One of the
windows straddles the 0/360-degree wraparound:

    [350, 10], [42, 60], [118, 136], [205, 225], [286, 305]

ArcBoost can peel off one circular arc per split, so five arcs need about five
splits. Native LightGBM on raw theta only has linear thresholds, so it has to
learn both ends of each window and needs roughly twice the split budget.

Outputs:
    examples/figures/multi_arc_efficiency.png
    examples/figures/multi_arc_efficiency.csv
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
ARCS = [
    (350.0, 10.0),
    (42.0, 60.0),
    (118.0, 136.0),
    (205.0, 225.0),
    (286.0, 305.0),
]
N_TRAIN = 15_000
N_TEST = 5_000
SEED = 7
MAX_SPLITS = 12


def in_any_arc(theta_degrees: np.ndarray) -> np.ndarray:
    theta = np.mod(theta_degrees, PERIOD)
    mask = np.zeros(theta.shape, dtype=bool)
    for start, end in ARCS:
        if start <= end:
            mask |= (theta >= start) & (theta <= end)
        else:
            mask |= (theta >= start) | (theta <= end)
    return mask


def make_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, PERIOD, size=n).astype(np.float32)
    y = in_any_arc(theta).astype(np.float64)
    return theta.reshape(-1, 1), y


def train_one_tree(lgb, X: np.ndarray, y: np.ndarray, split_budget: int, *, circular: bool):
    # use_missing=False so num_real_bins == max_bin (no NaN bin reserved). With
    # max_bin / max_bin_circular = 360 and PERIOD = 360, that gives bin width
    # exactly 1.0 deg, aligned with integer-degree arc edges. With the default
    # use_missing=True, one bin is reserved for NaN and bin width becomes
    # 360/359 ~ 1.003 deg, which puts arc edges INSIDE bins and creates mixed-
    # content leaves whose soft mean predictions cap RMSE at ~0.07 rather than
    # the algorithmic zero. The point of this demo is split-count efficiency,
    # not boundary precision, so we eliminate that bin-quantization residue.
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": split_budget + 1,
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
    dataset_params = {"use_missing": False}
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
    import arcboost
    import lightgbm as vanilla_lgb
    import matplotlib.pyplot as plt

    X_train, y_train = make_dataset(N_TRAIN, SEED)
    X_test, y_test = make_dataset(N_TEST, SEED + 1)
    grid = np.linspace(0.0, PERIOD, 1441, dtype=np.float32).reshape(-1, 1)
    y_grid = in_any_arc(grid[:, 0]).astype(np.float64)

    curves = {
        "Vanilla raw theta": {
            "color": "#d55e00",
            "library": vanilla_lgb,
            "circular": False,
            "rows": [],
        },
        "ArcBoost circular theta": {
            "color": "#0072b2",
            "library": arcboost,
            "circular": True,
            "rows": [],
        },
    }
    prediction_at_five = {}

    for split_budget in range(1, MAX_SPLITS + 1):
        for name, spec in curves.items():
            booster = train_one_tree(
                spec["library"],
                X_train,
                y_train,
                split_budget,
                circular=spec["circular"],
            )
            actual_splits = split_count(booster)
            test_rmse = rmse(booster.predict(X_test), y_test)
            spec["rows"].append(
                {
                    "model": name,
                    "requested_splits": split_budget,
                    "actual_splits": actual_splits,
                    "test_rmse": test_rmse,
                }
            )
            if split_budget == 5:
                prediction_at_five[name] = booster.predict(grid)

    csv_path = FIG_DIR / "multi_arc_efficiency.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "requested_splits", "actual_splits", "test_rmse"],
        )
        writer.writeheader()
        for spec in curves.values():
            writer.writerows(spec["rows"])

    fig, (ax_pred, ax_curve) = plt.subplots(2, 1, figsize=(11, 8.2), sharex=False)

    x_grid = grid[:, 0]
    for start, end in ARCS:
        if start <= end:
            ax_pred.axvspan(start, end, color="#d9ead3", alpha=0.8, linewidth=0)
        else:
            ax_pred.axvspan(0, end, color="#d9ead3", alpha=0.8, linewidth=0)
            ax_pred.axvspan(start, PERIOD, color="#d9ead3", alpha=0.8, linewidth=0)

    ax_pred.plot(x_grid, y_grid, color="#202020", linewidth=2.1, label="True five-arc target")
    for name, pred in prediction_at_five.items():
        color = curves[name]["color"]
        linestyle = "-" if "ArcBoost" in name else "--"
        row = curves[name]["rows"][4]
        ax_pred.plot(
            x_grid,
            pred,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=f"{name}, 5 splits (RMSE {row['test_rmse']:.3f})",
        )
    ax_pred.set_title("Five disjoint angle windows at the same 5-split budget")
    ax_pred.set_xlim(0.0, PERIOD)
    ax_pred.set_ylim(-0.08, 1.08)
    ax_pred.set_ylabel("prediction")
    ax_pred.grid(alpha=0.22)
    ax_pred.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)

    for name, spec in curves.items():
        xs = [row["actual_splits"] for row in spec["rows"]]
        ys = [row["test_rmse"] for row in spec["rows"]]
        ax_curve.plot(xs, ys, marker="o", linewidth=2.1, color=spec["color"], label=name)
    ax_curve.axvline(5, color="#202020", alpha=0.35, linewidth=1.2)
    ax_curve.text(5.15, 0.41, "5 target arcs", color="#202020", fontsize=10)
    ax_curve.set_title("Held-out RMSE versus one-tree split budget")
    ax_curve.set_xlabel("actual split count")
    ax_curve.set_ylabel("RMSE")
    ax_curve.set_xlim(1, MAX_SPLITS)
    ax_curve.set_ylim(0.0, 0.46)
    ax_curve.grid(alpha=0.22)
    ax_curve.legend(frameon=False)

    fig.suptitle("ArcBoost uses one circular split per arc; raw LightGBM learns two edges per arc")
    fig.tight_layout()
    fig_path = FIG_DIR / "multi_arc_efficiency.png"
    fig.savefig(fig_path, dpi=150)

    for name, spec in curves.items():
        row5 = spec["rows"][4]
        best_under_5 = min(row["test_rmse"] for row in spec["rows"] if row["actual_splits"] <= 5)
        print(f"{name}: RMSE at 5 splits = {row5['test_rmse']:.4f}; best <=5 splits = {best_under_5:.4f}")
    print(f"wrote {fig_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
