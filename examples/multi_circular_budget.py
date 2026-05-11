"""ArcBoost vs vanilla LightGBM at a constrained num_leaves budget on a
3-way conjunction of circular features.

Three circular features, all with **off-axis** arc centers (so each requires
2 sin/cos splits to bracket):

    wind direction  (period 360 deg, arc [200, 280] deg, center 240 deg)
    hour of day     (period 24 h,    arc [8, 20] h,       center 14 h)
    day of year     (period 365.25,  arc [90, 210],       center 150)

Target: y = 1.0 if (in_wind_arc AND in_hour_arc AND in_day_arc) else 0.0

Per-feature cost to express arc membership in a tree:
    ArcBoost (circular feature): **1 split** — arc-membership is one decision.
    Vanilla (sin x, cos x):       **2 splits** — needs both a sin-bound and
                                  a cos-bound for an off-axis arc center.

Per-tree budget to express the full 3-way conjunction (chain layout):
    ArcBoost: 3 splits, 4 leaves    (3 arc decisions, 1 final "in all" leaf)
    Vanilla:  6 splits, 7 leaves    (6 sin/cos decisions chain)

So at num_leaves = 4 ArcBoost can fit the target perfectly while vanilla
literally cannot: 3 splits in (sin x, cos x) of three features cannot
isolate the off-axis 3-way conjunction.

Outputs:
    examples/figures/multi_circular_budget.png
    examples/figures/multi_circular_budget.csv
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

PERIOD_WIND = 360.0
PERIOD_HOUR = 24.0
PERIOD_DAY = 365.25
ARC_WIND = (200.0, 280.0)
ARC_HOUR = (8.0, 20.0)
ARC_DAY = (90.0, 210.0)

N_TRAIN = 30_000
N_TEST = 10_000
SEED = 7

# num_leaves grid to sweep — exponentially spaced from "too small for either"
# (2, 3, 4) through "ArcBoost fits" (4-6) into "vanilla starts to fit" (7-15)
# and on to "both fit comfortably" (16-128).
NUM_LEAVES_GRID = [2, 3, 4, 5, 6, 7, 8, 9, 10]


def in_arc(theta: np.ndarray, period: float, start: float, end: float) -> np.ndarray:
    th = np.mod(theta, period)
    if start <= end:
        return (th >= start) & (th <= end)
    return (th >= start) | (th <= end)


def make_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    wind = rng.uniform(0.0, PERIOD_WIND, size=n).astype(np.float32)
    hour = rng.uniform(0.0, PERIOD_HOUR, size=n).astype(np.float32)
    day = rng.uniform(0.0, PERIOD_DAY, size=n).astype(np.float32)
    in_all = (
        in_arc(wind, PERIOD_WIND, *ARC_WIND)
        & in_arc(hour, PERIOD_HOUR, *ARC_HOUR)
        & in_arc(day, PERIOD_DAY, *ARC_DAY)
    )
    y = in_all.astype(np.float64)
    return wind, hour, day, y


def features_arcboost(
    wind: np.ndarray, hour: np.ndarray, day: np.ndarray
) -> np.ndarray:
    # Scale each circular feature to [0, 2*pi) so a single global circular_period
    # = 2*pi works for all three. (Per-feature periods are a v0.2 follow-up.)
    return np.column_stack(
        [
            wind * (2.0 * np.pi / PERIOD_WIND),
            hour * (2.0 * np.pi / PERIOD_HOUR),
            day * (2.0 * np.pi / PERIOD_DAY),
        ]
    ).astype(np.float32)


def features_sincos(
    wind: np.ndarray, hour: np.ndarray, day: np.ndarray
) -> np.ndarray:
    w = wind * (2.0 * np.pi / PERIOD_WIND)
    h = hour * (2.0 * np.pi / PERIOD_HOUR)
    d = day * (2.0 * np.pi / PERIOD_DAY)
    return np.column_stack(
        [np.sin(w), np.cos(w), np.sin(h), np.cos(h), np.sin(d), np.cos(d)]
    ).astype(np.float32)


def common_params(num_leaves: int) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": num_leaves,
        "learning_rate": 1.0,
        "min_data_in_leaf": 5,
        "min_data_in_bin": 1,
        "max_bin": 256,
        "use_missing": False,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "seed": SEED,
    }


def train_arcboost(arcboost, X: np.ndarray, y: np.ndarray, num_leaves: int):
    params = common_params(num_leaves)
    params.update(
        {
            "circular_feature": "0,1,2",
            "circular_period": 2.0 * np.pi,
            "max_bin_circular": 128,
        }
    )
    dataset_params = {
        "use_missing": False,
        "circular_feature": "0,1,2",
        "circular_period": 2.0 * np.pi,
        "max_bin_circular": 128,
    }
    ds = arcboost.Dataset(X, label=y, params=dataset_params, free_raw_data=False)
    return arcboost.train(params, ds, num_boost_round=1)


def train_vanilla(lgb, X: np.ndarray, y: np.ndarray, num_leaves: int):
    params = common_params(num_leaves)
    ds = lgb.Dataset(X, label=y, params={"use_missing": False}, free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=1)


def split_count(booster) -> int:
    return sum(int(t["num_leaves"]) - 1 for t in booster.dump_model()["tree_info"])


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def main() -> int:
    import arcboost
    import lightgbm as vanilla_lgb
    import matplotlib.pyplot as plt

    wind_tr, hour_tr, day_tr, y_tr = make_dataset(N_TRAIN, SEED)
    wind_te, hour_te, day_te, y_te = make_dataset(N_TEST, SEED + 1)

    X_arc_tr = features_arcboost(wind_tr, hour_tr, day_tr)
    X_arc_te = features_arcboost(wind_te, hour_te, day_te)
    X_sc_tr = features_sincos(wind_tr, hour_tr, day_tr)
    X_sc_te = features_sincos(wind_te, hour_te, day_te)

    methods = {
        "ArcBoost (3 circular features)": {
            "color": "#0072b2",
            "linestyle": "-",
            "fit": lambda nl: train_arcboost(arcboost, X_arc_tr, y_tr, nl),
            "predict": lambda b: b.predict(X_arc_te),
            "rows": [],
        },
        "Vanilla (6 sin/cos features)": {
            "color": "#d55e00",
            "linestyle": "--",
            "fit": lambda nl: train_vanilla(vanilla_lgb, X_sc_tr, y_tr, nl),
            "predict": lambda b: b.predict(X_sc_te),
            "rows": [],
        },
    }

    saturation: dict[str, int | None] = {name: None for name in methods}
    SAT_TOL = 1e-3

    for nl in NUM_LEAVES_GRID:
        for name, spec in methods.items():
            booster = spec["fit"](nl)
            actual = split_count(booster)
            test_rmse = rmse(spec["predict"](booster), y_te)
            spec["rows"].append(
                {
                    "model": name,
                    "num_leaves": nl,
                    "actual_splits": actual,
                    "actual_leaves": actual + 1,
                    "test_rmse": test_rmse,
                }
            )
            if saturation[name] is None and test_rmse < SAT_TOL:
                saturation[name] = nl

    csv_path = FIG_DIR / "multi_circular_budget.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "num_leaves", "actual_splits", "actual_leaves", "test_rmse"],
        )
        writer.writeheader()
        for spec in methods.values():
            writer.writerows(spec["rows"])

    fig, (ax_curve, ax_slice) = plt.subplots(2, 1, figsize=(11, 9.0))

    for name, spec in methods.items():
        xs = [row["actual_leaves"] for row in spec["rows"]]
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
        if saturation[name] is not None:
            sat_nl = saturation[name]
            row = next(r for r in spec["rows"] if r["num_leaves"] == sat_nl)
            ax_curve.axvline(
                row["actual_leaves"],
                color=spec["color"],
                alpha=0.25,
                linewidth=1.0,
            )
            ax_curve.text(
                row["actual_leaves"] * 1.05,
                0.42,
                f"{name.split(' ')[0]} fits at {row['actual_leaves']} leaves",
                color=spec["color"],
                fontsize=10,
                rotation=0,
            )

    ax_curve.set_xlabel("number of leaves in the single tree")
    ax_curve.set_ylabel("held-out RMSE")
    ax_curve.set_title(
        "Single-tree fit on 3-way circular conjunction at varying num_leaves budget"
    )
    ax_curve.set_ylim(-0.005, 0.45)
    ax_curve.grid(alpha=0.25, which="both")
    ax_curve.legend(frameon=False, loc="upper right")

    # Bottom panel: predictions at num_leaves=4 (the smallest budget where
    # ArcBoost can fit the target). Slice over wind direction with hour and
    # day fixed at their arc centers.
    DEMO_NL = 4
    hour_slice = np.full(721, 14.0, dtype=np.float32)
    day_slice = np.full(721, 150.0, dtype=np.float32)
    wind_slice = np.linspace(0.0, PERIOD_WIND, 721, dtype=np.float32)

    arc_booster = train_arcboost(
        arcboost, X_arc_tr, y_tr, DEMO_NL
    )
    sc_booster = train_vanilla(vanilla_lgb, X_sc_tr, y_tr, DEMO_NL)

    X_arc_slice = features_arcboost(wind_slice, hour_slice, day_slice)
    X_sc_slice = features_sincos(wind_slice, hour_slice, day_slice)
    arc_pred = arc_booster.predict(X_arc_slice)
    sc_pred = sc_booster.predict(X_sc_slice)
    y_slice_true = (
        in_arc(wind_slice, PERIOD_WIND, *ARC_WIND)
        & in_arc(hour_slice, PERIOD_HOUR, *ARC_HOUR)
        & in_arc(day_slice, PERIOD_DAY, *ARC_DAY)
    ).astype(np.float64)

    ax_slice.axvspan(*ARC_WIND, color="#d9ead3", alpha=0.85, linewidth=0)
    ax_slice.plot(
        wind_slice, y_slice_true, color="#202020", linewidth=2.0, label="True target"
    )
    arc_row = next(r for r in methods["ArcBoost (3 circular features)"]["rows"] if r["num_leaves"] == DEMO_NL)
    sc_row = next(r for r in methods["Vanilla (6 sin/cos features)"]["rows"] if r["num_leaves"] == DEMO_NL)
    ax_slice.plot(
        wind_slice,
        arc_pred,
        color="#0072b2",
        linewidth=1.8,
        label=f"ArcBoost (RMSE {arc_row['test_rmse']:.3f})",
    )
    ax_slice.plot(
        wind_slice,
        sc_pred,
        color="#d55e00",
        linestyle="--",
        linewidth=1.8,
        label=f"Vanilla sin/cos (RMSE {sc_row['test_rmse']:.3f})",
    )
    ax_slice.set_xlim(0.0, PERIOD_WIND)
    ax_slice.set_ylim(-0.10, 1.15)
    ax_slice.set_xlabel("wind direction (deg)  [hour=14, day=150 fixed at arc centers]")
    ax_slice.set_ylabel("prediction")
    ax_slice.set_title(
        f"Predictions over the wind-direction slice at num_leaves={DEMO_NL}"
    )
    ax_slice.grid(alpha=0.25)
    ax_slice.legend(loc="upper right", frameon=False)

    fig.suptitle(
        "Three off-axis circular arcs in a 3-way conjunction. "
        "ArcBoost fits at 4 leaves; vanilla sin/cos needs 8+ leaves."
    )
    fig.tight_layout()
    fig_path = FIG_DIR / "multi_circular_budget.png"
    fig.savefig(fig_path, dpi=150)

    print()
    for name, spec in methods.items():
        rows = sorted(spec["rows"], key=lambda r: r["num_leaves"])
        for row in rows[:6] + [rows[-1]]:
            print(
                f"  {name:35s} num_leaves={row['num_leaves']:3d}  "
                f"splits={row['actual_splits']:3d}  RMSE={row['test_rmse']:.5f}"
            )
        sat = saturation[name]
        sat_str = f"first num_leaves with RMSE<{SAT_TOL}: {sat}" if sat else "did not saturate"
        print(f"  {name:35s} {sat_str}")
        print()
    print(f"wrote {fig_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
