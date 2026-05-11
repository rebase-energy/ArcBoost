"""ArcBoost vs native LightGBM: splits-to-error efficiency on a 1D circular task.

Generates a synthetic regression dataset where the target is the union of two
disjoint arcs on the circle:

    Arc A: 60 degrees wide, *straddling* the 0/2*pi wraparound (centered at 0)
    Arc B: 60 degrees wide, centered at 3*pi/4 (off-axis: not aligned with
           either the sin or the cos coordinate)

    y = 1.0 if theta in (Arc A) U (Arc B), else 0.0, + Normal(0, 0.05)

Why these two arcs:
  - Arc A is the wraparound case: a single linear threshold on raw theta
    cannot express it. Native LightGBM with raw theta needs 2 splits per
    leaf to bracket the discontinuity at 0/2*pi.
  - Arc B is off-axis: sin/cos preprocessing must use *both* a sin-bound
    AND a cos-bound to isolate it (an axis-aligned rectangle in (sin, cos)
    space at this angle is a quadrant strip, not a single threshold).
  - ArcBoost gets each arc in 1 split — 2 splits total covers the full
    target topology, vs. ~4 splits per leaf for the linear baselines.

Three models are compared:

    1. ArcBoost with circular_feature=['theta']  (the contender — falls back
       to a clear runtime error in v0.1.0.dev0 because the arc-search finder
       is still a stub. Skipped automatically when not yet implemented.)
    2. Native LightGBM, raw theta as numeric                (worst-case baseline)
    3. Native LightGBM, [sin(theta), cos(theta)] features   (standard baseline)

For each model we train 500 boosting rounds at num_leaves=31, learning_rate=0.05,
log cumulative split count and held-out RMSE after every tree, and plot the
resulting Pareto curve (RMSE vs cumulative splits, log-x) averaged across 5
seeds with a stderr band.

Outputs:
    examples/figures/circular_efficiency.png
    examples/figures/circular_efficiency.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "examples" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 10_000
N_TREES = 500
NUM_LEAVES = 31
LEARNING_RATE = 0.05
MIN_DATA_IN_LEAF = 20
N_SEEDS = 5
NOISE_STD = 0.05
# Two non-overlapping arcs. Arc A straddles theta=0 (wraparound case).
# Arc B is centered at theta=3*pi/4 (off-axis: not aligned with sin or cos
# coordinate, so sin/cos preprocessing needs *two* axis-aligned bounds to
# bracket it). y = 1 inside either arc, 0 outside.
ARC_A_CENTER = 0.0
ARC_A_HALF_WIDTH = np.pi / 6.0
ARC_B_CENTER = 3.0 * np.pi / 4.0
ARC_B_HALF_WIDTH = np.pi / 6.0


def _circular_distance(theta: np.ndarray, center: float) -> np.ndarray:
    """Shortest angular distance from theta to center, on the circle."""
    diff = np.abs(theta - center) % (2.0 * np.pi)
    return np.minimum(diff, 2.0 * np.pi - diff)


def make_dataset(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    in_arc_a = _circular_distance(theta, ARC_A_CENTER) <= ARC_A_HALF_WIDTH
    in_arc_b = _circular_distance(theta, ARC_B_CENTER) <= ARC_B_HALF_WIDTH
    in_arc = in_arc_a | in_arc_b
    y = in_arc.astype(np.float64) + rng.normal(0.0, NOISE_STD, size=n)
    return theta, y


def split_count_per_tree(booster) -> np.ndarray:
    """Return an array of length n_trees with the number of splits in each tree.

    A LightGBM tree stores its splits as internal nodes; leaf count = splits + 1
    for binary trees. We use Booster.dump_model() to extract the structure since
    it's stable across versions and works for both vanilla LightGBM and the
    patched arcboost build.
    """
    dump = booster.dump_model()
    splits = []
    for tree_info in dump["tree_info"]:
        # num_leaves - 1 == number of internal split nodes for a binary tree.
        n_leaves = tree_info.get("num_leaves") or _count_leaves(tree_info["tree_structure"])
        splits.append(int(n_leaves) - 1)
    return np.asarray(splits, dtype=np.int64)


def _count_leaves(node) -> int:
    if "leaf_index" in node:
        return 1
    return _count_leaves(node["left_child"]) + _count_leaves(node["right_child"])


def per_round_rmse(booster, X_eval: np.ndarray, y_eval: np.ndarray, n_trees: int) -> np.ndarray:
    """Held-out RMSE after each cumulative number of trees in [1, n_trees]."""
    rmses = np.empty(n_trees, dtype=np.float64)
    for k in range(1, n_trees + 1):
        pred = booster.predict(X_eval, num_iteration=k)
        rmses[k - 1] = float(np.sqrt(np.mean((pred - y_eval) ** 2)))
    return rmses


def train_lightgbm(
    lgb,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    extra_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (cumulative_splits, rmse_per_round)."""
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": NUM_LEAVES,
        "learning_rate": LEARNING_RATE,
        "min_data_in_leaf": MIN_DATA_IN_LEAF,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        **extra_params,
    }
    booster = lgb.train(params, train_set, num_boost_round=N_TREES)
    splits = split_count_per_tree(booster)
    cum_splits = np.cumsum(splits)
    rmse = per_round_rmse(booster, X_eval, y_eval, N_TREES)
    return cum_splits, rmse


def main() -> int:
    try:
        import lightgbm as lgb_native
    except ImportError:
        print("error: lightgbm not installed. Run `pip install lightgbm>=4.6.0`.", file=sys.stderr)
        return 1

    try:
        import arcboost as lgb_arc

        arcboost_available = True
    except ImportError:
        lgb_arc = None
        arcboost_available = False
        print(
            "warning: arcboost not installed — running native baselines only. "
            "Build with `pip install -e .` to include the arcboost contender.",
            file=sys.stderr,
        )

    curves: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "native_raw": [],
        "native_sincos": [],
    }
    if arcboost_available:
        curves["arcboost"] = []

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        theta_tr, y_tr = make_dataset(N_SAMPLES, rng)
        theta_te, y_te = make_dataset(N_SAMPLES // 4, rng)

        # 1. Native LightGBM, raw theta.
        X_tr_raw = theta_tr.reshape(-1, 1).astype(np.float32)
        X_te_raw = theta_te.reshape(-1, 1).astype(np.float32)
        curves["native_raw"].append(
            train_lightgbm(lgb_native, X_tr_raw, y_tr, X_te_raw, y_te, extra_params={"seed": seed})
        )

        # 2. Native LightGBM, [sin(theta), cos(theta)].
        X_tr_sc = np.column_stack([np.sin(theta_tr), np.cos(theta_tr)]).astype(np.float32)
        X_te_sc = np.column_stack([np.sin(theta_te), np.cos(theta_te)]).astype(np.float32)
        curves["native_sincos"].append(
            train_lightgbm(lgb_native, X_tr_sc, y_tr, X_te_sc, y_te, extra_params={"seed": seed})
        )

        # 3. ArcBoost contender — only if the build is far enough along that
        # the circular split-finder is implemented (not the v0.1.0.dev0 stub).
        if arcboost_available:
            try:
                cum_splits, rmse = train_lightgbm(
                    lgb_arc,
                    X_tr_raw,
                    y_tr,
                    X_te_raw,
                    y_te,
                    extra_params={
                        "circular_feature": "0",
                        "circular_period": float(2.0 * np.pi),
                        # Match native LightGBM's max_bin=255 default so the
                        # comparison isolates topology (arc vs threshold) rather
                        # than bin-resolution. Lower this if arc-search runtime
                        # matters more than boundary precision.
                        "max_bin_circular": 255,
                        "seed": seed,
                    },
                )
                curves["arcboost"].append((cum_splits, rmse))
            except Exception as exc:  # noqa: BLE001 — runtime check, message is the value
                print(
                    f"info: arcboost circular path errored at seed={seed} ({exc}). "
                    "This is expected on v0.1.0.dev0 where the arc-search finder is a stub. "
                    "Skipping arcboost from the plot.",
                    file=sys.stderr,
                )
                curves.pop("arcboost", None)
                arcboost_available = False
                break

    summary_rows = []
    target_rmse_for_summary = None

    for label, runs in curves.items():
        if not runs:
            continue
        # Resample each (cum_splits, rmse) curve onto a common log-spaced split-grid.
        all_max = max(int(c[-1]) for c, _ in runs)
        grid = np.unique(
            np.round(np.geomspace(1, all_max, num=200)).astype(np.int64)
        )
        resampled = np.empty((len(runs), grid.size), dtype=np.float64)
        for i, (cum_splits, rmse) in enumerate(runs):
            # For each query split-count g, find the rmse of the smallest tree-count
            # whose cumulative splits >= g.
            idx = np.searchsorted(cum_splits, grid, side="left")
            idx = np.clip(idx, 0, rmse.size - 1)
            resampled[i] = rmse[idx]

        mean = resampled.mean(axis=0)
        stderr = resampled.std(axis=0, ddof=1) / np.sqrt(len(runs))
        for g, m, s in zip(grid, mean, stderr, strict=False):
            summary_rows.append({"model": label, "cum_splits": int(g), "rmse_mean": float(m), "rmse_stderr": float(s)})

        if target_rmse_for_summary is None:
            target_rmse_for_summary = float(np.percentile(mean, 25))

    csv_path = FIG_DIR / "circular_efficiency.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "cum_splits", "rmse_mean", "rmse_stderr"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib not installed — skipping figure. Run `pip install matplotlib` to enable.", file=sys.stderr)
        return 0

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {
        "native_raw": ("#999999", "LightGBM (raw $\\theta$)"),
        "native_sincos": ("#ff7f0e", "LightGBM ($\\sin\\theta, \\cos\\theta$)"),
        "arcboost": ("#1f77b4", "ArcBoost (circular_feature)"),
    }

    for label, runs in curves.items():
        if not runs:
            continue
        rows = [r for r in summary_rows if r["model"] == label]
        if not rows:
            continue
        grid = np.array([r["cum_splits"] for r in rows])
        mean = np.array([r["rmse_mean"] for r in rows])
        stderr = np.array([r["rmse_stderr"] for r in rows])
        color, display = palette.get(label, ("#666666", label))
        ax.plot(grid, mean, color=color, label=display, linewidth=1.8)
        ax.fill_between(grid, mean - stderr, mean + stderr, color=color, alpha=0.18)

    ax.set_xscale("log")
    ax.set_xlabel("Cumulative split count")
    ax.set_ylabel("Held-out RMSE")
    ax.set_title("ArcBoost circular-feature efficiency on 1D synthetic regression")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    fig_path = FIG_DIR / "circular_efficiency.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    print(f"wrote {fig_path}")

    if "arcboost" in curves and curves["arcboost"] and target_rmse_for_summary is not None:
        # Headline summary: at the same target RMSE, how many splits did each take?
        def first_to_target(label: str, target: float) -> int | None:
            rows = [r for r in summary_rows if r["model"] == label and r["rmse_mean"] <= target]
            return rows[0]["cum_splits"] if rows else None

        baseline = first_to_target("native_sincos", target_rmse_for_summary)
        contender = first_to_target("arcboost", target_rmse_for_summary)
        if baseline and contender:
            ratio = baseline / contender
            print(
                f"\n=== headline ===\n"
                f"target RMSE: {target_rmse_for_summary:.4f}\n"
                f"  arcboost reached it in           {contender:>7d} splits\n"
                f"  lightgbm (sin/cos) reached it in {baseline:>7d} splits  ({ratio:.2f}x more)"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
