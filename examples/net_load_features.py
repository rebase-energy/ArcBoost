"""Marginal feature plots for ArcBoost vs vanilla LightGBM on net load.

Same problem as `net_load.py` (4 cyclic + 2 numeric features predicting
behind-the-meter net load). After training both boosters, we visualize
*what each model has learned* by computing partial-dependence plots (PDPs)
for every cyclic feature, plus a 2D heatmap over the most informative
interaction (time-of-day x day-of-year).

Partial dependence at feature value v:
    PDP(v) = mean over training rows of  f(row with feature = v)

The result isolates the effect of one feature by averaging out the others.
We plot:
  - the **true** PDP (averaging the noiseless target with that substitution),
  - **ArcBoost**'s PDP,
  - **Vanilla LightGBM (sin/cos)**'s PDP,

so misalignment in any of the cyclic features is immediately visible.

Outputs:
    examples/figures/net_load_features.png
"""
from __future__ import annotations

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

from net_load import (  # noqa: E402
    PERIOD_TIME,
    PERIOD_YEAR,
    PERIOD_WEEK,
    PERIOD_AZ,
    N_TRAIN,
    SEED,
    _net_load_components,
    _solar_position,
    features_arcboost,
    features_sincos,
    make_dataset,
)

# For the feature plot we use a more generous budget than net_load.py's
# headline (which deliberately constrains num_leaves to expose the budget
# gap). Here we want both models near their *best* fit so any remaining PDP
# discrepancy is structural model bias, not under-fitting.
NUM_LEAVES_VIS = 32
N_TREES_VIS = 500
LEARNING_RATE_VIS = 0.05

# Smaller PDP sample so the plot regenerates quickly. PDP is an average so
# 800 samples is plenty (the variance is dominated by the per-row prediction
# variance, which averages to a tight estimate of E[f|x_j=v]).
N_PDP_SAMPLES = 800
N_GRID = 120


def consistent_pdp(predict_fn, sample_data: dict, feat_name: str, grid_values: np.ndarray, layout: str) -> np.ndarray:
    """Compute a PDP that keeps derived cyclic features physically consistent.

    Plain PDP substitutes only the target feature, leaving derived features
    (here: solar_azimuth, which is determined by time_of_day + day_of_year)
    at their original sample values. That feeds the model unphysical tuples
    it never saw during training (e.g., time=noon with sunset's azimuth).
    The resulting PDP misrepresents what the model has actually learned.

    When the substituted feature is one that determines a derived feature,
    we recompute the derived feature so the model sees a self-consistent
    input. Other independent features (cloud_cover, temperature, day_of_week)
    are left as-is — those are NOT derived from the cyclic ones.
    """
    n = len(next(iter(sample_data.values())))
    g = len(grid_values)
    all_d = {k: np.tile(arr, g) for k, arr in sample_data.items()}
    all_d[feat_name] = np.repeat(grid_values.astype(np.float32), n)
    if feat_name in ("time_of_day", "day_of_year"):
        _, azimuth = _solar_position(all_d["time_of_day"], all_d["day_of_year"])
        all_d["solar_azimuth"] = azimuth
    if layout == "arc":
        X = features_arcboost(all_d)
    else:
        X = features_sincos(all_d)
    preds = predict_fn(X)
    return preds.reshape(g, n).mean(axis=1)


def true_pdp(data: dict, feat_name: str, grid_values: np.ndarray) -> np.ndarray:
    """Compute the true PDP by replacing the named feature with each grid value
    and averaging the noiseless target across the sample.

    For features that *derive* solar_azimuth (time_of_day, day_of_year), we
    recompute the azimuth so the substitution is physically self-consistent.
    For solar_azimuth itself we override it directly (which is unphysical
    for some combinations, but is the same convention the model sees).
    """
    n = len(next(iter(data.values())))
    pdp = np.empty(len(grid_values), dtype=np.float64)
    for i, v in enumerate(grid_values):
        d = {k: arr.copy() for k, arr in data.items()}
        d[feat_name] = np.full(n, float(v), dtype=np.float32)
        if feat_name in ("time_of_day", "day_of_year"):
            altitude, azimuth = _solar_position(d["time_of_day"], d["day_of_year"])
            d["solar_azimuth"] = azimuth
        else:
            altitude, _ = _solar_position(d["time_of_day"], d["day_of_year"])
        demand, pv = _net_load_components(
            d["time_of_day"],
            d["day_of_year"],
            d["day_of_week"],
            altitude,
            d["solar_azimuth"],
            d["cloud_cover"],
            d["temperature"],
        )
        pdp[i] = float(np.mean(demand - pv))
    return pdp


def main() -> int:
    import arcboost
    import lightgbm as vanilla_lgb
    import matplotlib.pyplot as plt

    print("Generating dataset...")
    train_data, y_train = make_dataset(N_TRAIN, SEED)
    X_arc_train = features_arcboost(train_data)
    X_sc_train = features_sincos(train_data)

    common_p = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": NUM_LEAVES_VIS,
        "learning_rate": LEARNING_RATE_VIS,
        "min_data_in_leaf": 20,
        "min_data_in_bin": 1,
        "max_bin": 256,
        "use_missing": False,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "seed": SEED,
    }
    print(f"Training ArcBoost ({N_TREES_VIS} trees, num_leaves={NUM_LEAVES_VIS})...")
    arc_p = dict(
        common_p,
        circular_feature="0,1,2,3",
        circular_period=2.0 * np.pi,
        max_bin_circular=512,
    )
    arc_ds = arcboost.Dataset(
        X_arc_train,
        label=y_train,
        params={
            "use_missing": False,
            "circular_feature": "0,1,2,3",
            "circular_period": 2.0 * np.pi,
            "max_bin_circular": 512,
        },
        free_raw_data=False,
    )
    arc_booster = arcboost.train(arc_p, arc_ds, num_boost_round=N_TREES_VIS)

    print(f"Training Vanilla LightGBM ({N_TREES_VIS} trees, num_leaves={NUM_LEAVES_VIS})...")
    sc_ds = vanilla_lgb.Dataset(
        X_sc_train, label=y_train, params={"use_missing": False}, free_raw_data=False
    )
    sc_booster = vanilla_lgb.train(common_p, sc_ds, num_boost_round=N_TREES_VIS)

    rng = np.random.default_rng(SEED + 42)
    idx = rng.choice(N_TRAIN, size=N_PDP_SAMPLES, replace=False)
    sample_data = {k: arr[idx] for k, arr in train_data.items()}
    X_arc_sample = X_arc_train[idx]
    X_sc_sample = X_sc_train[idx]

    feature_configs = [
        ("time_of_day", "time of day (h)", 0, PERIOD_TIME),
        ("day_of_year", "day of year", 1, PERIOD_YEAR),
        ("day_of_week", "day of week", 2, PERIOD_WEEK),
        ("solar_azimuth", "solar azimuth (deg)", 3, PERIOD_AZ),
    ]

    fig = plt.figure(figsize=(17, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.25], hspace=0.45, wspace=0.20)
    pdp_axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    heat_ax_arc = fig.add_subplot(gs[2, 0])
    heat_ax_sc = fig.add_subplot(gs[2, 1])

    print("Computing PDPs (with consistent solar_azimuth recomputation)...")
    for ax, (feat_name, xlabel, feat_idx, period) in zip(pdp_axes, feature_configs):
        grid = np.linspace(0.0, period, N_GRID, dtype=np.float64)
        true_curve = true_pdp(sample_data, feat_name, grid)
        arc_curve = consistent_pdp(arc_booster.predict, sample_data, feat_name, grid, "arc")
        sc_curve = consistent_pdp(sc_booster.predict, sample_data, feat_name, grid, "sc")

        ax.plot(grid, true_curve, color="#202020", linewidth=2.4, label="True", zorder=3)
        ax.plot(
            grid,
            arc_curve,
            color="#0072b2",
            linewidth=1.9,
            label="ArcBoost",
            zorder=2,
        )
        ax.plot(
            grid,
            sc_curve,
            color="#d55e00",
            linestyle="--",
            linewidth=1.9,
            label="Vanilla sin/cos",
            zorder=2,
        )
        ax.set_xlim(0.0, period)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("avg net load (kW)", fontsize=11)
        ax.set_title(f"Partial dependence — {feat_name}", fontsize=12)
        ax.grid(alpha=0.28)
        ax.legend(frameon=False, fontsize=10, loc="best")
        ax.tick_params(labelsize=10)

    print("Computing 2D interaction heatmap...")
    n_grid_t = 64
    n_grid_d = 64
    grid_t = np.linspace(0.0, PERIOD_TIME, n_grid_t, dtype=np.float64)
    grid_d = np.linspace(0.0, PERIOD_YEAR, n_grid_d, dtype=np.float64)

    # Hold the other features at typical values (clear-ish day, weekday, mild temp).
    fixed_dow = 3.0
    fixed_cloud = 0.2
    base_n = N_PDP_SAMPLES // 4

    def heatmap_for(predict_fn, feat_layout: str) -> np.ndarray:
        H = np.empty((n_grid_d, n_grid_t), dtype=np.float64)
        rng_local = np.random.default_rng(SEED + 99)
        for i, d_val in enumerate(grid_d):
            for j, t_val in enumerate(grid_t):
                # Synthesize a small batch with these values to average over
                # cloud and temperature noise.
                temp = (
                    15.0
                    + 15.0 * np.sin(2 * np.pi * (d_val - 81.0) / PERIOD_YEAR)
                    + 5.0 * np.sin(2 * np.pi * (t_val - 14.0) / PERIOD_TIME)
                )
                temps = temp + rng_local.normal(0.0, 1.5, size=base_n)
                cloud = np.clip(rng_local.beta(2.0, 5.0, size=base_n), 0.0, 1.0)
                tt = np.full(base_n, t_val, dtype=np.float32)
                dd = np.full(base_n, d_val, dtype=np.float32)
                dw = np.full(base_n, fixed_dow, dtype=np.float32)
                _, az = _solar_position(tt, dd)
                if feat_layout == "arc":
                    X = np.column_stack(
                        [
                            tt * (2 * np.pi / PERIOD_TIME),
                            dd * (2 * np.pi / PERIOD_YEAR),
                            dw * (2 * np.pi / PERIOD_WEEK),
                            az * (2 * np.pi / PERIOD_AZ),
                            cloud.astype(np.float32),
                            temps.astype(np.float32),
                        ]
                    ).astype(np.float32)
                else:
                    cyclic = []
                    for arr, period in (
                        (tt, PERIOD_TIME),
                        (dd, PERIOD_YEAR),
                        (dw, PERIOD_WEEK),
                        (az, PERIOD_AZ),
                    ):
                        s = arr * (2 * np.pi / period)
                        cyclic.extend([np.sin(s), np.cos(s)])
                    X = np.column_stack(
                        cyclic + [cloud.astype(np.float32), temps.astype(np.float32)]
                    ).astype(np.float32)
                H[i, j] = predict_fn(X).mean()
        return H

    H_arc = heatmap_for(arc_booster.predict, "arc")
    H_sc = heatmap_for(sc_booster.predict, "sc")

    # Compute true heatmap for shared color scale reference.
    H_true = np.empty_like(H_arc)
    for i, d_val in enumerate(grid_d):
        for j, t_val in enumerate(grid_t):
            tt = np.array([t_val], dtype=np.float32)
            dd = np.array([d_val], dtype=np.float32)
            dw = np.array([fixed_dow], dtype=np.float32)
            altitude, azimuth = _solar_position(tt, dd)
            cloud_v = np.array([fixed_cloud], dtype=np.float32)
            temp_v = np.array(
                [
                    15.0
                    + 15.0 * np.sin(2 * np.pi * (d_val - 81.0) / PERIOD_YEAR)
                    + 5.0 * np.sin(2 * np.pi * (t_val - 14.0) / PERIOD_TIME)
                ],
                dtype=np.float32,
            )
            demand, pv = _net_load_components(
                tt, dd, dw, altitude, azimuth, cloud_v, temp_v
            )
            H_true[i, j] = float(demand[0] - pv[0])

    vmin = float(min(H_arc.min(), H_sc.min(), H_true.min()))
    vmax = float(max(H_arc.max(), H_sc.max(), H_true.max()))

    extent = (0.0, PERIOD_TIME, 0.0, PERIOD_YEAR)
    im_arc = heat_ax_arc.imshow(
        H_arc - H_true,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        extent=extent,
        vmin=-1.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    heat_ax_arc.set_xlabel("time of day (h)")
    heat_ax_arc.set_ylabel("day of year")
    heat_ax_arc.set_title("ArcBoost prediction error (predicted - true) — kW")
    plt.colorbar(im_arc, ax=heat_ax_arc, fraction=0.046, pad=0.04)

    im_sc = heat_ax_sc.imshow(
        H_sc - H_true,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        extent=extent,
        vmin=-1.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    heat_ax_sc.set_xlabel("time of day (h)")
    heat_ax_sc.set_ylabel("day of year")
    heat_ax_sc.set_title("Vanilla sin/cos prediction error (predicted - true) — kW")
    plt.colorbar(im_sc, ax=heat_ax_sc, fraction=0.046, pad=0.04)

    fig.suptitle(
        "What each model learned: per-cyclic-feature PDPs (top) and 2D prediction-error heatmaps "
        "over time-of-day x day-of-year (bottom)",
        fontsize=13,
        y=0.995,
    )
    fig_path = FIG_DIR / "net_load_features.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
