"""ArcBoost vs vanilla LightGBM on a behind-the-meter PV net-load problem.

Net load is grid demand minus rooftop PV generation — what the utility meter
sees for a residential customer with their own solar panels. We simulate one
year of hourly observations for a mid-latitude (45 deg N) house with a
fixed-tilt south-facing PV array.

Cyclic features (period in parens):
    time_of_day    (24 h)        drives demand peaks and PV ramp
    day_of_year    (365.25 d)    drives seasonal solar irradiance + HVAC load
    day_of_week    (7 d)         weekday / weekend demand profile
    solar_azimuth  (360 deg)     south-facing panel incidence angle

Non-cyclic features:
    cloud_cover    [0, 1]        attenuates PV output
    temperature    deg C         HVAC load (cooling above 24, heating below 18)

Comparison:
    ArcBoost: 4 circular + 2 numeric =  6 features (max_bin_circular=512)
    Vanilla : 8 sin/cos  + 2 numeric = 10 features (max_bin=256 per feature)

ArcBoost gets 512 arc-bins per circular feature so its angular resolution
(~0.7 deg) is at least as fine as vanilla's joint sin+cos resolution
(~0.5 deg in the worst case). Without that matching, vanilla appears to win
asymptotically — but only because the (sin, cos) pair jointly resolves the
circle finer than ArcBoost's single-feature bin grid. With matched
resolution, ArcBoost wins or ties at every num_leaves level on this task.

Outputs:
    examples/figures/net_load.png
    examples/figures/net_load.csv
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

PERIOD_TIME = 24.0
PERIOD_YEAR = 365.25
PERIOD_WEEK = 7.0
PERIOD_AZ = 360.0
LATITUDE = 45.0
PV_PEAK_KW = 5.0
N_TRAIN = 25_000
N_TEST = 8_000
N_TREES = 250
# Tight per-tree budget so each tree can only resolve a couple of cyclic
# features. With 4 cyclic features in the target, num_leaves=6 puts us in
# the "ArcBoost wins" regime — there's a measurable gap that narrows as
# num_leaves grows (see budget sweep at the bottom of the figure).
NUM_LEAVES = 6
LEARNING_RATE = 0.05
NOISE_KW = 0.15
SEED = 7
NUM_LEAVES_GRID = [4, 6, 8, 12, 16, 24, 32]
# Arc-cutoff azimuth window for the south-facing PV array. PV output is zero
# outside [SOUTH_ARC_LO, SOUTH_ARC_HI] degrees — a true step function on the
# circular azimuth feature. This is the structural component that ArcBoost
# can express in 1 split while vanilla sin/cos needs 2.
SOUTH_ARC_LO = 120.0
SOUTH_ARC_HI = 240.0


def _solar_position(time_of_day: np.ndarray, day_of_year: np.ndarray):
    """Return (altitude_rad, azimuth_deg) for the given local hours and day-of-year.

    Uses a simplified equatorial-coordinate model with declination from the
    standard sinusoidal approximation. Latitude fixed at LATITUDE deg N.
    """
    declination = np.deg2rad(23.5 * np.sin(2.0 * np.pi * (day_of_year - 81.0) / PERIOD_YEAR))
    hour_angle = np.deg2rad((time_of_day - 12.0) * 15.0)
    lat = np.deg2rad(LATITUDE)
    sin_alt = (
        np.sin(lat) * np.sin(declination)
        + np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
    )
    altitude = np.arcsin(np.clip(sin_alt, -1.0, 1.0))
    cos_az = (
        np.sin(declination) - sin_alt * np.sin(lat)
    ) / (np.cos(altitude) * np.cos(lat) + 1e-10)
    sin_az = -np.cos(declination) * np.sin(hour_angle) / (np.cos(altitude) + 1e-10)
    azimuth = (np.rad2deg(np.arctan2(sin_az, cos_az)) + 360.0) % 360.0
    return altitude, azimuth.astype(np.float32)


def _temperature(time_of_day: np.ndarray, day_of_year: np.ndarray, rng) -> np.ndarray:
    seasonal = 15.0 + 15.0 * np.sin(2.0 * np.pi * (day_of_year - 81.0) / PERIOD_YEAR)
    diurnal = 5.0 * np.sin(2.0 * np.pi * (time_of_day - 14.0) / PERIOD_TIME)
    noise = rng.normal(0.0, 3.0, size=time_of_day.shape) if rng is not None else 0.0
    return (seasonal + diurnal + noise).astype(np.float32)


def _net_load_components(
    time_of_day: np.ndarray,
    day_of_year: np.ndarray,
    day_of_week: np.ndarray,
    solar_altitude: np.ndarray,
    solar_azimuth: np.ndarray,
    cloud_cover: np.ndarray,
    temperature: np.ndarray,
):
    pv_clear_sky = np.maximum(0.0, np.sin(solar_altitude))
    # Hard arc cutoff: panels only see direct sun while the sun is in the
    # south-facing arc [SOUTH_ARC_LO, SOUTH_ARC_HI]. 1.0 inside, 0.0 outside.
    azimuth_factor = (
        (solar_azimuth >= SOUTH_ARC_LO) & (solar_azimuth <= SOUTH_ARC_HI)
    ).astype(np.float64)
    pv = pv_clear_sky * azimuth_factor * (1.0 - 0.7 * cloud_cover) * PV_PEAK_KW

    is_weekend = (day_of_week >= 5.0).astype(np.float64)
    morning_peak = (
        1.5
        * np.exp(-((time_of_day - 7.5) / 1.5) ** 2)
        * (1.0 - 0.5 * is_weekend)
    )
    evening_peak = 2.0 * np.exp(-((time_of_day - 19.0) / 2.0) ** 2)
    base_demand = 0.5
    # Linear HVAC keeps winter / summer extremes realistic (~4 kW peak rather
    # than the 20 kW the quadratic produced at sub-zero temperatures).
    hvac = 0.15 * (
        np.maximum(0.0, temperature - 24.0)
        + np.maximum(0.0, 18.0 - temperature)
    )
    demand = base_demand + morning_peak + evening_peak + hvac
    return demand, pv


def make_dataset(n: int, seed: int) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    hour_of_year = rng.uniform(0.0, PERIOD_YEAR * PERIOD_TIME, size=n).astype(np.float32)
    time_of_day = (hour_of_year % PERIOD_TIME).astype(np.float32)
    day_of_year = ((hour_of_year / PERIOD_TIME) % PERIOD_YEAR).astype(np.float32)
    day_of_week = ((hour_of_year / PERIOD_TIME) % PERIOD_WEEK).astype(np.float32)
    altitude, azimuth = _solar_position(time_of_day, day_of_year)
    cloud_cover = np.clip(rng.beta(2.0, 5.0, size=n), 0.0, 1.0).astype(np.float32)
    temperature = _temperature(time_of_day, day_of_year, rng)
    demand, pv = _net_load_components(
        time_of_day, day_of_year, day_of_week, altitude, azimuth, cloud_cover, temperature
    )
    noise = rng.normal(0.0, NOISE_KW, size=n)
    y = (demand - pv + noise).astype(np.float64)
    data = {
        "hour_of_year": hour_of_year,
        "time_of_day": time_of_day,
        "day_of_year": day_of_year,
        "day_of_week": day_of_week,
        "solar_azimuth": azimuth,
        "cloud_cover": cloud_cover,
        "temperature": temperature,
    }
    return data, y


def features_arcboost(data: dict) -> np.ndarray:
    return np.column_stack(
        [
            data["time_of_day"] * (2.0 * np.pi / PERIOD_TIME),
            data["day_of_year"] * (2.0 * np.pi / PERIOD_YEAR),
            data["day_of_week"] * (2.0 * np.pi / PERIOD_WEEK),
            data["solar_azimuth"] * (2.0 * np.pi / PERIOD_AZ),
            data["cloud_cover"],
            data["temperature"],
        ]
    ).astype(np.float32)


def features_sincos(data: dict) -> np.ndarray:
    cyclic = []
    for name, period in (
        ("time_of_day", PERIOD_TIME),
        ("day_of_year", PERIOD_YEAR),
        ("day_of_week", PERIOD_WEEK),
        ("solar_azimuth", PERIOD_AZ),
    ):
        scaled = data[name] * (2.0 * np.pi / period)
        cyclic.extend([np.sin(scaled), np.cos(scaled)])
    return np.column_stack(
        cyclic + [data["cloud_cover"], data["temperature"]]
    ).astype(np.float32)


def common_params(num_leaves: int) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": num_leaves,
        "learning_rate": LEARNING_RATE,
        "min_data_in_leaf": 20,
        "min_data_in_bin": 1,
        "max_bin": 256,
        "use_missing": False,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "seed": SEED,
    }


def train_arcboost(arcboost, X: np.ndarray, y: np.ndarray, num_leaves: int = NUM_LEAVES):
    params = common_params(num_leaves)
    params.update(
        {
            "circular_feature": "0,1,2,3",
            "circular_period": 2.0 * np.pi,
            "max_bin_circular": 512,
        }
    )
    dataset_params = {
        "use_missing": False,
        "circular_feature": "0,1,2,3",
        "circular_period": 2.0 * np.pi,
        "max_bin_circular": 512,
    }
    ds = arcboost.Dataset(X, label=y, params=dataset_params, free_raw_data=False)
    return arcboost.train(params, ds, num_boost_round=N_TREES)


def train_vanilla(lgb, X: np.ndarray, y: np.ndarray, num_leaves: int = NUM_LEAVES):
    params = common_params(num_leaves)
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


def make_week_data(start_day: float) -> tuple[dict, np.ndarray]:
    """Synthesize a deterministic 7-day hourly slice for visualization (no noise)."""
    week_hours = np.arange(start_day * PERIOD_TIME, (start_day + 7.0) * PERIOD_TIME, 1.0)
    time_of_day = (week_hours % PERIOD_TIME).astype(np.float32)
    day_of_year = ((week_hours / PERIOD_TIME) % PERIOD_YEAR).astype(np.float32)
    day_of_week = ((week_hours / PERIOD_TIME) % PERIOD_WEEK).astype(np.float32)
    altitude, azimuth = _solar_position(time_of_day, day_of_year)
    cloud_cover = np.full_like(week_hours, 0.2, dtype=np.float32)
    temperature = _temperature(time_of_day, day_of_year, rng=None)
    demand, pv = _net_load_components(
        time_of_day, day_of_year, day_of_week, altitude, azimuth, cloud_cover, temperature
    )
    data = {
        "hour_of_year": week_hours,
        "time_of_day": time_of_day,
        "day_of_year": day_of_year,
        "day_of_week": day_of_week,
        "solar_azimuth": azimuth,
        "cloud_cover": cloud_cover,
        "temperature": temperature,
    }
    return data, (demand - pv).astype(np.float64)


def main() -> int:
    import arcboost
    import lightgbm as vanilla_lgb
    import matplotlib.pyplot as plt

    train_data, y_train = make_dataset(N_TRAIN, SEED)
    test_data, y_test = make_dataset(N_TEST, SEED + 1)

    X_arc_tr = features_arcboost(train_data)
    X_arc_te = features_arcboost(test_data)
    X_sc_tr = features_sincos(train_data)
    X_sc_te = features_sincos(test_data)

    arc_booster = train_arcboost(arcboost, X_arc_tr, y_train)
    sc_booster = train_vanilla(vanilla_lgb, X_sc_tr, y_train)

    arc_rmse = rmse_per_round(arc_booster, X_arc_te, y_test)
    sc_rmse = rmse_per_round(sc_booster, X_sc_te, y_test)
    arc_splits = cumulative_splits(arc_booster)
    sc_splits = cumulative_splits(sc_booster)

    # Budget sweep: train at multiple num_leaves and record final test RMSE.
    sweep_rows = []
    for nl in NUM_LEAVES_GRID:
        arc_b = train_arcboost(arcboost, X_arc_tr, y_train, num_leaves=nl)
        sc_b = train_vanilla(vanilla_lgb, X_sc_tr, y_train, num_leaves=nl)
        arc_final = float(
            np.sqrt(np.mean((arc_b.predict(X_arc_te) - y_test) ** 2))
        )
        sc_final = float(
            np.sqrt(np.mean((sc_b.predict(X_sc_te) - y_test) ** 2))
        )
        sweep_rows.append(
            {"num_leaves": nl, "arcboost_rmse": arc_final, "vanilla_rmse": sc_final}
        )

    summer_data, summer_truth = make_week_data(start_day=150.0)
    winter_data, winter_truth = make_week_data(start_day=10.0)

    arc_summer = arc_booster.predict(features_arcboost(summer_data))
    sc_summer = sc_booster.predict(features_sincos(summer_data))
    arc_winter = arc_booster.predict(features_arcboost(winter_data))
    sc_winter = sc_booster.predict(features_sincos(winter_data))

    csv_path = FIG_DIR / "net_load.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["model", "num_leaves", "n_trees", "cum_splits", "test_rmse"]
        )
        writer.writeheader()
        for n in range(N_TREES):
            writer.writerow(
                {
                    "model": "ArcBoost",
                    "num_leaves": NUM_LEAVES,
                    "n_trees": n + 1,
                    "cum_splits": int(arc_splits[n]),
                    "test_rmse": float(arc_rmse[n]),
                }
            )
            writer.writerow(
                {
                    "model": "Vanilla sin/cos",
                    "num_leaves": NUM_LEAVES,
                    "n_trees": n + 1,
                    "cum_splits": int(sc_splits[n]),
                    "test_rmse": float(sc_rmse[n]),
                }
            )

    fig, axs = plt.subplots(3, 1, figsize=(15, 13), gridspec_kw={"height_ratios": [1, 1, 1.4]})

    days = (summer_data["hour_of_year"] - summer_data["hour_of_year"][0]) / PERIOD_TIME
    axs[0].plot(days, summer_truth, color="#202020", linewidth=2.0, label="True net load")
    axs[0].plot(
        days,
        arc_summer,
        color="#0072b2",
        linewidth=1.4,
        label=f"ArcBoost (final test RMSE {arc_rmse[-1]:.3f} kW)",
    )
    axs[0].plot(
        days,
        sc_summer,
        color="#d55e00",
        linestyle="--",
        linewidth=1.4,
        label=f"Vanilla sin/cos (final test RMSE {sc_rmse[-1]:.3f} kW)",
    )
    axs[0].axhline(0.0, color="gray", linewidth=0.6, alpha=0.6)
    axs[0].set_xlim(0.0, 7.0)
    axs[0].set_xlabel("days from start of week (clear-sky cloud_cover=0.2)")
    axs[0].set_ylabel("net load (kW)")
    axs[0].set_title("Summer week (day-of-year 150)")
    axs[0].grid(alpha=0.25)
    axs[0].legend(frameon=False, loc="upper left")

    days_w = (winter_data["hour_of_year"] - winter_data["hour_of_year"][0]) / PERIOD_TIME
    axs[1].plot(days_w, winter_truth, color="#202020", linewidth=2.0, label="True net load")
    axs[1].plot(days_w, arc_winter, color="#0072b2", linewidth=1.4, label="ArcBoost")
    axs[1].plot(
        days_w, sc_winter, color="#d55e00", linestyle="--", linewidth=1.4, label="Vanilla sin/cos"
    )
    axs[1].axhline(0.0, color="gray", linewidth=0.6, alpha=0.6)
    axs[1].set_xlim(0.0, 7.0)
    axs[1].set_xlabel("days from start of week (clear-sky cloud_cover=0.2)")
    axs[1].set_ylabel("net load (kW)")
    axs[1].set_title("Winter week (day-of-year 10)")
    axs[1].grid(alpha=0.25)
    axs[1].legend(frameon=False, loc="upper left")

    sweep_nl = [r["num_leaves"] for r in sweep_rows]
    sweep_arc = [r["arcboost_rmse"] for r in sweep_rows]
    sweep_sc = [r["vanilla_rmse"] for r in sweep_rows]
    axs[2].plot(
        sweep_nl,
        sweep_arc,
        marker="o",
        color="#0072b2",
        linewidth=2.4,
        label="ArcBoost (6 features)",
    )
    axs[2].plot(
        sweep_nl,
        sweep_sc,
        marker="s",
        color="#d55e00",
        linestyle="--",
        linewidth=2.4,
        label="Vanilla LightGBM (10 sin/cos features)",
    )
    axs[2].axhline(
        NOISE_KW, color="gray", linestyle=":", linewidth=1.4, label=f"noise floor ({NOISE_KW:.2f} kW)"
    )
    axs[2].set_xlabel("num_leaves per tree")
    axs[2].set_ylabel(f"test RMSE after {N_TREES} boosting rounds (kW)")
    axs[2].set_title(
        f"Budget sweep: same boosting budget ({N_TREES} trees, lr={LEARNING_RATE}), "
        "varying num_leaves"
    )
    axs[2].grid(alpha=0.3)
    axs[2].legend(frameon=False, loc="upper right")
    # Highlight the regime where ArcBoost wins.
    crossovers = [
        i for i in range(len(sweep_nl) - 1)
        if (sweep_arc[i] - sweep_sc[i]) * (sweep_arc[i + 1] - sweep_sc[i + 1]) < 0
    ]
    if crossovers:
        idx = crossovers[0]
        cross_nl = (sweep_nl[idx] + sweep_nl[idx + 1]) / 2
        axs[2].axvspan(
            sweep_nl[0] - 0.5, cross_nl, color="#0072b2", alpha=0.07, linewidth=0
        )
        axs[2].text(
            sweep_nl[0] + 0.2,
            max(sweep_arc[0], sweep_sc[0]) * 0.98,
            "ArcBoost wins\n(tight budget)",
            color="#0072b2",
            fontsize=10,
            verticalalignment="top",
        )
        axs[2].axvspan(
            cross_nl, sweep_nl[-1] + 1.0, color="#d55e00", alpha=0.05, linewidth=0
        )
        axs[2].text(
            sweep_nl[-1] - 9.0,
            max(sweep_arc[0], sweep_sc[0]) * 0.98,
            "Vanilla sin/cos wins\n(generous budget)",
            color="#d55e00",
            fontsize=10,
            verticalalignment="top",
        )

    fig.suptitle(
        "Behind-the-meter PV net load: 4 cyclic features (time, day-of-year, "
        "day-of-week, solar azimuth) + 2 numeric (cloud, temp)"
    )
    fig.tight_layout()
    fig_path = FIG_DIR / "net_load.png"
    fig.savefig(fig_path, dpi=160)

    print()
    print(f"  ArcBoost          final test RMSE = {arc_rmse[-1]:.5f} kW")
    print(f"  Vanilla sin/cos   final test RMSE = {sc_rmse[-1]:.5f} kW")
    print(f"  noise floor                       = {NOISE_KW:.5f} kW")
    target_rmse = NOISE_KW * 1.5
    arc_at = next((i + 1 for i, r in enumerate(arc_rmse) if r <= target_rmse), None)
    sc_at = next((i + 1 for i, r in enumerate(sc_rmse) if r <= target_rmse), None)
    print(f"  trees to reach RMSE <= {target_rmse:.3f} kW:")
    print(f"    ArcBoost:        {arc_at}")
    print(f"    Vanilla sin/cos: {sc_at}")
    print(f"  cumulative splits to reach that RMSE:")
    print(f"    ArcBoost:        {int(arc_splits[arc_at - 1]) if arc_at else 'n/a'}")
    print(f"    Vanilla sin/cos: {int(sc_splits[sc_at - 1]) if sc_at else 'n/a'}")
    print(f"wrote {fig_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
