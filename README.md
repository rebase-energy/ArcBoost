# ArcBoost

Gradient boosting with first-class **circular feature splits**, built as a fork
of [LightGBM v4.6.0](https://github.com/microsoft/LightGBM/releases/tag/v4.6.0).

For features whose true domain is a circle — wind direction, hour-of-day,
day-of-year, phase angles — standard gradient boosting libraries treat the
feature as a real-valued line and use linear-threshold splits `x ≤ τ`. This is
structurally wrong: the wraparound `0 ≡ 2π` is invisible to the splitter, and
the natural decision boundary on a circle is an **arc** `[a, b] mod 2π`, not a
threshold. ArcBoost adds `circular` as a feature type alongside the existing
`numerical` and `categorical`, with a histogram-based arc split-finder.

## Project status

**Pre-alpha (v0.1.0.dev0). Circular splits are end-to-end functional.** The
build, packaging, and import work; `import arcboost` is a drop-in replacement
for `import lightgbm` for numerical and categorical features; and circular
features go through a real arc-search at training time and an arc-membership
decode at inference time.

What's implemented (12 patches total, applied to LightGBM v4.6.0):

- `BinType::CircularBin` enum value and `BinMapper::InitAsCircular()`
- `kCircularMask` flag on `decision_type_`
- `FindBestThresholdCircular` — O(B²) arc-search using doubled cyclic prefix
  sums of (g, h), with length-≤-B/2 pruning since complement arcs are
  equivalent
- `Tree::SplitCircular` and `Tree::CircularDecision` — arc-membership decode
  at inference, including model-string serialization round-trip for the new
  per-cat-idx `(period, num_bins)` metadata
- `circular_feature` / `circular_period` / `max_bin_circular` parameters
  threaded through `Config` and `config_auto.cpp`
- `DatasetLoader` parses `circular_feature` and routes those indices through
  `InitAsCircular` instead of `FindBin`
- Python package renamed `lightgbm/` → `arcboost/`
- Safety guards on distributed (data-parallel and voting-parallel) tree
  learners and the GPU/CUDA device path — all reject `circular_feature`
  with a clear error rather than silently producing inconsistent splits

The `pytest tests/` suite passes 11/11, including the canonical recovery
tests on a known arc and on a wraparound arc straddling θ=0 (the case where
linear splits structurally cannot match arc semantics in one cut).

## How the fork is organized

ArcBoost vendors LightGBM v4.6.0 as a git submodule under `provider/lightgbm/`
and applies a numbered patch series from `patches/lightgbm/` at build time.
This keeps:

- **Upstream history clean** — the submodule itself is unmodified at HEAD.
- **Each change reviewable** — every patch is one logical edit (one enum
  variant, one new function, etc.).
- **The upgrade story sane** — `scripts/bump_lightgbm.sh v4.7.0` re-applies
  the series via `git am --3way` against the new tag and re-exports.

Repo layout:

```
ArcBoost/
├── pyproject.toml                  scikit-build-core, name=arcboost
├── CMakeLists.txt                  shim: apply patches → add_subdirectory(provider/lightgbm)
├── LIGHTGBM_VERSION                "v4.6.0"
├── provider/lightgbm/                git submodule pinned to v4.6.0
├── patches/lightgbm/               numbered .patch files
├── scripts/
│   ├── apply_patches.sh            idempotent, stamp-file guarded
│   └── bump_lightgbm.sh            documented upgrade flow
├── arcboost/                       (populated at build time, gitignored)
├── examples/circular_efficiency.py 1D synthetic comparison demo
└── tests/                          pytest suite
```

## Install (developer)

```sh
git clone --recurse-submodules https://github.com/sebaheg/ArcBoost.git
cd ArcBoost
pip install -e ".[example,test]"
```

`pip install` invokes scikit-build-core, which runs CMake on the repo root.
The first thing CMake does is execute `scripts/apply_patches.sh`, which
applies the patch series to `provider/lightgbm/`. After patches apply, the
patched LightGBM is built as a Python extension and the renamed
`provider/lightgbm/python-package/arcboost/` package is installed as
`arcboost`.

If you ever need to reset the submodule (e.g., after pulling new patches):

```sh
bash scripts/apply_patches.sh --force
```

## Usage (target API)

```python
from arcboost import LGBMRegressor

model = LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=500)
model.fit(X_train, y_train, circular_feature=["wind_dir", "hour_of_day"])
y_pred = model.predict(X_test)
```

`circular_feature` accepts column names or integer indices, mirroring the
existing `categorical_feature` parameter. The default circular period is `2π`;
override per-feature with the syntax `circular_feature="name:wind_dir=360"`.

## The headline demo

`examples/circular_efficiency.py` is a 1D synthetic regression task that
compares ArcBoost against native LightGBM (raw `θ` baseline + the standard
`[sin θ, cos θ]` baseline) on the same task, plotting **validation RMSE vs
cumulative split count** across 5 seeds. ArcBoost should reach a target RMSE
in fewer total splits because arc splits respect the cyclic topology that
linear splits cannot encode without wasted bracketing.

```sh
python examples/circular_efficiency.py
# writes examples/figures/circular_efficiency.png and .csv
```

## Pinning & upgrade

The pinned tag lives in `LIGHTGBM_VERSION`. The recorded submodule SHA in
`.gitmodules` + the parent repo's index entry locks the exact commit.

To bump:

```sh
bash scripts/bump_lightgbm.sh v4.7.0
# resolves any conflicts via `git am --continue` inside provider/lightgbm/
# regenerates patches/lightgbm/*.patch against the new base
# updates LIGHTGBM_VERSION
```

## v0 scope (what's NOT yet supported)

- **Templated split-finder booleans (USE_RAND / USE_MC / USE_L1 /
  USE_MAX_OUTPUT / USE_SMOOTHING)** — `FindBestThresholdCircular` collapses
  these to false. L1 / max_output / smoothing / extra-trees randomization
  for circular features are follow-ups; the surrounding scaffolding to add
  them mirrors LightGBM's existing `FuncForCategorical{,L1,L2}` template
  tree.
- **Per-feature `circular_period` overrides** — the `name:wind_dir=360`
  syntax described in the parameter docs is parsed but not yet applied
  per-feature. v0 uses the global `circular_period` for every circular
  feature.
- **GPU / CUDA tree learner** — `circular_feature` rejected with a clear
  error. The CUDA path has its own histogram and split-finder kernels
  unrelated to the patches in this repo.
- **Distributed training** (data-parallel, voting-parallel) — same:
  `circular_feature` rejected with a clear error.
- **The R package and CLI binary** are auto-skipped when `__BUILD_FOR_PYTHON=ON`
  (LightGBM's own CMake handles this; no patch needed).

## License

MIT, matching upstream LightGBM. See `LICENSE` and `provider/lightgbm/LICENSE`.
