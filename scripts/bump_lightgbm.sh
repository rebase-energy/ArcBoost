#!/usr/bin/env bash
# Bump the pinned LightGBM version and regenerate the patch series against it.
#
# Usage:
#   bash scripts/bump_lightgbm.sh v4.7.0
#
# Steps performed:
#   1. Fetch tags in the submodule.
#   2. Reset and apply the existing patch series via `git am --3way`. Conflicts
#      stop the script — resolve manually with `git am --continue` inside the
#      submodule, then re-run with --resume.
#   3. Re-export the patch series with `git format-patch <new_tag>..HEAD`.
#   4. Update LIGHTGBM_VERSION and stage the submodule SHA bump.
#
# The script is idempotent on success and fails loudly on conflict.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <new_lightgbm_tag> [--resume]" >&2
    exit 64
fi

NEW_TAG="$1"
RESUME=false
if [[ "${2:-}" == "--resume" ]]; then
    RESUME=true
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE="${REPO_ROOT}/provider/lightgbm"
PATCH_DIR="${REPO_ROOT}/patches/lightgbm"

echo "ArcBoost: bumping LightGBM to ${NEW_TAG}"

if ! ${RESUME}; then
    git -C "${SUBMODULE}" fetch --tags --quiet origin
    if ! git -C "${SUBMODULE}" rev-parse --verify "${NEW_TAG}" >/dev/null 2>&1; then
        echo "error: tag ${NEW_TAG} not found in upstream LightGBM." >&2
        exit 1
    fi

    git -C "${SUBMODULE}" reset --hard --quiet "${NEW_TAG}"
    git -C "${SUBMODULE}" clean -fdx --quiet
    git -C "${SUBMODULE}" config user.name  "ArcBoost Builder" >/dev/null
    git -C "${SUBMODULE}" config user.email "build@arcboost.local" >/dev/null

    shopt -s nullglob
    patches=("${PATCH_DIR}"/*.patch)
    shopt -u nullglob

    if [[ ${#patches[@]} -gt 0 ]]; then
        echo "applying ${#patches[@]} patch(es)..."
        if ! git -C "${SUBMODULE}" am --3way --keep-cr "${patches[@]}"; then
            cat <<EOF >&2

error: patch series failed to apply cleanly against ${NEW_TAG}.

resolve conflicts manually:
    cd ${SUBMODULE}
    # edit conflicting files
    git add <files>
    git am --continue
    # repeat for each conflicting patch

then re-run: bash scripts/bump_lightgbm.sh ${NEW_TAG} --resume
EOF
            exit 1
        fi
    fi
fi

# Re-export the series against the new base.
mkdir -p "${PATCH_DIR}"
rm -f "${PATCH_DIR}"/*.patch
git -C "${SUBMODULE}" format-patch "${NEW_TAG}..HEAD" \
    --output-directory "${PATCH_DIR}" \
    --zero-commit --no-signature

# Reset submodule back to the pinned tag — the build re-applies patches via
# scripts/apply_patches.sh. We don't keep the patched commits in the submodule.
git -C "${SUBMODULE}" reset --hard --quiet "${NEW_TAG}"

# Update version file and stage submodule SHA.
echo "${NEW_TAG}" > "${REPO_ROOT}/LIGHTGBM_VERSION"
rm -f "${REPO_ROOT}/.patches-applied.stamp"

cd "${REPO_ROOT}"
git add LIGHTGBM_VERSION provider/lightgbm patches/lightgbm

cat <<EOF

bump complete. review the regenerated patch series in patches/lightgbm/,
then run: pip install -e . && pytest

if everything passes, commit:
    git commit -m "Bump LightGBM to ${NEW_TAG}"
EOF
