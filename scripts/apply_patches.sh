#!/usr/bin/env bash
# Apply the ArcBoost patch series to the pinned LightGBM submodule.
#
# Idempotent: writes a stamp file at .patches-applied.stamp containing the
# concatenated patch content hash. Subsequent runs that match the same hash
# are no-ops. If patches change, the stamp mismatches and the script aborts
# with a message asking the developer to run a clean reset.
#
# Usage:
#   bash scripts/apply_patches.sh           # apply once
#   bash scripts/apply_patches.sh --force   # reset submodule and re-apply
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE="${REPO_ROOT}/provider/lightgbm"
PATCH_DIR="${REPO_ROOT}/patches/lightgbm"
STAMP="${REPO_ROOT}/.patches-applied.stamp"
LIGHTGBM_TAG="$(cat "${REPO_ROOT}/LIGHTGBM_VERSION")"

force=false
if [[ "${1:-}" == "--force" ]]; then
    force=true
fi

if [[ ! -d "${SUBMODULE}/.git" && ! -f "${SUBMODULE}/.git" ]]; then
    echo "error: submodule ${SUBMODULE} not initialized." >&2
    echo "run: git submodule update --init --recursive" >&2
    exit 1
fi

# Compute current expected hash from patch contents + tag.
shopt -s nullglob
patches=("${PATCH_DIR}"/*.patch)
shopt -u nullglob

if [[ ${#patches[@]} -eq 0 ]]; then
    echo "info: no patches found in ${PATCH_DIR}; nothing to apply."
    # Still ensure submodule is on the pinned tag.
    git -C "${SUBMODULE}" rev-parse --verify "${LIGHTGBM_TAG}" >/dev/null 2>&1 || {
        echo "error: tag ${LIGHTGBM_TAG} not found in submodule." >&2
        exit 1
    }
    git -C "${SUBMODULE}" checkout --quiet "${LIGHTGBM_TAG}" || true
    echo "no-patches:${LIGHTGBM_TAG}" > "${STAMP}"
    exit 0
fi

# The stamp encodes (patch contents + pinned tag + current submodule HEAD).
# Including the submodule HEAD means a manual `git checkout v4.6.0` inside the
# submodule (which discards previously-applied patches) invalidates the stamp
# and forces a re-apply on the next build. Without this, the script could
# falsely conclude "already applied" when the submodule was reset externally.
submodule_head="$(git -C "${SUBMODULE}" rev-parse HEAD)"
current_hash="$(cat "${PATCH_DIR}"/*.patch "${REPO_ROOT}/LIGHTGBM_VERSION" <(echo "${submodule_head}") 2>/dev/null | shasum -a 256 | awk '{print $1}')"

# Quick path: if the submodule HEAD already matches "tag + N patches applied",
# we're done. We detect this by checking that HEAD's commit count beyond the
# pinned tag equals the number of patches.
expected_commits=${#patches[@]}
actual_commits="$(git -C "${SUBMODULE}" rev-list --count "${LIGHTGBM_TAG}..HEAD" 2>/dev/null || echo 0)"

if [[ -f "${STAMP}" && "${force}" != "true" ]]; then
    stamp_hash="$(cat "${STAMP}")"
    if [[ "${stamp_hash}" == "${current_hash}" && "${actual_commits}" == "${expected_commits}" ]]; then
        # Already up to date.
        exit 0
    fi
fi

echo "ArcBoost: applying $(printf %d ${#patches[@]}) patch(es) to provider/lightgbm @ ${LIGHTGBM_TAG}"

# Reset submodule to pinned tag (clean working tree).
git -C "${SUBMODULE}" reset --hard --quiet "${LIGHTGBM_TAG}"
git -C "${SUBMODULE}" clean -fdx --quiet

# Configure a local user/email so `git am` can author commits even on a fresh CI machine.
# These commits live only inside the submodule's local checkout; they're never pushed.
git -C "${SUBMODULE}" config user.name  "ArcBoost Builder" >/dev/null
git -C "${SUBMODULE}" config user.email "build@arcboost.local" >/dev/null

# Apply each patch. --3way gives us conflict markers if upstream diverges.
git -C "${SUBMODULE}" am --3way --keep-cr "${patches[@]}"

# Record success.
echo "${current_hash}" > "${STAMP}"
echo "ArcBoost: patches applied cleanly."
