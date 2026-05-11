cmake_minimum_required(VERSION 3.18)
project(arcboost LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Apply patch series before configuring LightGBM.
#
# This must run at *configure* time (not build time) because LightGBM's own
# CMakeLists.txt — which we add_subdirectory below — discovers source files
# during configuration. Patches that add or rename files (e.g., 0009 renames
# python-package/lightgbm/ -> python-package/arcboost/) must be applied first.
#
# scripts/apply_patches.sh is idempotent (stamp-file guarded), so re-running
# CMake configure is cheap.
# -----------------------------------------------------------------------------
execute_process(
    COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/apply_patches.sh"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _arcboost_patch_rc
    OUTPUT_VARIABLE _arcboost_patch_out
    ERROR_VARIABLE  _arcboost_patch_err
)
if(NOT _arcboost_patch_rc EQUAL 0)
    message(FATAL_ERROR
        "ArcBoost: failed to apply LightGBM patches.\n"
        "stdout:\n${_arcboost_patch_out}\n"
        "stderr:\n${_arcboost_patch_err}")
endif()
message(STATUS "ArcBoost: LightGBM patches applied (or already up to date).")

# -----------------------------------------------------------------------------
# Build the patched LightGBM as a Python extension.
# __BUILD_FOR_PYTHON=ON tells LightGBM's CMake to produce the _lightgbm
# extension module suitable for wheel installation rather than the CLI binary.
# -----------------------------------------------------------------------------
set(__BUILD_FOR_PYTHON ON CACHE BOOL "" FORCE)
set(BUILD_CLI OFF CACHE BOOL "" FORCE)

add_subdirectory(provider/lightgbm)

# -----------------------------------------------------------------------------
# Install the compiled library at <wheel-install-dir>/lib/ as well as wherever
# the upstream rule put it. libpath.py searches Path(__file__).parents[0]/lib/,
# i.e., arcboost/lib/, so this places the dylib exactly where the Python loader
# expects to find it.
# -----------------------------------------------------------------------------
install(
    TARGETS _lightgbm
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
)
