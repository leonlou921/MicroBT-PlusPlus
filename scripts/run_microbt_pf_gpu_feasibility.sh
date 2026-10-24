#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
isolated_root="$(realpath -m "${MICROBT_PF_ISOLATED_ROOT:-/path/to/example}")"
cache_root="$(realpath -m "${MICROBT_PF_CACHE:-${isolated_root}/cache}")"
results_root="$(realpath -m "${nnUNet_results:-${isolated_root}/nnUNet_results}")"
preprocessed_root="$(realpath -m "${nnUNet_preprocessed:-/path/to/example}")"
run_root="$(realpath -m "${MICROBT_PF_GPU_RUN_ROOT:-${isolated_root}/gpu_feasibility}")"
artifact="$(realpath -m "${MICROBT_PF_GPU_ARTIFACT:-${isolated_root}/artifacts/gpu_feasibility.json}")"
runlock_root="$(realpath -m "${MICROBT_PF_RUNLOCK_ROOT:-${isolated_root}/runlocks}")"
python_bin="${MICROBT_PF_PYTHON:-python3}"
guard="${MICROBT_PF_GUARD:-${repo_root}/scripts/microbt_pf_guard.py}"
baseline="${MICROBT_PF_BASELINE:-${isolated_root}/bootstrap_snapshot.json}"

inside_isolated() {
    local candidate
    candidate="$(realpath -m "$1")"
    [[ "${candidate}" == "${isolated_root}" || "${candidate}" == "${isolated_root}/"* ]]
}

for writable in "${repo_root}" "${cache_root}" "${results_root}" "${run_root}" "${artifact}" "${runlock_root}"; do
    if ! inside_isolated "${writable}"; then
        exit 2
    fi
done
if [[ "${preprocessed_root}" != "/path/to/example" ]]; then
    exit 2
fi

cd "${repo_root}"
set +e
if [[ -x "${guard}" ]]; then
    "${guard}" verify --baseline "${baseline}" --mode gpu-smoke --destination "${repo_root}"
else
    "${python_bin}" "${guard}" verify --baseline "${baseline}" --mode gpu-smoke --destination "${repo_root}"
fi
guard_status=$?
set -e
if [[ ${guard_status} -ne 0 ]]; then
    exit 2
fi

mkdir -p "${cache_root}" "${results_root}" "${run_root}/A0" "${run_root}/A3" "$(dirname "${artifact}")" "${runlock_root}"
lock_directory="${runlock_root}/microbt_pf_active.lock"
if ! mkdir "${lock_directory}" 2>/dev/null; then
    exit 2
fi
cleanup_lock() {
    rmdir "${lock_directory}" 2>/dev/null || true
}
trap cleanup_lock EXIT INT TERM

export MICROBT_PF_CACHE="${cache_root}"
export nnUNet_results="${results_root}"
export nnUNet_preprocessed="${preprocessed_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

arguments=(
    --dataset 1
    --configuration 3d_fullres
    --plans nnUNetResEncUNetMPlans
    --variants A0 A3
    --seed 42
    --case-count 5
    --a0-output "${run_root}/A0"
    --a3-output "${run_root}/A3"
    --artifact "${artifact}"
)
if [[ -n "${MICROBT_PF_GPU_RUNNER:-}" ]]; then
    "${MICROBT_PF_GPU_RUNNER}" "${arguments[@]}"
else
    "${python_bin}" -m nnunetv2.scripts.microbt_pf_gpu_feasibility "${arguments[@]}"
fi
