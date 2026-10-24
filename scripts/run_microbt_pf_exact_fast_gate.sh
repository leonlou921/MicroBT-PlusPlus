#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
isolated_root="$(realpath -m "${MICROBT_PF_ISOLATED_ROOT:-/path/to/example}")"
artifact="$(realpath -m "${MICROBT_PF_EXACT_FAST_GATE_ARTIFACT:-${isolated_root}/artifacts/exact_fast_gate.json}")"
results_root="$(realpath -m "${nnUNet_results:-${isolated_root}/nnUNet_results_exact_fast_gate}")"
cache_root="$(realpath -m "${MICROBT_PF_CACHE:-${isolated_root}/cache}")"
preprocessed_root="$(realpath -m "${nnUNet_preprocessed:-/path/to/example}")"
python_bin="${MICROBT_PF_PYTHON:-python3}"
guard="${MICROBT_PF_GUARD:-${repo_root}/scripts/microbt_pf_guard.py}"
baseline="${MICROBT_PF_BASELINE:-${isolated_root}/bootstrap_snapshot.json}"
runner="${MICROBT_PF_GATE_RUNNER:-${python_bin}}"

inside_isolated() {
    local candidate
    candidate="$(realpath -m "$1")"
    [[ "${candidate}" == "${isolated_root}" || "${candidate}" == "${isolated_root}/"* ]]
}

for writable in "${repo_root}" "${artifact}" "${results_root}" "${cache_root}"; do
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
    "${guard}" verify --baseline "${baseline}" --mode gpu-gate --destination "${repo_root}"
else
    "${python_bin}" "${guard}" verify --baseline "${baseline}" --mode gpu-gate --destination "${repo_root}"
fi
guard_status=$?
set -e
if [[ ${guard_status} -ne 0 ]]; then
    exit 2
fi

mkdir -p "$(dirname "${artifact}")" "${results_root}" "${isolated_root}/runlocks"
lock_directory="${isolated_root}/runlocks/microbt_pf_exact_fast_gate.lock"
if ! mkdir "${lock_directory}" 2>/dev/null; then
    exit 2
fi
cleanup() {
    rmdir "${lock_directory}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export MICROBT_PF_CACHE="${cache_root}"
export MICROBT_PF_ISOLATED_ROOT="${isolated_root}"
export nnUNet_results="${results_root}"
export nnUNet_preprocessed="${preprocessed_root}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

arguments=(
    --artifact "${artifact}"
    --dataset 1
    --configuration 3d_fullres
    --plans nnUNetResEncUNetMPlans
    --fold 0
    --seed 42
)
if [[ -n "${MICROBT_PF_GATE_RUNNER:-}" ]]; then
    "${runner}" "${arguments[@]}"
else
    "${runner}" -m nnunetv2.scripts.microbt_pf_exact_fast_gate "${arguments[@]}"
fi
