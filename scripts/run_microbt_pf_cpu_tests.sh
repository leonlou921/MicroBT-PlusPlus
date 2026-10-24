#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
isolated_root="$(realpath -m "${MICROBT_PF_ISOLATED_ROOT:-/path/to/example}")"
cache_root="$(realpath -m "${MICROBT_PF_CACHE:-${isolated_root}/cache}")"
results_root="$(realpath -m "${nnUNet_results:-${isolated_root}/nnUNet_results}")"
preprocessed_root="$(realpath -m "${nnUNet_preprocessed:-/path/to/example}")"
python_bin="${MICROBT_PF_PYTHON:-python3}"
guard="${MICROBT_PF_GUARD:-${repo_root}/scripts/microbt_pf_guard.py}"
baseline="${MICROBT_PF_BASELINE:-${isolated_root}/bootstrap_snapshot.json}"

inside_isolated() {
    local candidate
    candidate="$(realpath -m "$1")"
    [[ "${candidate}" == "${isolated_root}" || "${candidate}" == "${isolated_root}/"* ]]
}

if ! inside_isolated "${repo_root}" || ! inside_isolated "${cache_root}" || ! inside_isolated "${results_root}"; then
    exit 2
fi
if [[ "${preprocessed_root}" != "/path/to/example" ]]; then
    exit 2
fi

cd "${repo_root}"
export MICROBT_PF_CACHE="${cache_root}"
export nnUNet_results="${results_root}"
export nnUNet_preprocessed="${preprocessed_root}"
export MICROBT_PF_TEST_ISOLATED_ROOT="${isolated_root}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${cache_root}" "${results_root}"

set +e
if [[ -x "${guard}" ]]; then
    "${guard}" verify --baseline "${baseline}" --mode cpu-test --destination "${repo_root}"
else
    "${python_bin}" "${guard}" verify --baseline "${baseline}" --mode cpu-test --destination "${repo_root}"
fi
guard_status=$?
set -e
if [[ ${guard_status} -ne 0 ]]; then
    exit 2
fi

if [[ -n "${MICROBT_PF_PYTEST_BIN:-}" ]]; then
    "${MICROBT_PF_PYTEST_BIN}" -q nnunetv2/tests
else
    nice -n 19 ionice -c3 "${python_bin}" -m pytest -q nnunetv2/tests
fi
