#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
isolated_root="$(realpath -m "${MICROBT_PF_ISOLATED_ROOT:-/path/to/example}")"
admission="$(realpath -m "${MICROBT_PF_SCHEDULED_ADMISSION:-${isolated_root}/artifacts/scheduled_exact_monitored_gate.json}")"
cache_root="$(realpath -m "${MICROBT_PF_CACHE:-${isolated_root}/cache}")"
results_root="$(realpath -m "${nnUNet_results:-${isolated_root}/nnUNet_results_scheduled_exact_200_monitored}")"
preprocessed_root="$(realpath -m "${nnUNet_preprocessed:-/path/to/example}")"
raw_root="$(realpath -m "${nnUNet_raw:-/path/to/example}")"
monitor_manifest="$(realpath -m "${MICROBT_PF_MONITOR_MANIFEST:-${repo_root}/monitor/fixed_cases_fold0.json}")"
monitor_output="$(realpath -m "${MICROBT_PF_MONITOR_OUTPUT:-${results_root}/Dataset001_Example/MicroBTPFTrainerA3ScheduledExact__nnUNetResEncUNetMPlans__3d_fullres/fold_0/monitor}")"
python_bin="${MICROBT_PF_PYTHON:-python3}"
train_bin="${MICROBT_PF_TRAIN_BIN:-${python_bin%/python}/nnUNetv2_train}"
guard="${MICROBT_PF_GUARD:-${repo_root}/scripts/microbt_pf_guard.py}"
baseline="${MICROBT_PF_BASELINE:-${isolated_root}/bootstrap_snapshot.json}"
trainer="MicroBTPFTrainerA3ScheduledExact"

inside_isolated() {
    local candidate
    candidate="$(realpath -m "$1")"
    [[ "${candidate}" == "${isolated_root}" || "${candidate}" == "${isolated_root}/"* ]]
}

for writable in "${repo_root}" "${admission}" "${cache_root}" "${results_root}" "${monitor_manifest}" "${monitor_output}"; do
    if ! inside_isolated "${writable}"; then
        exit 2
    fi
done
if [[ "${preprocessed_root}" != "/path/to/example" ]]; then
    exit 2
fi
if [[ "${raw_root}" != "/path/to/example" ]]; then
    exit 2
fi

cd "${repo_root}"
export MICROBT_PF_CACHE="${cache_root}"
export nnUNet_results="${results_root}"
export nnUNet_preprocessed="${preprocessed_root}"
export nnUNet_raw="${raw_root}"
export MICROBT_PF_MONITOR_MANIFEST="${monitor_manifest}"
export MICROBT_PF_MONITOR_OUTPUT="${monitor_output}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-8}"

"${python_bin}" -m nnunetv2.scripts.launch_microbt_pf_scheduled_exact --verify-only \
    --isolated-root "${isolated_root}" \
    --repo-root "${repo_root}" \
    --admission-artifact "${admission}" \
    --results-root "${results_root}" \
    --monitor-manifest "${monitor_manifest}" \
    --monitor-output "${monitor_output}"

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

mkdir -p "${cache_root}" "${results_root}" "${isolated_root}/runlocks"
lock_directory="${isolated_root}/runlocks/microbt_pf_scheduled_exact_train.lock"
if ! mkdir "${lock_directory}" 2>/dev/null; then
    exit 2
fi
cleanup_lock() {
    rmdir "${lock_directory}" 2>/dev/null || true
}
trap cleanup_lock EXIT INT TERM

"${train_bin}" 1 3d_fullres 0 -tr "${trainer}" -p nnUNetResEncUNetMPlans
