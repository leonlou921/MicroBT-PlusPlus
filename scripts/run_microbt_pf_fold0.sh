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
variant="${1:-}"

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

case "${variant}" in
    A0) trainer=MicroBTPFTrainerA0 ;;
    A1) trainer=MicroBTPFTrainerA1 ;;
    A2) trainer=MicroBTPFTrainerA2 ;;
    A3) trainer=MicroBTPFTrainerA3 ;;
    A0-D) trainer='' ;;
    *) exit 64 ;;
esac

cd "${repo_root}"
export MICROBT_PF_CACHE="${cache_root}"
export nnUNet_results="${results_root}"
export nnUNet_preprocessed="${preprocessed_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${cache_root}" "${results_root}" "${isolated_root}/runlocks"

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

lock_directory="${isolated_root}/runlocks/microbt_pf_active.lock"
if ! mkdir "${lock_directory}" 2>/dev/null; then
    exit 2
fi
cleanup_lock() {
    rmdir "${lock_directory}" 2>/dev/null || true
}
trap cleanup_lock EXIT INT TERM

if [[ "${variant}" == "A0-D" ]]; then
    a0_model="$(realpath -m "${MICROBT_PF_A0_MODEL_FOLDER:-${results_root}/Dataset001_Example/MicroBTPFTrainerA0__nnUNetResEncUNetMPlans__3d_fullres}")"
    inference_input="${MICROBT_PF_INFERENCE_INPUT:?MICROBT_PF_INFERENCE_INPUT is required for A0-D}"
    inference_output="$(realpath -m "${MICROBT_PF_INFERENCE_OUTPUT:?MICROBT_PF_INFERENCE_OUTPUT is required for A0-D}")"
    if ! inside_isolated "${a0_model}" || ! inside_isolated "${inference_output}"; then
        exit 2
    fi
    if [[ ! -f "${a0_model}/fold_0/checkpoint_final.pth" ]]; then
        exit 2
    fi
    mkdir -p "${inference_output}"
    if [[ -n "${MICROBT_PF_PREDICT_BIN:-}" ]]; then
        "${MICROBT_PF_PREDICT_BIN}" -i "${inference_input}" -o "${inference_output}" -m "${a0_model}" --folds 0 --checkpoint checkpoint_final.pth --a0-decoder --save-field --overwrite
    else
        "${python_bin}" -m nnunetv2.scripts.predict_microbt_pf -i "${inference_input}" -o "${inference_output}" -m "${a0_model}" --folds 0 --checkpoint checkpoint_final.pth --a0-decoder --save-field --overwrite
    fi
    exit 0
fi

train_bin="${MICROBT_PF_TRAIN_BIN:-nnUNetv2_train}"
"${train_bin}" 1 3d_fullres 0 -tr "${trainer}" -p nnUNetResEncUNetMPlans
