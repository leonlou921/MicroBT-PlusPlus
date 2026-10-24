#!/usr/bin/env bash
set -euo pipefail

isolated_root="${MICROBT_PF_ISOLATED_ROOT:-/path/to/example}"
repo_root="${isolated_root}/nnUNet"
model_folder="${isolated_root}/nnUNet_results_scheduled_exact_200_monitored/Dataset001_Example/MicroBTPFTrainerA3ScheduledExact__nnUNetResEncUNetMPlans__3d_fullres"
fold_root="${model_folder}/fold_0"
preprocessed_root="/path/to/example"
raw_root="/path/to/example"
output="${isolated_root}/fast_full203_paired"
python_bin="${MICROBT_PF_PYTHON:-python3}"

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export nnUNet_preprocessed="${preprocessed_root}"
export nnUNet_raw="${raw_root}"
export nnUNet_results="${isolated_root}/nnUNet_results_scheduled_exact_200_monitored"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

compatibility_arguments=()
if [[ -n "${MICROBT_PF_COMPATIBLE_EVALUATOR_SHA256:-}" ]]; then
  compatibility_arguments=(
    --compatible-evaluator-sha256 "${MICROBT_PF_COMPATIBLE_EVALUATOR_SHA256}"
  )
fi

exec "${python_bin}" "${repo_root}/scripts/launch_microbt_pf_fast_full203.py" \
  --isolated-root "${isolated_root}" \
  --repo-root "${repo_root}" \
  --model-folder "${model_folder}" \
  --training-log "${fold_root}/training_log_2026_8_5_14_47_14.txt" \
  --split "${preprocessed_root}/Dataset001_Example/splits_final.json" \
  --output "${output}" \
  --manifest "${isolated_root}/artifacts/fast_full203_posttrain_launch.json" \
  --python "${python_bin}" \
  --admission-subject "EXAMPLE__ExampleCase-00003-000" \
  "${compatibility_arguments[@]}"
