#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
python_bin=${MICROBT_UMCTF_PYTHON:-python3}
export nnUNet_n_proc_DA=8
if [[ ${1:-} == --print-contract ]]; then
  printf '%s\n' '{"architecture_carrier":"B0","causal_change":"support_to_atom_continuation_learning","da_workers":8,"decoder_min_atom_mass":0.25,"fixed8":"abolished","fold":"all","formal_epochs":300,"formal_evaluation_set":"test","from_scratch":true,"hard_survival_mass":0.25,"lr_schedule_horizon":300,"measure_weight":"0.25*lambda","pretrained":false,"random_seed":42,"schedule":"b0_linear_measure_quadratic_transport","test_case_count":251,"trainer":"MicroBTUnitMassSACLFinal300Trainer","training_case_count":1011,"transport_weight":"0.25*lambda^2"}'
  exit 0
fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--print-contract]" >&2; exit 2; }

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
# Configure local resources explicitly; no host or private dataset is assumed.
: "${MICROBT_DATASET_ID:?Set MICROBT_DATASET_ID to your local dataset number}"
: "${MICROBT_DATASET_NAME:?Set MICROBT_DATASET_NAME to your local dataset folder}"
: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${MICROBT_PF_CACHE:?Set MICROBT_PF_CACHE}"
results_root=${root}/nnUNet_results_umctf_b0_sacl_c2_final300_foldall_da8
model_folder=${results_root}/${MICROBT_DATASET_NAME}/MicroBTUnitMassSACLFinal300Trainer__nnUNetResEncUNetMPlans__3d_fullres/fold_all
log_file=${root}/logs/formal_umctf_sacl_c2_final300_foldall_scratch300.log

[[ ! -e ${results_root} ]] || { echo "refusing to overwrite ${results_root}" >&2; exit 2; }
pgrep -af 'run_phase_training_stage.py.*MicroBTUnitMassSACLFinal300Trainer' | grep -v grep >/dev/null && { echo "SACL trainer already active" >&2; exit 2; }

mkdir -p "${root}/logs" "${root}/artifacts"
export PYTHONPATH=${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export nnUNet_preprocessed
export nnUNet_raw
export nnUNet_results=${results_root}
export MICROBT_PF_CACHE
export MICROBT_UMCTF_TARGET_EPOCHS=300
export MICROBT_UMCTF_FORMAL_DATA_PROTOCOL=fold_all_train1011_test251
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export nnUNet_n_proc_DA=8
export PYTHONUNBUFFERED=1
unset MICROBT_PF_MONITOR_MANIFEST MICROBT_PF_MONITOR_OUTPUT

"${python_bin}" "${repo_root}/scripts/build_umctf_sacl_c2_source_manifest.py" verify
"${python_bin}" "${repo_root}/scripts/run_phase_training_stage.py" \
  --dataset "${MICROBT_DATASET_ID}" --configuration 3d_fullres --fold all \
  --trainer MicroBTUnitMassSACLFinal300Trainer --plans nnUNetResEncUNetMPlans \
  2>&1 | tee "${log_file}"

[[ -s ${model_folder}/checkpoint_final.pth ]]
sha256sum \
  "${model_folder}/checkpoint_best.pth" \
  "${model_folder}/checkpoint_final.pth" \
  "${model_folder}/checkpoint_phase_060.pth" \
  "${model_folder}/checkpoint_phase_100.pth" \
  "${model_folder}/checkpoint_phase_150.pth" \
  "${model_folder}/checkpoint_phase_200.pth" \
  "${model_folder}/checkpoint_phase_250.pth" \
  "${model_folder}/checkpoint_phase_300.pth" \
  > "${root}/artifacts/checkpoint_c2_final300_foldall_sha256.txt"
