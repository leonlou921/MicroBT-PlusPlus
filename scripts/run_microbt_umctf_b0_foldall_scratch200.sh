#!/usr/bin/env bash
set -euo pipefail

root=/path/to/example
python_bin=${MICROBT_UMCTF_PYTHON:-python3}
if [[ ${1:-} == --print-contract ]]; then
  printf '%s\n' '{"architecture_carrier":"B0","causal_change":"none","da_workers":8,"decoder_min_atom_mass":0.25,"fixed8":"abolished","fold":"all","formal_epochs":200,"formal_evaluation_set":"test","from_scratch":true,"hard_survival_mass":0.25,"lr_schedule_horizon":200,"pretrained":false,"random_seed":42,"test_case_count":251,"trainer":"MicroBTUnitMassSurvivalTrainerScratch60","training_case_count":1011}'
  exit 0
fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--print-contract]" >&2; exit 2; }

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
results_root=${root}/nnUNet_results_umctf_b0_foldall_da8_200
model_folder=${results_root}/Dataset001_Example/MicroBTUnitMassSurvivalTrainerScratch60__nnUNetResEncUNetMPlans__3d_fullres/fold_all
[[ ! -e ${results_root} ]] || { echo "refusing to overwrite ${results_root}" >&2; exit 2; }

export PYTHONPATH=${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export nnUNet_preprocessed=/path/to/example
export nnUNet_raw=/path/to/example
export nnUNet_results=${results_root}
export MICROBT_PF_CACHE=/path/to/example
export MICROBT_UMCTF_TARGET_EPOCHS=200
export MICROBT_UMCTF_FORMAL_DATA_PROTOCOL=fold_all_train1011_test251
export CUDA_VISIBLE_DEVICES=0
export nnUNet_n_proc_DA=8
export PYTHONUNBUFFERED=1
unset MICROBT_PF_MONITOR_MANIFEST MICROBT_PF_MONITOR_OUTPUT

"${python_bin}" "${repo_root}/scripts/build_umctf_sacl_c2_source_manifest.py" verify
"${python_bin}" "${repo_root}/scripts/run_phase_training_stage.py" \
  --dataset 1 --configuration 3d_fullres --fold all \
  --trainer MicroBTUnitMassSurvivalTrainerScratch60 --plans nnUNetResEncUNetMPlans \
  2>&1 | tee "${root}/logs/formal_umctf_b0_foldall_scratch200.log"

[[ -s ${model_folder}/checkpoint_final.pth ]]
sha256sum "${model_folder}"/checkpoint_{best,final,phase_060,phase_100,phase_150,phase_200}.pth \
  > "${root}/artifacts/checkpoint_b0_foldall_sha256.txt"
