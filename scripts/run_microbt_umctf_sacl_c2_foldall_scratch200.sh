#!/usr/bin/env bash
set -euo pipefail

isolated_root=/path/to/example
python_bin=${MICROBT_UMCTF_PYTHON:-python3}
export nnUNet_n_proc_DA=8
if [[ ${1:-} == --print-contract ]]; then
  printf '%s\n' '{"architecture_carrier":"B0","causal_change":"support_to_atom_continuation_learning","da_workers":8,"da_workers_policy":"hard_locked","decoder_min_atom_mass":0.25,"fixed8":"abolished","fold":"all","formal_epochs":200,"formal_evaluation_set":"test","from_scratch":true,"hard_survival_mass":0.25,"lr_schedule_horizon":200,"measure_weight":"0.25*lambda","pretrained":false,"random_seed":42,"schedule":"b0_linear_measure_quadratic_transport","test_case_count":251,"trainer":"MicroBTUnitMassSACLTrainerScratch60","training_case_count":1011,"transport_weight":"0.25*lambda^2"}'
  exit 0
fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--print-contract]" >&2; exit 2; }

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ ${repo_root} == ${isolated_root} ]] || { echo "repository outside isolation root" >&2; exit 2; }
results_root=${isolated_root}/nnUNet_results_umctf_b0_sacl_c2_foldall_da8
model_folder=${results_root}/Dataset001_Example/MicroBTUnitMassSACLTrainerScratch60__nnUNetResEncUNetMPlans__3d_fullres
fold_folder=${model_folder}/fold_all
log_file=${isolated_root}/logs/formal_umctf_sacl_c2_foldall_scratch200.log

[[ ! -e ${results_root} ]] || { echo "refusing to overwrite ${results_root}" >&2; exit 2; }
if pgrep -af 'run_phase_training_stage.py.*1.*3d_fullres.*all' | grep -v 'run_microbt_umctf_sacl_c2_foldall_scratch200.sh' >/dev/null; then
  echo "another Dataset001 fold-all trainer is active" >&2
  exit 2
fi

mkdir -p "${isolated_root}/logs" "${isolated_root}/artifacts"
export PYTHONPATH=${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export nnUNet_preprocessed=/path/to/example
export nnUNet_raw=/path/to/example
export nnUNet_results=${results_root}
export MICROBT_PF_CACHE=/path/to/example
export MICROBT_UMCTF_TARGET_EPOCHS=200
export MICROBT_UMCTF_FORMAL_DATA_PROTOCOL=fold_all_train1011_test251
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export nnUNet_n_proc_DA=8
export PYTHONUNBUFFERED=1
unset MICROBT_PF_MONITOR_MANIFEST MICROBT_PF_MONITOR_OUTPUT

"${python_bin}" "${repo_root}/scripts/build_umctf_sacl_c2_source_manifest.py" verify
"${python_bin}" "${repo_root}/scripts/run_phase_training_stage.py" \
  --dataset 1 --configuration 3d_fullres --fold all \
  --trainer MicroBTUnitMassSACLTrainerScratch60 --plans nnUNetResEncUNetMPlans \
  2>&1 | tee "${log_file}"

[[ -s ${fold_folder}/checkpoint_final.pth ]]
sha256sum "${fold_folder}/checkpoint_best.pth" "${fold_folder}/checkpoint_final.pth" \
  "${fold_folder}/checkpoint_phase_060.pth" "${fold_folder}/checkpoint_phase_100.pth" \
  "${fold_folder}/checkpoint_phase_150.pth" "${fold_folder}/checkpoint_phase_200.pth" \
  > "${isolated_root}/artifacts/checkpoint_foldall_sha256.txt"
