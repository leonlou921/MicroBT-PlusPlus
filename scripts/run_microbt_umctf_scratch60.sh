#!/usr/bin/env bash
set -euo pipefail

isolated_root=/path/to/example
python_bin=${MICROBT_UMCTF_PYTHON:-python3}
if [[ ${1:-} == --print-contract ]]; then
  printf '%s\n' '{"auto_continue_to200_on_pass":true,"exact_h0":false,"formal_epochs":60,"from_scratch":true,"full203_default":false,"isolated_root":"/path/to/example","pretrained":false,"random_seed":42,"trainer":"MicroBTUnitMassTrainerScratch60"}'
  exit 0
fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--print-contract]" >&2; exit 2; }
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ ${repo_root} == ${isolated_root} ]] || { echo "repository outside isolation root" >&2; exit 2; }
results_root=${isolated_root}/nnUNet_results_umctf
model_folder=${results_root}/Dataset001_Example/MicroBTUnitMassTrainerScratch60__nnUNetResEncUNetMPlans__3d_fullres
fold_folder=${model_folder}/fold_0
log_file=${isolated_root}/logs/formal_umctf_scratch60.log
gate_file=${isolated_root}/artifacts/umctf_stage_gate60.json
[[ ! -e ${results_root} ]] || { echo "refusing to overwrite ${results_root}" >&2; exit 2; }
if pgrep -af 'run_phase_training_stage.py.*1.*3d_fullres.*0' | grep -v 'run_microbt_umctf_scratch60.sh' >/dev/null; then
  echo "another Dataset001 fold-0 trainer is active" >&2; exit 2
fi
mkdir -p "${isolated_root}/logs" "${isolated_root}/artifacts"
export PYTHONPATH=${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export nnUNet_preprocessed=/path/to/example
export nnUNet_raw=/path/to/example
export nnUNet_results=${results_root}
export MICROBT_PF_CACHE=/path/to/example
export MICROBT_UMCTF_TARGET_EPOCHS=60
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-8}
export PYTHONUNBUFFERED=1
unset MICROBT_PF_MONITOR_MANIFEST MICROBT_PF_MONITOR_OUTPUT
"${python_bin}" "${repo_root}/scripts/run_phase_training_stage.py" \
  --dataset 1 --configuration 3d_fullres --fold 0 \
  --trainer MicroBTUnitMassTrainerScratch60 --plans nnUNetResEncUNetMPlans \
  2>&1 | tee "${log_file}"
[[ -s ${fold_folder}/checkpoint_final.pth ]]
sha256sum "${fold_folder}/checkpoint_best.pth" "${fold_folder}/checkpoint_final.pth" > "${isolated_root}/artifacts/checkpoint_sha256.txt"
"${python_bin}" "${repo_root}/scripts/evaluate_microbt_umctf_full_gate.py" \
  --model-folder "${model_folder}" --checkpoint checkpoint_final.pth \
  --manifest "${isolated_root}/input/fixed_cases_fold0.json" \
  --output "${gate_file}" --count-mae-lt 15
if "${python_bin}" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["full203_unlocked"] else 1)' "${gate_file}"; then
  exec "${repo_root}/scripts/continue_microbt_umctf_to200.sh"
fi

