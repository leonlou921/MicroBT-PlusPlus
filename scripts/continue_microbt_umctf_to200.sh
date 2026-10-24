#!/usr/bin/env bash
set -euo pipefail
isolated_root=/path/to/example
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
gate_file=${isolated_root}/artifacts/umctf_stage_gate60.json
python_bin=${MICROBT_UMCTF_PYTHON:-python3}
"${python_bin}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["full203_unlocked"] is True' "${gate_file}"
export PYTHONPATH=${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export nnUNet_preprocessed=/path/to/example
export nnUNet_raw=/path/to/example
export nnUNet_results=${isolated_root}/nnUNet_results_umctf
export MICROBT_PF_CACHE=/path/to/example
export MICROBT_UMCTF_TARGET_EPOCHS=200
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-8}
export PYTHONUNBUFFERED=1
"${python_bin}" "${repo_root}/scripts/run_phase_training_stage.py" \
  --continue-training --dataset 1 --configuration 3d_fullres --fold 0 \
  --trainer MicroBTUnitMassTrainerScratch60 --plans nnUNetResEncUNetMPlans \
  2>&1 | tee -a "${isolated_root}/logs/formal_umctf_scratch60.log"
