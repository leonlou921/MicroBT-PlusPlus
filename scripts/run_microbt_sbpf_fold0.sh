#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
isolated_root="/path/to/example"
source_root="/path/to/example"
source_repo="${source_root}/nnUNet"
source_checkpoint="${MICROBT_SBPF_SOURCE_CHECKPOINT:-${source_root}/nnUNet_results_scheduled_exact_200_monitored/Dataset001_Example/MicroBTPFTrainerA3ScheduledExact__nnUNetResEncUNetMPlans__3d_fullres/fold_0/checkpoint_best.pth}"
python_bin="${MICROBT_SBPF_PYTHON:-python3}"
train_bin="${MICROBT_SBPF_TRAIN_BIN:-${python_bin%/python}/nnUNetv2_train}"
preprocessed_root="/path/to/example"
raw_root="/path/to/example"
results_root="${isolated_root}/nnUNet_results"
model_folder="${results_root}/Dataset001_Example/MicroBTSBPFTrainerScheduledExact__nnUNetResEncUNetMPlans__3d_fullres"
fold_root="${model_folder}/fold_0"
migrated_checkpoint="${isolated_root}/artifacts/a3_best_to_sbpf.pth"
monitor_manifest="${repo_root}/monitor/fixed_cases_fold0.json"
monitor_output="${fold_root}/monitor"
evaluation_output="${isolated_root}/full203_paired"
evaluation_manifest="${isolated_root}/artifacts/full203_posttrain_launch.json"
trainer="MicroBTSBPFTrainerScheduledExact"

[[ "${repo_root}" == "${isolated_root}" ]] || { echo "repository is outside SB-PF isolation root"; exit 2; }
[[ -f "${source_checkpoint}" ]] || { echo "source checkpoint is missing"; exit 2; }
[[ -f "${monitor_manifest}" ]] || { echo "fixed-case monitor manifest is missing"; exit 2; }
[[ ! -e "${fold_root}" ]] || { echo "refusing to overwrite an existing SB-PF fold"; exit 2; }
[[ ! -e "${evaluation_output}" ]] || { echo "refusing to overwrite an existing full203 evaluation"; exit 2; }
[[ ! -e "${migrated_checkpoint}" ]] || { echo "refusing to overwrite migrated checkpoint"; exit 2; }

source_head_before="$(git -C "${source_repo}" rev-parse HEAD)"
source_status_before="$(git -C "${source_repo}" status --porcelain=v1 --untracked-files=no | sha256sum)"
mkdir -p "${isolated_root}/artifacts" "${isolated_root}/logs" "${results_root}"

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export nnUNet_preprocessed="${preprocessed_root}"
export nnUNet_raw="${raw_root}"
export nnUNet_results="${results_root}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-8}"
# The 1011-case A3 cache is immutable input, like nnUNet_preprocessed. Reusing it
# avoids rebuilding identical target fields and topology metadata during training.
export MICROBT_PF_CACHE="${source_root}/cache"
export MICROBT_PF_MONITOR_MANIFEST="${monitor_manifest}"
export MICROBT_PF_MONITOR_OUTPUT="${monitor_output}"

"${python_bin}" -m nnunetv2.training.nnUNetTrainer.microbt_sbpf_checkpoint \
  "${source_checkpoint}" "${migrated_checkpoint}"

[[ "$(git -C "${source_repo}" rev-parse HEAD)" == "${source_head_before}" ]] || { echo "source HEAD changed during preflight"; exit 2; }
[[ "$(git -C "${source_repo}" status --porcelain=v1 --untracked-files=no | sha256sum)" == "${source_status_before}" ]] || { echo "source worktree changed during preflight"; exit 2; }

"${train_bin}" 1 3d_fullres 0 \
  -tr "${trainer}" \
  -p nnUNetResEncUNetMPlans \
  -pretrained_weights "${migrated_checkpoint}"

[[ -f "${fold_root}/checkpoint_best.pth" ]] || { echo "training ended without checkpoint_best.pth"; exit 3; }
[[ -f "${fold_root}/checkpoint_final.pth" ]] || { echo "training ended without checkpoint_final.pth"; exit 3; }
shopt -s nullglob
training_logs=("${fold_root}"/training_log_*.txt)
(( ${#training_logs[@]} > 0 )) || { echo "training log is missing"; exit 3; }
training_log="${training_logs[${#training_logs[@]}-1]}"

"${python_bin}" "${repo_root}/scripts/launch_microbt_pf_fast_full203.py" \
  --isolated-root "${isolated_root}" \
  --repo-root "${repo_root}" \
  --model-folder "${model_folder}" \
  --training-log "${training_log}" \
  --split "${preprocessed_root}/Dataset001_Example/splits_final.json" \
  --output "${evaluation_output}" \
  --manifest "${evaluation_manifest}" \
  --python "${python_bin}" \
  --admission-subject "EXAMPLE__ExampleCase-00003-000"
