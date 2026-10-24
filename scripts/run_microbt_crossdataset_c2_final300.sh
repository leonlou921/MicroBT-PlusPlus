#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 6 ]] || { echo "usage: $0 DATASET_ID DATASET_NAME DATA_ROOT RESULTS_ROOT CACHE_ROOT MANIFEST" >&2; exit 2; }
dataset_id=$1
dataset_name=$2
data_root=$3
results_root=$4
cache_root=$5
manifest=$6
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
python_bin=${MICROBT_UMCTF_PYTHON:-python3}
trainer=MicroBTUnitMassSACLFinal300Trainer
model_folder=${results_root}/${dataset_name}/${trainer}__nnUNetResEncUNetMPlans__3d_fullres/fold_all
train_count=$(find "${data_root}/nnUNet_raw/${dataset_name}/labelsTr" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)
test_count=$(find "${data_root}/nnUNet_raw/${dataset_name}/labelsTs" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)

[[ ! -e ${results_root} ]] || { echo "refusing to overwrite ${results_root}" >&2; exit 2; }
if pgrep -af "run_phase_training_stage.py.*--dataset ${dataset_id}.*${trainer}" | grep -v grep >/dev/null; then
  echo "dataset ${dataset_id} final trainer already active" >&2
  exit 2
fi
mkdir -p "${results_root}" "${cache_root}/${dataset_name}"
export PYTHONPATH=${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export nnUNet_preprocessed=${data_root}/nnUNet_preprocessed
export nnUNet_raw=${data_root}/nnUNet_raw
export nnUNet_results=${results_root}
export MICROBT_PF_CACHE=${cache_root}/${dataset_name}
export MICROBT_UMCTF_TARGET_EPOCHS=300
export MICROBT_UMCTF_FORMAL_DATA_PROTOCOL=fold_all_train${train_count}_test${test_count}
export nnUNet_n_proc_DA=8
export PYTHONUNBUFFERED=1
unset MICROBT_PF_MONITOR_MANIFEST MICROBT_PF_MONITOR_OUTPUT

"${python_bin}" "${repo_root}/scripts/build_crossdataset_source_manifest.py" verify \
  --repo "${repo_root}" --data-root "${data_root}" --dataset-name "${dataset_name}" \
  --manifest "${manifest}"
"${python_bin}" -m nnunetv2.scripts.build_microbt_pf_metadata \
  --dataset "${dataset_id}" --configuration 3d_fullres \
  --plans nnUNetResEncUNetMPlans --workers 8
metadata_directory=${MICROBT_PF_CACHE}/${dataset_name}/nnUNetResEncUNetMPlans/3d_fullres
metadata_count=$(find "${metadata_directory}" -mindepth 2 -maxdepth 2 -type f -name metadata.json | wc -l)
[[ ${metadata_count} -eq ${train_count} ]] || {
  echo "PF metadata coverage ${metadata_count}/${train_count}" >&2
  exit 2
}
printf 'MicroBT cross-dataset frozen protocol: dataset=%s fold=all train=%s test=%s epochs=300 horizon=300 seed=42 trainer=%s\n' \
  "${dataset_name}" "${train_count}" "${test_count}" "${trainer}"
"${python_bin}" "${repo_root}/scripts/run_phase_training_stage.py" \
  --dataset "${dataset_id}" --configuration 3d_fullres --fold all \
  --trainer "${trainer}" --plans nnUNetResEncUNetMPlans

[[ -s ${model_folder}/checkpoint_phase_300.pth && -s ${model_folder}/checkpoint_final.pth ]]
"${python_bin}" "${repo_root}/scripts/verify_crossdataset_checkpoints.py" \
  --model-folder "${model_folder}" --expected-epoch 300 \
  --output "${results_root}/checkpoint_audit.json"
