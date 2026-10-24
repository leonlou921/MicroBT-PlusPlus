#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 5 ]] || { echo "usage: $0 MODEL_FOLDER CHECKPOINT OUTPUT IMAGES_TS LABELS_TS" >&2; exit 2; }
model_folder=$1
checkpoint=$2
output=$3
source_root=$4
labels_ts=$5
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
python_bin=${MICROBT_UMCTF_PYTHON:-python3}
checkpoint_path=${model_folder}/fold_all/${checkpoint}
[[ -s ${checkpoint_path} ]]
checkpoint_sha=$(sha256sum "${checkpoint_path}" | awk '{print $1}')
workers=${MICROBT_TEST251_WORKERS:-7}
[[ ${workers} =~ ^[0-9]+$ && ${workers} -ge 2 ]] || {
  echo "MICROBT_TEST251_WORKERS must be an integer >= 2" >&2
  exit 2
}
shard_root=${output}.shards
plan=${shard_root}/plan.json
merge_audit=${output}/shard_merge_audit.json
total_expected_count=$(find "${labels_ts}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)
[[ ${total_expected_count} -gt 0 ]] || { echo "empty labelsTs: ${labels_ts}" >&2; exit 2; }

export PYTHONPATH=${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export nnUNet_preprocessed=${NNUNET_PREPROCESSED_ROOT:?}
export nnUNet_raw=${NNUNET_RAW_ROOT:?}
export nnUNet_results=$(dirname "$(dirname "${model_folder}")")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

evaluator_sha=$("${python_bin}" -c "from scripts.evaluate_microbt_crossdataset_test import _test_evaluator_sha256; print(_test_evaluator_sha256())")
"${python_bin}" "${repo_root}/scripts/crossdataset_shard_accel.py" prepare \
  --labels-ts "${labels_ts}" --source-root "${source_root}" \
  --main-output "${output}" --shard-root "${shard_root}" \
  --workers "${workers}" --checkpoint-sha256 "${checkpoint_sha}" \
  --evaluator-sha256 "${evaluator_sha}"

pids=()
cleanup_workers() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_workers INT TERM

for shard_index in $(seq 0 $((workers - 1))); do
  shard_name=shard-${shard_index}
  subject_file=${shard_root}/${shard_name}-subjects.json
  shard_output=${shard_root}/${shard_name}
  shard_log=${shard_root}/${shard_name}.log
  expected_count=$("${python_bin}" -c "import json; print(json.load(open('${plan}'))['shards'][${shard_index}]['expected_count'])")
  if [[ ${expected_count} -eq 0 ]]; then
    continue
  fi
  "${python_bin}" "${repo_root}/scripts/evaluate_microbt_crossdataset_test.py" \
    --model-folder "${model_folder}" --checkpoint "${checkpoint}" \
    --checkpoint-sha256 "${checkpoint_sha}" \
    --source-root "${source_root}" --labels-ts "${labels_ts}" \
    --output-dir "${shard_output}" --expected-count "${expected_count}" \
    --total-expected-count "${total_expected_count}" --subject-ids-json "${subject_file}" \
    >"${shard_log}" 2>&1 &
  pids+=("$!")
done

worker_status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    worker_status=1
  fi
done
trap - INT TERM
[[ ${worker_status} -eq 0 ]] || { echo "one or more test251 shard workers failed" >&2; exit 3; }

merge_args=(
  merge --plan "${plan}" --main-output "${output}" --audit "${merge_audit}"
)
for shard_index in $(seq 0 $((workers - 1))); do
  merge_args+=(--shard-output "${shard_root}/shard-${shard_index}")
done
"${python_bin}" "${repo_root}/scripts/crossdataset_shard_accel.py" "${merge_args[@]}"

"${python_bin}" "${repo_root}/scripts/evaluate_microbt_crossdataset_test.py" \
  --model-folder "${model_folder}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-sha256 "${checkpoint_sha}" \
  --source-root "${source_root}" \
  --labels-ts "${labels_ts}" \
  --output-dir "${output}" \
  --expected-count "${total_expected_count}" \
  --total-expected-count "${total_expected_count}"

"${python_bin}" "${repo_root}/scripts/export_microbt_crossdataset_native.py" \
  --model-folder "${model_folder}" \
  --evaluation-output "${output}" \
  --workers "${MICROBT_NATIVE_EXPORT_WORKERS:-3}"
