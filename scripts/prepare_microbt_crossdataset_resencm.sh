#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 5 ]] || { echo "usage: $0 DATASET_ID DATASET_NAME DATA_ROOT AUDIT MANIFEST" >&2; exit 2; }
dataset_id=$1
dataset_name=$2
data_root=$3
audit=$4
manifest=$5
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
python_bin=${MICROBT_UMCTF_PYTHON:-python3}

export PYTHONPATH=${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export nnUNet_raw=${data_root}/nnUNet_raw
export nnUNet_preprocessed=${data_root}/nnUNet_preprocessed
export nnUNet_results=${MICROBT_CROSSDATA_DUMMY_RESULTS:-${repo_root}/_planning_results}

"${python_bin}" "${repo_root}/scripts/audit_crossdataset_data.py" \
  --dataset-id "${dataset_id}" --dataset-name "${dataset_name}" \
  --data-root "${data_root}" --output "${audit}.raw.json"

plan_path=${data_root}/nnUNet_preprocessed/${dataset_name}/nnUNetResEncUNetMPlans.json
if [[ -f "${plan_path}" ]] && \
  "${python_bin}" "${repo_root}/scripts/audit_crossdataset_data.py" \
    --dataset-id "${dataset_id}" --dataset-name "${dataset_name}" \
    --data-root "${data_root}" --output "${audit}" --require-preprocessed; then
  echo "preprocessing already complete; reusing frozen artifacts"
else
  "${python_bin}" -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints \
    -d "${dataset_id}" -pl nnUNetPlannerResEncM -c 3d_fullres -np 4 \
    --verify_dataset_integrity
  "${python_bin}" "${repo_root}/scripts/audit_crossdataset_data.py" \
    --dataset-id "${dataset_id}" --dataset-name "${dataset_name}" \
    --data-root "${data_root}" --output "${audit}" --require-preprocessed
fi
"${python_bin}" "${repo_root}/scripts/build_crossdataset_source_manifest.py" write \
  --repo "${repo_root}" --data-root "${data_root}" --dataset-name "${dataset_name}" \
  --manifest "${manifest}"
