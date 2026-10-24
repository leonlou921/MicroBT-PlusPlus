#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
python_bin=${MICROBT_UMCTF_PYTHON:-python3}
if [[ ${1:-} == --print-contract ]]; then
  printf '%s\n' '{"continue_training":true,"formal_epochs":300,"fold":"all","required_current_epoch":200,"scientific_schedule":"unchanged_endpoint_after_epoch29"}'
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
checkpoint=${model_folder}/checkpoint_latest.pth
log_file=${root}/logs/formal_umctf_sacl_c2_final300_foldall_resume200to300.log

[[ -s ${checkpoint} ]] || { echo "missing checkpoint_latest.pth" >&2; exit 2; }
pgrep -af 'run_phase_training_stage.py.*MicroBTUnitMassSACLFinal300Trainer' | grep -v grep >/dev/null && { echo "SACL trainer already active" >&2; exit 2; }

"${python_bin}" - "${checkpoint}" <<'PY'
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
epoch = int(checkpoint.get("current_epoch", -1))
optimizer_entries = len(checkpoint.get("optimizer_state", {}).get("state", {}))
if epoch != 200 or optimizer_entries != 256:
    raise SystemExit(f"checkpoint contract failed: epoch={epoch}, optimizer_entries={optimizer_entries}")
print(f"verified recovery checkpoint: epoch={epoch}, optimizer_entries={optimizer_entries}")
PY

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
  --continue-training --dataset "${MICROBT_DATASET_ID}" --configuration 3d_fullres --fold all \
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
