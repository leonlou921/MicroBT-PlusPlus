#!/usr/bin/env bash
set -euo pipefail

isolated_root="/path/to/example"
python_bin="${MICROBT_CTF_PYTHON:-python3}"

if [[ "${1:-}" == "--print-contract" ]]; then
  printf '%s\n' '{"exact_h0":false,"formal_epochs":60,"from_scratch":true,"full203_default":false,"isolated_root":"/path/to/example","pretrained":false,"random_seed":42,"snapshot_epochs":[15,30,40,50,60],"stage_gate_before_full203":true,"target_builder":"centroid_only_no_geodesic","trainer":"MicroBTCTFTrainerScratch60","transport_implementation":"vectorized_equal_instance"}'
  exit 0
fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--print-contract]" >&2; exit 2; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
dataset="Dataset001_Example"
plans="nnUNetResEncUNetMPlans"
configuration="3d_fullres"
trainer="MicroBTCTFTrainerScratch60"
results_root="${isolated_root}/nnUNet_results_ctf_scratch60"
model_folder="${results_root}/${dataset}/${trainer}__${plans}__${configuration}"
fold_folder="${model_folder}/fold_0"
log_file="${isolated_root}/logs/formal_ctf_scratch60.log"
exit_file="${isolated_root}/artifacts/formal_ctf_scratch60.exit"

[[ "${repo_root}" == "${isolated_root}" ]] || { echo "repository is outside CTF isolation root"; exit 2; }
for path in "${results_root}" "${exit_file}"; do
  [[ ! -e "${path}" ]] || { echo "refusing to overwrite ${path}"; exit 2; }
done
active_dataset001_fold0_trainer() {
  "${python_bin}" - <<'PY'
from pathlib import Path

patterns = (
    ("nnUNetv2_train", "1", "3d_fullres", "0"),
    ("run_phase_training_stage.py", "1", "3d_fullres", "0"),
)
for cmdline_path in Path("/proc").glob("[0-9]*/cmdline"):
    try:
        cmd = cmdline_path.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except (OSError, PermissionError):
        continue
    if "run_microbt_ctf_scratch60.sh" in cmd:
        continue
    if any(all(token in cmd for token in pattern) for pattern in patterns):
        print(cmdline_path.parent.name, cmd)
        raise SystemExit(0)
raise SystemExit(1)
PY
}
if active_dataset001_fold0_trainer; then
  echo "another Dataset001 fold-0 trainer is active" >&2
  exit 2
fi

mkdir -p "${isolated_root}/logs" "${isolated_root}/artifacts"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export nnUNet_preprocessed="/path/to/example"
export nnUNet_raw="/path/to/example"
export nnUNet_results="${results_root}"
export MICROBT_PF_CACHE="/path/to/example"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-8}"
unset MICROBT_PF_MONITOR_MANIFEST MICROBT_PF_MONITOR_OUTPUT

set +e
"${python_bin}" "${repo_root}/scripts/run_phase_training_stage.py" \
  --dataset 1 --configuration "${configuration}" --fold 0 \
  --trainer "${trainer}" --plans "${plans}" 2>&1 | tee "${log_file}"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "${status}" > "${exit_file}"
[[ ${status} -eq 0 ]] || exit "${status}"

[[ -s "${fold_folder}/checkpoint_final.pth" ]] || { echo "missing checkpoint_final"; exit 3; }
for epoch in 15 30 40 50 60; do
  [[ -s "${fold_folder}/checkpoint_phase_$(printf '%03d' "${epoch}").pth" ]] || exit 3
done
sha256sum "${fold_folder}/checkpoint_best.pth" "${fold_folder}/checkpoint_final.pth" \
  > "${isolated_root}/artifacts/checkpoint_sha256.txt"

# This evaluator is fail-closed: it can unlock full203 only after every
# segmentation, counting, proposal, centroid, box, and micro-lesion gate exists.
"${python_bin}" "${repo_root}/scripts/evaluate_microbt_ctf_stage_gate.py" \
  --model-folder "${model_folder}" \
  --checkpoint checkpoint_final.pth \
  --manifest "${isolated_root}/input/fixed_cases_fold0.json" \
  --output "${isolated_root}/artifacts/ctf_stage_gate60.json"
