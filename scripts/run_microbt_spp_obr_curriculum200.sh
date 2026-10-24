#!/usr/bin/env bash
set -euo pipefail

isolated_root="/path/to/example"
python_bin="${MICROBT_SPP_OBR_PYTHON:-python3}"
train_bin="${MICROBT_SPP_OBR_TRAIN_BIN:-${python_bin%/python}/nnUNetv2_train}"

if [[ "${1:-}" == "--print-contract" ]]; then
  printf '%s\n' '{"auto_full203":true,"formal_epochs":200,"from_scratch":true,"gate_epochs":30,"isolated_root":"/path/to/example","n_proc_da":8,"random_seed":42,"resume_same_optimizer":true,"trainer":"MicroBTSPPOBRCurriculumTrainerScratch200"}'
  exit 0
fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--print-contract]" >&2; exit 2; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
preprocessed_root="/path/to/example"
raw_root="/path/to/example"
cache_root="/path/to/example"
dataset="Dataset001_Example"
plans="nnUNetResEncUNetMPlans"
configuration="3d_fullres"
fold="0"
trainer="MicroBTSPPOBRCurriculumTrainerScratch200"
results_root="${isolated_root}/nnUNet_results_obr_curriculum200"
model_folder="${results_root}/${dataset}/${trainer}__${plans}__${configuration}"
fold_folder="${model_folder}/fold_0"
evaluation_output="${isolated_root}/full203_obr_curriculum200"
gate_manifest="${isolated_root}/artifacts/spp_obr_curriculum_gate30.json"
evaluation_manifest="${isolated_root}/artifacts/spp_obr_curriculum_full203.json"
split="${preprocessed_root}/${dataset}/splits_final.json"

active_fold0_trainer() {
  "${python_bin}" -c 'from pathlib import Path
for f in Path("/proc").glob("[0-9]*/cmdline"):
    try: a=[x.decode("utf-8",errors="replace") for x in f.read_bytes().split(b"\0") if x]
    except (OSError,PermissionError): continue
    for i,v in enumerate(a):
        if Path(v).name=="nnUNetv2_train" and a[i+1:i+4]==["1","3d_fullres","0"]: raise SystemExit(0)
raise SystemExit(1)'
}

[[ "${repo_root}" == "${isolated_root}" ]] || { echo "repository is outside curriculum isolation root"; exit 2; }
[[ -f "${split}" ]] || { echo "fold split is missing: ${split}"; exit 2; }
[[ -d "${cache_root}" ]] || { echo "precomputed target cache is missing: ${cache_root}"; exit 2; }
for path in "${results_root}" "${evaluation_output}" "${gate_manifest}" "${evaluation_manifest}"; do
  [[ ! -e "${path}" ]] || { echo "refusing to overwrite ${path}"; exit 2; }
done
if active_fold0_trainer; then
  echo "another fold-0 trainer is active"
  exit 2
fi

mkdir -p "${isolated_root}/artifacts" "${isolated_root}/logs"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export nnUNet_preprocessed="${preprocessed_root}"
export nnUNet_raw="${raw_root}"
export nnUNet_results="${results_root}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-8}"
export MICROBT_PF_CACHE="${cache_root}"
unset MICROBT_PF_MONITOR_MANIFEST MICROBT_PF_MONITOR_OUTPUT

export MICROBT_CURRICULUM_GATE_EPOCHS=30
"${train_bin}" 1 "${configuration}" "${fold}" -tr "${trainer}" -p "${plans}"

gate_log="$(ls -1t "${fold_folder}"/training_log_*.txt | head -1)"
"${python_bin}" "${repo_root}/scripts/check_microbt_spp_obr_curriculum_gate.py" \
  --log "${gate_log}" \
  --telemetry "${fold_folder}/monitor/learning_components.jsonl" \
  --output "${gate_manifest}"

unset MICROBT_CURRICULUM_GATE_EPOCHS
"${train_bin}" 1 "${configuration}" "${fold}" -tr "${trainer}" -p "${plans}" -c

[[ -s "${fold_folder}/checkpoint_best.pth" ]] || { echo "missing checkpoint_best"; exit 3; }
[[ -s "${fold_folder}/checkpoint_final.pth" ]] || { echo "missing checkpoint_final"; exit 3; }
best_hash="$(sha256sum "${fold_folder}/checkpoint_best.pth" | cut -d' ' -f1)"
final_hash="$(sha256sum "${fold_folder}/checkpoint_final.pth" | cut -d' ' -f1)"

"${python_bin}" "${repo_root}/nnunetv2/scripts/run_microbt_pf_full203.py" \
  --model-folder "${model_folder}" --split "${split}" --output "${evaluation_output}" \
  --checkpoint-best-sha256 "${best_hash}" --checkpoint-final-sha256 "${final_hash}" \
  --admission-subject "EXAMPLE__ExampleCase-00003-000" --device cuda

"${python_bin}" -c 'import json,sys; from pathlib import Path
out=Path(sys.argv[1]); root=Path(sys.argv[2]); payload={"status":"complete","from_scratch":True,"seed":42,"best":json.loads((root/"best"/"summary.json").read_text()),"final":json.loads((root/"final"/"summary.json").read_text())}; out.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=True)+"\n")' "${evaluation_manifest}" "${evaluation_output}"
