#!/usr/bin/env bash
set -euo pipefail

isolated_root="/path/to/example"
python_bin="${MICROBT_SPP_OBR_PYTHON:-python3}"

if [[ "${1:-}" == "--print-contract" ]]; then
  printf '%s\n' '{"auto_full203":true,"fixed_probe_epochs":[15,30,45,60],"fixed_probe_objects":52,"formal_epochs":200,"from_scratch":true,"hard_gate_epoch":60,"isolated_root":"/path/to/example","momentum_reset_epoch":15,"phase_learning_rate":0.003,"random_seed":42,"trainer":"MicroBTSPPOBRPhaseAlignedTrainerScratch200"}'
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
trainer="MicroBTSPPOBRPhaseAlignedTrainerScratch200"
results_root="${isolated_root}/nnUNet_results_obr_phasealigned200"
model_folder="${results_root}/${dataset}/${trainer}__${plans}__${configuration}"
fold_folder="${model_folder}/fold_0"
probe_root="${isolated_root}/fixed_object_probes"
evaluation_output="${isolated_root}/full203_obr_phasealigned200"
gate_manifest="${isolated_root}/artifacts/spp_obr_phase_aligned_gate60.json"
evaluation_manifest="${isolated_root}/artifacts/spp_obr_phase_aligned_full203.json"
combined_log="${isolated_root}/artifacts/training_log_epochs000_059.txt"
fixed_manifest="${isolated_root}/input/fixed_cases_fold0.json"
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

[[ "${repo_root}" == "${isolated_root}" ]] || { echo "repository is outside phase-aligned isolation root"; exit 2; }
[[ -f "${split}" ]] || { echo "fold split is missing: ${split}"; exit 2; }
[[ -d "${cache_root}" ]] || { echo "precomputed target cache is missing: ${cache_root}"; exit 2; }
[[ -f "${fixed_manifest}" ]] || { echo "fixed probe manifest is missing"; exit 2; }
for path in "${results_root}" "${probe_root}" "${evaluation_output}" "${gate_manifest}" "${evaluation_manifest}" "${combined_log}"; do
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

continuation=()
for stage in 15 30 45 60; do
  export MICROBT_PHASE_ALIGNED_STOP_EPOCHS="${stage}"
  "${python_bin}" "${repo_root}/scripts/run_phase_training_stage.py" \
    --dataset 1 --configuration "${configuration}" --fold "${fold}" \
    --trainer "${trainer}" --plans "${plans}" "${continuation[@]}"
  checkpoint_name="checkpoint_phase_$(printf '%03d' "${stage}").pth"
  [[ ! -e "${fold_folder}/${checkpoint_name}" ]] || { echo "refusing to overwrite phase checkpoint"; exit 2; }
  cp "${fold_folder}/checkpoint_final.pth" "${fold_folder}/${checkpoint_name}"
  "${python_bin}" -m diagnostic.run_lesion_patch_diagnostic \
    --model-folder "${model_folder}" \
    --manifest "${fixed_manifest}" \
    --checkpoint-name "${checkpoint_name}" \
    --output "${probe_root}/epoch_$(printf '%03d' "${stage}")" \
    --fold 0 --device cuda
  continuation=(--continue-training)
done
unset MICROBT_PHASE_ALIGNED_STOP_EPOCHS

cat "${fold_folder}"/training_log_*.txt > "${combined_log}"
"${python_bin}" "${repo_root}/scripts/check_microbt_spp_obr_phase_aligned_gate.py" \
  --log "${combined_log}" \
  --telemetry "${fold_folder}/monitor/learning_components.jsonl" \
  --probe-45 "${probe_root}/epoch_045/objects.csv" \
  --probe-60 "${probe_root}/epoch_060/objects.csv" \
  --output "${gate_manifest}"

"${python_bin}" "${repo_root}/scripts/run_phase_training_stage.py" \
  --dataset 1 --configuration "${configuration}" --fold "${fold}" \
  --trainer "${trainer}" --plans "${plans}" --continue-training

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
