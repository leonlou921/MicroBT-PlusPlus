#!/usr/bin/env bash
set -euo pipefail

isolated_root="/path/to/example"
python_bin="${MICROBT_SPP_OBR_PYTHON:-python3}"

if [[ "${1:-}" == "--print-contract" ]]; then
  printf '%s\n' '{"continuous_process":true,"exact_h0":false,"formal_epochs":60,"from_scratch":true,"full203":false,"isolated_root":"/path/to/example","random_seed":42,"snapshot_epochs":[15,30,35,40,45,50,55,60],"trainer":"MicroBTSPPOBRH0OffCausalTrainerScratch60"}'
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
trainer="MicroBTSPPOBRH0OffCausalTrainerScratch60"
results_root="${isolated_root}/nnUNet_results_obr_h0off_causal60"
model_folder="${results_root}/${dataset}/${trainer}__${plans}__${configuration}"
fold_folder="${model_folder}/fold_0"
probe_root="${isolated_root}/fixed_object_probes_h0off"
gate_manifest="${isolated_root}/artifacts/spp_obr_h0off_causal_gate60.json"
combined_log="${isolated_root}/artifacts/training_log_epochs000_059.txt"
fixed_manifest="${isolated_root}/input/fixed_cases_fold0.json"

active_fold0_trainer() {
  "${python_bin}" -c 'from pathlib import Path
for f in Path("/proc").glob("[0-9]*/cmdline"):
    try: a=[x.decode("utf-8",errors="replace") for x in f.read_bytes().split(b"\0") if x]
    except (OSError,PermissionError): continue
    for i,v in enumerate(a):
        if Path(v).name=="nnUNetv2_train" and a[i+1:i+4]==["1","3d_fullres","0"]: raise SystemExit(0)
raise SystemExit(1)'
}

[[ "${repo_root}" == "${isolated_root}" ]] || { echo "repository is outside H0-off isolation root"; exit 2; }
[[ -d "${cache_root}" ]] || { echo "precomputed target cache is missing: ${cache_root}"; exit 2; }
[[ -f "${fixed_manifest}" ]] || { echo "fixed probe manifest is missing"; exit 2; }
for path in "${results_root}" "${probe_root}" "${gate_manifest}" "${combined_log}"; do
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
unset MICROBT_PHASE_ALIGNED_STOP_EPOCHS MICROBT_CURRICULUM_GATE_EPOCHS

"${python_bin}" "${repo_root}/scripts/run_phase_training_stage.py" \
  --dataset 1 --configuration "${configuration}" --fold "${fold}" \
  --trainer "${trainer}" --plans "${plans}"

[[ -s "${fold_folder}/checkpoint_final.pth" ]] || { echo "missing checkpoint_final"; exit 3; }
for epoch in 15 30 35 40 45 50 55 60; do
  checkpoint_name="checkpoint_phase_$(printf '%03d' "${epoch}").pth"
  [[ -s "${fold_folder}/${checkpoint_name}" ]] || { echo "missing ${checkpoint_name}"; exit 3; }
  "${python_bin}" -m diagnostic.run_lesion_patch_diagnostic \
    --model-folder "${model_folder}" \
    --manifest "${fixed_manifest}" \
    --checkpoint-name "${checkpoint_name}" \
    --output "${probe_root}/epoch_$(printf '%03d' "${epoch}")" \
    --fold 0 --device cuda
done

cat "${fold_folder}"/training_log_*.txt > "${combined_log}"
"${python_bin}" "${repo_root}/scripts/check_microbt_spp_obr_h0off_gate.py" \
  --log "${combined_log}" \
  --telemetry "${fold_folder}/monitor/learning_components.jsonl" \
  --probe-45 "${probe_root}/epoch_045/objects.csv" \
  --probe-60 "${probe_root}/epoch_060/objects.csv" \
  --output "${gate_manifest}"
