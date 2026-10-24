#!/usr/bin/env bash
set -euo pipefail

isolated_root="/path/to/example"
python_bin="${MICROBT_SPP_OBR_PYTHON:-python3}"
train_bin="${MICROBT_SPP_OBR_TRAIN_BIN:-${python_bin%/python}/nnUNetv2_train}"

if [[ "${1:-}" == "--print-contract" ]]; then
  printf '%s\n' '{"auto_full203":true,"formal_epochs":200,"formal_trainer":"MicroBTSPPOBRTrainerScratch200","from_scratch":true,"fresh_optimizer":true,"gate_epochs":5,"gate_trainer":"MicroBTSPPOBRTrainerScratchGate5","isolated_root":"/path/to/example","random_seed":42}'
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
gate_trainer="MicroBTSPPOBRTrainerScratchGate5"
formal_trainer="MicroBTSPPOBRTrainerScratch200"
gate_results="${isolated_root}/nnUNet_results_obr_scratch_gate5"
formal_results="${isolated_root}/nnUNet_results_obr_scratch200"
gate_model="${gate_results}/${dataset}/${gate_trainer}__${plans}__${configuration}"
formal_model="${formal_results}/${dataset}/${formal_trainer}__${plans}__${configuration}"
gate_fold="${gate_model}/fold_0"
formal_fold="${formal_model}/fold_0"
evaluation_output="${isolated_root}/full203_obr_scratch200"
verification_manifest="${isolated_root}/artifacts/spp_obr_scratch_gate_manifest.json"
evaluation_manifest="${isolated_root}/artifacts/spp_obr_scratch_full203.json"
split="${preprocessed_root}/${dataset}/splits_final.json"

active_fold0_trainer() {
  "${python_bin}" - <<'PY'
from pathlib import Path

for command_file in Path("/proc").glob("[0-9]*/cmdline"):
    try:
        arguments = [
            value.decode("utf-8", errors="replace")
            for value in command_file.read_bytes().split(b"\0")
            if value
        ]
    except (OSError, PermissionError):
        continue
    for index, value in enumerate(arguments):
        if Path(value).name == "nnUNetv2_train" and arguments[index + 1:index + 4] == [
            "1", "3d_fullres", "0"
        ]:
            raise SystemExit(0)
raise SystemExit(1)
PY
}

[[ "${repo_root}" == "${isolated_root}" ]] || { echo "repository is outside OBR isolation root"; exit 2; }
[[ -f "${split}" ]] || { echo "fold split is missing: ${split}"; exit 2; }
[[ -d "${cache_root}" ]] || { echo "precomputed target cache is missing: ${cache_root}"; exit 2; }
for path in \
  "${gate_results}" \
  "${formal_results}" \
  "${evaluation_output}" \
  "${verification_manifest}" \
  "${evaluation_manifest}"; do
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-8}"
export MICROBT_PF_CACHE="${cache_root}"
unset MICROBT_PF_MONITOR_MANIFEST MICROBT_PF_MONITOR_OUTPUT

export nnUNet_results="${gate_results}"
"${train_bin}" 1 "${configuration}" "${fold}" \
  -tr "${gate_trainer}" -p "${plans}"

gate_log="$(ls -1t "${gate_fold}"/training_log_*.txt | head -1)"
"${python_bin}" - "${gate_log}" "${verification_manifest}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

log = Path(sys.argv[1])
output = Path(sys.argv[2])
text = log.read_text(encoding="utf-8", errors="replace")
if "checkpoint=none" not in text:
    raise SystemExit("scratch gate did not record checkpoint-free initialization")
rows = {}
current = None
exact = []
components = []
for line in text.splitlines():
    match = re.search(r": Epoch (\d+)\s*$", line)
    if match:
        current = int(match.group(1))
        rows.setdefault(current, {})
    if current is not None:
        match = re.search(r"train_loss ([0-9.eE+-]+)", line)
        if match:
            rows[current]["train_loss"] = float(match.group(1))
        match = re.search(r"Pseudo dice \[([^]]+)\]", line)
        if match:
            rows[current]["dice"] = [float(value.strip()) for value in match.group(1).split(",")]
        match = re.search(r"Epoch time: ([0-9.eE+-]+) s", line)
        if match:
            rows[current]["seconds"] = float(match.group(1))
    marker = "MicroBT-PF exact-fast telemetry: "
    if marker in line:
        exact.append(json.loads(line.split(marker, 1)[1]))
    marker = "MicroBT-PF component telemetry: "
    if marker in line:
        components.append(json.loads(line.split(marker, 1)[1]))

completed = [rows.get(epoch, {}) for epoch in range(5)]
if any(len(row.get("dice", ())) != 3 for row in completed):
    raise SystemExit("scratch gate is missing one or more three-region Dice rows")
if any(not math.isfinite(float(row.get("train_loss", float("nan")))) for row in completed):
    raise SystemExit("scratch gate has missing or nonfinite train loss")
if any(row.get("seconds", float("inf")) > 360.0 for row in completed[1:]):
    raise SystemExit("scratch gate exceeded the six-minute ordinary-epoch limit")
if len(exact) < 5 or any(item.get("nonfinite_steps") != 0 or item.get("skipped_steps") != 0 for item in exact[-5:]):
    raise SystemExit("scratch gate contains missing, nonfinite, or skipped optimizer telemetry")
train_components = [item for item in components if item.get("phase") == "train"]
if len(train_components) < 5:
    raise SystemExit("scratch gate is missing train component telemetry")
required = (
    "birth_reservation_raw_all_mean",
    "birth_existence_raw_all_mean",
    "birth_prominence_raw_all_mean",
)
if any(not math.isfinite(float(item[key])) for item in train_components[-5:] for key in required):
    raise SystemExit("scratch gate contains nonfinite reservation telemetry")
if not any(value > 0.0 for value in completed[-1]["dice"]):
    raise SystemExit("scratch gate shows no semantic learning signal by epoch 4")
payload = {
    "status": "passed",
    "from_scratch": True,
    "seed": 42,
    "epochs": completed,
    "last_exact_telemetry": exact[-1],
    "last_component_telemetry": train_components[-1],
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("SPP_OBR_SCRATCH_GATE_OK", json.dumps(payload, sort_keys=True))
PY

export nnUNet_results="${formal_results}"
"${train_bin}" 1 "${configuration}" "${fold}" \
  -tr "${formal_trainer}" -p "${plans}"

[[ -s "${formal_fold}/checkpoint_best.pth" ]] || { echo "missing checkpoint_best"; exit 3; }
[[ -s "${formal_fold}/checkpoint_final.pth" ]] || { echo "missing checkpoint_final"; exit 3; }
best_hash="$(sha256sum "${formal_fold}/checkpoint_best.pth" | cut -d' ' -f1)"
final_hash="$(sha256sum "${formal_fold}/checkpoint_final.pth" | cut -d' ' -f1)"

"${python_bin}" "${repo_root}/nnunetv2/scripts/run_microbt_pf_full203.py" \
  --model-folder "${formal_model}" --split "${split}" --output "${evaluation_output}" \
  --checkpoint-best-sha256 "${best_hash}" --checkpoint-final-sha256 "${final_hash}" \
  --admission-subject "EXAMPLE__ExampleCase-00003-000" --device cuda

"${python_bin}" - "${evaluation_manifest}" "${evaluation_output}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
root = Path(sys.argv[2])
payload = {
    "status": "complete",
    "from_scratch": True,
    "seed": 42,
    "best": json.loads((root / "best" / "summary.json").read_text(encoding="utf-8")),
    "final": json.loads((root / "final" / "summary.json").read_text(encoding="utf-8")),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
print("SPP_OBR_SCRATCH_FULL203_COMPLETE", output)
PY
