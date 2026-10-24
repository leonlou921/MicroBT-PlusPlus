#!/usr/bin/env bash
set -euo pipefail

isolated_root="/path/to/example"
source_root="/path/to/example"
source_repo="${source_root}/nnUNet"
source_checkpoint="${MICROBT_SPP_SOURCE_CHECKPOINT:-${source_root}/nnUNet_results_scheduled_exact_200_monitored/Dataset001_Example/MicroBTPFTrainerA3ScheduledExact__nnUNetResEncUNetMPlans__3d_fullres/fold_0/checkpoint_final.pth}"
python_bin="${MICROBT_SPP_PYTHON:-python3}"
train_bin="${MICROBT_SPP_TRAIN_BIN:-${python_bin%/python}/nnUNetv2_train}"

if [[ "${1:-}" == "--print-contract" ]]; then
  printf '%s\n' '{"auto_full203":true,"formal_epochs":200,"formal_trainer":"MicroBTSPPTrainerScheduledExact","gate_epochs":5,"gate_trainer":"MicroBTSPPTrainerGate5","isolated_root":"/path/to/example","source_checkpoint":"checkpoint_final.pth"}'
  exit 0
fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--print-contract]" >&2; exit 2; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
preprocessed_root="/path/to/example"
raw_root="/path/to/example"
gate_results="${isolated_root}/nnUNet_results_spp_gate5"
formal_results="${isolated_root}/nnUNet_results_spp_200"
dataset="Dataset001_Example"
plans="nnUNetResEncUNetMPlans"
configuration="3d_fullres"
fold="0"
gate_trainer="MicroBTSPPTrainerGate5"
formal_trainer="MicroBTSPPTrainerScheduledExact"
gate_model="${gate_results}/${dataset}/${gate_trainer}__${plans}__${configuration}"
gate_fold="${gate_model}/fold_0"
formal_model="${formal_results}/${dataset}/${formal_trainer}__${plans}__${configuration}"
formal_fold="${formal_model}/fold_0"
migrated_checkpoint="${isolated_root}/artifacts/pf_final_to_spp.pth"
gate_report="${isolated_root}/artifacts/spp_gate_report.json"
evaluation_output="${isolated_root}/full203_paired_spp"
evaluation_manifest="${isolated_root}/artifacts/full203_posttrain_spp.json"
monitor_manifest="${repo_root}/monitor/fixed_cases_fold0.json"

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

[[ "${repo_root}" == "${isolated_root}" ]] || { echo "repository is outside SPP isolation root"; exit 2; }
[[ -f "${source_checkpoint}" ]] || { echo "source checkpoint is missing: ${source_checkpoint}"; exit 2; }
[[ -f "${monitor_manifest}" ]] || { echo "fixed-case monitor manifest is missing"; exit 2; }
for path in \
  "${gate_results}" \
  "${formal_results}" \
  "${migrated_checkpoint}" \
  "${gate_report}" \
  "${evaluation_output}" \
  "${evaluation_manifest}"; do
  [[ ! -e "${path}" ]] || { echo "refusing to overwrite ${path}"; exit 2; }
done
if active_fold0_trainer; then
  echo "another fold-0 trainer is active"
  exit 2
fi

source_head_before="$(git -C "${source_repo}" rev-parse HEAD)"
source_status_before="$(git -C "${source_repo}" status --porcelain=v1 --untracked-files=no | sha256sum)"
mkdir -p "${isolated_root}/artifacts" "${isolated_root}/logs"

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export nnUNet_preprocessed="${preprocessed_root}"
export nnUNet_raw="${raw_root}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-8}"
export MICROBT_PF_CACHE="${source_root}/cache"
unset MICROBT_PF_MONITOR_MANIFEST MICROBT_PF_MONITOR_OUTPUT

"${python_bin}" -m nnunetv2.training.nnUNetTrainer.microbt_spp_checkpoint \
  "${source_checkpoint}" "${migrated_checkpoint}"

[[ "$(git -C "${source_repo}" rev-parse HEAD)" == "${source_head_before}" ]] || { echo "source HEAD changed"; exit 2; }
[[ "$(git -C "${source_repo}" status --porcelain=v1 --untracked-files=no | sha256sum)" == "${source_status_before}" ]] || { echo "source worktree changed"; exit 2; }

export nnUNet_results="${gate_results}"
"${train_bin}" 1 "${configuration}" "${fold}" \
  -tr "${gate_trainer}" -p "${plans}" \
  -pretrained_weights "${migrated_checkpoint}"

gate_log="$(ls -1t "${gate_fold}"/training_log_*.txt | head -1)"
"${python_bin}" - "${gate_log}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
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
    raise SystemExit("SPP gate is missing one or more three-region Dice rows")
if any(not all(math.isfinite(value) and value > 0 for value in row["dice"]) for row in completed[1:]):
    raise SystemExit("SPP gate contains nonpositive or nonfinite ordinary-epoch Dice")
means = [sum(row["dice"]) / 3.0 for row in completed]
if any(value < 0.8 * means[0] for value in means[1:]):
    raise SystemExit(f"SPP gate mean Dice fell below stability floor: {means}")
if any(row.get("seconds", float("inf")) > 360 for row in completed[1:]):
    raise SystemExit("SPP gate exceeded the six-minute ordinary-epoch limit")
if len(exact) < 5 or any(item.get("nonfinite_steps") != 0 or item.get("skipped_steps") != 0 for item in exact[-5:]):
    raise SystemExit("SPP gate contains missing, nonfinite, or skipped optimizer telemetry")
train_components = [item for item in components if item.get("phase") == "train"]
if len(train_components) < 5:
    raise SystemExit("SPP gate is missing train component telemetry")
if any(not math.isfinite(float(item["false_persistence_weighted_all_mean"])) for item in train_components[-5:]):
    raise SystemExit("SPP false-persistence telemetry is nonfinite")
print("SPP_GATE_CURVE_OK", json.dumps({"mean_dice": means}, sort_keys=True))
PY

"${python_bin}" - "${gate_model}" "${gate_report}" <<'PY'
import json
import sys
from pathlib import Path
import torch
from nnunetv2.scripts.run_microbt_pf_full203 import _admission_view, _infer_case, _make_dataset, _make_predictor

model = Path(sys.argv[1])
output = Path(sys.argv[2])
subject = "EXAMPLE__ExampleCase-00003-000"
predictor = _make_predictor(model, "checkpoint_final.pth", 0, fast=False, batch_size=1, device=torch.device("cuda"))
dataset = _make_dataset(predictor, (subject,))
restored, target, spacing, elapsed, peak = _infer_case(predictor, dataset, subject)
metrics = _admission_view(restored, target, spacing)["metrics"]
diagnostics = restored.get("diagnostics", {})
if int(restored["raw_logits"].shape[0]) != 4:
    raise SystemExit("SPP gate did not return four raw channels")
if diagnostics.get("marker_outside_support") != 0 or diagnostics.get("support_conserved") is not True:
    raise SystemExit(f"SPP support invariant failed: {diagnostics}")
target_count = int(metrics["target_count"])
predicted_count = int(restored["count"])
if predicted_count > max(3, 3 * target_count):
    raise SystemExit(f"SPP count gate failed: predicted={predicted_count}, target={target_count}")
singleton_count = sum(int(item["volume"]) <= 1 for item in restored["objects"])
singleton_fraction = singleton_count / max(1, predicted_count)
if singleton_fraction > 0.10:
    raise SystemExit(f"SPP singleton gate failed: {singleton_fraction}")
payload = {
    "status": "passed",
    "subject_id": subject,
    "predicted_count": predicted_count,
    "target_count": target_count,
    "singleton_fraction": singleton_fraction,
    "elapsed_seconds": float(elapsed),
    "peak_memory_bytes": int(peak),
    "diagnostics": diagnostics,
    "metrics": metrics,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
print("SPP_GATE_FIXED_CASE_OK", json.dumps({key: payload[key] for key in ("subject_id", "predicted_count", "target_count", "singleton_fraction")}, sort_keys=True))
PY

export nnUNet_results="${formal_results}"
export MICROBT_PF_MONITOR_MANIFEST="${monitor_manifest}"
export MICROBT_PF_MONITOR_OUTPUT="${formal_fold}/monitor"
"${train_bin}" 1 "${configuration}" "${fold}" \
  -tr "${formal_trainer}" -p "${plans}" \
  -pretrained_weights "${migrated_checkpoint}"

[[ -f "${formal_fold}/checkpoint_best.pth" ]] || { echo "formal SPP training lacks checkpoint_best"; exit 3; }
[[ -f "${formal_fold}/checkpoint_final.pth" ]] || { echo "formal SPP training lacks checkpoint_final"; exit 3; }
formal_log="$(ls -1t "${formal_fold}"/training_log_*.txt | head -1)"

"${python_bin}" "${repo_root}/scripts/launch_microbt_pf_fast_full203.py" \
  --isolated-root "${isolated_root}" \
  --repo-root "${repo_root}" \
  --model-folder "${formal_model}" \
  --training-log "${formal_log}" \
  --split "${preprocessed_root}/${dataset}/splits_final.json" \
  --output "${evaluation_output}" \
  --manifest "${evaluation_manifest}" \
  --python "${python_bin}" \
  --admission-subject "EXAMPLE__ExampleCase-00003-000"
