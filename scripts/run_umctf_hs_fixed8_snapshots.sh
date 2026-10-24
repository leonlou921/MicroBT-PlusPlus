#!/usr/bin/env bash
set -euo pipefail

root=/path/to/example
python_bin=${MICROBT_UMCTF_PYTHON:-python3}
model_folder=${root}/nnUNet_results_umctf_hs_da8/Dataset001_Example/MicroBTUnitMassSurvivalTrainerScratch60__nnUNetResEncUNetMPlans__3d_fullres
fold_folder=${model_folder}/fold_0
manifest=${root}/artifacts/umctf_hs_fixed8_checkpoint_manifest.json

if [[ ${1:-} == --print-contract ]]; then
  printf '%s\n' '{"cases":8,"checkpoint_epochs":[60,100,150,200],"count_mae_lt":15,"evaluator":"evaluate_microbt_umctf_full_gate.py","frozen_thresholds":true,"trainer":"MicroBTUnitMassSurvivalTrainerScratch60"}'
  exit 0
fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--print-contract]" >&2; exit 2; }
cd "${root}"
[[ $(<artifacts/formal_umctf_hs_pipeline.exit) == 0 ]] || {
  echo "formal training did not complete cleanly" >&2
  exit 2
}

"${python_bin}" - "${root}/artifacts/source_manifest_umctf_hs.json" <<'PY'
import hashlib
import json
import sys

data = json.load(open(sys.argv[1]))
bad = []
for path, expected in data["files"].items():
    actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if actual != expected:
        bad.append((path, expected, actual))
if bad:
    raise SystemExit(f"source manifest drift: {bad}")
PY

[[ ! -e ${manifest} ]] || { echo "refusing to overwrite ${manifest}" >&2; exit 2; }
for epoch in 60 100 150 200; do
  [[ ! -e artifacts/umctf_hs_fixed8_epoch${epoch}.json ]] || {
    echo "refusing to overwrite epoch ${epoch} evaluation" >&2
    exit 2
  }
done

"${python_bin}" - "${fold_folder}" "${manifest}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

fold = Path(sys.argv[1])
output = Path(sys.argv[2])
rows = []
for epoch in (60, 100, 150, 200):
    path = fold / f"checkpoint_phase_{epoch:03d}.pth"
    if not path.is_file():
        raise SystemExit(f"missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("current_epoch", -1)) != epoch:
        raise SystemExit(f"epoch mismatch: {path}")
    if checkpoint.get("trainer_name") != "MicroBTUnitMassSurvivalTrainerScratch60":
        raise SystemExit(f"trainer mismatch: {path}")
    rows.append({
        "epoch": epoch,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
output.write_text(json.dumps({"schema_version": 1, "checkpoints": rows}, indent=2, sort_keys=True) + "\n")
PY

export PYTHONPATH=${root}${PYTHONPATH:+:${PYTHONPATH}}
export nnUNet_preprocessed=/path/to/example
export nnUNet_raw=/path/to/example
export nnUNet_results=${root}/nnUNet_results_umctf_hs_da8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

for epoch in 60 100 150 200; do
  checkpoint=$(printf 'checkpoint_phase_%03d.pth' "${epoch}")
  expected=$("${python_bin}" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(next(x["sha256"] for x in d["checkpoints"] if x["epoch"] == int(sys.argv[2])))' "${manifest}" "${epoch}")
  actual=$(sha256sum "${fold_folder}/${checkpoint}" | awk '{print $1}')
  [[ ${actual} == "${expected}" ]] || { echo "checkpoint drift at epoch ${epoch}" >&2; exit 2; }
  set +e
  "${python_bin}" scripts/evaluate_microbt_umctf_full_gate.py \
    --model-folder "${model_folder}" \
    --checkpoint "${checkpoint}" \
    --manifest input/fixed_cases_fold0.json \
    --output "artifacts/umctf_hs_fixed8_epoch${epoch}.json" \
    --count-mae-lt 15 \
    > "logs/umctf_hs_fixed8_epoch${epoch}.log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "${rc}" > "artifacts/umctf_hs_fixed8_epoch${epoch}.exit"
  if [[ ${rc} -ne 0 && ${rc} -ne 4 ]]; then
    echo "infrastructure failure at epoch ${epoch}: rc=${rc}" >&2
    exit "${rc}"
  fi
  [[ -s artifacts/umctf_hs_fixed8_epoch${epoch}.json ]]
done

"${python_bin}" - <<'PY'
import json
from pathlib import Path

root = Path("artifacts")
rows = []
for epoch in (60, 100, 150, 200):
    data = json.loads((root / f"umctf_hs_fixed8_epoch{epoch}.json").read_text())
    rows.append({
        "epoch": epoch,
        "status": data["status"],
        "checkpoint_sha256": data["checkpoint_sha256"],
        "metrics": data["metrics"],
        "causal_matrix": data["causal_matrix"],
    })
(root / "umctf_hs_fixed8_snapshots_summary.json").write_text(
    json.dumps({"schema_version": 1, "status": "evaluation_complete", "snapshots": rows}, indent=2, sort_keys=True) + "\n"
)
PY
