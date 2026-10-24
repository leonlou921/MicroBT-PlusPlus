#!/usr/bin/env bash
set -euo pipefail

# User-authorized retrospective convergence extension. This intentionally
# bypasses only the historical Stage-60 effectiveness gate; the model,
# checkpoint, optimizer, data, seed, loss, augmentation, and PolyLR horizon
# remain frozen.
isolated_root=/path/to/example
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
python_bin=${MICROBT_UMCTF_PYTHON:-python3}
results_root=${isolated_root}/nnUNet_results_umctf
model_folder=${results_root}/Dataset001_Example/MicroBTUnitMassTrainerScratch60__nnUNetResEncUNetMPlans__3d_fullres
fold_folder=${model_folder}/fold_0
checkpoint=${fold_folder}/checkpoint_final.pth
expected_checkpoint_sha256=${MICROBT_EXPECTED_CHECKPOINT_SHA256:?Set the expected checksum of your own checkpoint}
gate_file=${isolated_root}/artifacts/umctf_stage_gate60.json
contract_file=${isolated_root}/artifacts/umctf_convergence_extension_contract.json
log_file=${isolated_root}/logs/formal_umctf_converge200.log

[[ ${repo_root} == ${isolated_root} ]] || {
  echo "repository outside frozen isolation root" >&2
  exit 2
}
[[ -x ${python_bin} ]] || { echo "missing python: ${python_bin}" >&2; exit 2; }
[[ -s ${checkpoint} ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 2; }
[[ -s ${gate_file} ]] || { echo "missing historical gate: ${gate_file}" >&2; exit 2; }

actual_checkpoint_sha256=$(sha256sum "${checkpoint}" | awk '{print $1}')
[[ ${actual_checkpoint_sha256} == ${expected_checkpoint_sha256} ]] || {
  echo "checkpoint SHA256 mismatch" >&2
  exit 2
}

(
  cd "${isolated_root}"
  sha256sum -c artifacts/implementation_sha256.txt
)

"${python_bin}" - "${checkpoint}" "${gate_file}" "${contract_file}" <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import json
import socket
import sys
from pathlib import Path

import torch

checkpoint_path = Path(sys.argv[1]).resolve()
gate_path = Path(sys.argv[2]).resolve()
contract_path = Path(sys.argv[3]).resolve()

checkpoint = torch.load(checkpoint_path, map_location="cpu")
gate = json.loads(gate_path.read_text())
if checkpoint.get("current_epoch") != 60:
    raise RuntimeError(f"expected current_epoch=60, got {checkpoint.get('current_epoch')}")
if checkpoint.get("trainer_name") != "MicroBTUnitMassTrainerScratch60":
    raise RuntimeError(f"unexpected trainer={checkpoint.get('trainer_name')}")
optimizer_state = checkpoint.get("optimizer_state", {})
if len(optimizer_state.get("state", {})) != 256:
    raise RuntimeError("optimizer state is incomplete")
groups = optimizer_state.get("param_groups", [])
if len(groups) != 1 or abs(float(groups[0]["lr"]) - 0.007300795862273964) > 1e-12:
    raise RuntimeError("epoch-60 optimizer LR does not match the frozen horizon-200 trajectory")
if gate.get("status") != "failed" or gate.get("full203_unlocked") is not False:
    raise RuntimeError("historical Stage-60 gate state drifted")

sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
contract = {
    "schema_version": 1,
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "purpose": "retrospective_convergence_extension",
    "single_causal_variable": "formal_stop_epoch_60_to_200",
    "historical_gate_bypassed": True,
    "historical_gate_status": gate["status"],
    "historical_full203_unlocked": gate["full203_unlocked"],
    "source_checkpoint": str(checkpoint_path),
    "source_checkpoint_sha256": sha256,
    "source_current_epoch": checkpoint["current_epoch"],
    "target_epoch": 200,
    "poly_lr_horizon": 200,
    "optimizer_state_entries": len(optimizer_state["state"]),
    "optimizer_lr_at_resume": float(groups[0]["lr"]),
    "frozen_controls": [
        "dataset",
        "fold",
        "seed",
        "optimizer",
        "loss",
        "augmentation",
        "model",
        "decoder",
        "fixed8_cases",
        "evaluator",
    ],
}
contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
print(json.dumps(contract, sort_keys=True))
PY

available_kib=$(df --output=avail -k /path/to/example | tail -n 1 | tr -d ' ')
[[ ${available_kib} =~ ^[0-9]+$ ]] || { echo "unable to read free disk" >&2; exit 2; }
(( available_kib >= 5 * 1024 * 1024 )) || {
  echo "less than 5 GiB free; refusing convergence extension" >&2
  exit 2
}

export PYTHONPATH=${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export nnUNet_preprocessed=/path/to/example
export nnUNet_raw=/path/to/example
export nnUNet_results=${results_root}
export MICROBT_PF_CACHE=/path/to/example
export MICROBT_UMCTF_TARGET_EPOCHS=200
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-8}
export PYTHONUNBUFFERED=1
unset MICROBT_PF_MONITOR_MANIFEST MICROBT_PF_MONITOR_OUTPUT

"${python_bin}" "${repo_root}/scripts/run_phase_training_stage.py" \
  --continue-training --dataset 1 --configuration 3d_fullres --fold 0 \
  --trainer MicroBTUnitMassTrainerScratch60 --plans nnUNetResEncUNetMPlans \
  2>&1 | tee -a "${log_file}"

"${python_bin}" - "${fold_folder}" "${isolated_root}/artifacts/umctf_convergence_extension_completion.json" <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import torch

fold = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
final_path = fold / "checkpoint_final.pth"
checkpoint = torch.load(final_path, map_location="cpu")
if checkpoint.get("current_epoch") != 200:
    raise RuntimeError(f"expected current_epoch=200, got {checkpoint.get('current_epoch')}")

required = [
    fold / "checkpoint_phase_100.pth",
    fold / "checkpoint_phase_150.pth",
    fold / "checkpoint_phase_200.pth",
    final_path,
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise RuntimeError(f"missing convergence checkpoints: {missing}")

hashes = {}
for path in required:
    hashes[path.name] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
payload = {
    "schema_version": 1,
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": "training_complete",
    "current_epoch": checkpoint["current_epoch"],
    "optimizer_state_entries": len(checkpoint.get("optimizer_state", {}).get("state", {})),
    "checkpoints": hashes,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
