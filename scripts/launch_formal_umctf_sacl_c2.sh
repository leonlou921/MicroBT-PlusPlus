#!/usr/bin/env bash
set -euo pipefail

root=/path/to/example
session=microbt_umctf_b0_sacl_c2_da8_200

if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists" >&2
  exit 2
fi
if pgrep -af 'MicroBTUnitMassSACLTrainerScratch60' | grep -v grep >/dev/null; then
  echo "SACL trainer already exists" >&2
  exit 2
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU already occupied" >&2
  exit 2
fi

tmux new-session -d -s "${session}" \
  "cd '${root}' && bash scripts/run_microbt_umctf_sacl_c2_scratch200.sh; status=\$?; printf '%s\n' \"\$status\" > '${root}/artifacts/formal_umctf_sacl_c2_pipeline.exit'; exit \$status"

sleep 5
tmux has-session -t "${session}"
pgrep -af 'MicroBTUnitMassSACLTrainerScratch60' | head -n 8
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
tail -n 30 "${root}/logs/formal_umctf_sacl_c2_scratch200.log" 2>/dev/null || true
