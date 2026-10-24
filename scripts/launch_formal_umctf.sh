#!/usr/bin/env bash
set -euo pipefail

root=/path/to/example
session=microbt_umctf60

if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists" >&2
  exit 2
fi
if pgrep -af 'MicroBTUnitMassTrainerScratch60' | grep -v grep >/dev/null; then
  echo "Unit-Mass trainer already exists" >&2
  exit 2
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU already occupied" >&2
  exit 2
fi

tmux new-session -d -s "${session}" \
  "cd '${root}' && bash scripts/run_microbt_umctf_scratch60.sh; status=\$?; printf '%s\n' \"\$status\" > '${root}/artifacts/formal_umctf_pipeline.exit'; exit \$status"

sleep 5
tmux has-session -t "${session}"
pgrep -af 'MicroBTUnitMassTrainerScratch60' | head -n 8
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
tail -n 30 "${root}/logs/formal_umctf_scratch60.log" 2>/dev/null || true
