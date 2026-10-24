#!/usr/bin/env bash
set -euo pipefail

root=/path/to/example
session=microbt_umctf_hs_da8_fixed8

[[ $(<"${root}/artifacts/formal_umctf_hs_pipeline.exit") == 0 ]] || {
  echo "training exit is not zero" >&2
  exit 2
}
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists" >&2
  exit 2
fi
if pgrep -af 'MicroBTUnitMassSurvivalTrainerScratch60' | grep -v grep >/dev/null; then
  echo "training process is still active" >&2
  exit 2
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU already occupied" >&2
  exit 2
fi

tmux new-session -d -s "${session}" \
  "cd '${root}' && bash scripts/run_umctf_hs_fixed8_snapshots.sh; status=\$?; printf '%s\n' \"\$status\" > '${root}/artifacts/launcher_umctf_hs_fixed8.exit'; exit \$status"

sleep 3
tmux has-session -t "${session}"
