#!/usr/bin/env bash
# Run a Hydra training on a vast.ai instance with auto-resume and optional
# Google Drive checkpoint sync.
#
# Usage:  bash tools/train_remote.sh runs/<name> [--set key=value ...] [--init-weights ...]
#
# - Resumes automatically if runs/<name>/last.pt exists (preemption-safe).
# - Uses all visible GPUs via torchrun when there is more than one.
# - If an rclone remote named 'gdrive' is configured, syncs checkpoints and
#   metrics to gdrive:hydra_runs/<name> every 5 minutes and once at the end.
set -uo pipefail
cd /workspace/hydra

RUN_DIR="${1:?usage: train_remote.sh RUN_DIR [--set ...]}"
shift || true

RESUME=()
if [ -f "$RUN_DIR/last.pt" ]; then
    RESUME=(--resume "$RUN_DIR/last.pt")
    echo "resuming from $RUN_DIR/last.pt"
fi

SYNC_PID=""
if command -v rclone >/dev/null && rclone listremotes 2>/dev/null | grep -q '^gdrive:'; then
    (
        while true; do
            rclone copy "$RUN_DIR" "gdrive:hydra_runs/$(basename "$RUN_DIR")" \
                --include "*.pt" --include "*.json" --include "*.jsonl" -q || true
            sleep 300
        done
    ) &
    SYNC_PID=$!
    echo "checkpoint sync to gdrive:hydra_runs/$(basename "$RUN_DIR") every 300s"
fi

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
COMMON=(--config configs/default.toml
        --set data.corpus_dir=/workspace/data/MHD
        --set data.num_workers=4
        --set run.run_dir="$RUN_DIR")

if [ "$NGPU" -gt 1 ]; then
    torchrun --nproc_per_node="$NGPU" -m hydra.cli train "${COMMON[@]}" "$@" "${RESUME[@]}"
else
    python -u -m hydra.cli train "${COMMON[@]}" "$@" "${RESUME[@]}"
fi
STATUS=$?

if [ -n "$SYNC_PID" ]; then
    kill "$SYNC_PID" 2>/dev/null
    rclone copy "$RUN_DIR" "gdrive:hydra_runs/$(basename "$RUN_DIR")" \
        --include "*.pt" --include "*.json" --include "*.jsonl" -q || true
fi
exit $STATUS
