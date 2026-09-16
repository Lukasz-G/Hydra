#!/usr/bin/env bash
# Run a chain of training stages on a rented box, each firing the instant the
# previous one succeeds.
#
# Usage (on the instance):
#   nohup bash tools/vast_pipeline.sh PIPE_NAME "stage1" "stage2" ... &
# where each stage is:
#   NAME|CONFIG|INIT_FROM      (INIT_FROM may be empty, or a run dir whose
#                               best.pt warm-starts this stage)
#
# Why this exists. Pretraining for stage 2 finished at 08:52 and the fine-tune
# it feeds was launched by hand at 12:00 -- three hours of a $0.35/h box doing
# nothing but waiting to be noticed, because the chain ran through a human.
# Anything that can be sequenced without judgement should be.
#
# What it guarantees:
#   * the next stage starts within seconds of the previous one's exit
#   * a stage that produced no best.pt STOPS the chain instead of feeding
#     garbage forward (a warm start from a failed run is worse than no run)
#   * every finished stage is slimmed to model_only.pt and swept immediately,
#     so an interrupted pipeline loses at most the stage in flight
#   * PIPELINE_DONE / PIPELINE_FAILED is the last line of the log, so a local
#     watcher can block on it and wake the operator exactly once
set -uo pipefail
cd /workspace/hydra

PIPE="${1:?usage: vast_pipeline.sh PIPE_NAME STAGE...}"
shift
LOG="/workspace/${PIPE}_results.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

say () { echo "$*" | tee -a "$LOG"; }

say "PIPELINE_START $PIPE $(date -u +%H:%M:%S)"

for spec in "$@"; do
    name="${spec%%|*}"; rest="${spec#*|}"
    cfg="${rest%%|*}";  init="${rest#*|}"
    run="runs/$name"

    if grep -q "STAGE_DONE $name " "$LOG" 2>/dev/null; then
        say "=== SKIP $name (already done) ==="; continue
    fi

    extra=()
    if [ -n "$init" ]; then
        # refuse to warm-start from a stage that did not finish
        if [ ! -f "$init/best.pt" ]; then
            say "PIPELINE_FAILED $name: warm-start source $init/best.pt missing"
            exit 1
        fi
        extra=(--init-weights "$init/best.pt")
    fi

    say "=== $name  $(date -u +%H:%M:%S) ==="
    python -u -m hydra.cli train --config "$cfg" "${extra[@]}" \
        > "/workspace/$name.log" 2>&1
    if [ ! -f "$run/best.pt" ]; then
        say "PIPELINE_FAILED $name (no best.pt; see /workspace/$name.log)"
        exit 1
    fi

    # slim immediately: best.pt carries optimiser state and is ~3x the size,
    # too slow to pull when a box is about to be destroyed
    python - "$run" <<'PY'
import sys, torch
r = sys.argv[1]
p = torch.load(f"{r}/best.pt", map_location="cpu")
torch.save({"model": p["model"], "config": p["config"]}, f"{r}/model_only.pt")
PY
    # a pretraining stage has no tagging heads, so there is nothing to sweep
    if ! grep -q "pretrain_mlm = true" "$cfg"; then
        python -u tools/sweep_eval.py "$run" >> "$LOG" 2>&1
    fi
    say "STAGE_DONE $name  $(date -u +%H:%M:%S)"
done

say "PIPELINE_DONE $PIPE $(date -u +%H:%M:%S)"
