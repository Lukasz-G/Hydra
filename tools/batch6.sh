#!/usr/bin/env bash
# Batch 6: the combined-tag ablation and the explicit item-count head.
#
# Both answer questions worth more than the ~0.3pp effects recent runs chased:
# three seeds of the SAME config spread 0.42pp on lemma, 0.86pp on multi-item
# lemma and 1.11pp on OOV lemma, so anything smaller than that is unmeasurable
# with one run. Hence three seeds each.
#
#   A. combined-tag ablation (paper SS6.2) -- from scratch, 3 seeds.
#      Baseline is the EXISTING s_joint at 3 seeds, so only this arm is needed.
#      Paper-critical, so it runs FIRST: credit is tight (~$8.20 at launch for
#      a ~$7.20 batch) and a credit-out must not land mid-experiment.
#   B. explicit item-count head -- warm-started from s_crf, 3 seeds.
#      Targets the diagnosed 13% under-segmentation on multi-item tokens.
#
# Every stage writes a slim model_only.pt and appends to ONE results log as
# soon as it finishes, so an interrupted batch loses at most the run in
# flight. (batch5, 2026-09-08: a destroy with no pull lost a whole batch.)
set -uo pipefail
cd /workspace/hydra

LOG=/workspace/batch6_results.log
SEEDS="1337 1 2"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

stage () {                      # stage <name> <config> <seed> [init-weights]
    local name="$1" cfg="$2" seed="$3" init="${4:-}"
    local run="runs/$name"
    if grep -q "STAGE_DONE $name" "$LOG" 2>/dev/null; then
        echo "=== SKIP $name (already done) ===" | tee -a "$LOG"; return 0
    fi
    echo "=== $name  $(date +%H:%M) ===" | tee -a "$LOG"
    local extra=()
    [ -n "$init" ] && extra=(--init-weights "$init")
    python -u -m hydra.cli train --config "$cfg" \
        --set run.run_dir="$run" --set run.seed="$seed" \
        "${extra[@]}" > "/workspace/$name.log" 2>&1
    if [ ! -f "$run/best.pt" ]; then
        echo "STAGE_FAILED $name (no best.pt)" | tee -a "$LOG"; return 1
    fi
    # slim checkpoint immediately: full best.pt carries optimiser state and is
    # ~3x the size, too slow to pull when credit is running out
    python - "$run" <<'PY'
import sys, torch
r = sys.argv[1]
p = torch.load(f"{r}/best.pt", map_location="cpu")
torch.save({"model": p["model"], "config": p["config"]}, f"{r}/model_only.pt")
PY
    python -u tools/sweep_eval.py "$run" >> "$LOG" 2>&1
    echo "STAGE_DONE $name  $(date +%H:%M)" | tee -a "$LOG"
}

echo "BATCH6_START $(date)" | tee -a "$LOG"

# A -- combined-tag ablation, from scratch (paper-critical, runs first)
for s in $SEEDS; do
    stage "comb_s$s" configs/stratified_combined_remote.toml "$s"
done

# B -- explicit item-count head, warm-started from s_crf
for s in $SEEDS; do
    stage "cnt_s$s" configs/stratified_count_remote.toml "$s" /workspace/s_crf/model_only.pt
done

echo "BATCH6_DONE $(date)" | tee -a "$LOG"
