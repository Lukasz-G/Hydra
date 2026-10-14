#!/usr/bin/env bash
# Batch 7b: RNNTagger's per-epoch wall time, measured properly.
#
# The first attempt produced 45 epoch lines all carrying the SAME timestamp.
# tools/rnn_train_patience.py runs its child with -u, but was not itself run
# with -u, so its own stdout block-buffered into the pipe and every line
# reached the timestamper in one flush at kill time. Per-epoch deltas were
# therefore unrecoverable. `python -u` fixes it.
#
# Worse, the aborted run never reached its pkill, so a stray rnn-train.py
# kept the GPU for roughly the first half of the Pie training: Pie's epochs
# 1-30 ran at 18.8 min against 9.8 min for epochs 41-70, a 1.92x penalty.
# Only uncontended epochs are worth quoting as a cost.
#
# This runs each trainer for a fixed wall budget and then stops it, rather
# than waiting for a pattern to appear a given number of times. No counting,
# nothing to mis-parse. The kill travels in its own command whose line
# mentions no other copy of the pattern.
#
#   setsid nohup bash tools/batch7b.sh > /workspace/b7b_driver.log 2>&1 < /dev/null &
#
# Pulls nothing, destroys nothing.
set -u
cd /workspace/hydra
RESULTS=/workspace/batch7_results.log
RT=/workspace/RNNTagger
BUDGET=${BUDGET:-1200}          # seconds of training per system

say() { echo "=== B7B $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS"; }
stamp() { awk "{ printf \"%d %s\n\", systime(), \$0; fflush() }"; }

run_for() {   # run_for LABEL KILLPATTERN cmd...
    local label=$1 pattern=$2
    shift 2
    say "$label"
    echo "B7B_BUDGET $label seconds=$BUDGET" | tee -a "$RESULTS"
    "$@" 2>&1 | stamp >> "$RESULTS" &
    local pid=$!
    local i
    for i in $(seq 1 "$((BUDGET / 10))"); do
        sleep 10
        kill -0 "$pid" 2>/dev/null || break
    done
    pkill -f "$pattern" || true
    wait "$pid" 2>/dev/null || true
    say "${label}_END"
}

run_for rnn_tagger_timing "rnn-trai[n]\.py" \
    python -u tools/rnn_train_patience.py --patience 99 --min-epochs 99 -- \
    "$RT/PyRNN/rnn-train.py" --gpu 0 \
    --char_embedding_size 100 --char_recurrent_size 400 \
    --word_recurrent_size 400 --word_rnn_depth 2 --dropout_rate 0.5 \
    /workspace/rnnt/stratified/tagger/train.tsv \
    /workspace/rnnt/stratified/tagger/dev.tsv \
    /workspace/t2_rnn_tagger.par

sleep 20

run_for rnn_lemma_timing "nmt-trai[n]\.py" \
    python -u tools/rnn_train_patience.py --patience 99 --min-epochs 99 -- \
    "$RT/PyNMT/nmt-train.py" --gpu 0 --tie_embeddings \
    --word_emb_size 100 --enc_rnn_size 400 --dec_rnn_size 400 \
    --enc_depth 2 --dec_depth 2 --dropout_rate 0.5 \
    --max_src_vocab_size 500 --max_tgt_vocab_size 500 \
    /workspace/rnnt/stratified/lemmatizer/train.src \
    /workspace/rnnt/stratified/lemmatizer/train.tgt \
    /workspace/rnnt/stratified/lemmatizer/dev.src \
    /workspace/rnnt/stratified/lemmatizer/dev.tgt \
    /workspace/t2_rnn_lem.par

say "BATCH7B_DONE"
echo "RNNTagger timings measured on an idle GPU. Nothing pulled, nothing destroyed." \
    | tee -a "$RESULTS"
