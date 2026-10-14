#!/usr/bin/env bash
# Batch 7a: Job A again, after three bugs of mine in tools/batch7.sh.
#
#   1. hydra.cli train takes --config and a run.run_dir setting, not a
#      positional run directory. tools/train_remote.sh supplies both, which
#      is how batch5 invoked it; call that instead of the module.
#   2. `grep -c PATTERN FILE || echo 0` emits TWO lines when grep matches
#      nothing (its own 0, then the echo), so the arithmetic comparison died
#      with "syntax error in expression" and both RNNTagger timings aborted
#      at once. `grep PATTERN FILE | wc -l` always prints one number and
#      always exits 0.
#   3. A failed stage left the trainer's partial output in the log, which
#      is harmless but made the failure hard to read. Each stage here keeps
#      its own section marker.
#
# Timings are only worth having on an idle GPU, so this waits for the Pie
# run already in flight to finish before it starts. Do not touch
# /workspace/GO_CHUNK until this has written BATCH7A_DONE, or the chunk
# training will contend with the measurement.
#
#   setsid nohup bash tools/batch7a.sh > /workspace/b7a_driver.log 2>&1 < /dev/null &
#
# Pulls nothing, destroys nothing.
set -u
cd /workspace/hydra
RESULTS=/workspace/batch7_results.log
MARK=/workspace/markers7
RT=/workspace/RNNTagger
mkdir -p "$MARK"

say() { echo "=== B7A $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS"; }

stage() {
    local name=$1; shift
    if [ -f "$MARK/$name.done" ]; then echo "skip $name (done)"; return 0; fi
    say "$name"
    if "$@" >> "$RESULTS" 2>&1; then
        touch "$MARK/$name.done"; echo "STAGE_OK $name" | tee -a "$RESULTS"
    else
        rc=$?
        echo "STAGE_FAILED $name rc=$rc" | tee -a "$RESULTS"
        touch "$MARK/$name.failed"; return 1
    fi
}

timed() {
    local label=$1; shift
    local t0 t1 rc
    t0=$(date +%s); "$@"; rc=$?; t1=$(date +%s)
    echo "TIMING $label wall_seconds=$((t1 - t0)) rc=$rc" | tee -a "$RESULTS"
    return $rc
}

stamp() { awk "{ printf \"%d %s\n\", systime(), \$0; fflush() }"; }

# -------------------------------------------------- wait for an idle GPU
wait_for_pie() {
    local i
    for i in $(seq 1 2400); do        # up to 20 h
        if [ -f "$MARK/B_pie_stratified.done" ] || [ -f "$MARK/B_pie_stratified.failed" ]; then
            echo "pie stratified finished; GPU is free" | tee -a "$RESULTS"
            sleep 20
            return 0
        fi
        sleep 30
    done
    echo "gave up waiting for the pie run" | tee -a "$RESULTS"
    return 1
}
wait_for_pie || exit 1

# ---------------------------------------------------------- Hydra, 3 epochs
a_hydra() {
    rm -rf runs/t_hydra
    bash tools/train_remote.sh runs/t_hydra \
        --set data.split_mode=stratified \
        --set data.metadata_csv=meta/rem_metadata.csv \
        --set data.halo=64 \
        --set model.lemma_classifier=true \
        --set model.ctx_self_attention=true \
        --set model.joint_tag=true \
        --set train.batch_chunks=12 \
        --set train.amp_dtype=bf16 \
        --set train.max_epochs=3 \
        --set train.patience=999 2>&1 | stamp
    echo "## hydra epoch_seconds from metrics.jsonl"
    python -u tools/b7_epochs.py runs/t_hydra/metrics.jsonl
}
stage A2_hydra timed A2_hydra_3epochs a_hydra

# ------------------------------------------------- RNNTagger, 3 epoch lines
# Neither trainer has an --epochs flag, so run it and stop it once three
# epoch lines have appeared. The kill pattern is bracketed and appears
# nowhere else on this command line.
run_until_epochs() {   # run_until_epochs NEEDLE PATTERN NEEDED cmd...
    local needle=$1 pattern=$2 needed=$3
    shift 3
    local before have pid i
    before=$(grep -c "$needle" "$RESULTS" | head -1)
    "$@" 2>&1 | stamp &
    pid=$!
    for i in $(seq 1 480); do
        sleep 30
        have=$(grep "$needle" "$RESULTS" | wc -l)
        if [ "$((have - before))" -ge "$needed" ]; then break; fi
        if ! kill -0 "$pid" 2>/dev/null; then break; fi
    done
    pkill -f "$pattern" || true
    wait "$pid" 2>/dev/null || true
    echo "## stamped epoch lines above; the deltas are the per-epoch times"
}

a_rnn_tagger() {
    run_until_epochs "DevAccuracy" "rnn-trai[n]\.py" 3 \
        python tools/rnn_train_patience.py --patience 99 --min-epochs 99 -- \
        "$RT/PyRNN/rnn-train.py" --gpu 0 \
        --char_embedding_size 100 --char_recurrent_size 400 \
        --word_recurrent_size 400 --word_rnn_depth 2 --dropout_rate 0.5 \
        /workspace/rnnt/stratified/tagger/train.tsv \
        /workspace/rnnt/stratified/tagger/dev.tsv \
        /workspace/t_rnn_tagger.par
}
stage A2_rnn_tagger timed A2_rnn_tagger a_rnn_tagger

a_rnn_lemma() {
    run_until_epochs "DevAccuracy\|dev accuracy\|Dev loss" "nmt-trai[n]\.py" 3 \
        python tools/rnn_train_patience.py --patience 99 --min-epochs 99 -- \
        "$RT/PyNMT/nmt-train.py" --gpu 0 --tie_embeddings \
        --word_emb_size 100 --enc_rnn_size 400 --dec_rnn_size 400 \
        --enc_depth 2 --dec_depth 2 --dropout_rate 0.5 \
        --max_src_vocab_size 500 --max_tgt_vocab_size 500 \
        /workspace/rnnt/stratified/lemmatizer/train.src \
        /workspace/rnnt/stratified/lemmatizer/train.tgt \
        /workspace/rnnt/stratified/lemmatizer/dev.src \
        /workspace/rnnt/stratified/lemmatizer/dev.tgt \
        /workspace/t_rnn_lem.par
}
stage A2_rnn_lemma timed A2_rnn_lemma a_rnn_lemma

say "BATCH7A_DONE"
echo "Job A measured. Nothing pulled, nothing destroyed." | tee -a "$RESULTS"
