#!/usr/bin/env bash
# Batch 7: the two things batch5 lost when instance 49727924 was destroyed
# before its log was pulled.
#
#   Job A  per-epoch wall time and inference throughput for Hydra, Pie and
#          RNNTagger on identical data and one instance type, which is the
#          cost comparison section 5 cannot currently state.
#   Job B  a full Pie retrain that SAVES ITS PREDICTIONS, so Pie's Lemma+POS
#          and joint cells can be scored by tools/score_baseline.py instead
#          of bounded. Scoring Pie with our own scorer also removes the
#          asymmetry in Table 2, where Pie was read off its native report
#          while RNNTagger went through ours.
#
# Run on the instance from /workspace/hydra:
#   setsid nohup bash tools/batch7.sh > /workspace/b7_driver.log 2>&1 < /dev/null &
# then verify it outlived the ssh session:
#   ps -eo pid,ppid,args | grep "[b]atch7.sh"      # ppid MUST be 1
#
# Requires: tools/vast_setup.sh done, and /workspace/up/{s_joint,c_joint}
# holding config.json + split.json + vocab.json scp'd from local.
#
# This script PULLS NOTHING and DESTROYS NOTHING. It writes numbers to
# /workspace/batch7_results.log as it goes, so the log is worth more than the
# checkpoints if the box dies. A human reads it and then destroys.
#
# The chunk-protocol Pie run is gated on /workspace/GO_CHUNK existing: at
# $0.31-0.34/h the two full runs together cost more than the account held
# when this was launched.
set -u
cd /workspace/hydra
RESULTS=/workspace/batch7_results.log
MARK=/workspace/markers7
PIE=/workspace/pie_env/bin
RT=/workspace/RNNTagger
mkdir -p "$MARK" /workspace/pie_pred

say() { echo "=== B7 $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS"; }

stage() {  # stage NAME cmd...
    local name=$1; shift
    if [ -f "$MARK/$name.done" ]; then echo "skip $name (done)"; return 0; fi
    say "$name"
    if "$@" >> "$RESULTS" 2>&1; then
        touch "$MARK/$name.done"
        echo "STAGE_OK $name" | tee -a "$RESULTS"
    else
        rc=$?
        echo "STAGE_FAILED $name rc=$rc" | tee -a "$RESULTS"
        touch "$MARK/$name.failed"
        return 1
    fi
}

timed() {  # timed LABEL cmd...   wall seconds for the whole command
    local label=$1; shift
    local t0 t1 rc
    t0=$(date +%s)
    "$@"
    rc=$?
    t1=$(date +%s)
    echo "TIMING $label wall_seconds=$((t1 - t0)) rc=$rc" | tee -a "$RESULTS"
    return $rc
}

# Prefix every line with epoch seconds, so an external trainer's own progress
# lines become per-epoch timings. Both RNNTagger trainers and pie print one
# line per epoch, and the deltas between them are what batch5 lost.
stamp() { awk "{ printf \"%d %s\n\", systime(), \$0; fflush() }"; }

# ---------------------------------------------------------- stage 0: what box
record_env() {
    echo "## environment"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    python -u tools/b7_env.py
}
stage env record_env

# ------------------------------------------------------ stage 1: identical data
prep_data() {
    python tools/convert_to_pie.py /workspace/up/s_joint /workspace/pie/stratified
    python tools/convert_to_pie.py /workspace/up/c_joint /workspace/pie/chunk
    for p in stratified chunk; do
        python tools/convert_to_rnntagger.py /workspace/pie/$p /workspace/rnnt/$p
        python tools/convert_to_rnntagger.py /workspace/pie/$p /workspace/rnnt/${p}_pos --tag=pos
    done
    wc -l /workspace/pie/stratified/*.tsv /workspace/pie/chunk/*.tsv
}
stage data prep_data

pie_install() {
    python -m venv /workspace/pie_env
    "$PIE/pip" install -q papie
    "$PIE/pip" show papie | head -3
}
stage pie_env pie_install

# What the installed pie actually offers, so the prediction step is written
# against it rather than guessed at.
discover_pie() {
    echo "## pie CLI"
    "$PIE/pie" --help 2>&1 | head -40
    echo "## pie tag"
    "$PIE/pie" tag --help 2>&1 | head -40
    echo "## pie python surface"
    "$PIE/python" -u tools/b7_pie_api.py
}
stage discover discover_pie

rnn_fetch() {
    command -v perl >/dev/null || { apt-get update -qq && apt-get install -y -qq perl; }
    cd /workspace
    if [ ! -f rnntagger.zip ]; then
        curl -sSL -o rnntagger.zip \
            "https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/data/RNNTagger-1.5.0.zip"
    fi
    if [ ! -d RNNTagger ]; then
        python -c "import zipfile; zipfile.ZipFile('rnntagger.zip').extractall()"
    fi
    ls RNNTagger | head
    cd /workspace/hydra
}
stage rnn_fetch rnn_fetch

# ======================================================== JOB A: per-epoch cost
# Three epochs, not one: an epoch time has variance and a median of three is
# worth the extra minutes. Time to the best dev score matters more than time
# per epoch and needs a full run, which Job B supplies for Pie.

jobA_hydra() {
    rm -rf runs/t_hydra
    python -u -m hydra.cli train runs/t_hydra \
        --set data.corpus_dir=/workspace/data/MHD \
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
stage A_hydra timed A_hydra_3epochs jobA_hydra

# Neither RNNTagger trainer has an --epochs flag (both default to a fixed 50
# with no early stop), so take three epoch lines and stop the trainer. The
# kill pattern is bracketed AND this command line mentions no other literal
# copy of it, per the pkill rule in the vast.ai notes.
jobA_rnn() {  # jobA_rnn NAME PATTERN NEEDED cmd...
    local name=$1 pattern=$2 needed=$3
    shift 3
    local before pid i have
    before=$(grep -c "DevAccuracy\|dev accuracy\|Dev " "$RESULTS" 2>/dev/null || echo 0)
    "$@" 2>&1 | stamp &
    pid=$!
    for i in $(seq 1 480); do
        sleep 30
        have=$(grep -c "DevAccuracy\|dev accuracy\|Dev " "$RESULTS" 2>/dev/null || echo 0)
        if [ "$((have - before))" -ge "$needed" ]; then break; fi
        if ! kill -0 "$pid" 2>/dev/null; then break; fi
    done
    pkill -f "$pattern" || true
    wait "$pid" 2>/dev/null || true
    echo "## $name: stamped epoch lines above; deltas are the per-epoch times"
}

a_rnn_tagger() {
    jobA_rnn rnn_tagger "rnn-trai[n]\.py" 3 \
        python tools/rnn_train_patience.py --patience 99 --min-epochs 99 -- \
        "$RT/PyRNN/rnn-train.py" --gpu 0 \
        --char_embedding_size 100 --char_recurrent_size 400 \
        --word_recurrent_size 400 --word_rnn_depth 2 --dropout_rate 0.5 \
        /workspace/rnnt/stratified/tagger/train.tsv \
        /workspace/rnnt/stratified/tagger/dev.tsv \
        /workspace/t_rnn_tagger.par
}
stage A_rnn_tagger timed A_rnn_tagger a_rnn_tagger

a_rnn_lemma() {
    jobA_rnn rnn_lemma "nmt-trai[n]\.py" 3 \
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
stage A_rnn_lemma timed A_rnn_lemma a_rnn_lemma

# ================================================= JOB B: Pie, with predictions
pie_full() {  # pie_full PROTO
    local proto=$1
    "$PIE/python" tools/make_pie_settings.py \
        /workspace/pie/$proto /workspace/pie_models/$proto \
        /workspace/pie/$proto/settings.json pie_$proto
    "$PIE/pie" train /workspace/pie/$proto/settings.json 2>&1 | stamp
    echo "## model files written"
    ls -l /workspace/pie_models/$proto || true
}
stage B_pie_stratified timed B_pie_stratified pie_full stratified

# The chunk run is the other 12-13 hours. Start it by touching the gate:
#   touch /workspace/GO_CHUNK
wait_for_chunk_gate() {
    local i
    for i in $(seq 1 960); do          # up to 8 h of waiting, then give up
        if [ -f /workspace/GO_CHUNK ]; then return 0; fi
        sleep 30
    done
    echo "no GO_CHUNK after 8 h; skipping the chunk protocol" | tee -a "$RESULTS"
    return 1
}
if wait_for_chunk_gate; then
    stage B_pie_chunk timed B_pie_chunk pie_full chunk
fi

say "BATCH7_DONE"
echo "NOTHING PULLED, NOTHING DESTROYED. Fetch $RESULTS and /workspace/pie_models first." \
    | tee -a "$RESULTS"
