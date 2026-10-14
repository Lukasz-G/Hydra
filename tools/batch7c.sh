#!/usr/bin/env bash
# Batch 7c: inference throughput for all three systems on the same test set.
#
# Training cost is what a developer pays once. Inference cost is what a user
# pays every time the tagger is turned on a new manuscript, and nobody
# reports it. Pie is already measured at 8,893 tokens/s by
# tools/pie_predict.py; this adds Hydra and RNNTagger on the same GPU and
# the same material.
#
# Throughput does not depend on how well a model is trained, only on its
# shape, so the partially trained parameter files from the timing runs give
# the same answer as fully trained ones would.
#
# Waits for the RNNTagger training measurement to finish: a throughput
# figure taken from a busy GPU is worth nothing, which the Pie run already
# demonstrated the hard way.
#
#   setsid nohup bash tools/batch7c.sh > /workspace/b7c_driver.log 2>&1 < /dev/null &
#
# Pulls nothing, destroys nothing.
set -u
cd /workspace/hydra
RESULTS=/workspace/batch7_results.log
RT=/workspace/RNNTagger
D=/workspace/rnnt/stratified

say() { echo "=== B7C $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS"; }

wait_idle() {
    local i
    for i in $(seq 1 240); do
        if grep -q "BATCH7B_DONE" "$RESULTS" 2>/dev/null; then
            sleep 20
            return 0
        fi
        sleep 15
    done
    echo "gave up waiting for the training measurement" | tee -a "$RESULTS"
    return 1
}
wait_idle || exit 1

N_TOK=$(awk -F"\t" 'NF>=3' /workspace/pie/stratified/test.tsv | wc -l)
echo "B7C_TEST_TOKENS $N_TOK" | tee -a "$RESULTS"

# ----------------------------------------------------------------- Hydra
say hydra_inference
HYDRA_CK=""
for c in runs/t_hydra/best.pt runs/t_hydra/last.pt; do
    [ -f "$c" ] && HYDRA_CK="$c" && break
done
if [ -n "$HYDRA_CK" ]; then
    echo "checkpoint $HYDRA_CK" | tee -a "$RESULTS"
    t0=$(date +%s)
    python -u -m hydra.cli eval --model "$HYDRA_CK" --split test >> "$RESULTS" 2>&1
    t1=$(date +%s)
    echo "HYDRA_INFERENCE_SECONDS $((t1 - t0))" | tee -a "$RESULTS"
    awk -v s="$((t1 - t0))" -v n="$N_TOK" \
        'BEGIN { printf "HYDRA_INFERENCE_TOKENS_PER_SEC %.0f\n", n / s }' | tee -a "$RESULTS"
else
    echo "no hydra checkpoint; skipped" | tee -a "$RESULTS"
fi

# ------------------------------------------------------------- RNNTagger
# The published chain: tag, reformat into the lemmatiser's input, translate,
# then look the lemma up. Timed as a whole, because that is what a user runs.
say rnntagger_inference
if [ -f /workspace/t2_rnn_tagger.par ] && [ -f /workspace/t2_rnn_lem.par ]; then
    cut -f1 "$D/tagger/test.tsv" > "$D/t2.tok"
    t0=$(date +%s)
    python -u "$RT/PyRNN/rnn-annotate.py" --gpu 0 /workspace/t2_rnn_tagger.par \
        "$D/t2.tok" > "$D/t2.tagged" 2>> "$RESULTS"
    t1=$(date +%s)
    echo "RNNT_TAGGER_SECONDS $((t1 - t0))" | tee -a "$RESULTS"

    perl "$RT/scripts/reformat.pl" "$D/t2.tagged" > "$D/t2.types" 2>> "$RESULTS"
    t2=$(date +%s)
    python -u "$RT/PyNMT/nmt-translate.py" --gpu 0 --print_source /workspace/t2_rnn_lem.par \
        "$D/t2.types" > "$D/t2.lemmas" 2>> "$RESULTS"
    t3=$(date +%s)
    echo "RNNT_LEMMATISER_SECONDS $((t3 - t2))" | tee -a "$RESULTS"
    echo "RNNT_CHAIN_SECONDS $((t3 - t0))" | tee -a "$RESULTS"
    awk -v s="$((t3 - t0))" -v n="$N_TOK" \
        'BEGIN { printf "RNNT_INFERENCE_TOKENS_PER_SEC %.0f\n", n / s }' | tee -a "$RESULTS"
else
    echo "no RNNTagger parameter files; skipped" | tee -a "$RESULTS"
fi

say BATCH7C_DONE
echo "Inference measured for every system. Nothing pulled, nothing destroyed." \
    | tee -a "$RESULTS"
