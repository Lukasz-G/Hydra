#!/usr/bin/env bash
# Batch 5: normalised-target pretraining, regularisation bundle, Pie and
# RNNTagger baselines — all on the exact published splits.
#
# Run on the instance from /workspace/hydra:
#   setsid nohup bash tools/batch5.sh > /workspace/batch5_driver.log 2>&1 &
#
# Requires: bootstrap done (repo + ReM corpus), and /workspace/up/{s_joint,c_joint}
# holding config.json + split.json (+ vocab.json for c_joint) scp'd from local.
# Stages append to /workspace/batch_results.log and drop markers in
# /workspace/markers/; a failed stage marks FAILED and the chain continues.
set -u
cd /workspace/hydra
RESULTS=/workspace/batch_results.log
MARK=/workspace/markers
mkdir -p "$MARK"
say() { echo "=== B5 $* $(date +%H:%M) ===" | tee -a "$RESULTS"; }

stage() {  # stage NAME cmd...
    local name=$1; shift
    if [ -f "$MARK/$name.done" ]; then echo "skip $name (done)"; return 0; fi
    say "$name"
    if "$@" >> "$RESULTS" 2>&1; then
        touch "$MARK/$name.done"
    else
        echo "STAGE_FAILED $name" | tee -a "$RESULTS"
        touch "$MARK/$name.failed"
        return 1
    fi
}

# ---------------------------------------------------------------- stage 0: prep
# RNNTagger download runs in the background while everything else trains
if [ ! -f /workspace/rnntagger.zip ] && [ ! -f "$MARK/rnnzip.done" ]; then
    (curl -sSL -o /workspace/rnntagger.zip \
        "https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/data/RNNTagger-1.5.0.zip" \
        && touch "$MARK/rnnzip.done") &
fi

prep_mhdbdb() {
    rm -rf /workspace/mhdbdb_src /workspace/mhdbdb_txt
    git clone -q --depth 1 \
        https://github.com/Middle-High-German-Conceptual-Database/plain-txt-Texte \
        /workspace/mhdbdb_src
    mkdir -p /workspace/mhdbdb_txt
    find /workspace/mhdbdb_src -name '*.txt' -exec cp {} /workspace/mhdbdb_txt/ \;
    python tools/prepare_raw_corpus.py /workspace/mhdbdb_txt \
        /workspace/data/mhdbdb_clean meta/mhdbdb_exclude.txt
}
stage mhdbdb prep_mhdbdb

prep_baseline_data() {
    python tools/convert_to_pie.py /workspace/up/s_joint /workspace/pie/stratified
    python tools/convert_to_pie.py /workspace/up/c_joint /workspace/pie/chunk
    for p in stratified chunk; do
        python tools/convert_to_rnntagger.py /workspace/pie/$p /workspace/rnnt/$p
        python tools/convert_to_rnntagger.py /workspace/pie/$p /workspace/rnnt/${p}_pos --tag=pos
    done
}
stage baseline_data prep_baseline_data

stage pie_env bash -c '
    python -m venv /workspace/pie_env &&
    /workspace/pie_env/bin/pip install -q nlp-pie'

# ------------------------------------------------------- stage 1: pre_norm MLM
# (train_remote.sh supplies corpus_dir/num_workers/run_dir and auto-resume)
BIG="--set model.d_char=96 --set model.d_tok=384 --set model.d_model=768 \
 --set model.d_dec=320 --set model.char_tcn_dilations=[1,2,4,8,8] \
 --set model.ctx_tcn_dilations=[1,2,4,8,16] --set model.lemma_classifier=true \
 --set model.ctx_self_attention=true --set model.masked_lm=true \
 --set model.joint_tag=true --set train.batch_chunks=8"
STRAT="--set data.split_mode=stratified --set data.metadata_csv=meta/rem_metadata.csv \
 --set data.halo=64"
NORM="--set data.norm_lookup=meta/norm_lookup.tsv"

stage pre_norm bash tools/train_remote.sh runs/pre_norm $BIG $STRAT $NORM \
    --set data.extra_train_dir=/workspace/data/mhdbdb_clean \
    --set data.spelling_noise=1.0 --set data.spelling_noise_strength=0.3 \
    --set data.noise_rules=meta/rem_layers/noise_rules.json \
    --set data.word_type_min_freq=5 \
    --set model.pretrain_mlm=true \
    --set train.max_epochs=12 --set train.patience=999

# ------------------------------------------------ stage 2: s_norm fine-tune
stage s_norm bash tools/train_remote.sh runs/s_norm $BIG $STRAT $NORM \
    --set data.vocab_file=runs/pre_norm/vocab.json \
    --set train.max_epochs=30 --set train.patience=10 \
    --set train.cls_head_lr_mult=3.0 \
    --init-weights runs/pre_norm/last.pt
stage s_norm_sweep python tools/sweep_eval.py runs/s_norm

# --------------------------------------- stage 3+4: regularisation, small recipe
SMALL="--set model.ctx_tcn_dilations=[1,2,4,8,16] --set model.lemma_classifier=true \
 --set model.ctx_self_attention=true --set model.masked_lm=true \
 --set model.joint_tag=true --set train.batch_chunks=12"
REG="--set loss.label_smoothing=0.1 --set train.ema_decay=0.999 \
 --set data.spelling_noise=0.15 --set data.noise_rules=meta/rem_layers/noise_rules.json"

stage s_reg bash tools/train_remote.sh runs/s_reg $SMALL $STRAT $REG \
    --set train.max_epochs=30
stage s_reg_sweep python tools/sweep_eval.py runs/s_reg

stage c_reg bash tools/train_remote.sh runs/c_reg $SMALL $REG \
    --set data.split_mode=chunk --set data.halo=64 \
    --set train.max_epochs=30
stage c_reg_sweep python tools/sweep_eval.py runs/c_reg

# ---------------------------------------------------------- stage 5: Pie
run_pie() {
    local proto=$1
    /workspace/pie_env/bin/python tools/make_pie_settings.py \
        /workspace/pie/$proto /workspace/pie_models/$proto \
        /workspace/pie/$proto/settings.json pie_$proto
    /workspace/pie_env/bin/pie train /workspace/pie/$proto/settings.json
}
stage pie_stratified run_pie stratified
stage pie_chunk run_pie chunk

# ------------------------------------------------------ stage 6: RNNTagger
stage rnn_unzip bash -c '
    command -v perl >/dev/null || { apt-get update -qq && apt-get install -y -qq perl; }
    for i in $(seq 1 120); do [ -f "$0/rnnzip.done" ] && break; sleep 30; done
    [ -f "$0/rnnzip.done" ] || { echo "RNNTagger download never finished"; exit 1; }
    cd /workspace && python -c "import zipfile; zipfile.ZipFile(\"rnntagger.zip\").extractall()"
' "$MARK"

RT=/workspace/RNNTagger
train_tagger() {  # train_tagger DATA_DIR PARAM
    python $RT/PyRNN/rnn-train.py --gpu 0 \
        --char_embedding_size 100 --char_recurrent_size 400 \
        --word_recurrent_size 400 --word_rnn_depth 2 --dropout_rate 0.5 \
        "$1/tagger/train.tsv" "$1/tagger/dev.tsv" "$2"
}
train_lem() {  # train_lem DATA_DIR PARAM
    python $RT/PyNMT/nmt-train.py --gpu 0 --tie_embeddings \
        --word_emb_size 100 --enc_rnn_size 400 --dec_rnn_size 400 \
        --enc_depth 2 --dec_depth 2 --dropout_rate 0.5 \
        "$1/lemmatizer/train.src" "$1/lemmatizer/train.tgt" \
        "$1/lemmatizer/dev.src" "$1/lemmatizer/dev.tgt" "$2"
}
eval_chain() {  # eval_chain PROTO  (posmorph variant, end-to-end predicted tags)
    local d=/workspace/rnnt/$1
    cut -f1 $d/tagger/test.tsv > $d/test.tok
    python $RT/PyRNN/rnn-annotate.py --gpu 0 $d/tagger.par $d/test.tok > $d/test.tagged
    perl $RT/scripts/reformat.pl $d/test.tagged > $d/test.types
    python $RT/PyNMT/nmt-translate.py --gpu 0 --print_source $d/lem.par \
        $d/test.types > $d/test.lemmas
    perl $RT/scripts/lemma-lookup.pl $d/test.lemmas $d/test.tagged > $d/test.pred
    python tools/score_baseline.py /workspace/pie/$1/test.tsv $d/test.pred \
        /workspace/pie/$1/train.tsv
}
for proto in stratified chunk; do
    stage rnn_tagger_$proto train_tagger /workspace/rnnt/$proto /workspace/rnnt/$proto/tagger.par
    stage rnn_tagger_${proto}_pos train_tagger /workspace/rnnt/${proto}_pos /workspace/rnnt/${proto}_pos/tagger.par
    stage rnn_lem_$proto train_lem /workspace/rnnt/$proto /workspace/rnnt/$proto/lem.par
    stage rnn_eval_$proto eval_chain $proto
    stage rnn_eval_${proto}_pos bash -c "
        d=/workspace/rnnt/${proto}_pos
        cut -f1 \$d/tagger/test.tsv > \$d/test.tok
        python $RT/PyRNN/rnn-annotate.py --gpu 0 \$d/tagger.par \$d/test.tok > \$d/test.tagged
        python tools/score_baseline.py /workspace/pie/$proto/test.tsv \$d/test.tagged \
            /workspace/pie/$proto/train.tsv"
done

echo "BATCH5_DONE $(date +%H:%M)" | tee -a "$RESULTS"
